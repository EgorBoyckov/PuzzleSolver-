#!/usr/bin/env python3
"""Приёмка привязки к образцу (locate) и глобальной сборки (solve) на
синтетическом датасете.

Критерии этапа 2 (только сходство с образцом, топ-3 ячейки):
  - текстурные зоны: >= 90%
  - однотонные зоны: >= 50%
Итог сборки (locate + форма сторон, взаимно однозначная раскладка): доля
деталей, поставленных в верную ячейку с верным поворотом — отдельно по
зонам. Зона ячейки ("текстурная"/"однотонная") определяется по локальной
дисперсии яркости образца.

Использование:
    python scripts/stage2_acceptance.py --pieces 2000 [--seed 42]
    python scripts/stage2_acceptance.py --pieces 5000 --seed 7 --workers 2
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "server"))
sys.path.insert(0, str(REPO_ROOT / "server" / "tests"))

from app.pipeline.describe import describe_piece  # noqa: E402
from app.pipeline.locate import build_reference_index, locate_appearances, piece_appearance  # noqa: E402
from app.pipeline.preprocess import preprocess_frame  # noqa: E402
from app.pipeline.segment import segment_pieces  # noqa: E402
from app.pipeline.solve import solve_layout  # noqa: E402
from app.synth.dataset import SyntheticDatasetConfig, generate_dataset  # noqa: E402
from pipeline_test_utils import match_pieces_to_ground_truth, rotation_matches_gt  # noqa: E402

_ctx: dict = {}


def _init(out_dir: str, meta: dict) -> None:
    _ctx["out_dir"] = Path(out_dir)
    _ctx["meta"] = meta
    reference_bgr = cv2.imread(str(Path(out_dir) / "catalog" / "box.jpg"))
    _ctx["index"] = build_reference_index(reference_bgr, meta["rows"], meta["cols"])


def _process_batch(batch_summary: dict):
    """preprocess -> segment -> describe -> признаки для locate одной партии
    (в отдельном процессе; кадр дальше не передаётся)."""
    out_dir, meta, index = _ctx["out_dir"], _ctx["meta"], _ctx["index"]
    batch_number = batch_summary["batch_number"]
    with (out_dir / batch_summary["annotations"]).open(encoding="utf-8") as f:
        gt_batch = json.load(f)
    frame, rectified, valid_mask = preprocess_frame(out_dir / batch_summary["image"], "acceptance", batch_number)
    if not frame.accepted:
        return batch_number, frame.rejection_reason, [], [], {}
    pieces = segment_pieces(
        rectified, frame.mm_per_pixel, "acceptance", batch_number,
        marker_bbox_px=frame.marker_bbox_px, valid_mask=valid_mask,
    )
    used, apps = [], []
    for p in pieces:
        describe_piece(p, rectified, frame.mm_per_pixel)
        app = piece_appearance(p, rectified, index)
        if app is not None:
            used.append(p)
            apps.append(app)
    tolerance_px = 0.5 * meta["piece_size_mm"] / frame.mm_per_pixel
    matches = match_pieces_to_ground_truth(
        pieces, gt_batch, frame.marker_bbox_px, meta["table_px_per_mm"], 1.0 / frame.mm_per_pixel, tolerance_px
    )
    gt_by_id = {gt["id"]: gt for gt in gt_batch["pieces"]}
    truth = {piece.id: gt_by_id[gt_id] for gt_id, (piece, _d) in matches.items()}
    return batch_number, None, used, apps, truth, len(pieces)


def main() -> None:
    parser = argparse.ArgumentParser(description="Проверка locate + глобальной сборки на синтетике")
    parser.add_argument("--pieces", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--var-threshold", type=float, default=150.0, help="Порог дисперсии для текстурная/однотонная зона")
    parser.add_argument("--workers", type=int, default=2, help="Процессов для обработки партий (каждый ~1-2 ГБ ОЗУ)")
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()

    out_dir = Path(args.out) if args.out else REPO_ROOT / "data" / "debug" / "stage2_acceptance"
    config = SyntheticDatasetConfig(total_pieces=args.pieces, out_dir=out_dir, seed=args.seed)

    t0 = time.time()
    meta = generate_dataset(config)
    print(f"Датасет: {meta['total_pieces_actual']} деталей, {meta['num_batches']} партий, "
          f"сетка {meta['rows']}x{meta['cols']} (сгенерировано за {time.time()-t0:.1f}с)")

    reference_bgr = cv2.imread(str(out_dir / "catalog" / "box.jpg"))
    index = build_reference_index(reference_bgr, meta["rows"], meta["cols"])
    gray_ref = cv2.cvtColor(reference_bgr, cv2.COLOR_BGR2GRAY)

    t1 = time.time()
    pieces, apps, truth, found = [], [], {}, 0
    with ProcessPoolExecutor(args.workers, initializer=_init, initargs=(str(out_dir), meta)) as ex:
        for res in ex.map(_process_batch, meta["batches"]):
            batch_number, rejected = res[0], res[1]
            if rejected:
                print(f"  batch {batch_number}: ОТБРАКОВАН ({rejected})")
                continue
            _, _, used, batch_apps, batch_truth, n_found = res
            pieces.extend(used)
            apps.extend(batch_apps)
            truth.update(batch_truth)
            found += n_found
    t_catalog = time.time() - t1
    print(f"Каталогизация: {found} деталей найдено, {len(pieces)} с углами/сторонами, "
          f"{len(truth)} сопоставлено с разметкой ({t_catalog:.1f}с, {args.workers} проц.)")

    t2 = time.time()
    located = locate_appearances(pieces, apps, index)
    t_locate = time.time() - t2
    t3 = time.time()
    layout = solve_layout(located)
    t_solve = time.time() - t3

    def zone(r: int, c: int) -> bool:
        x0, y0, x1, y1 = index.cell_bounds_px(r, c)
        return float(gray_ref[y0:y1, x0:x1].var()) >= args.var_threshold

    stats = {True: [0, 0, 0], False: [0, 0, 0]}
    for p in located.pieces:
        gt = truth.get(p.id)
        if gt is None or not p.location_candidates:
            continue
        rc = (gt["grid_row"], gt["grid_col"])
        cands = [(r, c) for r, c, _rot, _conf in p.location_candidates]
        s = stats[zone(*rc)]
        s[0] += 1
        s[1] += cands[0] == rc
        s[2] += rc in cands

    kind_total = sum(1 for p in pieces if p.id in truth)
    kind_ok = sum(1 for p in pieces if p.id in truth and p.kind is not None and p.kind.value == truth[p.id]["kind"])

    asm = {True: [0, 0, 0], False: [0, 0, 0]}
    for pl in layout.placements:
        p = located.pieces[pl.piece_index]
        gt = truth.get(p.id)
        if gt is None:
            continue
        s = asm[zone(gt["grid_row"], gt["grid_col"])]
        s[0] += 1
        if (gt["grid_row"], gt["grid_col"]) == (pl.row, pl.col):
            s[1] += 1
            s[2] += rotation_matches_gt(p, pl.rotation * 90, gt["rotation_deg"])

    def pct(a: int, b: int) -> float:
        return 100.0 * a / max(b, 1)

    total_cells = meta["rows"] * meta["cols"]
    tn, t1_, t3_ = stats[True]
    mn, m1, m3 = stats[False]
    an = asm[True][0] + asm[False][0]
    arot = asm[True][2] + asm[False][2]
    print()
    print("=" * 78)
    print(f"Этап 1: найдено {found} деталей на {total_cells} ячеек; тип угол/край/центр верен у "
          f"{pct(kind_ok, kind_total):.3f}% сопоставленных с разметкой")
    print("locate (только сходство с образцом):")
    print(f"  Текстурные зоны: n={tn:5d}  top1={pct(t1_, tn):6.2f}%  top3={pct(t3_, tn):6.2f}%  критерий top3>=90%  {'OK' if t3_ >= 0.9 * tn else 'FAIL'}")
    print(f"  Однотонные зоны: n={mn:5d}  top1={pct(m1, mn):6.2f}%  top3={pct(m3, mn):6.2f}%  критерий top3>=50%  {'OK' if m3 >= 0.5 * mn else 'FAIL'}")
    print("Глобальная сборка (верная ячейка / верная ячейка+поворот):")
    for name, key in (("Текстурные зоны", True), ("Однотонные зоны", False)):
        n, cell_ok, rot_ok = asm[key]
        print(f"  {name}: n={n:5d}  ячейка={pct(cell_ok, n):6.2f}%  ячейка+поворот={pct(rot_ok, n):6.2f}%")
    print(f"  ВСЕГО: {arot} из {an} деталей на своём месте с верным поворотом ({pct(arot, an):.2f}%), "
          f"найдено деталей {found} из {total_cells}")
    print(f"Время: каталогизация {t_catalog:.0f}с, locate {t_locate:.1f}с, сборка {t_solve:.1f}с")
    print("=" * 78)


if __name__ == "__main__":
    main()
