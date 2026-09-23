#!/usr/bin/env python3
"""Честная проверка этапа 3 (match + plan + recognize) на синтетическом
датасете. В отличие от stage1/stage2_acceptance.py, здесь нет строгого
критерия приёмки из технического задания — точный текст критериев этапа
match/plan/recognize не был доступен в контексте этой сессии (в отличие
от stage1 ≥99.5%/≥99.9% и stage2 ≥90%/≥50%, явно указанных в ТЗ и
процитированных ранее). Числа ниже — честные измеренные показатели
реализации, а не проверка против опубликованного порога.

Измеряется:
  - match: точность топ-1 совпадения стороны — доля рёбер-кандидатов
    (build_edge_matches), чьи детали действительно физически смежные по
    эталонной раскладке (манхэттенское расстояние клеток сетки == 1).
  - plan: доля рамочных деталей (corner/edge) среди первых N шагов плана
    (frame_first должно проявляться на практике, не только в юнит-тестах).
  - recognize: точность самоидентификации — повторное распознавание того
    же кадра, что использовался для каталогизации, по одной геометрии
    (без доступа к исходному сопоставлению id<->деталь).

Использование:
    python scripts/stage3_acceptance.py --pieces 600 [--seed 42]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "server"))

from app.pipeline.describe import describe_piece  # noqa: E402
from app.pipeline.locate import build_reference_index, locate_pieces  # noqa: E402
from app.pipeline.match import build_edge_matches  # noqa: E402
from app.pipeline.plan import next_steps_with_catalog  # noqa: E402
from app.pipeline.preprocess import preprocess_frame  # noqa: E402
from app.pipeline.recognize import recognize_pieces_on_frame  # noqa: E402
from app.pipeline.schemas import PieceKind  # noqa: E402
from app.pipeline.segment import segment_pieces  # noqa: E402
from app.pipeline.solve import solve_layout  # noqa: E402

sys.path.insert(0, str(REPO_ROOT / "server" / "tests"))
from pipeline_test_utils import match_pieces_to_ground_truth  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Честная проверка этапа 3 (match+plan+recognize)")
    parser.add_argument("--pieces", type=int, default=600)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()

    out_dir = Path(args.out) if args.out else REPO_ROOT / "data" / "debug" / "stage3_acceptance"
    from app.synth.dataset import SyntheticDatasetConfig, generate_dataset

    config = SyntheticDatasetConfig(total_pieces=args.pieces, out_dir=out_dir, seed=args.seed)

    t0 = time.time()
    meta = generate_dataset(config)
    print(f"Датасет: {meta['total_pieces_actual']} деталей, {meta['num_batches']} партий, "
          f"сетка {meta['rows']}x{meta['cols']} (сгенерировано за {time.time()-t0:.1f}с)")

    reference_bgr = cv2.imread(str(out_dir / "catalog" / "box.jpg"))
    ref_index = build_reference_index(reference_bgr, meta["rows"], meta["cols"])

    all_pieces = []
    frame_lookup = {}
    gt_lookup: dict[str, tuple[int, int]] = {}
    first_batch_rectified = None
    first_batch_mm_per_pixel = None

    for batch_summary in meta["batches"]:
        batch_number = batch_summary["batch_number"]
        with (out_dir / batch_summary["annotations"]).open(encoding="utf-8") as f:
            gt_batch = json.load(f)
        photo = out_dir / batch_summary["image"]
        frame, rectified, valid_mask = preprocess_frame(photo, "acceptance", batch_number)
        if not frame.accepted:
            print(f"  batch {batch_number}: ОТБРАКОВАН ({frame.rejection_reason})")
            continue

        pieces = segment_pieces(
            rectified, frame.mm_per_pixel, "acceptance", batch_number,
            marker_bbox_px=frame.marker_bbox_px, valid_mask=valid_mask,
        )
        for p in pieces:
            describe_piece(p, rectified, frame.mm_per_pixel)

        tolerance_px = 0.5 * meta["piece_size_mm"] / frame.mm_per_pixel
        matches = match_pieces_to_ground_truth(
            pieces, gt_batch, frame.marker_bbox_px, meta["table_px_per_mm"], 1.0 / frame.mm_per_pixel, tolerance_px
        )
        gt_rc = {gt["id"]: (gt["grid_row"], gt["grid_col"]) for gt in gt_batch["pieces"]}
        for gt_id, (piece, _dist) in matches.items():
            gt_lookup[piece.id] = gt_rc[gt_id]

        frame_lookup[batch_number] = rectified
        all_pieces.extend(pieces)
        print(f"  batch {batch_number}: {len(pieces)} деталей обработано")
        if first_batch_rectified is None:
            first_batch_rectified, first_batch_mm_per_pixel = rectified, frame.mm_per_pixel

    t1 = time.time()
    # Позиции деталей для match — из итоговой раскладки глобальной сборки.
    layout = solve_layout(locate_pieces(all_pieces, frame_lookup, ref_index))
    print(f"locate + сборка заняли {time.time()-t1:.1f}с, поставлено {len(layout.placements)} деталей")

    # --- match ---
    t2 = time.time()
    edges = build_edge_matches(all_pieces)
    checked = correct = 0
    for e in edges:
        rc_a, rc_b = gt_lookup.get(e.piece_a), gt_lookup.get(e.piece_b)
        if rc_a is None or rc_b is None:
            continue
        checked += 1
        manhattan = abs(rc_a[0] - rc_b[0]) + abs(rc_a[1] - rc_b[1])
        correct += manhattan == 1
    match_precision = correct / checked if checked else 0.0
    print(f"match: {len(edges)} рёбер-кандидатов за {time.time()-t2:.1f}с, "
          f"точность топ-1 = {correct}/{checked} = {match_precision:.2%}")

    # --- plan ---
    steps = next_steps_with_catalog(edges, all_pieces, max_steps=len(all_pieces))
    pieces_by_id = {p.id: p for p in all_pieces}
    frame_kinds = (PieceKind.CORNER, PieceKind.EDGE)
    n_check = min(50, len(steps))
    early_frame_fraction = (
        sum(
            1
            for s in steps[:n_check]
            if pieces_by_id.get(s.piece_a, None) and pieces_by_id[s.piece_a].kind in frame_kinds
            or pieces_by_id.get(s.piece_b, None) and pieces_by_id[s.piece_b].kind in frame_kinds
        )
        / n_check
        if n_check
        else 0.0
    )
    total_frame_pieces = sum(1 for p in all_pieces if p.kind in frame_kinds)
    print(f"plan: {len(steps)} шагов; среди первых {n_check} рамочные детали участвуют в "
          f"{early_frame_fraction:.2%} шагов (всего рамочных деталей в каталоге: {total_frame_pieces}/{len(all_pieces)})")

    # --- recognize (самоидентификация на том же кадре первой партии) ---
    catalog_for_recognize = [p for p in all_pieces if not p.is_suspect and len(p.sides) == 4]
    t3 = time.time()
    results = recognize_pieces_on_frame(
        first_batch_rectified, catalog_for_recognize, first_batch_mm_per_pixel, puzzle_id="acceptance", batch_number=0,
    )
    first_batch_ids = {p.id for p in catalog_for_recognize if p.batch_number == meta["batches"][0]["batch_number"]}
    recognized_correct = len(set(results.keys()) & first_batch_ids)
    recognize_rate = recognized_correct / len(first_batch_ids) if first_batch_ids else 0.0
    print(f"recognize: самоидентификация {recognized_correct}/{len(first_batch_ids)} = "
          f"{recognize_rate:.2%} за {time.time()-t3:.1f}с (та же партия, что и в каталоге)")

    print()
    print("=" * 70)
    print(f"match  топ-1 точность:            {match_precision:.2%}  (n={checked})")
    print(f"plan   доля рамочных в первых {n_check:3d}: {early_frame_fraction:.2%}")
    print(f"recognize самоидентификация:      {recognize_rate:.2%}  (n={len(first_batch_ids)})")
    print("=" * 70)


if __name__ == "__main__":
    main()
