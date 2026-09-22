#!/usr/bin/env python3
"""Проверка критерия приёмки этапа 2 (locate) на синтетическом датасете:
  - топ-3 в текстурных зонах: >= 90%
  - топ-3 в однотонных зонах: >= 50%

Прогоняет полный конвейер (preprocess -> segment -> describe -> locate,
включая цветокоррекцию образца и повторный поиск для неуверенных деталей)
и сравнивает результат с точной разметкой генератора. Зона ячейки образца
("текстурная"/"однотонная") определяется по локальной дисперсии яркости.

Использование:
    python scripts/stage2_acceptance.py --pieces 2000 [--seed 42]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "server"))

from app.core.config import get_config  # noqa: E402
from app.pipeline.describe import describe_piece  # noqa: E402
from app.pipeline.locate import build_reference_index, locate_piece, refit_reference_colors  # noqa: E402
from app.pipeline.preprocess import preprocess_frame  # noqa: E402
from app.pipeline.segment import segment_pieces  # noqa: E402
from app.synth.dataset import SyntheticDatasetConfig, generate_dataset  # noqa: E402

sys.path.insert(0, str(REPO_ROOT / "server" / "tests"))
from pipeline_test_utils import match_pieces_to_ground_truth  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Проверка критерия приёмки этапа 2 (locate)")
    parser.add_argument("--pieces", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--var-threshold", type=float, default=150.0, help="Порог дисперсии для текстурная/однотонная зона")
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()

    out_dir = Path(args.out) if args.out else REPO_ROOT / "data" / "debug" / "stage2_acceptance"
    config = SyntheticDatasetConfig(total_pieces=args.pieces, out_dir=out_dir, seed=args.seed)

    t0 = time.time()
    meta = generate_dataset(config)
    print(f"Датасет: {meta['total_pieces_actual']} деталей, {meta['num_batches']} партий, "
          f"сетка {meta['rows']}x{meta['cols']} (сгенерировано за {time.time()-t0:.1f}с)")

    reference_bgr = cv2.imread(str(out_dir / "catalog" / "box.jpg"))
    ref_index = build_reference_index(reference_bgr, meta["rows"], meta["cols"])
    gray_ref = cv2.cvtColor(reference_bgr, cv2.COLOR_BGR2GRAY)

    all_pieces = []
    frame_lookup = {}
    per_batch_gt = {}
    per_batch_frame = {}

    lc = get_config().section("locate")

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
            locate_piece(p, rectified, ref_index)

        frame_lookup[batch_number] = rectified
        per_batch_gt[batch_number] = gt_batch
        per_batch_frame[batch_number] = frame
        all_pieces.extend(pieces)
        print(f"  batch {batch_number}: {len(pieces)} деталей обработано")

    t1 = time.time()
    refitted = refit_reference_colors(
        ref_index, all_pieces, frame_lookup,
        confidence_threshold=lc["color_refit_confidence_threshold"],
        min_samples=lc["color_refit_min_samples"],
    )
    if refitted is not None:
        ref_index = refitted
        low_conf = [p for p in all_pieces if not p.location_candidates or p.location_candidates[0][3] < lc["color_refit_confidence_threshold"]]
        print(f"Цветокоррекция применена, повторный поиск для {len(low_conf)} неуверенных деталей...")
        for p in low_conf:
            locate_piece(p, frame_lookup[p.batch_number], ref_index)
    print(f"locate + цветокоррекция заняли {time.time()-t1:.1f}с")

    tex_top1 = tex_top3 = tex_n = 0
    mono_top1 = mono_top3 = mono_n = 0

    for batch_number, gt_batch in per_batch_gt.items():
        frame = per_batch_frame[batch_number]
        batch_pieces = [p for p in all_pieces if p.batch_number == batch_number]
        tolerance_px = 0.5 * meta["piece_size_mm"] / frame.mm_per_pixel
        matches = match_pieces_to_ground_truth(
            batch_pieces, gt_batch, frame.marker_bbox_px, meta["table_px_per_mm"], 1.0 / frame.mm_per_pixel, tolerance_px
        )
        gt_rc = {gt["id"]: (gt["grid_row"], gt["grid_col"]) for gt in gt_batch["pieces"]}

        for gt_id, (piece, _dist) in matches.items():
            if not piece.location_candidates:
                continue
            r, c = gt_rc[gt_id]
            x0, y0, x1, y1 = ref_index.cell_bounds_px(r, c)
            var = float(gray_ref[y0:y1, x0:x1].var())
            cands = [(rr, cc) for rr, cc, _rot, _conf in piece.location_candidates]
            is_top1, is_top3 = cands[0] == (r, c), (r, c) in cands
            if var >= args.var_threshold:
                tex_n += 1
                tex_top1 += is_top1
                tex_top3 += is_top3
            else:
                mono_n += 1
                mono_top1 += is_top1
                mono_top3 += is_top3

    tex_rate3 = tex_top3 / tex_n if tex_n else 0.0
    mono_rate3 = mono_top3 / mono_n if mono_n else 0.0

    print()
    print("=" * 70)
    print(f"Текстурные зоны:   n={tex_n:5d}  top1={tex_top1/max(tex_n,1)*100:.2f}%  top3={tex_rate3*100:.2f}%  критерий >=90%  {'OK' if tex_rate3>=0.90 else 'FAIL'}")
    print(f"Однотонные зоны:   n={mono_n:5d}  top1={mono_top1/max(mono_n,1)*100:.2f}%  top3={mono_rate3*100:.2f}%  критерий >=50%  {'OK' if mono_rate3>=0.50 else 'FAIL'}")
    print("=" * 70)


if __name__ == "__main__":
    main()
