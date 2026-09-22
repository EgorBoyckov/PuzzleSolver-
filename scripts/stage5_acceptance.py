#!/usr/bin/env python3
"""Честная проверка этапа 5 (no_reference: сборка без образца) на
синтетическом датасете. Как и для этапа 3, точный текст критериев приёмки
этого этапа не был доступен в контексте этой сессии — ниже честные
измеренные показатели, а не проверка против опубликованного порога.

Измеряется:
  - build_frame_chain: доля последовательных пар в собранной цепочке рамки,
    чьи детали действительно физически соседние по эталонной раскладке
    (манхэттенское расстояние клеток сетки == 1), и доля найденных рамочных
    деталей, вошедших в цепочку.
  - cluster_islands: структурная статистика (число островов, их размеры) —
    нет прямого способа объективно оценить "качество" кластеризации без
    эталонной сегментации картинки коробки на смысловые зоны, которой в
    синтетическом датасете нет.

Использование:
    python scripts/stage5_acceptance.py --pieces 600 [--seed 42]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "server"))

from app.pipeline.describe import describe_piece  # noqa: E402
from app.pipeline.no_reference import build_frame_chain, cluster_islands  # noqa: E402
from app.pipeline.preprocess import preprocess_frame  # noqa: E402
from app.pipeline.schemas import PieceKind  # noqa: E402
from app.pipeline.segment import segment_pieces  # noqa: E402

sys.path.insert(0, str(REPO_ROOT / "server" / "tests"))
from pipeline_test_utils import match_pieces_to_ground_truth  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Честная проверка этапа 5 (no_reference)")
    parser.add_argument("--pieces", type=int, default=600)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()

    out_dir = Path(args.out) if args.out else REPO_ROOT / "data" / "debug" / "stage5_acceptance"
    from app.synth.dataset import SyntheticDatasetConfig, generate_dataset

    config = SyntheticDatasetConfig(total_pieces=args.pieces, out_dir=out_dir, seed=args.seed)

    t0 = time.time()
    meta = generate_dataset(config)
    print(f"Датасет: {meta['total_pieces_actual']} деталей, {meta['num_batches']} партий, "
          f"сетка {meta['rows']}x{meta['cols']} (сгенерировано за {time.time()-t0:.1f}с)")

    all_pieces = []
    gt_lookup: dict[str, tuple[int, int]] = {}

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

        all_pieces.extend(pieces)
        print(f"  batch {batch_number}: {len(pieces)} деталей обработано")

    # --- build_frame_chain ---
    t1 = time.time()
    chain = build_frame_chain(all_pieces)
    frame_pieces_total = sum(1 for p in all_pieces if p.kind in (PieceKind.CORNER, PieceKind.EDGE) and len(p.sides) == 4)

    checked = correct = 0
    for a, b in zip(chain, chain[1:]):
        rc_a, rc_b = gt_lookup.get(a), gt_lookup.get(b)
        if rc_a is None or rc_b is None:
            continue
        checked += 1
        manhattan = abs(rc_a[0] - rc_b[0]) + abs(rc_a[1] - rc_b[1])
        correct += manhattan == 1
    chain_precision = correct / checked if checked else 0.0
    coverage = len(set(chain)) / frame_pieces_total if frame_pieces_total else 0.0
    print(f"build_frame_chain: {len(chain)} деталей в цепочке из {frame_pieces_total} рамочных "
          f"(покрытие {coverage:.2%}) за {time.time()-t1:.1f}с, точность соседства {correct}/{checked} = {chain_precision:.2%}")

    # --- cluster_islands ---
    t2 = time.time()
    islands = cluster_islands(all_pieces)
    sizes = sorted((len(v) for v in islands.values()), reverse=True)
    interior_total = sum(1 for p in all_pieces if p.kind == PieceKind.CENTER and p.embedding)
    print(f"cluster_islands: {len(islands)} островов из {interior_total} внутренних деталей за {time.time()-t2:.1f}с, "
          f"размеры {sizes}")

    print()
    print("=" * 70)
    print(f"build_frame_chain: покрытие {coverage:.2%}, точность соседства {chain_precision:.2%} (n={checked})")
    print(f"cluster_islands:   {len(islands)} островов, размеры {sizes}")
    print("=" * 70)


if __name__ == "__main__":
    main()
