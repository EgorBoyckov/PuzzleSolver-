#!/usr/bin/env python3
"""Проверка критериев приёмки этапа 1 на синтетическом датасете:
  - сегментация: ≥ 99.5% деталей найдено
  - классификация угол/край/центр: ≥ 99.9%

Генерирует синтетический пазл реалистичного размера партий (60-120 деталей
на кадр, как в протоколе съёмки), прогоняет preprocess -> segment -> describe
по всем партиям и сравнивает результат с точной разметкой генератора.

Использование:
    python scripts/stage1_acceptance.py --pieces 2000 [--seed 42]
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
from app.pipeline.preprocess import preprocess_frame  # noqa: E402
from app.pipeline.segment import segment_pieces  # noqa: E402
from app.synth.dataset import SyntheticDatasetConfig, generate_dataset  # noqa: E402

sys.path.insert(0, str(REPO_ROOT / "server" / "tests"))
from pipeline_test_utils import match_pieces_to_ground_truth  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Проверка критериев приёмки этапа 1")
    parser.add_argument("--pieces", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=str, default=None, help="Папка датасета (иначе — временная)")
    args = parser.parse_args()

    out_dir = Path(args.out) if args.out else REPO_ROOT / "data" / "debug" / "stage1_acceptance"
    config = SyntheticDatasetConfig(total_pieces=args.pieces, out_dir=out_dir, seed=args.seed)

    t0 = time.time()
    meta = generate_dataset(config)
    print(f"Датасет: {meta['total_pieces_actual']} деталей, {meta['num_batches']} партий "
          f"(сгенерировано за {time.time()-t0:.1f}с)")

    total_gt = 0
    total_found = 0
    total_matched_for_kind = 0
    total_kind_correct = 0
    kind_confusion: dict[str, dict[str, int]] = {}

    for batch_summary in meta["batches"]:
        batch_number = batch_summary["batch_number"]
        with (out_dir / batch_summary["annotations"]).open(encoding="utf-8") as f:
            gt_batch = json.load(f)

        photo = out_dir / batch_summary["image"]
        frame, rectified, valid_mask = preprocess_frame(photo, "acceptance", batch_number)
        if not frame.accepted:
            print(f"  batch {batch_number}: ОТБРАКОВАН ({frame.rejection_reason}) — считаем все детали не найденными")
            total_gt += len(gt_batch["pieces"])
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
        gt_kind_by_id = {gt["id"]: gt["kind"] for gt in gt_batch["pieces"]}

        batch_found = len(matches)
        batch_kind_correct = 0
        for gt_id, (piece, _dist) in matches.items():
            gt_kind = gt_kind_by_id[gt_id]
            detected_kind = piece.kind.value if piece.kind else "none"
            kind_confusion.setdefault(gt_kind, {}).setdefault(detected_kind, 0)
            kind_confusion[gt_kind][detected_kind] += 1
            if detected_kind == gt_kind:
                batch_kind_correct += 1

        total_gt += len(gt_batch["pieces"])
        total_found += batch_found
        total_matched_for_kind += batch_found
        total_kind_correct += batch_kind_correct

        print(
            f"  batch {batch_number}: {batch_found}/{len(gt_batch['pieces'])} найдено, "
            f"kind {batch_kind_correct}/{batch_found} верно"
        )

    found_rate = total_found / total_gt if total_gt else 0.0
    kind_accuracy = total_kind_correct / total_matched_for_kind if total_matched_for_kind else 0.0

    print()
    print("=" * 60)
    print(f"Всего эталонных деталей: {total_gt}")
    print(f"Найдено (segment):        {total_found}  ({found_rate*100:.3f}%)  критерий: >= 99.5%  {'OK' if found_rate>=0.995 else 'FAIL'}")
    print(f"Классификация (describe): {total_kind_correct}/{total_matched_for_kind}  ({kind_accuracy*100:.3f}%)  критерий: >= 99.9%  {'OK' if kind_accuracy>=0.999 else 'FAIL'}")
    print("=" * 60)
    print("Матрица ошибок классификации (истина -> предсказано):")
    for gt_kind, row in kind_confusion.items():
        print(f"  {gt_kind}: {row}")


if __name__ == "__main__":
    main()
