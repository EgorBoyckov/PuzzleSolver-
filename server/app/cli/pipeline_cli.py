"""CLI для прогона конвейера (preprocess -> segment -> describe) на папке с
фото партий, без веб-клиента.

Пример:
    python -m app.cli.pipeline_cli --input path/to/batch_photos --puzzle-id demo --out data/debug/pipeline_run
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from app.core.logging import setup_logging
from app.pipeline.describe import describe_piece
from app.pipeline.preprocess import preprocess_frame
from app.pipeline.schemas import PieceKind
from app.pipeline.segment import segment_pieces

logger = logging.getLogger(__name__)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def run_pipeline_on_folder(input_dir: Path, puzzle_id: str, out_dir: Path) -> dict:
    debug_dir = out_dir / "debug"
    catalog_dir = out_dir / "catalog"
    debug_dir.mkdir(parents=True, exist_ok=True)
    catalog_dir.mkdir(parents=True, exist_ok=True)

    photos = sorted(p for p in input_dir.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)
    if not photos:
        raise FileNotFoundError(f"В папке {input_dir} не найдено изображений ({IMAGE_EXTENSIONS})")

    all_pieces = []
    frames_report = []

    for batch_number, photo_path in enumerate(photos, start=1):
        frame, rectified, valid_mask = preprocess_frame(
            photo_path, puzzle_id, batch_number, debug_dir=debug_dir / "preprocess"
        )
        frames_report.append(frame.model_dump())

        if not frame.accepted or rectified is None:
            logger.warning("batch %d (%s) отбракован: %s", batch_number, photo_path.name, frame.rejection_reason)
            continue

        pieces = segment_pieces(
            rectified,
            frame.mm_per_pixel,
            puzzle_id,
            batch_number,
            marker_bbox_px=frame.marker_bbox_px,
            valid_mask=valid_mask,
            output_dir=catalog_dir,
            debug_dir=debug_dir / "segment",
        )

        for piece in pieces:
            describe_piece(piece, rectified, frame.mm_per_pixel, debug_dir=debug_dir / "describe")

        all_pieces.extend(pieces)
        logger.info("batch %d (%s): %d деталей", batch_number, photo_path.name, len(pieces))

    kind_counts = {k.value: 0 for k in PieceKind}
    kind_counts["unknown"] = 0
    for p in all_pieces:
        kind_counts[p.kind.value if p.kind else "unknown"] += 1

    summary = {
        "puzzle_id": puzzle_id,
        "frames_total": len(photos),
        "frames_accepted": sum(1 for f in frames_report if f["accepted"]),
        "frames_rejected": sum(1 for f in frames_report if not f["accepted"]),
        "pieces_found": len(all_pieces),
        "pieces_suspect": sum(1 for p in all_pieces if p.is_suspect),
        "kind_counts": kind_counts,
        "frames": frames_report,
    }

    with (catalog_dir / "pieces.json").open("w", encoding="utf-8") as f:
        json.dump([p.model_dump() for p in all_pieces], f, ensure_ascii=False, indent=2)
    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    return summary


def main() -> None:
    setup_logging()
    parser = argparse.ArgumentParser(description="Прогон конвейера PuzzleVision на папке с фото партий")
    parser.add_argument("--input", type=str, required=True, help="Папка с фото партий (jpg/png)")
    parser.add_argument("--puzzle-id", type=str, default="cli-run", help="ID пазла (для нумерации деталей)")
    parser.add_argument("--out", type=str, required=True, help="Папка вывода (каталог + отладочные изображения)")
    args = parser.parse_args()

    summary = run_pipeline_on_folder(Path(args.input), args.puzzle_id, Path(args.out))
    logger.info("Готово: %s", json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
