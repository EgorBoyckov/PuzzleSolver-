"""CLI для прогона конвейера (preprocess -> segment -> describe [-> locate])
на папке с фото партий, без веб-клиента.

Примеры:
    python -m app.cli.pipeline_cli --input path/to/batch_photos --puzzle-id demo --out data/debug/pipeline_run
    python -m app.cli.pipeline_cli --input path/to/batch_photos --puzzle-id demo --out data/debug/pipeline_run \
        --reference path/to/box.jpg --grid-rows 40 --grid-cols 50
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import cv2
import numpy as np

from app.core.config import get_config
from app.core.logging import setup_logging
from app.pipeline.describe import describe_piece
from app.pipeline.locate import ReferenceGridIndex, build_reference_index, locate_piece, refit_reference_colors
from app.pipeline.match import build_edge_matches
from app.pipeline.plan import next_steps_with_catalog
from app.pipeline.preprocess import preprocess_frame
from app.pipeline.schemas import PieceKind
from app.pipeline.segment import segment_pieces

logger = logging.getLogger(__name__)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def run_pipeline_on_folder(
    input_dir: Path,
    puzzle_id: str,
    out_dir: Path,
    reference_path: Path | None = None,
    grid_rows: int | None = None,
    grid_cols: int | None = None,
    skip_match: bool = False,
    max_steps: int = 10000,
) -> dict:
    debug_dir = out_dir / "debug"
    catalog_dir = out_dir / "catalog"
    debug_dir.mkdir(parents=True, exist_ok=True)
    catalog_dir.mkdir(parents=True, exist_ok=True)

    photos = sorted(p for p in input_dir.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)
    if not photos:
        raise FileNotFoundError(f"В папке {input_dir} не найдено изображений ({IMAGE_EXTENSIONS})")

    ref_index: ReferenceGridIndex | None = None
    frame_lookup: dict[int, np.ndarray] = {}
    if reference_path is not None:
        if not (grid_rows and grid_cols):
            raise ValueError("--reference требует --grid-rows и --grid-cols")
        reference_bgr = cv2.imread(str(reference_path))
        if reference_bgr is None:
            raise FileNotFoundError(f"Не удалось прочитать образец: {reference_path}")
        ref_index = build_reference_index(reference_bgr, grid_rows, grid_cols)
        logger.info("locate: индекс образца построен (%dx%d ячеек)", grid_rows, grid_cols)

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
            if ref_index is not None:
                locate_piece(piece, rectified, ref_index)

        if ref_index is not None:
            frame_lookup[batch_number] = rectified

        all_pieces.extend(pieces)
        logger.info("batch %d (%s): %d деталей", batch_number, photo_path.name, len(pieces))

    if ref_index is not None and all_pieces:
        lc = get_config().section("locate")
        refitted = refit_reference_colors(
            ref_index,
            all_pieces,
            frame_lookup,
            confidence_threshold=lc["color_refit_confidence_threshold"],
            min_samples=lc["color_refit_min_samples"],
        )
        if refitted is not None:
            ref_index = refitted
            low_confidence = [
                p
                for p in all_pieces
                if not p.location_candidates or p.location_candidates[0][3] < lc["color_refit_confidence_threshold"]
            ]
            logger.info("locate: повторный поиск после цветокоррекции для %d деталей", len(low_confidence))
            for piece in low_confidence:
                rectified = frame_lookup.get(piece.batch_number)
                if rectified is not None:
                    locate_piece(piece, rectified, ref_index)

    kind_counts = {k.value: 0 for k in PieceKind}
    kind_counts["unknown"] = 0
    for p in all_pieces:
        kind_counts[p.kind.value if p.kind else "unknown"] += 1

    edges_count = steps_count = None
    if not skip_match and len(all_pieces) >= 2:
        edges = build_edge_matches(all_pieces)
        steps = next_steps_with_catalog(edges, all_pieces, max_steps=max_steps)
        edges_count, steps_count = len(edges), len(steps)
        with (catalog_dir / "steps.json").open("w", encoding="utf-8") as f:
            json.dump([s.model_dump() for s in steps], f, ensure_ascii=False, indent=2)
        logger.info("match+plan: %d кандидатов-рёбер, %d шагов сборки", edges_count, steps_count)

    summary = {
        "puzzle_id": puzzle_id,
        "frames_total": len(photos),
        "frames_accepted": sum(1 for f in frames_report if f["accepted"]),
        "frames_rejected": sum(1 for f in frames_report if not f["accepted"]),
        "pieces_found": len(all_pieces),
        "pieces_suspect": sum(1 for p in all_pieces if p.is_suspect),
        "kind_counts": kind_counts,
        "located": sum(1 for p in all_pieces if p.location_candidates) if ref_index is not None else None,
        "edge_matches": edges_count,
        "assembly_steps": steps_count,
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
    parser.add_argument("--reference", type=str, default=None, help="Фото коробки — если задано, запускается locate")
    parser.add_argument("--grid-rows", type=int, default=None, help="Число строк сетки образца")
    parser.add_argument("--grid-cols", type=int, default=None, help="Число столбцов сетки образца")
    parser.add_argument("--skip-match", action="store_true", help="Не считать сопоставление сторон и план сборки")
    parser.add_argument("--max-steps", type=int, default=10000, help="Максимум шагов сборки в плане")
    args = parser.parse_args()

    summary = run_pipeline_on_folder(
        Path(args.input),
        args.puzzle_id,
        Path(args.out),
        reference_path=Path(args.reference) if args.reference else None,
        grid_rows=args.grid_rows,
        grid_cols=args.grid_cols,
        skip_match=args.skip_match,
        max_steps=args.max_steps,
    )
    logger.info("Готово: %s", json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
