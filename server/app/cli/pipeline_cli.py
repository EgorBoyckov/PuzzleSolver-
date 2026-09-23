"""CLI для прогона конвейера (preprocess -> segment -> describe [-> locate ->
solve]) на папке с фото партий, без веб-клиента.

С образцом (--reference) после каталогизации всех партий детали
привязываются к картинке коробки (locate) и раскладываются по сетке
глобальной сборкой (solve): результат — catalog/layout.json (для каждой
ячейки: деталь, поворот, уверенность) и debug/assembled.jpg (картинка,
собранная из самих деталей — ошибки видны глазом).

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

from app.core.logging import setup_logging
from app.pipeline.describe import describe_piece
from app.pipeline.locate import PieceAppearance, ReferenceGridIndex, build_reference_index, locate_appearances, piece_appearance, render_layout
from app.pipeline.preprocess import preprocess_frame
from app.pipeline.schemas import PieceKind
from app.pipeline.segment import segment_pieces
from app.pipeline.solve import solve_layout

logger = logging.getLogger(__name__)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def run_pipeline_on_folder(
    input_dir: Path,
    puzzle_id: str,
    out_dir: Path,
    reference_path: Path | None = None,
    grid_rows: int | None = None,
    grid_cols: int | None = None,
) -> dict:
    debug_dir = out_dir / "debug"
    catalog_dir = out_dir / "catalog"
    debug_dir.mkdir(parents=True, exist_ok=True)
    catalog_dir.mkdir(parents=True, exist_ok=True)

    photos = sorted(p for p in input_dir.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)
    if not photos:
        raise FileNotFoundError(f"В папке {input_dir} не найдено изображений ({IMAGE_EXTENSIONS})")

    ref_index: ReferenceGridIndex | None = None
    # Признаки для locate считаются сразу по кадру партии — кадры целиком в
    # памяти не держим (на 5000 деталях это полтора гигабайта).
    located_pieces = []
    appearances: list[PieceAppearance] = []
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
                app = piece_appearance(piece, rectified, ref_index)
                if app is not None:
                    located_pieces.append(piece)
                    appearances.append(app)

        all_pieces.extend(pieces)
        logger.info("batch %d (%s): %d деталей", batch_number, photo_path.name, len(pieces))

    layout_summary = None
    if ref_index is not None and appearances:
        located = locate_appearances(located_pieces, appearances, ref_index)
        layout = solve_layout(located)
        grid = [[None] * ref_index.cols for _ in range(ref_index.rows)]
        for pl in layout.placements:
            piece = located.pieces[pl.piece_index]
            grid[pl.row][pl.col] = {
                "piece_id": piece.id,
                "rotation_deg": pl.rotation * 90,
                "confidence": round(pl.confidence, 4),
                "source": pl.source,
            }
        with (catalog_dir / "layout.json").open("w", encoding="utf-8") as f:
            json.dump({"rows": ref_index.rows, "cols": ref_index.cols, "cells": grid}, f, ensure_ascii=False, indent=1)
        cv2.imwrite(str(debug_dir / "assembled.jpg"), render_layout(located, layout.placements))
        sources = [pl.source for pl in layout.placements]
        layout_summary = {
            "cells": ref_index.rows * ref_index.cols,
            "placed": len(layout.placements),
            "by_source": {src: sources.count(src) for src in ("anchor", "growth", "fallback")},
        }

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
        "located": sum(1 for p in all_pieces if p.location_candidates) if ref_index is not None else None,
        "layout": layout_summary,
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
    parser.add_argument("--reference", type=str, default=None, help="Фото коробки — если задано, запускаются locate и глобальная сборка")
    parser.add_argument("--grid-rows", type=int, default=None, help="Число строк сетки образца")
    parser.add_argument("--grid-cols", type=int, default=None, help="Число столбцов сетки образца")
    args = parser.parse_args()

    summary = run_pipeline_on_folder(
        Path(args.input),
        args.puzzle_id,
        Path(args.out),
        reference_path=Path(args.reference) if args.reference else None,
        grid_rows=args.grid_rows,
        grid_cols=args.grid_cols,
    )
    logger.info("Готово: %s", json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
