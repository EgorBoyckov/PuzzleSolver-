"""CLI генератора синтетического датасета.

Пример:
    python -m app.synth.cli --pieces 500 --out data/debug/synth_demo --seed 7
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

from app.core.config import get_config
from app.core.logging import setup_logging
from app.synth.dataset import SyntheticDatasetConfig, generate_dataset

logger = logging.getLogger(__name__)


def build_config_from_args(args: argparse.Namespace) -> SyntheticDatasetConfig:
    app_config = get_config()
    capture = app_config.section("capture")
    synth = app_config.section("synth")
    aruco = app_config.section("aruco")

    return SyntheticDatasetConfig(
        total_pieces=args.pieces,
        out_dir=Path(args.out),
        piece_size_mm=synth.get("piece_size_mm", 25.0),
        aspect_ratio=tuple(synth.get("default_aspect_ratio", [4, 3])),
        source_image_path=Path(args.source) if args.source else None,
        batch_size_min=args.batch_min or capture.get("batch_size_min", 60),
        batch_size_max=args.batch_max or capture.get("batch_size_max", 120),
        min_gap_mm=capture.get("min_gap_mm", 10.0),
        max_gap_mm=capture.get("max_gap_mm", 15.0),
        aruco_dictionary=aruco.get("dictionary", "DICT_4X4_50"),
        aruco_marker_id=aruco.get("marker_id", 0),
        aruco_marker_size_mm=aruco.get("marker_size_mm", 100.0),
        background_color_bgr=tuple(synth.get("background_color_bgr", [40, 34, 30])),
        lighting_gradient_strength=synth.get("lighting_gradient_strength", 0.25),
        noise_sigma=synth.get("noise_sigma", 4.0),
        jpeg_quality=synth.get("jpeg_quality", 92),
        tab_params=synth.get("tab_shape", {}),
        seed=args.seed if args.seed is not None else synth.get("seed", 42),
    )


def main() -> None:
    setup_logging()
    parser = argparse.ArgumentParser(description="Генератор синтетического датасета PuzzleVision")
    parser.add_argument("--pieces", type=int, required=True, help="Целевое число деталей пазла")
    parser.add_argument("--out", type=str, required=True, help="Папка вывода датасета")
    parser.add_argument("--source", type=str, default=None, help="Путь к картинке коробки (иначе — процедурная генерация)")
    parser.add_argument("--batch-min", type=int, default=None, help="Мин. деталей на кадр партии")
    parser.add_argument("--batch-max", type=int, default=None, help="Макс. деталей на кадр партии")
    parser.add_argument("--seed", type=int, default=None, help="Seed генератора случайных чисел")
    args = parser.parse_args()

    config = build_config_from_args(args)
    meta = generate_dataset(config)
    logger.info("Готово: %s", meta)


if __name__ == "__main__":
    main()
