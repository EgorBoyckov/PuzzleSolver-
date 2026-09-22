#!/usr/bin/env python3
"""Сгенерировать иконки PWA (favicon + manifest icons) из формы детали пазла.

Использует тот же генератор форм, что и синтетический датасет — маленькая,
но приятная деталь: иконка приложения буквально нарисована тем же кодом,
который умеет описывать реальные детали пазла.

Запуск: python scripts/gen_icons.py
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "server"))

import cv2
import numpy as np

from app.synth.piece_shapes import build_puzzle_grid

OUT_DIR = REPO_ROOT / "client" / "public" / "icons"
BG_COLOR = (196, 120, 46)  # BGR — тёплый оранжевый, под тон акцентной кнопки в UI
PIECE_COLOR = (255, 255, 255)


def render_icon(size_px: int) -> np.ndarray:
    rng = np.random.default_rng(1)
    grid = build_puzzle_grid(rows=1, cols=1, cell_size_mm=100.0, rng=rng, tab_params={"waist_depth_ratio": 0.0})
    # Одна деталь 1x1 — все стороны straight; возьмём вместо неё деталь из
    # сетки 2x2 (есть настоящие выступы/впадины), кусок (0,0).
    grid = build_puzzle_grid(rows=2, cols=2, cell_size_mm=100.0, rng=rng, tab_params={"waist_depth_ratio": 0.0})
    piece = grid.piece(0, 0)
    contour = piece.local_contour()

    pad = 0.12
    xmax, ymax = contour.max(axis=0)
    scale = size_px * (1 - 2 * pad) / max(xmax, ymax)
    pts = (contour * scale + size_px * pad).astype(np.int32)

    canvas = np.zeros((size_px, size_px, 3), dtype=np.uint8)
    canvas[:, :] = BG_COLOR
    cv2.fillPoly(canvas, [pts], PIECE_COLOR)
    return canvas


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for size in (192, 512):
        img = render_icon(size)
        out_path = OUT_DIR / f"icon-{size}.png"
        cv2.imwrite(str(out_path), img)
        print("wrote", out_path)

    favicon = render_icon(64)
    cv2.imwrite(str(REPO_ROOT / "client" / "public" / "favicon.png"), favicon)
    print("wrote favicon.png")


if __name__ == "__main__":
    main()
