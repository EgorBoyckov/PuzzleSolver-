#!/usr/bin/env python3
"""Сгенерировать PDF с ArUco-маркером для печати на листе A4.

Параметры (словарь, id, размер маркера в мм) берутся из config.yaml,
раздел `aruco`, чтобы PDF всегда соответствовал тому, что ищет preprocess.

Текст рисуется через Pillow с шрифтом DejaVu Sans (server/assets/fonts) —
у cv2.putText (Hershey-шрифты) нет кириллицы, а подписи на маркере русские.

Использование:
    python scripts/print_marker.py [--out marker.pdf]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "server"))

import cv2
from PIL import Image, ImageDraw, ImageFont

from app.core.config import get_config

ARUCO_DICT_NAMES = {
    "DICT_4X4_50": cv2.aruco.DICT_4X4_50,
    "DICT_5X5_50": cv2.aruco.DICT_5X5_50,
    "DICT_6X6_50": cv2.aruco.DICT_6X6_50,
}

A4_MM = (210.0, 297.0)
DPI = 300
FONT_DIR = REPO_ROOT / "server" / "assets" / "fonts"


def mm_to_px(mm: float, dpi: int = DPI) -> int:
    return int(round(mm / 25.4 * dpi))


def _marker_bitmap(marker_size_mm: float, dictionary_name: str, marker_id: int) -> Image.Image:
    aruco_dict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT_NAMES[dictionary_name])
    marker_px = mm_to_px(marker_size_mm)
    marker_img = cv2.aruco.generateImageMarker(aruco_dict, marker_id, marker_px)
    return Image.fromarray(marker_img).convert("RGB")


def title_lines(dictionary_name: str, marker_id: int) -> list[str]:
    return [
        "PuzzleVision — калибровочный маркер ArUco",
        f"({dictionary_name}, id={marker_id})",
    ]


def instruction_lines(marker_size_mm: float) -> list[str]:
    return [
        "Как использовать:",
        "1. Распечатать без масштабирования: печать «Actual size» / 100%,",
        "   а не «По размеру страницы».",
        f"2. Проверить линейкой: сторона квадрата маркера должна быть ровно {marker_size_mm:.0f} мм.",
        "3. Наклеить лист на плотную подложку (картон), чтобы он не",
        "   сворачивался и не мялся.",
        "4. Перед съёмкой каждой партии деталей класть маркер в угол стола так,",
        "   чтобы он целиком попадал в кадр вместе с деталями.",
        "5. Маркер используется для определения масштаба (мм/пиксель), выравнивания",
        "   перспективы и баланса белого — не закрывайте его деталями и не мните.",
    ]


def title_font() -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(str(FONT_DIR / "DejaVuSans-Bold.ttf"), mm_to_px(5))


def body_font() -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(str(FONT_DIR / "DejaVuSans.ttf"), mm_to_px(4))


def build_marker_page(marker_size_mm: float, dictionary_name: str, marker_id: int) -> Image.Image:
    page_w_px, page_h_px = mm_to_px(A4_MM[0]), mm_to_px(A4_MM[1])
    page = Image.new("RGB", (page_w_px, page_h_px), "white")
    draw = ImageDraw.Draw(page)

    t_font = title_font()
    b_font = body_font()

    y_title = mm_to_px(15)
    for line in title_lines(dictionary_name, marker_id):
        draw.text((mm_to_px(15), y_title), line, fill="black", font=t_font)
        y_title += mm_to_px(8)

    marker = _marker_bitmap(marker_size_mm, dictionary_name, marker_id)
    marker_px = marker.size[0]
    x0 = (page_w_px - marker_px) // 2
    y0 = mm_to_px(40)
    page.paste(marker, (x0, y0))

    size_label = f"{marker_size_mm:.0f} мм"
    draw.text((x0, y0 - mm_to_px(9)), size_label, fill="black", font=b_font)

    instructions = instruction_lines(marker_size_mm)
    y = y0 + marker_px + mm_to_px(15)
    line_h = mm_to_px(7)
    for line in instructions:
        draw.text((mm_to_px(15), y), line, fill="black", font=b_font)
        y += line_h

    return page


def main() -> None:
    parser = argparse.ArgumentParser(description="Генерация PDF с ArUco-маркером для печати")
    parser.add_argument("--out", type=str, default="marker_a4.pdf")
    args = parser.parse_args()

    config = get_config()
    aruco_cfg = config.section("aruco")
    page = build_marker_page(
        marker_size_mm=aruco_cfg.get("marker_size_mm", 100.0),
        dictionary_name=aruco_cfg.get("dictionary", "DICT_4X4_50"),
        marker_id=aruco_cfg.get("marker_id", 0),
    )

    out_path = Path(args.out)
    page.save(out_path, "PDF", resolution=DPI)
    print(f"Готово: {out_path}")


if __name__ == "__main__":
    main()
