"""Тесты генератора PDF с ArUco-маркером (scripts/print_marker.py).

Проверяем то, что один раз уже было реально сломано при разработке:
кириллица через cv2.putText не рендерится, а длинные строки инструкции
могут вылезать за поля страницы A4.
"""
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest
from PIL import Image, ImageDraw

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from print_marker import (  # noqa: E402
    A4_MM,
    DPI,
    body_font,
    build_marker_page,
    instruction_lines,
    mm_to_px,
    title_font,
    title_lines,
)


@pytest.fixture(scope="module")
def page():
    return build_marker_page(marker_size_mm=100.0, dictionary_name="DICT_4X4_50", marker_id=0)


def test_page_size_matches_a4_at_dpi(page):
    assert page.size == (mm_to_px(A4_MM[0]), mm_to_px(A4_MM[1]))


def test_marker_detectable_in_generated_page(page):
    arr = cv2.cvtColor(np.array(page), cv2.COLOR_RGB2BGR)
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    detector = cv2.aruco.ArucoDetector(aruco_dict, cv2.aruco.DetectorParameters())
    corners, ids, _ = detector.detectMarkers(arr)
    assert ids is not None
    assert 0 in ids.flatten()


def test_marker_side_length_matches_requested_mm(page):
    arr = cv2.cvtColor(np.array(page), cv2.COLOR_RGB2BGR)
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    detector = cv2.aruco.ArucoDetector(aruco_dict, cv2.aruco.DetectorParameters())
    corners, ids, _ = detector.detectMarkers(arr)
    side_px = np.linalg.norm(corners[0][0][0] - corners[0][0][1])
    side_mm = side_px / DPI * 25.4
    assert abs(side_mm - 100.0) < 1.0


def test_no_text_line_overflows_page_margins():
    """Регрессия: раньше длинные строки инструкции вылезали за правое поле."""
    dummy = Image.new("RGB", (10, 10))
    draw = ImageDraw.Draw(dummy)
    t_font, b_font = title_font(), body_font()
    usable_w = mm_to_px(A4_MM[0] - 30)  # 15мм отступ с каждой стороны

    lines_with_fonts = [(t, t_font) for t in title_lines("DICT_4X4_50", 0)] + [
        (t, b_font) for t in instruction_lines(100.0)
    ]
    for text, font in lines_with_fonts:
        width = draw.textlength(text, font=font)
        assert width < usable_w, f"line overflows page margin: {text!r} ({width}px >= {usable_w}px)"
