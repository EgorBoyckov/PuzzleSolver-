import cv2
import numpy as np
import pytest

from app.synth.piece_shapes import build_puzzle_grid, classify_kind


@pytest.fixture
def grid():
    rng = np.random.default_rng(123)
    return build_puzzle_grid(rows=6, cols=8, cell_size_mm=25.0, rng=rng)


def test_classify_kind():
    assert classify_kind(2) == "corner"
    assert classify_kind(1) == "edge"
    assert classify_kind(0) == "center"


def test_piece_counts_and_kinds(grid):
    assert len(grid) == 6 * 8
    kinds = {}
    for p in grid:
        kinds[p.kind] = kinds.get(p.kind, 0) + 1
    assert kinds["corner"] == 4
    assert kinds["edge"] == 2 * (6 - 2) + 2 * (8 - 2)
    assert kinds["center"] == (6 - 2) * (8 - 2)
    assert kinds["corner"] + kinds["edge"] + kinds["center"] == len(grid)


def test_straight_side_count_matches_kind(grid):
    kind_to_count = {"corner": 2, "edge": 1, "center": 0}
    for p in grid:
        straight = sum(1 for s in p.sides if s.kind == "straight")
        assert straight == kind_to_count[p.kind], (p.row, p.col, p.kind, straight)


def test_boundary_pieces_have_straight_sides_on_boundary(grid):
    for p in grid:
        for s in p.sides:
            on_boundary = (
                (s.index == 0 and p.row == 0)
                or (s.index == 2 and p.row == grid.rows - 1)
                or (s.index == 3 and p.col == 0)
                or (s.index == 1 and p.col == grid.cols - 1)
            )
            if s.kind == "straight":
                assert on_boundary, f"straight side not on puzzle boundary: {p.row},{p.col},{s.index}"


def test_contours_are_closed_simple_polygons_with_reasonable_area(grid):
    areas = []
    for p in grid:
        contour = p.contour.astype(np.float32)
        area = cv2.contourArea(contour)
        assert area > 0
        areas.append(area)
    median = float(np.median(areas))
    cell_area = 25.0 * 25.0
    assert abs(median - cell_area) / cell_area < 0.1
    for a in areas:
        assert abs(a - median) / median < 0.4  # ±40% допуск как в config.segment.area_tolerance_fraction


def test_neighboring_sides_are_complementary_and_share_the_same_cut_line(grid):
    for r in range(grid.rows):
        for c in range(grid.cols - 1):
            left = grid.piece(r, c)
            right = grid.piece(r, c + 1)
            left_right_side = next(s for s in left.sides if s.index == 1)
            right_left_side = next(s for s in right.sides if s.index == 3)
            assert {left_right_side.kind, right_left_side.kind} == {"tab", "blank"}
            np.testing.assert_allclose(left_right_side.points, right_left_side.points[::-1], atol=1e-9)

    for r in range(grid.rows - 1):
        for c in range(grid.cols):
            top = grid.piece(r, c)
            bottom = grid.piece(r + 1, c)
            top_bottom_side = next(s for s in top.sides if s.index == 2)
            bottom_top_side = next(s for s in bottom.sides if s.index == 0)
            assert {top_bottom_side.kind, bottom_top_side.kind} == {"tab", "blank"}
            np.testing.assert_allclose(top_bottom_side.points, bottom_top_side.points[::-1], atol=1e-9)


def test_no_self_intersection_via_shapely_like_rasterization(grid):
    # Без внешней зависимости shapely: проверяем через растеризацию, что
    # площадь контура (cv2.contourArea, формула шнурования) близка к площади
    # растеризованной маски — большое расхождение означает самопересечения.
    upscale = 10.0  # снижаем ошибку квантования при округлении мм-координат до целых пикселей
    for p in grid:
        local = p.local_contour() * upscale
        xmax, ymax = local.max(axis=0)
        w, h = int(np.ceil(xmax)) + 2, int(np.ceil(ymax)) + 2
        mask = np.zeros((h, w), dtype=np.uint8)
        pts = np.round(local).astype(np.int32)
        cv2.fillPoly(mask, [pts], 255)
        raster_area = float((mask > 0).sum())
        shoelace_area = abs(cv2.contourArea(local.astype(np.float32)))
        assert raster_area > 0
        assert abs(raster_area - shoelace_area) / shoelace_area < 0.03
