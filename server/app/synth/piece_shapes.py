"""Геометрия классических форм деталей пазла (выступ/впадина/прямой край).

Строит сетку rows x cols деталей на условной "решённой" раскладке (в мм).
Каждая внутренняя граница между соседними ячейками — это ОДНА физическая
линия разреза: она же является выступом для одной детали и ровно той же
(общей) впадиной для соседней — поэтому форма генерируется один раз на
ребро сетки, а не отдельно для каждой стороны каждой детали.

Профиль выступа/впадины — гладкая "шейка-бульба-шейка" из суммы приподнятых
косинусов (C1-гладкая, обнуляется на границах сегмента, поэтому линия
разреза без самопересечений и стыкуется с соседними сторонами без разрывов).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class SideGeom:
    index: int  # 0=top, 1=right, 2=bottom, 3=left (по часовой стрелке в image-координатах)
    kind: str  # "straight" | "tab" | "blank"
    points: np.ndarray  # (N, 2) абсолютные координаты в мм, по ходу контура детали


@dataclass
class PieceGeometry:
    row: int
    col: int
    kind: str  # "corner" | "edge" | "center"
    sides: list[SideGeom]
    contour: np.ndarray  # (M, 2) замкнутый полигон в мм, абсолютные координаты решённой раскладки

    @property
    def bbox(self) -> tuple[float, float, float, float]:
        xmin, ymin = self.contour.min(axis=0)
        xmax, ymax = self.contour.max(axis=0)
        return float(xmin), float(ymin), float(xmax), float(ymax)

    def local_contour(self) -> np.ndarray:
        """Контур со смещением так, чтобы bbox начинался в (0, 0)."""
        xmin, ymin, _, _ = self.bbox
        return self.contour - np.array([xmin, ymin])


@dataclass
class PuzzleGrid:
    rows: int
    cols: int
    cell_size_mm: float
    pieces: dict[tuple[int, int], PieceGeometry] = field(default_factory=dict)

    @property
    def width_mm(self) -> float:
        return self.cols * self.cell_size_mm

    @property
    def height_mm(self) -> float:
        return self.rows * self.cell_size_mm

    def piece(self, row: int, col: int) -> PieceGeometry:
        return self.pieces[(row, col)]

    def __iter__(self):
        return iter(self.pieces.values())

    def __len__(self) -> int:
        return len(self.pieces)


def classify_kind(straight_count: int) -> str:
    if straight_count == 2:
        return "corner"
    if straight_count == 1:
        return "edge"
    return "center"


def _raised_cosine(d: np.ndarray, half_width: float) -> np.ndarray:
    out = np.zeros_like(d)
    if half_width <= 0:
        return out
    mask = np.abs(d) < half_width
    out[mask] = 0.5 * (1.0 + np.cos(np.pi * d[mask] / half_width))
    return out


def _bump_profile(
    u: np.ndarray,
    length: float,
    rng: np.random.Generator,
    depth_ratio: float,
    center_jitter: float,
    half_width_ratio: float,
    waist_w_ratio: float,
    waist_depth_ratio: float,
    depth_jitter: float,
) -> np.ndarray:
    """Безразмерный профиль "шейка-бульба-шейка" для одного ребра сетки."""
    center = length * (0.5 + rng.uniform(-center_jitter, center_jitter))
    half_width = length * half_width_ratio * (1.0 + rng.uniform(-0.1, 0.1))
    waist_w = length * waist_w_ratio
    depth = length * depth_ratio * (1.0 + rng.uniform(-depth_jitter, depth_jitter))
    waist_depth = depth * waist_depth_ratio

    main = _raised_cosine(u - center, half_width) * depth
    left_waist = _raised_cosine(u - (center - half_width - waist_w), waist_w) * waist_depth
    right_waist = _raised_cosine(u - (center + half_width + waist_w), waist_w) * waist_depth
    return main - left_waist - right_waist


def _cubic_bezier(p0: np.ndarray, p1: np.ndarray, p2: np.ndarray, p3: np.ndarray, n: int, include_start: bool) -> np.ndarray:
    t = np.linspace(0.0, 1.0, n + 1)[:, None]
    if not include_start:
        t = t[1:]
    return (1 - t) ** 3 * p0 + 3 * (1 - t) ** 2 * t * p1 + 3 * (1 - t) * t**2 * p2 + t**3 * p3


def _classic_knob_curve(length: float, n_points: int, rng: np.random.Generator, tab_size: float, jitter: float) -> np.ndarray:
    """Классический «грибовидный» замок: узкая шейка и круглая головка шире
    шейки (с поднутрением — кривая НЕ является функцией y(x)), как у
    настоящих вырубных пазлов. Три кубических сплайна Безье по 10 опорным
    точкам (известная схема процедурных генераторов пазлов): t — размер
    замка (доля стороны), случайные a..e — индивидуальная асимметрия ребра
    (сдвиг головки вдоль стороны, наклон шейки, лёгкая волна у углов).

    Возвращает (N, 2): x вдоль ребра [0, length], y — смещение (головка
    в сторону +y)."""
    t = tab_size
    a, b, c, d, e = (rng.uniform(-jitter, jitter) for _ in range(5))
    pts = np.array(
        [
            [0.0, 0.0],
            [0.2, a],
            [0.5 + b + d, -t + c],
            [0.5 - t + b, t + c],
            [0.5 - 2.0 * t + b - d, 3.0 * t + c],
            [0.5 + 2.0 * t + b - d, 3.0 * t + c],
            [0.5 + t + b, t + c],
            [0.5 + b + d, -t + c],
            [0.8, e],
            [1.0, 0.0],
        ]
    ) * length
    per_seg = max(8, n_points // 3)
    seg1 = _cubic_bezier(pts[0], pts[1], pts[2], pts[3], per_seg, include_start=True)
    seg2 = _cubic_bezier(pts[3], pts[4], pts[5], pts[6], per_seg, include_start=False)
    seg3 = _cubic_bezier(pts[6], pts[7], pts[8], pts[9], per_seg, include_start=False)
    return np.concatenate([seg1, seg2, seg3], axis=0)


def _generate_edge_curve(
    length: float,
    n_points: int,
    bulge_sign: int,
    rng: np.random.Generator,
    tab_params: dict,
) -> np.ndarray:
    """Кривая ребра в локальной системе: x вдоль ребра [0,length], y — перпендикулярное смещение."""
    if bulge_sign == 0:
        u = np.array([0.0, length])
        v = np.array([0.0, 0.0])
        return np.stack([u, v], axis=1)

    if tab_params.get("style", "classic") == "classic":
        curve = _classic_knob_curve(length, n_points, rng, tab_params["tab_size"], tab_params["jitter"])
        curve[:, 1] *= bulge_sign
        return curve

    u = np.linspace(0.0, length, n_points)
    profile = _bump_profile(
        u,
        length,
        rng,
        depth_ratio=tab_params["depth_ratio"],
        center_jitter=tab_params["center_jitter"],
        half_width_ratio=tab_params["half_width_ratio"],
        waist_w_ratio=tab_params["waist_w_ratio"],
        waist_depth_ratio=tab_params["waist_depth_ratio"],
        depth_jitter=tab_params["depth_jitter"],
    )
    v = bulge_sign * profile
    return np.stack([u, v], axis=1)


DEFAULT_TAB_PARAMS = {
    # "classic" — грибовидный замок с шейкой (как у настоящих пазлов);
    # "bump" — прежний гладкий горб без шейки (оставлен для сравнения).
    "style": "classic",
    "tab_size": 0.1,
    "jitter": 0.035,
    "depth_ratio": 0.22,
    "center_jitter": 0.06,
    "half_width_ratio": 0.24,
    "waist_w_ratio": 0.05,
    "waist_depth_ratio": 0.0,
    "depth_jitter": 0.15,
    "side_curve_points": 96,
}


def build_puzzle_grid(
    rows: int,
    cols: int,
    cell_size_mm: float,
    rng: np.random.Generator,
    tab_params: dict | None = None,
) -> PuzzleGrid:
    """Построить полную сетку деталей с формами выступов/впадин.

    Возвращает PuzzleGrid с контуром каждой детали в мм в системе координат
    "решённого" пазла (0,0)-(cols*cell, rows*cell) — она же используется как
    эталонная раскладка при вырезании текстур из картинки коробки.
    """
    params = {**DEFAULT_TAB_PARAMS, **(tab_params or {})}
    n_pts = params["side_curve_points"]
    P = cell_size_mm

    # bulge_h[r][c] — знак горизонтального ребра между piece(r-1,c) и piece(r,c), r=1..rows-1
    # bulge_v[r][c] — знак вертикального ребра между piece(r,c-1) и piece(r,c), c=1..cols-1
    bulge_h = rng.choice([-1, 1], size=(rows, cols))
    bulge_v = rng.choice([-1, 1], size=(rows, cols))

    # Кэш точек общих рёбер, чтобы соседние детали делили ровно одну и ту же линию разреза.
    h_edge_points: dict[tuple[int, int], np.ndarray] = {}
    v_edge_points: dict[tuple[int, int], np.ndarray] = {}

    def horizontal_edge(r: int, c: int) -> np.ndarray:
        """Точки ребра на y = r*P между x=c*P и x=(c+1)*P, слева направо."""
        key = (r, c)
        if key not in h_edge_points:
            sign = int(bulge_h[r, c])
            local = _generate_edge_curve(P, n_pts, sign, rng, params)
            xs = c * P + local[:, 0]
            ys = r * P + local[:, 1]
            h_edge_points[key] = np.stack([xs, ys], axis=1)
        return h_edge_points[key]

    def vertical_edge(r: int, c: int) -> np.ndarray:
        """Точки ребра на x = c*P между y=r*P и y=(r+1)*P, сверху вниз."""
        key = (r, c)
        if key not in v_edge_points:
            sign = int(bulge_v[r, c])
            local = _generate_edge_curve(P, n_pts, sign, rng, params)
            xs = c * P + local[:, 1]
            ys = r * P + local[:, 0]
            v_edge_points[key] = np.stack([xs, ys], axis=1)
        return v_edge_points[key]

    grid = PuzzleGrid(rows=rows, cols=cols, cell_size_mm=cell_size_mm)

    for r in range(rows):
        for c in range(cols):
            # top
            if r == 0:
                top_pts = np.array([[c * P, 0.0], [(c + 1) * P, 0.0]])
                top_kind = "straight"
            else:
                top_pts = horizontal_edge(r, c)
                top_kind = "tab" if bulge_h[r, c] == -1 else "blank"
                # горизонтальное ребро строится в направлении "вниз положительно";
                # если знак -1, бугор торчит вверх -> в ячейку (r-1,c) снизу -> у piece(r,c) сверху это tab.

            # right
            if c == cols - 1:
                right_pts = np.array([[(c + 1) * P, r * P], [(c + 1) * P, (r + 1) * P]])
                right_kind = "straight"
            else:
                right_pts = vertical_edge(r, c + 1)
                right_kind = "tab" if bulge_v[r, c + 1] == 1 else "blank"

            # bottom (reverse of horizontal_edge(r+1, c): right -> left)
            if r == rows - 1:
                bottom_pts = np.array([[(c + 1) * P, (r + 1) * P], [c * P, (r + 1) * P]])
                bottom_kind = "straight"
            else:
                bottom_pts = horizontal_edge(r + 1, c)[::-1]
                bottom_kind = "tab" if bulge_h[r + 1, c] == 1 else "blank"

            # left (reverse of vertical_edge(r, c): bottom -> top)
            if c == 0:
                left_pts = np.array([[c * P, (r + 1) * P], [c * P, r * P]])
                left_kind = "straight"
            else:
                left_pts = vertical_edge(r, c)[::-1]
                left_kind = "tab" if bulge_v[r, c] == -1 else "blank"

            sides = [
                SideGeom(0, top_kind, top_pts),
                SideGeom(1, right_kind, right_pts),
                SideGeom(2, bottom_kind, bottom_pts),
                SideGeom(3, left_kind, left_pts),
            ]
            straight_count = sum(1 for s in sides if s.kind == "straight")
            kind = classify_kind(straight_count)

            contour = np.concatenate(
                [top_pts, right_pts[1:], bottom_pts[1:], left_pts[1:-1]], axis=0
            )

            grid.pieces[(r, c)] = PieceGeometry(row=r, col=c, kind=kind, sides=sides, contour=contour)

    return grid
