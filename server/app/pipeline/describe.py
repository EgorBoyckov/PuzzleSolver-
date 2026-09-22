"""Модуль 3: описание детали — углы, стороны, форма, цвет, эмбеддинг.

Углы ищутся как топ-4 самых острых вершины выпуклой оболочки контура (см.
docstring _find_corners): выступ выпуклый и добавляет на оболочку пологую
вершину, впадина вогнутая и на оболочку не попадает, поэтому настоящие
90°-углы детали резко острее любой другой вершины оболочки — устойчиво даже
при сильно несимметричных выступах/впадинах на разных сторонах. Контур
делится на 4 стороны по найденным углам; тип стороны — по максимальному
отклонению от прямой линии между её концами (порог
config.describe.straight_side_deviation_ratio), знак отклонения (наружу/
внутрь от центра детали) отличает выступ от впадины.

Эмбеддинг лица детали — временный лёгкий дескриптор (цветовая гистограмма
Lab + Hu-моменты формы), а не DINOv2: настоящая модель подключается на
этапе 2, когда эмбеддинг реально нужен для привязки к образцу (locate) и
кластеризации в режиме без образца — тащить тяжёлую ML-зависимость раньше,
чем она используется, нет смысла.
"""
from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np

from app.core.config import get_config
from app.pipeline.embedding import compute_placeholder_embedding
from app.pipeline.schemas import PieceKind, PieceRecord, Point2D, Side, SideKind

logger = logging.getLogger(__name__)

_SIDE_DEBUG_COLORS = {
    SideKind.STRAIGHT: (200, 200, 200),
    SideKind.TAB: (0, 200, 0),
    SideKind.BLANK: (0, 120, 255),
}


def _classify_piece_kind(straight_count: int) -> PieceKind:
    if straight_count >= 2:
        return PieceKind.CORNER
    if straight_count == 1:
        return PieceKind.EDGE
    return PieceKind.CENTER


def _find_corners(contour: np.ndarray, ds: dict) -> list[int] | None:
    """Топ-4 самых острых вершины выпуклой оболочки контура.

    Раньше углы искались привязкой к 4 точкам cv2.minAreaRect (смещается на
    несимметричных выступах) и затем скан по всему контуру (иногда цеплялся
    за точку на пологой дуге выступа, где локальное окно кривизны случайно
    давало сравнимый с углом счёт). Выпуклая оболочка устойчивее по
    построению: выступ — выпуклая деталь контура и добавляет свою вершину на
    оболочку, но эта вершина пологая (тупой угол поворота); впадина вогнутая
    и вообще не попадает на оболочку. Настоящие 90°-углы детали дают резко
    более острый поворот, чем вершина на пике любого выступа — поэтому топ-4
    самых острых вершин оболочки почти всегда и есть 4 угла детали.
    """
    n = len(contour)
    if n < 16:
        return None

    contour32 = contour.astype(np.float32)
    hull_idx = cv2.convexHull(contour32, returnPoints=False)
    if hull_idx is None or len(hull_idx) < 4:
        return None
    hull_idx = sorted(int(i) for i in hull_idx.flatten())

    # JPEG-артефакты/шум сегментации дают на оболочке множество мелких
    # ложных вершин (однопиксельные зазубрины), среди которых острый угол
    # поворота может случайно оказаться выше, чем у настоящего угла детали.
    # approxPolyDP по точкам оболочки схлопывает такой шум, сохраняя крупные
    # геометрические особенности (настоящие углы и пики выступов, десятки px).
    hull_points = contour32[hull_idx].reshape(-1, 1, 2)
    perimeter = cv2.arcLength(contour32, True)
    epsilon = ds["corner_hull_approx_epsilon_fraction"] * perimeter
    approx = cv2.approxPolyDP(hull_points, epsilon, True).reshape(-1, 2)
    if len(approx) < 4:
        return None

    m = len(approx)
    scored: list[tuple[float, int]] = []
    for k in range(m):
        p_prev, p_cur, p_next = approx[(k - 1) % m], approx[k], approx[(k + 1) % m]
        v1, v2 = p_prev - p_cur, p_next - p_cur
        cos_a = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-9)
        angle = np.degrees(np.arccos(np.clip(cos_a, -1.0, 1.0)))
        nearest_contour_idx = int(np.argmin(np.linalg.norm(contour32 - p_cur, axis=1)))
        scored.append((180.0 - angle, nearest_contour_idx))

    if len(scored) < 4:
        return None
    scored.sort(key=lambda t: -t[0])
    top4_idx = sorted(i for _, i in scored[:4])
    if len(set(top4_idx)) != 4:
        return None
    return top4_idx


def _rectangularity_deviation_deg(contour: np.ndarray, indices: list[int]) -> float:
    pts = contour[indices]
    max_dev = 0.0
    for i in range(4):
        p_prev, p, p_next = pts[(i - 1) % 4], pts[i], pts[(i + 1) % 4]
        v1, v2 = p_prev - p, p_next - p
        cos_a = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-9)
        angle = np.degrees(np.arccos(np.clip(cos_a, -1.0, 1.0)))
        max_dev = max(max_dev, abs(angle - 90.0))
    return max_dev


def _split_into_sides(contour: np.ndarray, corner_indices: list[int]) -> list[np.ndarray]:
    idxs = sorted(corner_indices)
    sides = []
    for k in range(4):
        i0, i1 = idxs[k], idxs[(k + 1) % 4]
        pts = contour[i0 : i1 + 1] if i1 > i0 else np.concatenate([contour[i0:], contour[: i1 + 1]])
        sides.append(pts)
    return sides


def _arc_length(pts: np.ndarray) -> np.ndarray:
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    return np.concatenate([[0.0], np.cumsum(seg)])


def _resample_by_arclength(pts: np.ndarray, n: int) -> np.ndarray:
    cum = _arc_length(pts)
    total = cum[-1]
    if total <= 1e-9:
        return np.tile(pts[0], (n, 1))
    targets = np.linspace(0, total, n)
    xs = np.interp(targets, cum, pts[:, 0])
    ys = np.interp(targets, cum, pts[:, 1])
    return np.stack([xs, ys], axis=1)


def _tangents(resampled: np.ndarray) -> np.ndarray:
    tang = np.gradient(resampled, axis=0)
    norms = np.linalg.norm(tang, axis=1, keepdims=True)
    norms[norms < 1e-9] = 1.0
    return tang / norms


def _classify_side(pts: np.ndarray, centroid: np.ndarray, mm_per_pixel: float, ds: dict) -> tuple[SideKind, np.ndarray]:
    p0, p1 = pts[0], pts[-1]
    baseline = p1 - p0
    length = float(np.linalg.norm(baseline))
    n_curve = int(ds["side_curve_points"])

    if length < 1e-6:
        return SideKind.STRAIGHT, np.zeros((n_curve, 2))

    direction = baseline / length
    normal = np.array([-direction[1], direction[0]])
    if np.dot(pts.mean(axis=0) - centroid, normal) < 0:
        normal = -normal

    rel = pts - p0
    x_local = rel @ direction
    y_local = rel @ normal

    max_dev = float(np.max(np.abs(y_local)))
    deviation_ratio = max_dev / length

    curve_local = np.stack([x_local, y_local], axis=1)
    resampled_local = _resample_by_arclength(curve_local, n_curve)

    if deviation_ratio < ds["straight_side_deviation_ratio"]:
        kind = SideKind.STRAIGHT
    else:
        idx_max = int(np.argmax(np.abs(y_local)))
        kind = SideKind.TAB if y_local[idx_max] > 0 else SideKind.BLANK

    return kind, resampled_local * mm_per_pixel


def _sample_color_strip(
    lab_crop: np.ndarray, crop_offset: np.ndarray, pts: np.ndarray, centroid: np.ndarray, mm_per_pixel: float, ds: dict
) -> list[tuple[float, float, float]]:
    n_samples = int(ds["color_strip_samples"])
    width_px = ds["color_strip_width_mm"] / mm_per_pixel
    resampled = _resample_by_arclength(pts, n_samples)
    tangents = _tangents(resampled)
    h, w = lab_crop.shape[:2]
    half_patch = max(1, int(round(width_px * 0.25)))

    strip: list[tuple[float, float, float]] = []
    for p, t in zip(resampled, tangents):
        normal = np.array([-t[1], t[0]])
        if np.dot(p - centroid, normal) < 0:
            normal = -normal
        sample_pt = p - normal * (width_px * 0.6) - crop_offset  # сдвиг внутрь + в локальные координаты кропа
        xi, yi = int(round(sample_pt[0])), int(round(sample_pt[1]))
        x0, x1 = max(0, xi - half_patch), min(w, xi + half_patch + 1)
        y0, y1 = max(0, yi - half_patch), min(h, yi + half_patch + 1)
        if x1 <= x0 or y1 <= y0:
            strip.append((0.0, 0.0, 0.0))
            continue
        patch = lab_crop[y0:y1, x0:x1].reshape(-1, 3).astype(np.float32)
        mean_lab = patch.mean(axis=0)
        strip.append((float(mean_lab[0]), float(mean_lab[1]), float(mean_lab[2])))
    return strip


def _save_debug_overlay(frame_bgr: np.ndarray, piece: PieceRecord, contour: np.ndarray, corner_indices: list[int], debug_dir: Path) -> None:
    x, y, w, h = cv2.boundingRect(contour.astype(np.int32))
    pad = 15
    x0, y0 = max(0, x - pad), max(0, y - pad)
    x1, y1 = min(frame_bgr.shape[1], x + w + pad), min(frame_bgr.shape[0], y + h + pad)
    crop = frame_bgr[y0:y1, x0:x1].copy()
    offset = np.array([x0, y0])

    sides_pts = _split_into_sides(contour, sorted(corner_indices))
    for side, s in zip(piece.sides, sides_pts):
        pts_local = (s - offset).astype(np.int32)
        color = _SIDE_DEBUG_COLORS.get(side.kind, (255, 255, 255))
        cv2.polylines(crop, [pts_local], False, color, 3)

    for ci in corner_indices:
        cx, cy = (contour[ci] - offset).astype(int)
        cv2.circle(crop, (int(cx), int(cy)), 5, (0, 0, 255), -1)

    label = f"{piece.id} {piece.kind.value if piece.kind else '?'}"
    cv2.putText(crop, label, (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    debug_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(debug_dir / f"{piece.id}_describe.jpg"), crop)


def describe_piece(
    piece: PieceRecord,
    frame_bgr: np.ndarray,
    mm_per_pixel: float,
    debug_dir: Path | None = None,
) -> PieceRecord:
    """Найти 4 угла, разбить контур на стороны, классифицировать каждую
    (прямая/выступ/впадина), посчитать нормализованную кривую, цветовую
    полосу в Lab и эмбеддинг лица детали. Определяет piece.kind."""
    cfg = get_config()
    ds = cfg.section("describe")

    contour = np.array([[p.x, p.y] for p in piece.contour_px], dtype=np.float64)
    if len(contour) < 16:
        piece.is_suspect = True
        piece.suspect_reason = piece.suspect_reason or "contour_too_small"
        return piece

    corner_indices = _find_corners(contour, ds)
    if corner_indices is None:
        piece.is_suspect = True
        piece.suspect_reason = piece.suspect_reason or "corner_detection_failed"
        logger.warning("describe: не удалось найти 4 угла для %s", piece.id)
        return piece

    if _rectangularity_deviation_deg(contour, corner_indices) > ds["corner_rectangularity_tolerance_deg"]:
        piece.is_suspect = True
        piece.suspect_reason = piece.suspect_reason or "non_rectangular"

    piece.corners_px = [Point2D(x=float(contour[i][0]), y=float(contour[i][1])) for i in corner_indices]

    sides_pts = _split_into_sides(contour, corner_indices)
    centroid = contour.mean(axis=0)

    # Работаем с маленьким кропом вокруг детали, а не с целым кадром: cv2.cvtColor
    # на полном 2000-5000px кадре на каждую из ~100 деталей партии — основной
    # источник тормозов (секунды на партию вместо долей секунды на деталь).
    xmin, ymin = contour.min(axis=0)
    xmax, ymax = contour.max(axis=0)
    pad = int(ds["color_strip_width_mm"] / mm_per_pixel) + 10
    cx0 = max(0, int(xmin) - pad)
    cy0 = max(0, int(ymin) - pad)
    cx1 = min(frame_bgr.shape[1], int(xmax) + pad)
    cy1 = min(frame_bgr.shape[0], int(ymax) + pad)
    crop_offset = np.array([cx0, cy0], dtype=np.float64)
    bgr_crop = frame_bgr[cy0:cy1, cx0:cx1]
    lab_crop = cv2.cvtColor(bgr_crop, cv2.COLOR_BGR2LAB)

    sides: list[Side] = []
    straight_count = 0
    for idx, pts in enumerate(sides_pts):
        kind, curve_mm = _classify_side(pts, centroid, mm_per_pixel, ds)
        if kind == SideKind.STRAIGHT:
            straight_count += 1
        color_strip = _sample_color_strip(lab_crop, crop_offset, pts, centroid, mm_per_pixel, ds)
        sides.append(
            Side(
                index=idx,
                kind=kind,
                curve=[Point2D(x=float(px), y=float(py)) for px, py in curve_mm],
                color_strip_lab=color_strip,
            )
        )

    piece.sides = sides
    piece.kind = _classify_piece_kind(straight_count)
    piece.embedding = compute_placeholder_embedding(bgr_crop, contour - crop_offset)

    if debug_dir is not None:
        _save_debug_overlay(frame_bgr, piece, contour, corner_indices, debug_dir)

    return piece
