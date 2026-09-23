"""Модуль 3: описание детали — углы, стороны, форма, цвет, эмбеддинг.

Углы ищутся как точки, из которых контур уходит двумя прямыми «плечами» под
~90°, с перебором четвёрок кандидатов на согласованный прямоугольник (см.
docstring _find_corners) — острота вершины оболочки не годится: вершина
головки замка бывает острее настоящего угла. Контур
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

import itertools
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


def _resample_closed(contour: np.ndarray, n: int) -> tuple[np.ndarray, float]:
    pts = np.vstack([contour, contour[:1]])
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    t = np.linspace(0.0, cum[-1], n, endpoint=False)
    return np.stack([np.interp(t, cum, pts[:, 0]), np.interp(t, cum, pts[:, 1])], axis=1), float(cum[-1])


def _arm_directions(pts: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Для каждой точки замкнутого равномерного контура — направление и
    непрямолинейность (RMS отклонения от прямой) «плеча» из k следующих
    (вперёд) и k предыдущих (назад) точек. Направления — от точки наружу."""
    n = len(pts)
    offsets = np.arange(0, k + 1)
    result = []
    for sign in (+1, -1):
        idx = (np.arange(n)[:, None] + sign * offsets[None, :]) % n
        arms = pts[idx]                                   # (n, k+1, 2)
        centered = arms - arms.mean(axis=1, keepdims=True)
        cov = np.einsum("nki,nkj->nij", centered, centered) / (k + 1)
        evals, evecs = np.linalg.eigh(cov)                 # по возрастанию
        d = evecs[:, :, 1]
        flip = np.einsum("ni,ni->n", arms[:, -1] - arms[:, 0], d) < 0
        d[flip] *= -1
        result += [d, np.sqrt(np.clip(evals[:, 0], 0.0, None))]
    return result[0], result[1], result[2], result[3]


def _find_corners(contour: np.ndarray, ds: dict) -> tuple[list[int], np.ndarray] | None:
    """4 угла детали: индексы в контуре + уточнённые субпиксельные координаты.

    Прежний способ (4 самых острых вершины выпуклой оболочки) систематически
    ошибался: если у угла обе соседние стороны с выступами, оболочка обходит
    его по касательным к головкам замков, и вершина головки на оболочке
    оказывается острее настоящего угла — на синтетике 2000 деталей все 4
    угла были верны лишь у ~26% деталей. Хуже того, 4 вершины головок у
    детали с четырьмя выступами образуют почти идеальный квадрат того же
    размера, так что проверка «прямоугольности» такую ошибку не ловит.

    Настоящий угол отличается не остротой, а тем, что из него контур уходит
    двумя ПРЯМЫМИ отрезками под ~90° (вершина головки — дуга). Поэтому:
      1. контур равномерно передискретизируется; для каждой точки
         подгоняются прямые к «плечам» длиной arm_fraction·L вперёд/назад
         (L = sqrt(площади) — оценка стороны);
      2. кандидаты — локальные минимумы штрафа (|угол-90°|, кривизна плеч,
         выпуклость);
      3. из кандидатов перебором выбирается четвёрка, образующая
         прямоугольник с равными противоположными сторонами, площадью как у
         детали и — главное — с плечами, направленными ВДОЛЬ сторон
         четырёхугольника (у «ромба» из головок замков касательная в
         вершине головки идёт под 45° к его сторонам);
      4. каждый угол уточняется пересечением прямых обоих плеч — это
         компенсирует скругление углов (морфология сегментации, реальные
         вырубные углы тоже слегка скруглены).
    """
    if len(contour) < 16:
        return None
    area = abs(cv2.contourArea(contour.astype(np.float32)))
    if area <= 1.0:
        return None
    side_est = float(np.sqrt(area))
    n = int(ds["corner_resample_points"])
    pts, perimeter = _resample_closed(contour, n)
    step = perimeter / n
    k = max(3, int(round(float(ds["corner_arm_fraction"]) * side_est / step)))

    fwd, rms_f, bwd, rms_b = _arm_directions(pts, k)
    angle = np.degrees(np.arccos(np.clip(np.einsum("ni,ni->n", fwd, bwd), -1.0, 1.0)))
    orient = np.sign(cv2.contourArea(pts.astype(np.float32), oriented=True)) or 1.0
    cross = bwd[:, 0] * fwd[:, 1] - bwd[:, 1] * fwd[:, 0]
    convex = np.sign(cross) == -orient
    straightness = (rms_f + rms_b) / (0.02 * side_est)
    point_score = ((angle - 90.0) / 12.0) ** 2 + straightness**2 + np.where(convex, 0.0, 50.0)

    min_sep = 0.8 * k
    candidates: list[int] = []
    for i in np.argsort(point_score):
        if len(candidates) >= int(ds["corner_max_candidates"]):
            break
        if all(min(abs(int(i) - j), n - abs(int(i) - j)) > min_sep for j in candidates):
            candidates.append(int(i))
    if len(candidates) < 4:
        return None
    cand = np.array(sorted(candidates))

    combos = np.array(list(itertools.combinations(range(len(cand)), 4)))
    ids = cand[combos]                                    # (m, 4), по ходу контура
    q = pts[ids]                                          # (m, 4, 2)
    e = np.roll(q, -1, axis=1) - q
    sl = np.linalg.norm(e, axis=2) + 1e-9
    eu = e / sl[:, :, None]
    prev = -np.roll(eu, 1, axis=1)
    q_angle = np.degrees(np.arccos(np.clip(np.sum(eu * prev, axis=2), -1.0, 1.0)))
    s_angle = np.sum(((q_angle - 90.0) / 8.0) ** 2, axis=1)
    mean_len = sl.mean(axis=1)
    s_len = (((sl[:, 0] - sl[:, 2]) / mean_len / 0.08) ** 2 + ((sl[:, 1] - sl[:, 3]) / mean_len / 0.08) ** 2)
    s_aspect = (np.log((sl[:, 0] + sl[:, 2]) / (sl[:, 1] + sl[:, 3])) / 0.35) ** 2
    arm_f = np.degrees(np.arccos(np.clip(np.sum(fwd[ids] * eu, axis=2), -1.0, 1.0)))
    arm_b = np.degrees(np.arccos(np.clip(np.sum(bwd[ids] * prev, axis=2), -1.0, 1.0)))
    s_arm = np.sum((arm_f / 10.0) ** 2 + (arm_b / 10.0) ** 2, axis=1)
    quad_area = 0.5 * np.abs(np.sum(q[:, :, 0] * np.roll(q[:, :, 1], -1, axis=1) - np.roll(q[:, :, 0], -1, axis=1) * q[:, :, 1], axis=1))
    s_area = (np.log(np.maximum(quad_area, 1e-9) / area) / 0.25) ** 2
    total = s_angle + s_len + s_aspect + s_arm + s_area + point_score[ids].sum(axis=1)
    best = ids[int(np.argmin(total))]

    refined = []
    inner = np.arange(max(1, k // 4), k + 1)
    for i in best:
        lines = []
        for sign in (+1, -1):
            arm = pts[(i + sign * inner) % n]
            c = arm.mean(axis=0)
            _, _, vt = np.linalg.svd(arm - c, full_matrices=False)
            lines.append((c, vt[0]))
        (c1, d1), (c2, d2) = lines
        system = np.array([d1, -d2]).T
        point = pts[i]
        if abs(np.linalg.det(system)) > 1e-6:
            t = np.linalg.solve(system, c2 - c1)
            candidate = c1 + t[0] * d1
            if np.linalg.norm(candidate - pts[i]) < 0.1 * side_est:
                point = candidate
        refined.append(point)

    contour_idx = [int(np.argmin(np.linalg.norm(contour - pts[i], axis=1))) for i in best]
    order = np.argsort(contour_idx)
    contour_idx = [contour_idx[j] for j in order]
    if len(set(contour_idx)) != 4:
        return None
    return contour_idx, np.array([refined[j] for j in order])


def _rectangularity_deviation_deg(pts: np.ndarray) -> float:
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

    # Единая ориентация обхода контура для всех деталей (знак ориентированной
    # площади OpenCV < 0): от неё зависит, в каком порядке идут углы/стороны,
    # и то, что у двух соседних деталей общая линия разреза проходится в
    # противоположных направлениях — на этом строятся locate (поворот) и
    # сравнение формы сторон в match.
    if cv2.contourArea(contour.astype(np.float32), oriented=True) > 0:
        contour = contour[::-1].copy()
        piece.contour_px = [Point2D(x=float(x), y=float(y)) for x, y in contour]

    found = _find_corners(contour, ds)
    if found is None:
        piece.is_suspect = True
        piece.suspect_reason = piece.suspect_reason or "corner_detection_failed"
        logger.warning("describe: не удалось найти 4 угла для %s", piece.id)
        return piece

    corner_indices, corner_points = found
    if _rectangularity_deviation_deg(corner_points) > ds["corner_rectangularity_tolerance_deg"]:
        piece.is_suspect = True
        piece.suspect_reason = piece.suspect_reason or "non_rectangular"

    piece.corners_px = [Point2D(x=float(x), y=float(y)) for x, y in corner_points]

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
