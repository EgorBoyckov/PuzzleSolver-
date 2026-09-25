"""Модуль 2: сегментация деталей на кадре партии.

Первый проход — дешёвый порог по цвету фона (однотонный стол) + морфология;
компоненты, чья площадь заметно больше медианной, считаются слипшимися и
разделяются водоразделом (cv2.watershed), иначе помечаются сомнительными.
SAM2 для сложных случаев подключается отдельно (config.segment.sam2.enabled,
пока не реализовано — см. TODO ниже) и в этой функции не используется, пока
выключен.
"""
from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np

from app.core.config import get_config
from app.pipeline.schemas import PieceRecord

logger = logging.getLogger(__name__)

_PALETTE = [
    (66, 133, 244), (52, 168, 83), (251, 188, 5), (234, 67, 53),
    (154, 88, 235), (0, 172, 193), (255, 112, 67), (124, 179, 66),
]


def _estimate_background_bgr(image: np.ndarray, sample_size: int = 200) -> np.ndarray:
    small = cv2.resize(image, (sample_size, sample_size), interpolation=cv2.INTER_AREA)
    return np.median(small.reshape(-1, 3), axis=0)


def _background_distance(image: np.ndarray, background_bgr: np.ndarray) -> np.ndarray:
    """Евклидово расстояние в Lab от цвета фона — непрерывная «карта
    детальности» пикселя (float32)."""
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)
    bg_lab = cv2.cvtColor(np.uint8([[background_bgr]]), cv2.COLOR_BGR2LAB)[0, 0].astype(np.float32)
    return np.linalg.norm(lab - bg_lab, axis=2)


def _foreground_mask(image: np.ndarray, background_bgr: np.ndarray, distance_threshold: float) -> np.ndarray:
    dist = _background_distance(image, background_bgr)
    return (dist > distance_threshold).astype(np.uint8) * 255


def refine_contour_subpixel(dist: np.ndarray, contour: np.ndarray, search_px: float = 2.5, step_px: float = 0.25) -> np.ndarray:
    """Уточнить контур до субпикселя: каждая точка сдвигается вдоль нормали
    туда, где карта расстояния до фона пересекает середину между уровнем
    детали (изнутри) и фона (снаружи).

    Контур cv2.findContours целочисленный и после морфологии гуляет на ±1 px
    (0.17 мм при 6 px/мм) — это того же порядка, что различия формы замков
    соседних деталей, и прямо ограничивает сравнение сторон (match).
    Середина перепада (а не фиксированный порог) не смещается от контраста
    детали с фоном: тёмная и светлая кромка дают одну и ту же линию реза.
    """
    pts = contour.reshape(-1, 2).astype(np.float64)
    n = len(pts)
    if n < 8:
        return pts
    # Нормаль по сглаженной касательной (±3 точки), наружу — по ориентации контура.
    k = 3
    tang = np.roll(pts, -k, axis=0) - np.roll(pts, k, axis=0)
    tang /= np.linalg.norm(tang, axis=1, keepdims=True) + 1e-9
    # (t_y, -t_x) смотрит внутрь при отрицательной ориентированной площади
    # (так OpenCV обходит внешние контуры) — разворачиваем наружу.
    normal = np.stack([tang[:, 1], -tang[:, 0]], axis=1)
    if cv2.contourArea(pts.astype(np.float32), oriented=True) < 0:
        normal = -normal
    offsets = np.arange(-search_px - 1.0, search_px + 1.0 + 1e-9, step_px)
    samples = pts[:, None, :] + offsets[None, :, None] * normal[:, None, :]      # (n, m, 2)
    prof = cv2.remap(
        dist, samples[..., 0].astype(np.float32), samples[..., 1].astype(np.float32),
        interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE,
    )                                                                            # (n, m)
    inner = np.median(prof[:, offsets <= -search_px], axis=1)
    outer = np.median(prof[:, offsets >= search_px], axis=1)
    level = 0.5 * (inner + outer)
    above = prof >= level[:, None]
    # Ближайший к исходной точке переход «деталь -> фон» при движении наружу.
    crossing = above[:, :-1] & ~above[:, 1:]
    center = np.argmin(np.abs(offsets))
    refined = pts.copy()
    idx = np.arange(len(offsets) - 1)
    for i in np.where(crossing.any(axis=1) & (inner - outer > 10.0))[0]:
        cand = idx[crossing[i]]
        j = cand[np.argmin(np.abs(cand - center))]
        v0, v1 = prof[i, j], prof[i, j + 1]
        t = offsets[j] + step_px * (v0 - level[i]) / max(v0 - v1, 1e-6)
        if abs(t) <= search_px:
            refined[i] = pts[i] + t * normal[i]
    return refined


def _marker_paper_region(
    image: np.ndarray,
    marker_bbox_px: tuple[float, float, float, float],
    search_margin_px: float,
    white_luma_threshold: int = 200,
) -> np.ndarray:
    """Найти область белого листа A4 вокруг маркера заливкой от его границ,
    а не жёстким отступом в мм: печатный лист может быть любого размера
    (полный A4 или обрезанный), фиксированный отступ либо не докрывает белое
    поле, либо (что хуже) вырезает настоящие детали рядом с маркером."""
    h, w = image.shape[:2]
    x0, y0, x1, y1 = marker_bbox_px
    sx0 = max(0, int(x0 - search_margin_px))
    sy0 = max(0, int(y0 - search_margin_px))
    sx1 = min(w, int(x1 + search_margin_px))
    sy1 = min(h, int(y1 + search_margin_px))

    window_gray = cv2.cvtColor(image[sy0:sy1, sx0:sx1], cv2.COLOR_BGR2GRAY)
    white_mask = (window_gray > white_luma_threshold).astype(np.uint8) * 255

    mx0, my0 = max(0, int(x0 - sx0)), max(0, int(y0 - sy0))
    mx1, my1 = int(x1 - sx0), int(y1 - sy0)
    seed_mask = np.zeros_like(white_mask)
    seed_mask[my0:my1, mx0:mx1] = 255

    combined = cv2.bitwise_or(white_mask, seed_mask)
    n_labels, labels = cv2.connectedComponents(combined)
    cy, cx = (my0 + my1) // 2, (mx0 + mx1) // 2
    seed_label = labels[min(cy, labels.shape[0] - 1), min(cx, labels.shape[1] - 1)]

    region = np.zeros_like(white_mask)
    if seed_label != 0:
        region[labels == seed_label] = 255
    else:
        region = seed_mask

    region = cv2.dilate(region, np.ones((9, 9), np.uint8))

    full_mask = np.zeros((h, w), dtype=np.uint8)
    full_mask[sy0:sy1, sx0:sx1] = region
    return full_mask


def _apply_marker_exclusion(
    mask: np.ndarray,
    image: np.ndarray,
    marker_bbox_px: tuple[float, float, float, float] | None,
    safety_margin_px: float,
    search_margin_px: float,
) -> None:
    if marker_bbox_px is None:
        return
    paper_region = _marker_paper_region(image, marker_bbox_px, search_margin_px)
    mask[paper_region > 0] = 0

    # Небольшой фиксированный запас вокруг самого маркера — на случай, если
    # лист обрезан вплотную и заливка по белому почти ничего не захватила.
    x0, y0, x1, y1 = marker_bbox_px
    h, w = mask.shape[:2]
    ex0 = max(0, int(x0 - safety_margin_px))
    ey0 = max(0, int(y0 - safety_margin_px))
    ex1 = min(w, int(x1 + safety_margin_px))
    ey1 = min(h, int(y1 + safety_margin_px))
    mask[ey0:ey1, ex0:ex1] = 0


def _split_merged_component(image: np.ndarray, comp_mask: np.ndarray) -> list[np.ndarray]:
    """Разделить слипшийся компонент водоразделом. Возвращает список масок-частей."""
    dist = cv2.distanceTransform(comp_mask, cv2.DIST_L2, 5)
    if dist.max() <= 0:
        return [comp_mask]
    _, sure_fg = cv2.threshold(dist, 0.5 * dist.max(), 255, 0)
    sure_fg = sure_fg.astype(np.uint8)

    n_seeds, seed_labels = cv2.connectedComponents(sure_fg)
    if n_seeds <= 2:  # фон + один пик -> разделить не на что
        return [comp_mask]

    sure_bg = cv2.dilate(comp_mask, np.ones((3, 3), np.uint8), iterations=3)
    unknown = cv2.subtract(sure_bg, sure_fg)

    markers = seed_labels + 1
    markers[unknown > 0] = 0

    color_img = cv2.cvtColor(comp_mask, cv2.COLOR_GRAY2BGR) if image is None else image
    markers = cv2.watershed(color_img, markers.astype(np.int32))

    parts = []
    for label in range(2, n_seeds + 1):
        part_mask = np.zeros_like(comp_mask)
        part_mask[markers == label] = 255
        if part_mask.any():
            parts.append(part_mask)
    return parts if parts else [comp_mask]


def _largest_contour(mask: np.ndarray) -> np.ndarray | None:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None
    return max(contours, key=cv2.contourArea)


def _is_cut_by_frame(contour: np.ndarray, image_shape: tuple[int, int], margin_px: float) -> bool:
    h, w = image_shape[:2]
    x, y, cw, ch = cv2.boundingRect(contour)
    return x <= margin_px or y <= margin_px or (x + cw) >= (w - margin_px) or (y + ch) >= (h - margin_px)


def _save_thumbnail(image: np.ndarray, mask: np.ndarray, contour: np.ndarray, out_path: Path, pad_px: int = 6) -> None:
    x, y, w, h = cv2.boundingRect(contour)
    x0, y0 = max(0, x - pad_px), max(0, y - pad_px)
    x1, y1 = min(image.shape[1], x + w + pad_px), min(image.shape[0], y + h + pad_px)
    crop = image[y0:y1, x0:x1]
    crop_mask = mask[y0:y1, x0:x1]
    rgba = np.dstack([crop, crop_mask])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), rgba)


def segment_pieces(
    image: np.ndarray,
    mm_per_pixel: float,
    puzzle_id: str,
    batch_number: int,
    marker_bbox_px: tuple[float, float, float, float] | None = None,
    valid_mask: np.ndarray | None = None,
    output_dir: Path | None = None,
    debug_dir: Path | None = None,
) -> list[PieceRecord]:
    """Найти контур каждой детали на однотонном фоне, разделить слипшиеся
    водоразделом, пометить сомнительные (площадь вне ±40% от медианы, деталь
    обрезана краем кадра)."""
    cfg = get_config()
    sg = cfg.section("segment")

    if sg.get("sam2", {}).get("enabled", False):
        # TODO(этап 1+): интеграция SAM2 с точками-подсказками из fast-path
        # для сложных кадров (перекрытия, нетипичный фон). Пока используется
        # только дешёвый цветовой порог ниже.
        logger.warning("segment.sam2.enabled=true, но SAM2 ещё не реализован — использую color_threshold")

    background_bgr = _estimate_background_bgr(image)
    dist = _background_distance(image, background_bgr)
    mask = (dist > sg["background_color_distance_threshold"]).astype(np.uint8) * 255
    _apply_marker_exclusion(
        mask,
        image,
        marker_bbox_px,
        safety_margin_px=sg["marker_exclusion_margin_mm"] / mm_per_pixel,
        search_margin_px=sg["marker_paper_search_margin_mm"] / mm_per_pixel,
    )
    if valid_mask is not None:
        # Область вне проекции исходного кадра (чёрная рамка после
        # выпрямления перспективы preprocess'ом) — не деталь, исключаем.
        mask[valid_mask == 0] = 0

    kernel_size = max(3, int(round(sg["morph_kernel_mm"] / mm_per_pixel)) | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    min_area_px = sg["min_component_area_mm2"] / (mm_per_pixel ** 2)
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)

    component_masks = []
    for label in range(1, n_labels):
        area = stats[label, cv2.CC_STAT_AREA]
        if area < min_area_px:
            continue
        component_masks.append((labels == label).astype(np.uint8) * 255)

    if not component_masks:
        logger.warning("segment: не найдено ни одной детали на кадре batch=%d", batch_number)
        return []

    raw_areas = [float(cv2.countNonZero(m)) for m in component_masks]
    median_area = float(np.median(raw_areas))
    tol = sg["area_tolerance_fraction"]
    cluster_ratio = sg["cluster_area_ratio_threshold"]

    final_masks: list[tuple[np.ndarray, bool, str | None]] = []
    for comp_mask, area in zip(component_masks, raw_areas):
        if area > median_area * cluster_ratio:
            parts = _split_merged_component(image, comp_mask)
            if len(parts) > 1:
                for part in parts:
                    part_area = cv2.countNonZero(part)
                    ok = abs(part_area - median_area) / median_area <= tol
                    final_masks.append((part, not ok, None if ok else "odd_area"))
                continue
            final_masks.append((comp_mask, True, "merged"))
        elif abs(area - median_area) / median_area > tol:
            final_masks.append((comp_mask, True, "odd_area"))
        else:
            final_masks.append((comp_mask, False, None))

    border_margin_px = sg["frame_border_margin_mm"] / mm_per_pixel

    # Порядок сверху вниз, слева направо — стабильная и предсказуемая нумерация партии.
    items = []
    for comp_mask, is_suspect, reason in final_masks:
        contour = _largest_contour(comp_mask)
        if contour is None or len(contour) < 3:
            continue
        x, y, _, _ = cv2.boundingRect(contour)
        items.append((y, x, comp_mask, contour, is_suspect, reason))
    items.sort(key=lambda t: (t[0] // max(1, int(median_area ** 0.5)), t[1]))

    pieces: list[PieceRecord] = []
    debug_overlay = image.copy() if debug_dir is not None else None

    for i, (_, _, comp_mask, contour, is_suspect, reason) in enumerate(items, start=1):
        if _is_cut_by_frame(contour, image.shape, border_margin_px):
            is_suspect = True
            reason = reason or "cut_by_frame"

        piece_id = f"B{batch_number:02d}-{i:03d}"
        if sg.get("subpixel_contour", True):
            contour_pts = refine_contour_subpixel(dist, contour)
        else:
            contour_pts = contour.reshape(-1, 2).astype(float)

        thumb_path = None
        if output_dir is not None:
            thumb_path = output_dir / "pieces" / f"{piece_id}.png"
            _save_thumbnail(image, comp_mask, contour, thumb_path)

        pieces.append(
            PieceRecord(
                id=piece_id,
                puzzle_id=puzzle_id,
                batch_number=batch_number,
                number_in_batch=i,
                contour_px=[{"x": float(px), "y": float(py)} for px, py in contour_pts],
                thumbnail_path=str(thumb_path) if thumb_path else None,
                is_suspect=is_suspect,
                suspect_reason=reason,
            )
        )

        if debug_overlay is not None:
            color = (0, 0, 255) if is_suspect else _PALETTE[i % len(_PALETTE)]
            cv2.drawContours(debug_overlay, [contour], -1, color, 2)
            cx, cy = contour_pts.mean(axis=0).astype(int)
            cv2.putText(debug_overlay, str(i), (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)

    if debug_overlay is not None:
        debug_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(debug_dir / f"batch_{batch_number:02d}_segment_debug.jpg"), debug_overlay)
        cv2.imwrite(str(debug_dir / f"batch_{batch_number:02d}_mask.png"), mask)

    logger.info(
        "segment batch=%d: %d деталей (%d сомнительных)",
        batch_number,
        len(pieces),
        sum(1 for p in pieces if p.is_suspect),
    )
    return pieces
