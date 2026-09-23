"""Модуль 1: предобработка кадра партии деталей.

Находит ArUco-маркер, выпрямляет перспективу в вид строго сверху (гомография
по 4 углам маркера — плоскость стола и маркер копланарны, поэтому та же
гомография выпрямляет и все детали на столе), считает масштаб мм/пиксель,
делает баланс белого по белому полю листа A4 вокруг маркера и отбраковывает
размытые/пересвеченные кадры или кадры без маркера.
"""
from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np

from app.core.config import get_config
from app.pipeline.schemas import BatchFrame

logger = logging.getLogger(__name__)

ARUCO_DICT_NAMES = {
    "DICT_4X4_50": cv2.aruco.DICT_4X4_50,
    "DICT_5X5_50": cv2.aruco.DICT_5X5_50,
    "DICT_6X6_50": cv2.aruco.DICT_6X6_50,
}


def _reject(puzzle_id: str, batch_number: int, image_path: Path, reason: str, marker_found: bool) -> BatchFrame:
    logger.info("frame rejected (%s): %s", reason, image_path)
    return BatchFrame(
        puzzle_id=puzzle_id,
        batch_number=batch_number,
        image_path=str(image_path),
        mm_per_pixel=0.0,
        marker_found=marker_found,
        accepted=False,
        rejection_reason=reason,
    )


def _blur_variance(gray: np.ndarray) -> float:
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def _overexposed_fraction(gray: np.ndarray, luma_threshold: int, exclude_mask: np.ndarray | None = None) -> float:
    if exclude_mask is not None:
        considered = gray[exclude_mask == 0]
        if considered.size == 0:
            return 0.0
        return float((considered > luma_threshold).mean())
    return float((gray > luma_threshold).mean())


def _marker_exclusion_mask(shape: tuple[int, int], marker_corners: np.ndarray, expand_factor: float) -> np.ndarray:
    """Маска региона белого листа A4 вокруг маркера — легитимно яркая область,
    её не нужно учитывать при отбраковке кадра как пересвеченного."""
    mask = np.zeros(shape, dtype=np.uint8)
    center = marker_corners.mean(axis=0)
    side = float(np.mean(np.linalg.norm(marker_corners - np.roll(marker_corners, -1, axis=0), axis=1)))
    half = side * expand_factor / 2.0
    x0 = max(0, int(center[0] - half))
    y0 = max(0, int(center[1] - half))
    x1 = min(shape[1], int(center[0] + half))
    y1 = min(shape[0], int(center[1] + half))
    mask[y0:y1, x0:x1] = 255
    return mask


def _detect_marker(img: np.ndarray, dictionary_name: str, marker_id: int) -> np.ndarray | None:
    """Вернуть 4 угла маркера (порядок cv2.aruco: TL, TR, BR, BL) или None."""
    aruco_dict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT_NAMES[dictionary_name])
    detector = cv2.aruco.ArucoDetector(aruco_dict, cv2.aruco.DetectorParameters())
    corners, ids, _ = detector.detectMarkers(img)
    if ids is None:
        return None
    ids_flat = ids.flatten().tolist()
    if marker_id not in ids_flat:
        return None
    idx = ids_flat.index(marker_id)
    return corners[idx][0].astype(np.float32)


def detect_marker_and_scale(
    img: np.ndarray, dictionary_name: str, marker_id: int, marker_size_mm: float
) -> tuple[np.ndarray | None, float | None]:
    """Как _detect_marker, но также оценивает масштаб мм/пиксель ПРЯМО в
    исходном (невыпрямленном) кадре — в отличие от preprocess_frame, где
    масштаб после гомографии — всегда фиксированная config.preprocess.
    output_px_per_mm константа (гомография её нормирует), масштаб на сыром
    кадре зависит от расстояния/угла конкретной съёмки и оценивается по
    видимому размеру маркера в пикселях. Нужно для этапа 6 (AR-оверлей
    поверх живого видео камеры, где перерисовывать через гомографию каждый
    кадр не имеет смысла — оверлей должен лечь на ТО ЖЕ изображение,
    которое видит пользователь)."""
    corners = _detect_marker(img, dictionary_name, marker_id)
    if corners is None:
        return None, None
    side_px = float(np.mean(np.linalg.norm(corners - np.roll(corners, -1, axis=0), axis=1)))
    if side_px <= 0:
        return corners, None
    return corners, marker_size_mm / side_px


def _white_balance_gains(
    rectified: np.ndarray,
    marker_dst: np.ndarray,
    ring_px: float,
    gain_clamp: tuple[float, float],
) -> np.ndarray:
    """Оценить множители баланса белого по кольцу белой бумаги вокруг маркера."""
    h, w = rectified.shape[:2]
    x0, y0 = marker_dst.min(axis=0)
    x1, y1 = marker_dst.max(axis=0)
    ox0 = max(0, int(x0 - ring_px))
    oy0 = max(0, int(y0 - ring_px))
    ox1 = min(w, int(x1 + ring_px))
    oy1 = min(h, int(y1 + ring_px))

    ring_mask = np.zeros((h, w), dtype=np.uint8)
    ring_mask[oy0:oy1, ox0:ox1] = 255
    ring_mask[int(y0) : int(y1), int(x0) : int(x1)] = 0  # вырезать сам маркер (чёрно-белый паттерн)

    pixels = rectified[ring_mask > 0]
    if pixels.size == 0:
        return np.array([1.0, 1.0, 1.0])

    mean_bgr = pixels.reshape(-1, 3).mean(axis=0)
    mean_bgr = np.clip(mean_bgr, 1.0, 255.0)
    target = 245.0
    gains = target / mean_bgr
    return np.clip(gains, gain_clamp[0], gain_clamp[1])


def preprocess_frame(
    image_path: Path,
    puzzle_id: str,
    batch_number: int,
    debug_dir: Path | None = None,
) -> tuple[BatchFrame, np.ndarray | None, np.ndarray | None]:
    """Найти ArUco-маркер, выпрямить перспективу, посчитать мм/пиксель,
    сделать баланс белого и отбраковать размытые/пересвеченные кадры.

    Возвращает (BatchFrame с метаданными, выпрямленное BGR-изображение,
    маска валидных пикселей кадра — 0 в области вне проекции исходного фото,
    которую сегментация должна исключить). Изображение и маска — None, если
    кадр отбракован.
    """
    cfg = get_config()
    pp = cfg.section("preprocess")
    aruco_cfg = cfg.section("aruco")

    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(f"Не удалось прочитать изображение: {image_path}")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    blur_var = _blur_variance(gray)
    if blur_var < pp["blur_variance_threshold"]:
        return _reject(puzzle_id, batch_number, image_path, "blurry", marker_found=False), None, None

    marker_corners = _detect_marker(img, aruco_cfg["dictionary"], aruco_cfg["marker_id"])
    if marker_corners is None:
        return _reject(puzzle_id, batch_number, image_path, "marker_not_found", marker_found=False), None, None

    # Белый лист A4 вокруг маркера легитимно яркий — не считаем его "пересветом".
    marker_mask = _marker_exclusion_mask(gray.shape, marker_corners, pp["overexposure_marker_exclusion_factor"])
    overexposed = _overexposed_fraction(gray, pp["overexposure_luma"], exclude_mask=marker_mask)
    if overexposed > pp["overexposure_fraction_threshold"]:
        return _reject(puzzle_id, batch_number, image_path, "overexposed", marker_found=True), None, None

    marker_size_mm = float(aruco_cfg["marker_size_mm"])
    out_px_per_mm = float(pp["output_px_per_mm"])
    margin_px = float(pp["output_margin_mm"]) * out_px_per_mm
    marker_side_px = marker_size_mm * out_px_per_mm

    # Гомография по маркеру: устанавливает правильный МАСШТАБ и поворот, но
    # исходное смещение (marker_corners -> dst_provisional) может унести часть
    # стола (и весь холст) в отрицательные координаты или оставить лишнее
    # пустое поле — величина сдвига зависит от того, где именно относительно
    # своего угла лежит маркер на конкретном фото. Поэтому считаем финальный
    # размер и сдвиг холста от РЕАЛЬНОЙ проекции углов исходного кадра, а не
    # от предположения "выходной кадр ~ входной, растянутый по масштабу".
    dst_provisional = np.array(
        [[0, 0], [marker_side_px, 0], [marker_side_px, marker_side_px], [0, marker_side_px]],
        dtype=np.float32,
    )
    H_provisional = cv2.getPerspectiveTransform(marker_corners, dst_provisional)

    h_in, w_in = img.shape[:2]
    src_corners = np.array([[0, 0], [w_in, 0], [w_in, h_in], [0, h_in]], dtype=np.float32).reshape(-1, 1, 2)
    mapped_corners = cv2.perspectiveTransform(src_corners, H_provisional).reshape(-1, 2)

    min_xy = mapped_corners.min(axis=0)
    max_xy = mapped_corners.max(axis=0)
    shift = -min_xy + margin_px

    T = np.array([[1, 0, shift[0]], [0, 1, shift[1]], [0, 0, 1]], dtype=np.float64)
    H = T @ H_provisional

    out_w = min(int(np.ceil(max_xy[0] - min_xy[0])) + 2 * int(margin_px), pp["max_output_canvas_px"])
    out_h = min(int(np.ceil(max_xy[1] - min_xy[1])) + 2 * int(margin_px), pp["max_output_canvas_px"])

    rectified = cv2.warpPerspective(img, H, (out_w, out_h))

    # warpPerspective заливает область холста вне проекции исходного кадра
    # чёрным (BORDER_CONSTANT) — эта рамка иначе резко отличается от фона
    # стола и ошибочно засчитывается сегментацией как "деталь". Прогоняем ту
    # же гомографию по маске "здесь есть реальные пиксели кадра", чтобы
    # segment_pieces мог явно исключить эту область, а не гадать по цвету.
    valid_mask = cv2.warpPerspective(
        np.full((h_in, w_in), 255, dtype=np.uint8), H, (out_w, out_h), flags=cv2.INTER_NEAREST
    )
    erode_px = max(3, int(round(2.0 * out_px_per_mm)))
    valid_mask = cv2.erode(valid_mask, np.ones((erode_px, erode_px), np.uint8))

    dst = dst_provisional + shift  # фактические углы маркера в выпрямленном кадре

    gains = _white_balance_gains(
        rectified,
        dst,
        ring_px=float(pp["white_balance_ring_mm"]) * out_px_per_mm,
        gain_clamp=tuple(pp["white_balance_gain_clamp"]),
    )
    rectified = np.clip(rectified.astype(np.float32) * gains, 0, 255).astype(np.uint8)

    marker_bbox = (
        float(dst[:, 0].min()),
        float(dst[:, 1].min()),
        float(dst[:, 0].max()),
        float(dst[:, 1].max()),
    )

    frame = BatchFrame(
        puzzle_id=puzzle_id,
        batch_number=batch_number,
        image_path=str(image_path),
        mm_per_pixel=1.0 / out_px_per_mm,
        marker_found=True,
        accepted=True,
        rejection_reason=None,
        marker_bbox_px=marker_bbox,
    )

    if debug_dir is not None:
        debug_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(debug_dir / f"batch_{batch_number:02d}_rectified.jpg"), rectified)
        overlay = img.copy()
        cv2.polylines(overlay, [marker_corners.astype(np.int32)], True, (0, 0, 255), 3)
        cv2.imwrite(str(debug_dir / f"batch_{batch_number:02d}_marker_debug.jpg"), overlay)

    logger.info(
        "preprocessed %s: blur=%.1f overexposed=%.3f mm_per_px=%.4f canvas=%dx%d",
        image_path,
        blur_var,
        overexposed,
        frame.mm_per_pixel,
        out_w,
        out_h,
    )
    return frame, rectified, valid_mask
