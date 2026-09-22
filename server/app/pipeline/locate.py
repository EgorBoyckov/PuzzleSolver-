"""Модуль 4: привязка детали к образцу — ключевой шаг конвейера.

Картинка коробки делится на сетку по числу деталей пазла. Для каждой детали:

1. Быстрый отбор: топ-K ближайших по эмбеддингу ячеек (тот же дескриптор
   цвета+формы, что в describe/embedding.py — он уже инвариантен к повороту,
   так что канонизация ориентации для этого шага не нужна).
2. Точное сравнение: деталь канонизируется (поворачивается так, чтобы её
   собственный угол 0->1 лёг горизонтально — убирает произвольный угол, под
   которым она лежала на столе) и сравнивается с каждым кандидатом цветным
   (BGR, не оттенки серого — цвет точек текстуры образца сильно
   дискриминативен, серая яркость почти всё это теряет) маскированным
   шаблонным сопоставлением (cv2.matchTemplate, TM_CCORR_NORMED + маска по
   силуэту детали) в 4 поворотах на 90° (0/90/180/270 — оставшаяся
   неоднозначность "какой угол считать первым"), со скользящим поиском в
   расширенной области ячейки (выступы детали торчат за номинальную границу
   ячейки образца, а у соседних ячеек паддинг-окна перекрываются — поэтому
   близкие по счёту кандидаты дополнительно ранжируются небольшой добавкой
   от эмбеддинга, см. embedding_tiebreak_weight).

Цветокоррекция образца (`refit_reference_colors`) подгоняет цвет картинки
коробки под то, как реально выглядят сфотографированные детали — используя
только уверенные совпадения первого прохода — и пересобирает индекс для
повторного поиска по неуверенным деталям.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import cv2
import numpy as np

from app.core.config import get_config
from app.pipeline.embedding import compute_embedding_for_patch
from app.pipeline.schemas import PieceRecord

logger = logging.getLogger(__name__)


@dataclass
class ReferenceGridIndex:
    reference_bgr: np.ndarray
    rows: int
    cols: int
    cell_w: float
    cell_h: float
    embeddings: np.ndarray  # (rows*cols, D)

    def cell_bounds_px(self, row: int, col: int) -> tuple[int, int, int, int]:
        x0, y0 = int(round(col * self.cell_w)), int(round(row * self.cell_h))
        x1, y1 = int(round((col + 1) * self.cell_w)), int(round((row + 1) * self.cell_h))
        return x0, y0, x1, y1


def build_reference_index(reference_bgr: np.ndarray, rows: int, cols: int) -> ReferenceGridIndex:
    h, w = reference_bgr.shape[:2]
    cell_w, cell_h = w / cols, h / rows
    index = ReferenceGridIndex(reference_bgr, rows, cols, cell_w, cell_h, np.zeros((rows * cols, 1)))

    embeddings = []
    for r in range(rows):
        for c in range(cols):
            x0, y0, x1, y1 = index.cell_bounds_px(r, c)
            patch = reference_bgr[y0:y1, x0:x1]
            if patch.size == 0:
                embeddings.append(np.zeros(71))  # 8*8*8 hist + 7 Hu moments
                continue
            embeddings.append(compute_embedding_for_patch(patch))
    index.embeddings = np.array(embeddings, dtype=np.float64)
    return index


def _canonical_piece_crop(frame_bgr: np.ndarray, piece: PieceRecord, target_size: tuple[int, int]) -> np.ndarray:
    """Повернуть и вырезать деталь так, чтобы угол corners_px[0]->[1] был
    горизонтален (снимает произвольный угол укладки на столе), вырезать по
    контуру (с альфа-маской) и привести к target_size (w, h)."""
    corners = np.array([[p.x, p.y] for p in piece.corners_px], dtype=np.float64)
    contour = np.array([[p.x, p.y] for p in piece.contour_px], dtype=np.float64)

    edge = corners[1] - corners[0]
    angle_deg = np.degrees(np.arctan2(edge[1], edge[0]))

    centroid = contour.mean(axis=0)
    xmin, ymin = contour.min(axis=0)
    xmax, ymax = contour.max(axis=0)
    diag = float(np.hypot(xmax - xmin, ymax - ymin))
    pad = diag * 0.6 + 5

    cx0 = max(0, int(centroid[0] - diag / 2 - pad))
    cy0 = max(0, int(centroid[1] - diag / 2 - pad))
    cx1 = min(frame_bgr.shape[1], int(centroid[0] + diag / 2 + pad))
    cy1 = min(frame_bgr.shape[0], int(centroid[1] + diag / 2 + pad))
    local_bgr = frame_bgr[cy0:cy1, cx0:cx1]
    local_contour = contour - np.array([cx0, cy0])
    local_center = centroid - np.array([cx0, cy0])

    h0, w0 = local_bgr.shape[:2]
    M = cv2.getRotationMatrix2D(tuple(local_center), angle_deg, 1.0)
    cos, sin = abs(M[0, 0]), abs(M[0, 1])
    new_w = max(1, int(np.ceil(h0 * sin + w0 * cos)))
    new_h = max(1, int(np.ceil(h0 * cos + w0 * sin)))
    M[0, 2] += new_w / 2.0 - local_center[0]
    M[1, 2] += new_h / 2.0 - local_center[1]

    rotated_bgr = cv2.warpAffine(local_bgr, M, (new_w, new_h))
    rotated_contour = local_contour @ M[:, :2].T + M[:, 2]

    mask = np.zeros((new_h, new_w), dtype=np.uint8)
    cv2.fillPoly(mask, [rotated_contour.astype(np.int32)], 255)

    rx0, ry0 = rotated_contour.min(axis=0)
    rx1, ry1 = rotated_contour.max(axis=0)
    rx0, ry0 = max(0, int(rx0)), max(0, int(ry0))
    rx1, ry1 = min(new_w, int(np.ceil(rx1))), min(new_h, int(np.ceil(ry1)))
    tight_bgr = rotated_bgr[ry0:ry1, rx0:rx1]
    tight_mask = mask[ry0:ry1, rx0:rx1]
    if tight_bgr.size == 0:
        tight_bgr = rotated_bgr
        tight_mask = mask

    # Равномерный масштаб (один и тот же коэффициент по X и Y), затем
    # дополнение прозрачным фоном до target_size — раньше resize растягивал
    # bbox детали НЕЗАВИСИМО по ширине/высоте до квадрата, что при типично
    # неквадратном bbox (выступы/впадины редко симметричны по обеим осям)
    # заметно искажало внутренний узор и портило шаблонное сопоставление.
    th, tw = tight_bgr.shape[:2]
    scale = min(target_size[0] / max(tw, 1), target_size[1] / max(th, 1))
    new_tw, new_th = max(1, int(round(tw * scale))), max(1, int(round(th * scale)))
    scaled_bgr = cv2.resize(tight_bgr, (new_tw, new_th), interpolation=cv2.INTER_AREA)
    scaled_mask = cv2.resize(tight_mask, (new_tw, new_th), interpolation=cv2.INTER_NEAREST)

    out_bgr = np.zeros((target_size[1], target_size[0], 3), dtype=np.uint8)
    out_mask = np.zeros((target_size[1], target_size[0]), dtype=np.uint8)
    ox, oy = (target_size[0] - new_tw) // 2, (target_size[1] - new_th) // 2
    out_bgr[oy : oy + new_th, ox : ox + new_tw] = scaled_bgr
    out_mask[oy : oy + new_th, ox : ox + new_tw] = scaled_mask
    return np.dstack([out_bgr, out_mask])


def _rotate_gray(gray: np.ndarray, degrees: int) -> np.ndarray:
    degrees = degrees % 360
    if degrees == 0:
        return gray
    if degrees == 90:
        return cv2.rotate(gray, cv2.ROTATE_90_CLOCKWISE)
    if degrees == 180:
        return cv2.rotate(gray, cv2.ROTATE_180)
    if degrees == 270:
        return cv2.rotate(gray, cv2.ROTATE_90_COUNTERCLOCKWISE)
    raise ValueError(f"Неподдерживаемый поворот: {degrees}")


def locate_piece(
    piece: PieceRecord,
    frame_bgr: np.ndarray,
    reference_index: ReferenceGridIndex,
) -> None:
    """Сравнить деталь с участками образца в 4 поворотах: сперва топ-K по
    эмбеддингу, затем точное сравнение шаблонным сопоставлением. Заполняет
    piece.location_candidates (топ-3 ячейки с уверенностью и поворотом)."""
    cfg = get_config()
    lc = cfg.section("locate")

    if piece.embedding is None or len(piece.corners_px) != 4:
        return

    piece_emb = np.array(piece.embedding, dtype=np.float64)
    if piece_emb.shape[0] != reference_index.embeddings.shape[1]:
        logger.warning("locate: размерность эмбеддинга детали %s не совпадает с индексом образца", piece.id)
        return

    dists = np.linalg.norm(reference_index.embeddings - piece_emb, axis=1)
    total_cells = reference_index.rows * reference_index.cols
    top_k_raw = round(float(lc["top_k_fraction"]) * total_cells)
    top_k_clamped = max(int(lc["top_k_min"]), min(int(lc["top_k_max"]), top_k_raw))
    top_k = min(top_k_clamped, dists.shape[0])
    candidate_idx = np.argsort(dists)[:top_k]
    cand_dist_min, cand_dist_max = float(dists[candidate_idx].min()), float(dists[candidate_idx].max())
    cand_dist_range = max(cand_dist_max - cand_dist_min, 1e-9)

    cell_factor = float(lc["canonical_crop_cell_factor"])
    cell_px = (
        max(24, int(round(reference_index.cell_w * cell_factor))),
        max(24, int(round(reference_index.cell_h * cell_factor))),
    )
    canonical = _canonical_piece_crop(frame_bgr, piece, cell_px)
    canonical_bgr = canonical[:, :, :3]
    canonical_alpha = canonical[:, :, 3]

    # Небольшое размытие перед корреляцией: снижает чувствительность к
    # субпиксельным неточностям выравнивания (поворот/масштаб канонизации
    # никогда не идеальны), сохраняя структуру деталей крупнее пары пикселей.
    # Сравниваем по цвету (BGR), а не в оттенках серого — оттенок точек
    # текстуры образца несёт много различающей информации, которую серая
    # яркость отбрасывает.
    blur_sigma = float(lc["match_blur_sigma_px"])
    canonical_bgr = cv2.GaussianBlur(canonical_bgr, (0, 0), sigmaX=blur_sigma)

    rotations = [int(r) for r in lc["rotations_deg"]]
    rotated_masks = {rot: _rotate_gray(canonical_alpha, rot) for rot in rotations}
    rotated_templates = {rot: _rotate_gray(canonical_bgr, rot) for rot in rotations}

    pad_frac = float(lc["search_padding_fraction"])
    ref_h, ref_w = reference_index.reference_bgr.shape[:2]
    scored: list[tuple[float, int, int, int]] = []

    for idx in candidate_idx:
        row, col = divmod(int(idx), reference_index.cols)
        x0, y0, x1, y1 = reference_index.cell_bounds_px(row, col)
        pad_x = int((x1 - x0) * pad_frac)
        pad_y = int((y1 - y0) * pad_frac)
        # Клетки у края сетки: если паддинга не хватает с одной стороны
        # (упирается в границу образца), компенсируем его с другой стороны,
        # иначе область поиска сжимается почти до размера самого шаблона и
        # скользящему сопоставлению банально негде искать правильное
        # смещение (только 2-4 позиции вместо полноценного окна поиска).
        px0, px1 = x0 - pad_x, x1 + pad_x
        if px0 < 0:
            px1 += -px0
            px0 = 0
        if px1 > ref_w:
            px0 -= px1 - ref_w
            px1 = ref_w
        px0 = max(0, px0)

        py0, py1 = y0 - pad_y, y1 + pad_y
        if py0 < 0:
            py1 += -py0
            py0 = 0
        if py1 > ref_h:
            py0 -= py1 - ref_h
            py1 = ref_h
        py0 = max(0, py0)

        ref_patch = reference_index.reference_bgr[py0:py1, px0:px1]
        if ref_patch.size == 0:
            continue
        ref_bgr = cv2.GaussianBlur(ref_patch, (0, 0), sigmaX=blur_sigma)

        best_rot, best_score = rotations[0], -1.0
        for rot, template in rotated_templates.items():
            if template.shape[0] > ref_bgr.shape[0] or template.shape[1] > ref_bgr.shape[1]:
                continue
            # Маскированное сопоставление (TM_CCOEFF_NORMED маску не
            # поддерживает) — иначе фон стола вокруг детали (за пределами
            # альфа-маски) попадает в корреляцию и полностью её портит.
            mask = rotated_masks[rot]
            result = cv2.matchTemplate(ref_bgr, template, cv2.TM_CCORR_NORMED, mask=mask)
            score = float(result.max())
            if score > best_score:
                best_score, best_rot = score, rot
        # Эмбеддинг как tie-breaker: у соседних клеток корреляция часто почти
        # идентична (перекрывающиеся паддинг-окна), а эмбеддинг обычно верно
        # ранжирует истинную ячейку среди первых кандидатов — небольшая
        # добавка склоняет выбор в её пользу, не переворачивая явных лидеров.
        embedding_bonus = float(lc["embedding_tiebreak_weight"]) * (1.0 - (dists[idx] - cand_dist_min) / cand_dist_range)
        combined_score = best_score + embedding_bonus
        scored.append((combined_score, row, col, best_rot))

    scored.sort(key=lambda t: -t[0])
    top_n = int(lc["top_k_result_cells"])
    # TM_CCORR_NORMED на uint8-изображениях (всегда >=0) даёт результат в [0,1].
    piece.location_candidates = [
        (row, col, rot, max(0.0, min(1.0, score))) for score, row, col, rot in scored[:top_n]
    ]


def refit_reference_colors(
    reference_index: ReferenceGridIndex,
    pieces: list[PieceRecord],
    frame_lookup: dict[int, np.ndarray],
    confidence_threshold: float,
    min_samples: int,
) -> ReferenceGridIndex | None:
    """Подогнать цвет образца по уверенным совпадениям первого прохода:
    линейная (per-channel) коррекция BGR-каналов образца так, чтобы средний
    цвет уверенно найденных деталей был ближе к среднему цвету их ячеек.
    Возвращает пересобранный индекс или None, если уверенных совпадений
    недостаточно для устойчивой оценки.
    """
    confident = [
        p
        for p in pieces
        if p.location_candidates and p.location_candidates[0][3] >= confidence_threshold and p.batch_number in frame_lookup
    ]
    if len(confident) < min_samples:
        logger.info("locate: недостаточно уверенных деталей для цветокоррекции (%d < %d)", len(confident), min_samples)
        return None

    ratios = []
    for p in confident:
        frame_bgr = frame_lookup[p.batch_number]
        row, col, _rot, _conf = p.location_candidates[0]
        x0, y0, x1, y1 = reference_index.cell_bounds_px(row, col)
        ref_patch = reference_index.reference_bgr[y0:y1, x0:x1]
        if ref_patch.size == 0:
            continue
        ref_mean = ref_patch.reshape(-1, 3).mean(axis=0)

        contour = np.array([[pt.x, pt.y] for pt in p.contour_px], dtype=np.int32)
        mask = np.zeros(frame_bgr.shape[:2], dtype=np.uint8)
        cv2.fillPoly(mask, [contour], 255)
        piece_pixels = frame_bgr[mask > 0]
        if piece_pixels.size == 0:
            continue
        piece_mean = piece_pixels.reshape(-1, 3).mean(axis=0)

        ratios.append(piece_mean / np.clip(ref_mean, 1.0, 255.0))

    if len(ratios) < min_samples:
        return None

    avg_ratio = np.clip(np.median(np.array(ratios), axis=0), 0.4, 2.5)
    corrected = np.clip(reference_index.reference_bgr.astype(np.float64) * avg_ratio, 0, 255).astype(np.uint8)
    logger.info("locate: цветокоррекция образца по %d деталям, коэффициенты BGR=%s", len(ratios), avg_ratio.round(3))
    return build_reference_index(corrected, reference_index.rows, reference_index.cols)
