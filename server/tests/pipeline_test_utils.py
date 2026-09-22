"""Сопоставление найденных деталей с эталонной разметкой синтетического
датасета — для измерения реальных метрик (found rate, точность
классификации) в тестах segment/describe, а не только "не упало с исключением".

Эталонные координаты (`center_px`) синтетического датасета лежат в системе
координат исходного холста (масштаб table_px_per_mm), а найденные детали —
в системе координат выпрямленного preprocess'ом кадра (масштаб
output_px_per_mm, свой сдвиг). Обе системы жёстко привязаны к маркеру, так
что пересчёт — это перенос вектора "точка минус угол маркера" из одного
масштаба в другой.
"""
from __future__ import annotations

import math

import numpy as np

from app.pipeline.schemas import PieceRecord


def expected_position_in_rectified(
    gt_center_px: tuple[float, float],
    gt_marker_topleft_px: tuple[float, float],
    table_px_per_mm: float,
    frame_marker_bbox_px: tuple[float, float, float, float],
    out_px_per_mm: float,
) -> tuple[float, float]:
    scale = out_px_per_mm / table_px_per_mm
    dx = (gt_center_px[0] - gt_marker_topleft_px[0]) * scale
    dy = (gt_center_px[1] - gt_marker_topleft_px[1]) * scale
    return frame_marker_bbox_px[0] + dx, frame_marker_bbox_px[1] + dy


def piece_centroid(piece: PieceRecord) -> tuple[float, float]:
    xs = [p.x for p in piece.contour_px]
    ys = [p.y for p in piece.contour_px]
    return float(np.mean(xs)), float(np.mean(ys))


def match_pieces_to_ground_truth(
    detected: list[PieceRecord],
    gt_batch: dict,
    frame_marker_bbox_px: tuple[float, float, float, float],
    table_px_per_mm: float,
    out_px_per_mm: float,
    tolerance_px: float,
) -> dict[str, tuple[PieceRecord, float]]:
    """Вернуть {gt_piece_id: (найденная_деталь, расстояние_px)} — только для
    эталонных деталей, у которых нашёлся детектированный кандидат в пределах
    tolerance_px (по умолчанию берём лучший/ближайший кандидат)."""
    gt_marker_topleft = tuple(gt_batch["marker"]["corners_px"][0])
    detected_info = [(piece_centroid(p), p) for p in detected]

    used_idx: set[int] = set()
    matches: dict[str, tuple[PieceRecord, float]] = {}

    gt_pieces = sorted(
        gt_batch["pieces"],
        key=lambda gt: 0,  # порядок эталона не важен для жадного сопоставления
    )
    for gt in gt_pieces:
        expected = expected_position_in_rectified(
            tuple(gt["center_px"]), gt_marker_topleft, table_px_per_mm, frame_marker_bbox_px, out_px_per_mm
        )
        best_idx, best_dist = None, math.inf
        for i, (center, _piece) in enumerate(detected_info):
            if i in used_idx:
                continue
            d = math.hypot(center[0] - expected[0], center[1] - expected[1])
            if d < best_dist:
                best_dist, best_idx = d, i
        if best_idx is not None and best_dist <= tolerance_px:
            used_idx.add(best_idx)
            matches[gt["id"]] = (detected_info[best_idx][1], best_dist)

    return matches
