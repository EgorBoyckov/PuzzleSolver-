"""Дескриптор лица детали, общий для describe (каталогизация) и locate
(привязка к образцу) — обе стороны сравнения (деталь и ячейка образца)
должны кодироваться одной и той же функцией.

Временная лёгкая замена DINOv2 (цветовая гистограмма Lab + Hu-моменты формы):
оба компонента естественно инвариантны к повороту (гистограмма не зависит от
пространственного расположения пикселей, Hu-моменты инвариантны к повороту
по построению), поэтому эмбеддинг детали, снятой под произвольным углом на
столе, сравним с эмбеддингом ячейки образца без предварительного выравнивания
ориентации — выравнивание нужно только на этапе точного сравнения (locate.py).
Настоящая модель подключается, когда для неё появится GPU-инференс в проде.
"""
from __future__ import annotations

import cv2
import numpy as np


def compute_placeholder_embedding(bgr_crop: np.ndarray, local_contour: np.ndarray) -> list[float]:
    mask = np.zeros(bgr_crop.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [local_contour.astype(np.int32)], 255)
    lab = cv2.cvtColor(bgr_crop, cv2.COLOR_BGR2LAB)
    hist = cv2.calcHist([lab], [0, 1, 2], mask, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    hist = cv2.normalize(hist, None).flatten()

    moments = cv2.HuMoments(cv2.moments(local_contour)).flatten()
    moments = np.sign(moments) * np.log1p(np.abs(moments) * 1e6)

    return np.concatenate([hist, moments]).astype(float).tolist()


def compute_embedding_for_patch(bgr_patch: np.ndarray) -> list[float]:
    """То же самое для прямоугольного патча БЕЗ явного контура (например,
    ячейки образца) — контур берётся равным границе патча."""
    h, w = bgr_patch.shape[:2]
    rect_contour = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.float64)
    return compute_placeholder_embedding(bgr_patch, rect_contour)
