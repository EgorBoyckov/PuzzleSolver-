"""Процедурная генерация "картинки коробки", если пользователь не передал свою.

Даёт достаточно текстуры (границы форм, вариация цвета), чтобы алгоритмы
locate/describe следующих этапов имели на чём тестироваться, и не требует
внешних файлов изображений для запуска синтетического датасета.
"""
from __future__ import annotations

import numpy as np


def generate_box_art(width_px: int, height_px: int, rng: np.random.Generator) -> np.ndarray:
    """Вернуть BGR uint8 изображение width_px x height_px с случайными
    цветными пятнами (имитация сюжетной картинки пазла)."""
    img = np.zeros((height_px, width_px, 3), dtype=np.float32)

    # Плавный цветовой градиент-подложка.
    yy, xx = np.mgrid[0:height_px, 0:width_px].astype(np.float32)
    base_color_a = rng.uniform(20, 235, size=3)
    base_color_b = rng.uniform(20, 235, size=3)
    t = (xx / width_px + yy / height_px) / 2.0
    for ch in range(3):
        img[:, :, ch] = base_color_a[ch] * (1 - t) + base_color_b[ch] * t

    # Случайные "объекты" — размытые цветные эллипсы/пятна разных масштабов.
    n_blobs = rng.integers(25, 60)
    for _ in range(n_blobs):
        cx = rng.uniform(0, width_px)
        cy = rng.uniform(0, height_px)
        rx = rng.uniform(0.03, 0.25) * width_px
        ry = rng.uniform(0.03, 0.25) * height_px
        color = rng.uniform(0, 255, size=3)
        alpha = rng.uniform(0.25, 0.6)
        mask = ((xx - cx) / max(rx, 1)) ** 2 + ((yy - cy) / max(ry, 1)) ** 2 <= 1.0
        img[mask] = img[mask] * (1 - alpha) + color * alpha

    img = np.clip(img, 0, 255).astype(np.uint8)

    import cv2

    img = cv2.GaussianBlur(img, (0, 0), sigmaX=max(1.0, width_px / 400))
    return img
