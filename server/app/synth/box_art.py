"""Процедурная генерация "картинки коробки", если пользователь не передал свою.

Две составляющие, как у настоящей картинки пазла:
  - крупные плавные цветовые зоны (аналог неба/воды — с "локальным" минимумом
    высокочастотной информации, "однотонные зоны" из критериев приёмки locate);
  - множество мелких контрастных пятен ("текстурные зоны" — аналог построек/
    надписей/мелких объектов), размер которых меньше ячейки сетки деталей,
    чтобы соседние ячейки отличались друг от друга и было на чём проверять
    привязку к образцу по локальной текстуре.

Не требует внешних файлов изображений для запуска синтетического датасета.
"""
from __future__ import annotations

import cv2
import numpy as np


def generate_box_art(
    width_px: int,
    height_px: int,
    rng: np.random.Generator,
    textured_fraction: float = 0.55,
) -> np.ndarray:
    """Вернуть BGR uint8 изображение width_px x height_px.

    textured_fraction — примерная доля площади с мелкой текстурой (остальное
    остаётся плавными однотонными зонами) — используется тестами/скриптом
    приёмки этапа 2, чтобы отдельно посчитать точность locate по текстурным
    и однотонным зонам, как того требуют критерии приёмки.
    """
    img = np.zeros((height_px, width_px, 3), dtype=np.float32)
    yy, xx = np.mgrid[0:height_px, 0:width_px].astype(np.float32)

    # 1. Плавный цветовой градиент-подложка.
    base_color_a = rng.uniform(20, 235, size=3)
    base_color_b = rng.uniform(20, 235, size=3)
    t = (xx / width_px + yy / height_px) / 2.0
    for ch in range(3):
        img[:, :, ch] = base_color_a[ch] * (1 - t) + base_color_b[ch] * t

    # 2. Крупные плавные цветовые зоны (однотонные зоны — небо/вода).
    n_blobs = int(rng.integers(15, 30))
    for _ in range(n_blobs):
        cx, cy = rng.uniform(0, width_px), rng.uniform(0, height_px)
        rx = rng.uniform(0.08, 0.28) * width_px
        ry = rng.uniform(0.08, 0.28) * height_px
        color = rng.uniform(0, 255, size=3)
        alpha = rng.uniform(0.25, 0.55)
        mask = ((xx - cx) / max(rx, 1)) ** 2 + ((yy - cy) / max(ry, 1)) ** 2 <= 1.0
        img[mask] = img[mask] * (1 - alpha) + color * alpha

    img = cv2.GaussianBlur(img, (0, 0), sigmaX=max(1.0, width_px / 400))

    # 3. Область с мелкой текстурой — несколько крупных неправильных пятен
    # (объединение больших кругов), внутрь которых кладутся мелкие детали.
    textured_mask = np.zeros((height_px, width_px), dtype=bool)
    target_area = textured_fraction * width_px * height_px
    covered = 0.0
    guard = 0
    while covered < target_area and guard < 200:
        guard += 1
        cx, cy = rng.uniform(0, width_px), rng.uniform(0, height_px)
        r = rng.uniform(0.15, 0.32) * min(width_px, height_px)
        region = (xx - cx) ** 2 + (yy - cy) ** 2 <= r * r
        newly = region & ~textured_mask
        covered += float(newly.sum())
        textured_mask |= region

    # 4. Мелкие контрастные пятна ("постройки/надписи") — размер меньше
    # ожидаемой ячейки сетки (единицы-десятки px), только внутри textured_mask.
    textured_px = int(textured_mask.sum())
    n_fine = min(4000, max(150, textured_px // 900))
    ys_idx, xs_idx = np.where(textured_mask)
    if len(xs_idx) > 0:
        pick = rng.integers(0, len(xs_idx), size=n_fine)
        for i in pick:
            cx, cy = float(xs_idx[i]), float(ys_idx[i])
            r = rng.uniform(8.0, 20.0)
            color = rng.uniform(0, 255, size=3)
            x0, x1 = max(0, int(cx - r)), min(width_px, int(cx + r) + 1)
            y0, y1 = max(0, int(cy - r)), min(height_px, int(cy + r) + 1)
            if x1 <= x0 or y1 <= y0:
                continue
            sub_xx, sub_yy = xx[y0:y1, x0:x1], yy[y0:y1, x0:x1]
            sub_mask = (sub_xx - cx) ** 2 + (sub_yy - cy) ** 2 <= r * r
            alpha = rng.uniform(0.5, 0.9)
            region_img = img[y0:y1, x0:x1]
            region_img[sub_mask] = region_img[sub_mask] * (1 - alpha) + color * alpha

    img = np.clip(img, 0, 255).astype(np.uint8)
    img = cv2.GaussianBlur(img, (0, 0), sigmaX=max(0.5, width_px / 1200))
    return img
