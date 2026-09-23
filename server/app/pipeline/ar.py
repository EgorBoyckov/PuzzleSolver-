"""Модуль этапа 6: лёгкий AR-оверлей поверх живого видео камеры.

Полноценный WebXR ('immersive-ar', с мировым трекингом и якорями) требует
реального AR-совместимого устройства (ARCore на Android) для тестирования
— в этой среде разработки такого устройства нет ни физически, ни через
эмулятор, поэтому здесь реализован более простой и универсально
тестируемый вариант: клиент периодически снимает кадр с ЖИВОГО видео
камеры (тот же поток, что видит пользователь) и получает от сервера
координаты узнанных деталей ПРЯМО В ЭТОМ КАДРЕ — координаты рисуются
поверх <video> холстом <canvas> без какой-либо гомографии/трансформации.
Работает в любом браузере с доступом к камере, не требует WebXR API;
цена — оверлей не "прилипает" к столу при повороте камеры между кадрами
(нет мирового трекинга), только per-кадр наложение, которое обновляется
на каждый новый снимок.

ARCore напрямую (нативный SDK) в это не входит — проект по условиям ТЗ
работает как PWA в Chrome без нативного приложения, WebXR остаётся
единственным браузерным путём к "настоящему" AR и может быть добавлен
позже как прогрессивное улучшение, если/когда появится возможность
протестировать на реальном устройстве.
"""
from __future__ import annotations

import numpy as np

from app.core.config import get_config
from app.pipeline.preprocess import detect_marker_and_scale
from app.pipeline.recognize import recognize_pieces_on_frame
from app.pipeline.schemas import PieceRecord


def recognize_raw_frame(
    image_bgr: np.ndarray, catalog: list[PieceRecord], puzzle_id: str
) -> tuple[bool, dict[str, tuple[float, float, int]]]:
    """Узнать детали каталога прямо на сыром (невыпрямленном) кадре камеры.

    Масштаб (мм/пиксель) оценивается по видимому размеру ArUco-маркера в
    ЭТОМ кадре — точность ниже, чем после полной гомографии в
    preprocess_frame (не компенсирует перспективные искажения по всему
    кадру, только у самого маркера), но этого достаточно для визуальной
    подсказки "где какая деталь", а не для точных измерений.

    Возвращает (маркер_найден, {piece_id: (x_px, y_px, rotation_deg)}) —
    False и пустой словарь, если маркер не попал в кадр (пользователю
    нужно навести камеру так, чтобы маркер был виден, как и при обычной
    съёмке партий)."""
    cfg = get_config()
    aruco_cfg = cfg.section("aruco")
    _marker_corners, mm_per_pixel = detect_marker_and_scale(
        image_bgr, aruco_cfg["dictionary"], aruco_cfg["marker_id"], float(aruco_cfg["marker_size_mm"])
    )
    if mm_per_pixel is None:
        return False, {}

    results = recognize_pieces_on_frame(image_bgr, catalog, mm_per_pixel, puzzle_id=puzzle_id, batch_number=0)
    return True, results
