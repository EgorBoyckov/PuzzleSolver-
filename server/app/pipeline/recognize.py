"""Модуль 7: узнавание деталей на новом фото стола (реализация — этап 3)."""
from __future__ import annotations

import numpy as np

from app.pipeline.schemas import PieceRecord


def recognize_pieces_on_frame(image: np.ndarray, catalog: list[PieceRecord]) -> dict[str, tuple[float, float, int]]:
    """Сегментировать новый кадр, посчитать отпечаток (форма 4 сторон,
    инвариантная к повороту, + эмбеддинг), найти в каталоге через FAISS.

    Возвращает {piece_id: (x_px, y_px, rotation_deg)} для каждой узнанной детали.
    Реализация появляется на этапе 3.
    """
    raise NotImplementedError("recognize_pieces_on_frame реализуется на этапе 3")
