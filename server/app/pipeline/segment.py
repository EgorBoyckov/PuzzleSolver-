"""Модуль 2: сегментация деталей на кадре партии (реализация — этап 1)."""
from __future__ import annotations

import numpy as np

from app.pipeline.schemas import PieceRecord


def segment_pieces(image: np.ndarray, mm_per_pixel: float, puzzle_id: str, batch_number: int) -> list[PieceRecord]:
    """Найти контур каждой детали на однотонном фоне, разделить слипшиеся
    водоразделом, пометить сомнительные (площадь вне ±40% от медианы).

    Реализация появляется на этапе 1.
    """
    raise NotImplementedError("segment_pieces реализуется на этапе 1")
