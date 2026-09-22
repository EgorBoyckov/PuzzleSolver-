"""Модуль 3: описание детали — углы, стороны, форма, цвет, эмбеддинг (этап 1)."""
from __future__ import annotations

import numpy as np

from app.pipeline.schemas import PieceRecord


def describe_piece(piece: PieceRecord, face_image: np.ndarray) -> PieceRecord:
    """Найти 4 угла, разбить контур на стороны, классифицировать каждую
    (прямая/выступ/впадина), посчитать нормализованную кривую, цветовую
    полосу в Lab и эмбеддинг лица детали (DINOv2). Определяет piece.kind.

    Реализация появляется на этапе 1.
    """
    raise NotImplementedError("describe_piece реализуется на этапе 1")
