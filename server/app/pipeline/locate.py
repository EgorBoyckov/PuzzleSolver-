"""Модуль 4: привязка детали к образцу (реализация — этап 2)."""
from __future__ import annotations

import numpy as np

from app.pipeline.schemas import PieceRecord, PuzzleMeta


def locate_piece(piece: PieceRecord, reference_image: np.ndarray, puzzle: PuzzleMeta) -> PieceRecord:
    """Сравнить деталь с участками образца в 4 поворотах: сперва топ-50 по
    эмбеддингу, затем точное сравнение (SuperPoint+LightGlue/шаблоны).
    Заполняет piece.location_candidates (топ-3 ячейки с уверенностью).

    Реализация появляется на этапе 2.
    """
    raise NotImplementedError("locate_piece реализуется на этапе 2")
