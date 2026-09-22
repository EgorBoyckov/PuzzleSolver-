"""Модуль 5: сопоставление сторон деталей (реализация — этап 3)."""
from __future__ import annotations

from app.pipeline.schemas import EdgeMatch, PieceRecord


def score_side_pair(piece_a: PieceRecord, side_a: int, piece_b: PieceRecord, side_b: int) -> EdgeMatch:
    """S = w1*D_shape(A, зеркальная B) + w2*D_color(ΔE) + w3*D_pos.
    Выступ сравнивается только со впадиной. Веса берутся из config.match.weights.

    Реализация появляется на этапе 3.
    """
    raise NotImplementedError("score_side_pair реализуется на этапе 3")
