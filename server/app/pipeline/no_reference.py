"""Модуль 8: запасной режим сборки без образца (реализация — этап 5)."""
from __future__ import annotations

from app.pipeline.schemas import PieceRecord


def build_frame_chain(pieces: list[PieceRecord]) -> list[str]:
    """Собрать рамку по цепочке краевых деталей (прямая сторона + совпадение
    соседних краёв), без привязки к образцу.

    Реализация появляется на этапе 5.
    """
    raise NotImplementedError("build_frame_chain реализуется на этапе 5")


def cluster_islands(pieces: list[PieceRecord]) -> dict[str, list[str]]:
    """Кластеризовать внутренние детали по цвету/текстуре (эмбеддинги DINOv2)
    на "острова" — каждый кластер соответствует лотку.

    Реализация появляется на этапе 5.
    """
    raise NotImplementedError("cluster_islands реализуется на этапе 5")
