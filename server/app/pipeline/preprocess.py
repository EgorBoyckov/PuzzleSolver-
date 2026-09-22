"""Модуль 1: предобработка кадра (реализация — этап 1).

Интерфейс зафиксирован уже на этапе 0, чтобы синтетический датасет и тесты
следующих этапов могли на него ориентироваться.
"""
from __future__ import annotations

from pathlib import Path

from app.pipeline.schemas import BatchFrame


def preprocess_frame(image_path: Path, puzzle_id: str, batch_number: int) -> BatchFrame:
    """Найти ArUco-маркер, выпрямить перспективу, посчитать мм/пиксель,
    сделать баланс белого и отбраковать размытые/пересвеченные кадры.

    Реализация появляется на этапе 1.
    """
    raise NotImplementedError("preprocess_frame реализуется на этапе 1")
