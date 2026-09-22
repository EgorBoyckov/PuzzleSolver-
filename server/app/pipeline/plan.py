"""Модуль 6: планировщик шагов сборки (реализация — этап 3)."""
from __future__ import annotations

from app.pipeline.schemas import AssemblyStep, EdgeMatch, StepStatus


def next_steps(candidate_edges: list[EdgeMatch], max_steps: int = 5) -> list[AssemblyStep]:
    """Жадный выбор шагов по приоритету "уверенность x удобство": сначала
    рамка, затем текстурные острова, в конце однотонные зоны.

    Реализация появляется на этапе 3.
    """
    raise NotImplementedError("next_steps реализуется на этапе 3")


def apply_feedback(step: AssemblyStep, status: StepStatus) -> None:
    """"Готово" фиксирует ребро графа соседств; "Не подходит" удаляет ребро
    и запускает пересчёт кандидатов.

    Реализация появляется на этапе 3.
    """
    raise NotImplementedError("apply_feedback реализуется на этапе 3")
