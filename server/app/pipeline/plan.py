"""Модуль 6: планировщик шагов сборки.

Жадный выбор: все кандидаты-рёбра сортируются по приоритету "уверенность x
удобство" и берутся по убыванию, пока каждая сторона участвует не более чем
в одном принятом шаге (деталь физически можно приставить к соседу только
одной своей стороной за раз) и не набрано max_steps.

"Удобство" отражает порядок из ТЗ (frame_first: сначала рамка, потом
текстурные острова, потом однотонные зоны) без отдельного явного признака
зоны: рамочная деталь узнаётся по kind (corner/edge — хотя бы одна прямая
сторона); для остальных наличие конечного D_pos (см. match.py) — прокси
"текстурности": позиция уверенно определилась только когда локальный
патч детали достаточно уникален для locate, что как раз характерно для
текстурных, а не однотонных участков образца.
"""
from __future__ import annotations

import math

from app.core.config import get_config
from app.pipeline.schemas import AssemblyStep, EdgeMatch, PieceKind, PieceRecord, StepStatus


def _convenience(edge: EdgeMatch, pieces_by_id: dict[str, PieceRecord], frame_first: bool) -> float:
    if frame_first:
        piece_a = pieces_by_id.get(edge.piece_a)
        piece_b = pieces_by_id.get(edge.piece_b)
        is_frame = any(p is not None and p.kind in (PieceKind.CORNER, PieceKind.EDGE) for p in (piece_a, piece_b))
        if is_frame:
            return 1.0
    return 0.7 if math.isfinite(edge.d_pos) else 0.3


def _sector(edge: EdgeMatch, pieces_by_id: dict[str, PieceRecord]) -> str:
    piece_a = pieces_by_id.get(edge.piece_a)
    piece_b = pieces_by_id.get(edge.piece_b)
    if any(p is not None and p.kind in (PieceKind.CORNER, PieceKind.EDGE) for p in (piece_a, piece_b)):
        return "frame"
    return "textured" if math.isfinite(edge.d_pos) else "monochrome"


def _greedy_select(
    candidate_edges: list[EdgeMatch], pieces_by_id: dict[str, PieceRecord], max_steps: int
) -> list[AssemblyStep]:
    cfg = get_config()
    pc = cfg.section("plan")
    w_conf = float(pc["priority_confidence_weight"])
    w_conv = float(pc["priority_convenience_weight"])
    frame_first = bool(pc["frame_first"])

    scored = [(w_conf * e.score + w_conv * _convenience(e, pieces_by_id, frame_first), e) for e in candidate_edges]
    scored.sort(key=lambda t: -t[0])

    used_sides: set[tuple[str, int]] = set()
    steps: list[AssemblyStep] = []
    for _priority, edge in scored:
        if len(steps) >= max_steps:
            break
        key_a, key_b = (edge.piece_a, edge.side_a), (edge.piece_b, edge.side_b)
        if key_a in used_sides or key_b in used_sides:
            continue
        used_sides.add(key_a)
        used_sides.add(key_b)
        steps.append(
            AssemblyStep(
                step_number=len(steps) + 1,
                piece_a=edge.piece_a,
                piece_b=edge.piece_b,
                side_a=edge.side_a,
                side_b=edge.side_b,
                rotation_deg=((edge.side_a + 2 - edge.side_b) % 4) * 90,
                sector=_sector(edge, pieces_by_id),
                confidence=edge.score,
            )
        )
    return steps


def next_steps(candidate_edges: list[EdgeMatch], max_steps: int = 5) -> list[AssemblyStep]:
    """Жадный выбор шагов по приоритету "уверенность x удобство". Без
    доступа к каталогу деталей "рамка" не распознаётся по PieceKind (в
    EdgeMatch его нет) — удобство тогда опирается только на наличие
    позиционного сигнала (D_pos). Для точной классификации
    рамка/текстура/однотон используйте next_steps_with_catalog."""
    return _greedy_select(candidate_edges, pieces_by_id={}, max_steps=max_steps)


def next_steps_with_catalog(
    candidate_edges: list[EdgeMatch], catalog: list[PieceRecord], max_steps: int = 5
) -> list[AssemblyStep]:
    """Тот же жадный выбор, что next_steps, но с доступом к каталогу деталей —
    даёт точную классификацию "рамка/текстура/однотон" по PieceKind вместо
    приближения только по D_pos. Предпочтительный вход для реального
    использования (pipeline_cli/API)."""
    return _greedy_select(candidate_edges, pieces_by_id={p.id: p for p in catalog}, max_steps=max_steps)


def apply_feedback(step: AssemblyStep, status: StepStatus) -> None:
    """"Готово" фиксирует ребро графа соседств; "Не подходит" удаляет ребро
    и запускает пересчёт кандидатов.

    Пересчёт кандидатов по факту отклонения требует состояния сессии сборки
    (какие рёбра уже отклонены -> исключить из следующего build_edge_matches),
    которого пока нет — оно появляется вместе с полным сквозным сценарием
    веб-клиента (этап 4). Здесь фиксируется статус самого шага; вызывающий
    код (будущая сессия) отвечает за то, чтобы не предлагать тот же шаг
    повторно после REJECTED."""
    step.status = status
