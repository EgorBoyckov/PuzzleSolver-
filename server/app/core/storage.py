"""Персистентность состояния пазла между запросами API (этап 4).

Хранилище — один JSON-файл на пазл (data/puzzles/<id>/state.json),
переиспользующий существующие pydantic-модели конвейера напрямую. Этого
достаточно для домашнего сервера с одним активным пользователем; поле
server.database_url в config.yaml остаётся зарезервированным под реальную
БД (SQLite/Postgres), если понадобится многопользовательский доступ или
конкурентная запись — пока в этом нет необходимости (тот же принцип, что
и с DINOv2/FAISS-заглушками в describe/locate: не тащить инфраструктуру
раньше, чем она реально нужна).
"""
from __future__ import annotations

import json
from pathlib import Path
from threading import Lock

from pydantic import BaseModel

from app.pipeline.schemas import AssemblyStep, PieceRecord, PuzzleMeta


class RejectedEdge(BaseModel):
    piece_a: str
    side_a: int
    piece_b: str
    side_b: int


class PuzzleState(BaseModel):
    meta: PuzzleMeta
    pieces: list[PieceRecord] = []
    steps: list[AssemblyStep] = []
    rejected_edges: list[RejectedEdge] = []
    next_batch_number: int = 1


class PuzzleStore:
    """Один экземпляр на процесс сервера; блокировка — на запись состояния
    ОДНОГО пазла, не глобально (разные пазлы не мешают друг другу)."""

    def __init__(self, puzzles_dir: Path):
        self._dir = puzzles_dir
        self._locks: dict[str, Lock] = {}

    def _lock_for(self, puzzle_id: str) -> Lock:
        return self._locks.setdefault(puzzle_id, Lock())

    def _state_path(self, puzzle_id: str) -> Path:
        return self._dir / puzzle_id / "state.json"

    def reference_path(self, puzzle_id: str) -> Path:
        return self._dir / puzzle_id / "reference.jpg"

    def batches_dir(self, puzzle_id: str) -> Path:
        return self._dir / puzzle_id / "batches"

    def batch_photo_path(self, puzzle_id: str, batch_number: int) -> Path:
        return self.batches_dir(puzzle_id) / f"batch_{batch_number:04d}.jpg"

    def rectified_path(self, puzzle_id: str, batch_number: int) -> Path:
        return self.batches_dir(puzzle_id) / f"batch_{batch_number:04d}_rectified.png"

    def exists(self, puzzle_id: str) -> bool:
        return self._state_path(puzzle_id).exists()

    def create(self, meta: PuzzleMeta) -> PuzzleState:
        state = PuzzleState(meta=meta)
        self.save(state)
        return state

    def load(self, puzzle_id: str) -> PuzzleState | None:
        path = self._state_path(puzzle_id)
        if not path.exists():
            return None
        return PuzzleState.model_validate_json(path.read_text(encoding="utf-8"))

    def save(self, state: PuzzleState) -> None:
        path = self._state_path(state.meta.id)
        with self._lock_for(state.meta.id):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(state.model_dump(), ensure_ascii=False, indent=2), encoding="utf-8")

    def list_ids(self) -> list[str]:
        if not self._dir.exists():
            return []
        return sorted(p.name for p in self._dir.iterdir() if p.is_dir() and (p / "state.json").exists())
