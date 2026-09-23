"""Сквозной API этапа 4: создание пазла -> загрузка партий фото -> план
сборки -> обратная связь по шагам -> узнавание деталей на новом фото.

Оборачивает конвейер этапов 1-3 (preprocess/segment/describe/locate/match/
plan/recognize) в HTTP-эндпоинты с персистентностью между запросами
(app.core.storage.PuzzleStore). Индекс образца (ReferenceGridIndex, с
эмбеддингами по ячейкам) кэшируется в памяти процесса по puzzle_id — не
JSON-сериализуем и пересобирается из сохранённого reference.jpg при
перезапуске сервера.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone

import cv2
import numpy as np
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from pydantic import BaseModel

from app.core.config import get_config
from app.core.storage import PuzzleState, PuzzleStore, RejectedEdge
from app.pipeline.ar import recognize_raw_frame
from app.pipeline.describe import describe_piece
from app.pipeline.locate import ReferenceGridIndex, build_reference_index, locate_piece, refit_reference_colors
from app.pipeline.match import find_side_candidates
from app.pipeline.no_reference import build_frame_chain, cluster_islands
from app.pipeline.plan import apply_feedback, next_steps_with_catalog
from app.pipeline.preprocess import preprocess_frame
from app.pipeline.recognize import recognize_pieces_on_frame
from app.pipeline.schemas import AssemblyStep, PieceKind, PuzzleMeta, SideKind, StepStatus
from app.pipeline.segment import segment_pieces

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/puzzles", tags=["puzzles"])

_ref_index_cache: dict[str, ReferenceGridIndex] = {}


def _get_store() -> PuzzleStore:
    config = get_config()
    return PuzzleStore(config.path(config.server.data_dir) / "puzzles")


def _load_or_404(store: PuzzleStore, puzzle_id: str) -> PuzzleState:
    state = store.load(puzzle_id)
    if state is None:
        raise HTTPException(404, f"Пазл '{puzzle_id}' не найден")
    return state


def _get_ref_index(store: PuzzleStore, state: PuzzleState) -> ReferenceGridIndex | None:
    if not state.meta.has_reference or not state.meta.grid_rows or not state.meta.grid_cols:
        return None
    cached = _ref_index_cache.get(state.meta.id)
    if cached is not None:
        return cached
    ref_path = store.reference_path(state.meta.id)
    if not ref_path.exists():
        return None
    ref_bgr = cv2.imread(str(ref_path))
    if ref_bgr is None:
        return None
    idx = build_reference_index(ref_bgr, state.meta.grid_rows, state.meta.grid_cols)
    _ref_index_cache[state.meta.id] = idx
    return idx


def _load_frame_lookup(store: PuzzleStore, puzzle_id: str, batch_numbers: set[int]) -> dict[int, np.ndarray]:
    lookup: dict[int, np.ndarray] = {}
    for bn in batch_numbers:
        path = store.rectified_path(puzzle_id, bn)
        if path.exists():
            img = cv2.imread(str(path))
            if img is not None:
                lookup[bn] = img
    return lookup


def _recompute_locate(store: PuzzleStore, state: PuzzleState) -> None:
    """Цветокоррекция образца по всем уверенным деталям каталога (не только
    последней партии) + повторный поиск для неуверенных — перезагружает
    выпрямленные кадры прошлых партий с диска, так как между HTTP-запросами
    они не живут в памяти."""
    ref_index = _get_ref_index(store, state)
    if ref_index is None or not state.pieces:
        return
    lc = get_config().section("locate")
    batch_numbers = {p.batch_number for p in state.pieces}
    frame_lookup = _load_frame_lookup(store, state.meta.id, batch_numbers)
    refitted = refit_reference_colors(
        ref_index, state.pieces, frame_lookup,
        confidence_threshold=lc["color_refit_confidence_threshold"],
        min_samples=lc["color_refit_min_samples"],
    )
    if refitted is None:
        return
    _ref_index_cache[state.meta.id] = refitted
    low_conf = [
        p for p in state.pieces
        if not p.location_candidates or p.location_candidates[0][3] < lc["color_refit_confidence_threshold"]
    ]
    for p in low_conf:
        frame = frame_lookup.get(p.batch_number)
        if frame is not None:
            locate_piece(p, frame, refitted)


def _recompute_match_and_plan(state: PuzzleState) -> None:
    """Пересобрать кандидатов-рёбер и план, ИСКЛЮЧАЯ: (a) рёбра, отклонённые
    через feedback (в обе стороны пары), (b) стороны, уже "занятые"
    подтверждённым (DONE) шагом — деталь физически не может позже
    присоединиться той же стороной куда-то ещё. Статусы уже существующих
    шагов (DONE/REJECTED) переносятся на пересчитанный план по ключу
    (piece_a,side_a,piece_b,side_b); подтверждённые шаги, чья пара сторон
    выпала из новых кандидатов (обе стороны уже заняты), сохраняются как
    есть — историю подтверждённых соединений терять нельзя."""
    rejected_keys: set[tuple[str, int, str, int]] = set()
    for r in state.rejected_edges:
        rejected_keys.add((r.piece_a, r.side_a, r.piece_b, r.side_b))
        rejected_keys.add((r.piece_b, r.side_b, r.piece_a, r.side_a))

    done_steps = [s for s in state.steps if s.status == StepStatus.DONE]
    done_sides = {(s.piece_a, s.side_a) for s in done_steps} | {(s.piece_b, s.side_b) for s in done_steps}

    edges: list = []
    for piece in state.pieces:
        for side in piece.sides:
            if side.kind not in (SideKind.TAB, SideKind.BLANK):
                continue
            if (piece.id, side.index) in done_sides:
                continue
            for candidate in find_side_candidates(piece, side.index, state.pieces):
                key = (candidate.piece_a, candidate.side_a, candidate.piece_b, candidate.side_b)
                if key in rejected_keys or (candidate.piece_b, candidate.side_b) in done_sides:
                    continue
                edges.append(candidate)
                break

    old_by_key = {(s.piece_a, s.side_a, s.piece_b, s.side_b): s for s in state.steps}
    new_steps = next_steps_with_catalog(edges, state.pieces, max_steps=max(len(state.pieces), 1))
    for s in new_steps:
        old = old_by_key.get((s.piece_a, s.side_a, s.piece_b, s.side_b))
        if old is not None:
            s.status = old.status

    kept_keys = {(s.piece_a, s.side_a, s.piece_b, s.side_b) for s in new_steps}
    for s in done_steps:
        if (s.piece_a, s.side_a, s.piece_b, s.side_b) not in kept_keys:
            new_steps.append(s)

    for i, s in enumerate(new_steps, start=1):
        s.step_number = i
    state.steps = new_steps


# --- схемы запросов/ответов ---


class CreatePuzzleResponse(BaseModel):
    id: str
    has_reference: bool
    grid_rows: int | None
    grid_cols: int | None


class BatchUploadResponse(BaseModel):
    batch_number: int
    accepted: bool
    rejection_reason: str | None
    pieces_found_in_batch: int
    pieces_total: int
    steps_total: int


class PuzzleStatus(BaseModel):
    id: str
    has_reference: bool
    total_pieces: int
    batches_uploaded: int
    located: int | None
    kind_counts: dict[str, int]
    steps_total: int
    steps_pending: int
    frame_chain_length: int | None
    island_count: int | None


class FeedbackRequest(BaseModel):
    status: StepStatus


class RecognizedPiece(BaseModel):
    piece_id: str
    x_px: float
    y_px: float
    rotation_deg: int


class NoReferenceResult(BaseModel):
    frame_chain: list[str]
    islands: dict[str, list[str]]


class ARFrameResult(BaseModel):
    marker_found: bool
    image_width: int
    image_height: int
    pieces: list[RecognizedPiece]


# --- эндпоинты ---


@router.post("", response_model=CreatePuzzleResponse)
async def create_puzzle(
    puzzle_id: str = Form(...),
    grid_rows: int | None = Form(None),
    grid_cols: int | None = Form(None),
    reference: UploadFile | None = File(None),
) -> CreatePuzzleResponse:
    store = _get_store()
    if store.exists(puzzle_id):
        raise HTTPException(409, f"Пазл '{puzzle_id}' уже существует")

    has_reference = reference is not None
    if has_reference and not (grid_rows and grid_cols):
        raise HTTPException(400, "Для образца нужно указать grid_rows и grid_cols")

    meta = PuzzleMeta(
        id=puzzle_id, total_pieces=0, has_reference=has_reference,
        grid_rows=grid_rows, grid_cols=grid_cols,
        created_at=datetime.now(timezone.utc).isoformat(),
    )
    state = store.create(meta)

    if reference is not None:
        ref_path = store.reference_path(puzzle_id)
        ref_path.parent.mkdir(parents=True, exist_ok=True)
        ref_path.write_bytes(await reference.read())
        state.meta.box_image_path = str(ref_path)
        store.save(state)

    logger.info("puzzle '%s' создан (has_reference=%s, grid=%sx%s)", puzzle_id, has_reference, grid_rows, grid_cols)
    return CreatePuzzleResponse(id=puzzle_id, has_reference=has_reference, grid_rows=grid_rows, grid_cols=grid_cols)


@router.post("/{puzzle_id}/batches", response_model=BatchUploadResponse)
async def upload_batch(puzzle_id: str, file: UploadFile) -> BatchUploadResponse:
    store = _get_store()
    state = _load_or_404(store, puzzle_id)

    batch_number = state.next_batch_number
    photo_path = store.batch_photo_path(puzzle_id, batch_number)
    photo_path.parent.mkdir(parents=True, exist_ok=True)
    photo_path.write_bytes(await file.read())

    frame, rectified, valid_mask = preprocess_frame(photo_path, puzzle_id, batch_number)
    state.next_batch_number += 1

    if not frame.accepted or rectified is None:
        store.save(state)
        return BatchUploadResponse(
            batch_number=batch_number, accepted=False, rejection_reason=frame.rejection_reason,
            pieces_found_in_batch=0, pieces_total=len(state.pieces), steps_total=len(state.steps),
        )

    cv2.imwrite(str(store.rectified_path(puzzle_id, batch_number)), rectified)

    pieces = segment_pieces(
        rectified, frame.mm_per_pixel, puzzle_id, batch_number,
        marker_bbox_px=frame.marker_bbox_px, valid_mask=valid_mask,
    )
    ref_index = _get_ref_index(store, state)
    for p in pieces:
        describe_piece(p, rectified, frame.mm_per_pixel)
        if ref_index is not None:
            locate_piece(p, rectified, ref_index)

    state.pieces.extend(pieces)
    state.meta.total_pieces = len(state.pieces)

    if ref_index is not None:
        _recompute_locate(store, state)
        _recompute_match_and_plan(state)
    else:
        # Без образца позиционного сигнала нет -> рамка цепочкой по форме/
        # цвету шва, внутренние детали кластеризуются на "острова" для
        # раскладки по лоткам (см. app.pipeline.no_reference).
        state.frame_chain = build_frame_chain(state.pieces)
        state.islands = cluster_islands(state.pieces)
    store.save(state)

    logger.info("puzzle '%s' batch %d: %d деталей (всего %d)", puzzle_id, batch_number, len(pieces), len(state.pieces))
    return BatchUploadResponse(
        batch_number=batch_number, accepted=True, rejection_reason=None,
        pieces_found_in_batch=len(pieces), pieces_total=len(state.pieces), steps_total=len(state.steps),
    )


@router.get("/{puzzle_id}", response_model=PuzzleStatus)
async def get_puzzle(puzzle_id: str) -> PuzzleStatus:
    store = _get_store()
    state = _load_or_404(store, puzzle_id)

    kind_counts = {k.value: 0 for k in PieceKind}
    kind_counts["unknown"] = 0
    for p in state.pieces:
        kind_counts[p.kind.value if p.kind else "unknown"] += 1

    return PuzzleStatus(
        id=puzzle_id, has_reference=state.meta.has_reference, total_pieces=len(state.pieces),
        batches_uploaded=state.next_batch_number - 1,
        located=sum(1 for p in state.pieces if p.location_candidates) if state.meta.has_reference else None,
        kind_counts=kind_counts, steps_total=len(state.steps),
        steps_pending=sum(1 for s in state.steps if s.status == StepStatus.PENDING),
        frame_chain_length=len(state.frame_chain) if not state.meta.has_reference else None,
        island_count=len(state.islands) if not state.meta.has_reference else None,
    )


@router.get("/{puzzle_id}/no_reference", response_model=NoReferenceResult)
async def get_no_reference(puzzle_id: str) -> NoReferenceResult:
    store = _get_store()
    state = _load_or_404(store, puzzle_id)
    if state.meta.has_reference:
        raise HTTPException(400, "У пазла есть образец — используйте /steps, а не /no_reference")
    return NoReferenceResult(frame_chain=state.frame_chain, islands=state.islands)


@router.get("/{puzzle_id}/steps", response_model=list[AssemblyStep])
async def get_steps(puzzle_id: str, limit: int = 20, status: StepStatus = StepStatus.PENDING) -> list[AssemblyStep]:
    store = _get_store()
    state = _load_or_404(store, puzzle_id)
    return [s for s in state.steps if s.status == status][:limit]


@router.post("/{puzzle_id}/steps/{step_number}/feedback", response_model=list[AssemblyStep])
async def post_feedback(puzzle_id: str, step_number: int, body: FeedbackRequest) -> list[AssemblyStep]:
    store = _get_store()
    state = _load_or_404(store, puzzle_id)

    step = next((s for s in state.steps if s.step_number == step_number), None)
    if step is None:
        raise HTTPException(404, f"Шаг {step_number} не найден")

    apply_feedback(step, body.status)
    if body.status == StepStatus.REJECTED:
        state.rejected_edges.append(
            RejectedEdge(piece_a=step.piece_a, side_a=step.side_a, piece_b=step.piece_b, side_b=step.side_b)
        )
        _recompute_match_and_plan(state)

    store.save(state)
    return [s for s in state.steps if s.status == StepStatus.PENDING][:20]


@router.post("/{puzzle_id}/recognize", response_model=list[RecognizedPiece])
async def recognize(puzzle_id: str, file: UploadFile) -> list[RecognizedPiece]:
    store = _get_store()
    state = _load_or_404(store, puzzle_id)

    tmp_path = store.batches_dir(puzzle_id) / "_recognize_upload.jpg"
    tmp_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path.write_bytes(await file.read())

    frame, rectified, _valid_mask = preprocess_frame(tmp_path, puzzle_id, 0)
    if not frame.accepted or rectified is None:
        raise HTTPException(400, f"Кадр отбракован: {frame.rejection_reason}")

    catalog = [p for p in state.pieces if not p.is_suspect and len(p.sides) == 4]
    results = recognize_pieces_on_frame(rectified, catalog, frame.mm_per_pixel, puzzle_id=puzzle_id, batch_number=0)
    return [RecognizedPiece(piece_id=pid, x_px=x, y_px=y, rotation_deg=rot) for pid, (x, y, rot) in results.items()]


@router.post("/{puzzle_id}/ar_frame", response_model=ARFrameResult)
async def ar_frame(puzzle_id: str, file: UploadFile) -> ARFrameResult:
    """Этап 6: узнать детали каталога прямо на сыром кадре живого видео
    камеры (без выпрямления по маркеру — координаты должны совпадать с
    тем, что видит пользователь на экране, для оверлея). В отличие от
    остальных эндпоинтов, декодирует кадр прямо из памяти, не сохраняя на
    диск — этот эндпоинт дергается часто (раз в 1-2с при активном
    AR-режиме), в отличие от загрузки партий."""
    store = _get_store()
    state = _load_or_404(store, puzzle_id)

    content = await file.read()
    image_bgr = cv2.imdecode(np.frombuffer(content, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise HTTPException(400, "Не удалось декодировать изображение")

    catalog = [p for p in state.pieces if not p.is_suspect and len(p.sides) == 4]
    marker_found, results = recognize_raw_frame(image_bgr, catalog, puzzle_id)
    h, w = image_bgr.shape[:2]
    return ARFrameResult(
        marker_found=marker_found, image_width=w, image_height=h,
        pieces=[RecognizedPiece(piece_id=pid, x_px=x, y_px=y, rotation_deg=rot) for pid, (x, y, rot) in results.items()],
    )
