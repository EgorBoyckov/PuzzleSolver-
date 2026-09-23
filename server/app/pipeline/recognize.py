"""Модуль 7: узнавание деталей на новом фото стола.

Нужен, когда пользователь фотографирует детали ПОСЛЕ каталогизации (уже
разложенные, частично собранные — режим подсказки "что это за деталь") и
системе нужно сопоставить каждую увиденную деталь с её ID в каталоге, не
зная заранее ни расположения, ни того, каким углом она сейчас повёрнута к
камере.

Два прохода, как и в locate.py: быстрый отбор top-K по эмбеддингу (уже
инвариантен к повороту — см. app/pipeline/embedding.py), затем точное
сравнение формы 4 сторон — но, в отличие от locate (где сравнивается
УЖЕ канонизированная по image processing деталь с ячейкой образца),
здесь заранее неизвестно, какая сторона детали на фото соответствует
стороне 0 в каталоге: contour tracing даёт произвольную стартовую точку
на новом фото, даже если бы физическая ориентация детали была той же.
Поэтому форма сравнивается по всем 4 циклическим сдвигам (поворотам на
90°) индексов сторон, а не по одному выравниванию — решение того же
типа, что и локальный перебор поворотов в locate.locate_piece.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from app.core.config import get_config
from app.pipeline.describe import describe_piece
from app.pipeline.schemas import PieceRecord, SideKind
from app.pipeline.segment import segment_pieces


def _curve_y(piece: PieceRecord, side_idx: int) -> np.ndarray:
    return np.array([p.y for p in piece.sides[side_idx].curve], dtype=np.float64)


def _same_side_distance_mm(piece_a: PieceRecord, side_a: int, piece_b: PieceRecord, side_b: int, penalty_mm: float) -> float:
    """Расстояние между профилем ОДНОЙ И ТОЙ ЖЕ физической стороны, снятой
    дважды (каталог + новое фото) — в отличие от match.shape_distance_mm,
    здесь НЕ зеркалим (это не два конца шва, а один и тот же край с двух
    фотографий), знак и порядок точек совпадают напрямую."""
    ka, kb = piece_a.sides[side_a].kind, piece_b.sides[side_b].kind
    if ka != kb:
        return penalty_mm
    if ka == SideKind.STRAIGHT:
        return 0.0
    ya, yb = _curve_y(piece_a, side_a), _curve_y(piece_b, side_b)
    if len(ya) == 0 or len(ya) != len(yb):
        return penalty_mm
    return float(np.sqrt(np.mean((ya - yb) ** 2)))


def fingerprint_distance(catalog_piece: PieceRecord, candidate: PieceRecord, penalty_mm: float) -> tuple[float, int]:
    """Минимальное суммарное расстояние формы по 4 сторонам среди всех 4
    циклических поворотов индексации кандидата. Возвращает (расстояние,
    поворот_deg) — поворот, на который повёрнут candidate относительно
    catalog_piece (candidate.side[(i+rot/90)%4] соответствует catalog.side[i])."""
    if len(catalog_piece.sides) != 4 or len(candidate.sides) != 4:
        return math.inf, 0
    best_dist, best_rot = math.inf, 0
    for rot_steps in range(4):
        total = 0.0
        for i in range(4):
            total += _same_side_distance_mm(catalog_piece, i, candidate, (i + rot_steps) % 4, penalty_mm)
        if total < best_dist:
            best_dist, best_rot = total, rot_steps * 90
    return best_dist, best_rot


def recognize_pieces_on_frame(
    image: np.ndarray,
    catalog: list[PieceRecord],
    mm_per_pixel: float,
    puzzle_id: str = "recognize",
    batch_number: int = 0,
    debug_dir: Path | None = None,
) -> dict[str, tuple[float, float, int]]:
    """Сегментировать новый кадр, посчитать отпечаток (форма 4 сторон,
    инвариантная к повороту, + эмбеддинг), найти в каталоге через
    приближение FAISS (numpy top-K по эмбеддингу + точный перебор поворотов
    по форме — реальный FAISS-индекс подключается вместе с остальными
    тяжёлыми ML-зависимостями, см. requirements.txt).

    Возвращает {piece_id: (x_px, y_px, rotation_deg)} для каждой узнанной
    детали — координаты и поворот в системе координат ПЕРЕДАННОГО кадра
    image (не образца и не каталога), поворот — на сколько градусов деталь
    сейчас повёрнута относительно того, как она лежала при каталогизации.
    """
    cfg = get_config()
    rc = cfg.section("recognize")
    embedding_top_k = int(rc["embedding_top_k"])
    penalty_mm = float(rc["kind_mismatch_penalty_mm"])

    detected = segment_pieces(image, mm_per_pixel, puzzle_id, batch_number, debug_dir=debug_dir)
    catalog_with_embedding = [p for p in catalog if p.embedding is not None and len(p.sides) == 4]
    if not catalog_with_embedding:
        return {}
    catalog_embeddings = np.array([p.embedding for p in catalog_with_embedding], dtype=np.float64)

    results: dict[str, tuple[float, float, int]] = {}
    for piece in detected:
        describe_piece(piece, image, mm_per_pixel, debug_dir=debug_dir)
        if piece.embedding is None or len(piece.sides) != 4:
            continue

        emb = np.array(piece.embedding, dtype=np.float64)
        if emb.shape[0] != catalog_embeddings.shape[1]:
            continue
        dists = np.linalg.norm(catalog_embeddings - emb, axis=1)
        top_k = min(embedding_top_k, dists.shape[0])
        candidate_idx = np.argsort(dists)[:top_k]

        best_id, best_dist, best_rot = None, math.inf, 0
        for idx in candidate_idx:
            cat_piece = catalog_with_embedding[int(idx)]
            dist, rot = fingerprint_distance(cat_piece, piece, penalty_mm)
            if dist < best_dist:
                best_dist, best_id, best_rot = dist, cat_piece.id, rot

        if best_id is None:
            continue
        xs = [pt.x for pt in piece.contour_px]
        ys = [pt.y for pt in piece.contour_px]
        results[best_id] = (float(np.mean(xs)), float(np.mean(ys)), best_rot)

    return results
