"""Модуль 8: запасной режим сборки без образца.

Без картинки коробки нет позиционного сигнала (locate.location_candidates),
но остальной конвейер уже собирает всё нужное для двух независимых задач:

  - рамка собирается цепочкой через match.py (форма+цвет шва) — сама
    функция D_pos уже корректно откатывается к чистой форме+цвету, когда
    позиции нет (см. match.score_side_pair), так что дополнительной логики
    сопоставления сторон здесь не нужно, только обход по рамочным деталям;
  - внутренние детали без рамки и без образца сортировать по сторонам
    некуда — их можно только сгруппировать по похожести (цвет/текстура
    эмбеддинга) на "острова", чтобы пользователь разложил их по лоткам и
    начал собирать каждый остров независимо, как обычно делают руками.
"""
from __future__ import annotations

import numpy as np

from app.core.config import get_config
from app.pipeline.match import find_side_candidates
from app.pipeline.schemas import PieceKind, PieceRecord, SideKind

FRAME_KINDS = (PieceKind.CORNER, PieceKind.EDGE)


def _frame_relevant_sides(piece: PieceRecord) -> list[int]:
    """Стороны детали, идущие ВДОЛЬ рамки (не в интерьер пазла).

    Деталь-угол: 2 смежные прямые стороны (сам угол) -> обе оставшиеся
    стороны идут вдоль двух сторон рамки. Деталь-край: 1 прямая сторона
    (сама граница) -> вдоль рамки идут две СОСЕДНИЕ с ней стороны, а
    противоположная прямой смотрит внутрь пазла и в цепочку не входит.
    """
    straight = [s.index for s in piece.sides if s.kind == SideKind.STRAIGHT]
    if len(straight) >= 2:
        return [s.index for s in piece.sides if s.kind != SideKind.STRAIGHT]
    if len(straight) == 1:
        i = straight[0]
        return [(i - 1) % 4, (i + 1) % 4]
    return []


def build_frame_chain(pieces: list[PieceRecord]) -> list[str]:
    """Собрать рамку по цепочке краевых деталей (прямая сторона + совпадение
    соседних краёв по форме/цвету — match.find_side_candidates, ограниченный
    только рамочными деталями), без привязки к образцу.

    Идёт от произвольного угла вдоль рамки, каждый раз выбирая лучшего
    кандидата среди ещё не посещённых рамочных деталей для "исходящей"
    стороны (та из двух рамочных сторон текущей детали, что не использована
    для входа). Останавливается, когда цепочка замыкается на стартовой
    детали (полная рамка) или кандидатов больше не находится (после чего
    возвращает то, что собрано — например, если часть рамочных деталей не
    была найдена/описана).
    """
    frame_pieces = [p for p in pieces if p.kind in FRAME_KINDS and len(p.sides) == 4]
    if not frame_pieces:
        return []
    by_id = {p.id: p for p in frame_pieces}

    corners = [p for p in frame_pieces if p.kind == PieceKind.CORNER]
    start = corners[0] if corners else frame_pieces[0]
    start_frame_sides = _frame_relevant_sides(start)
    if not start_frame_sides:
        return [start.id]

    chain = [start.id]
    visited = {start.id}
    current = start
    outgoing_side = start_frame_sides[0]

    while True:
        candidates = find_side_candidates(current, outgoing_side, frame_pieces)
        next_match = next((c for c in candidates if c.piece_b == start.id or c.piece_b not in visited), None)
        if next_match is None:
            break

        if next_match.piece_b == start.id:
            if len(chain) > 2:
                chain.append(start.id)
            break

        next_piece = by_id.get(next_match.piece_b)
        if next_piece is None:
            break

        next_frame_sides = _frame_relevant_sides(next_piece)
        outgoing_candidates = [s for s in next_frame_sides if s != next_match.side_b]
        if not outgoing_candidates:
            break

        chain.append(next_piece.id)
        visited.add(next_piece.id)
        current = next_piece
        outgoing_side = outgoing_candidates[0]

        if len(visited) >= len(frame_pieces):
            break

    return chain


def _kmeans(points: np.ndarray, k: int, seed: int, max_iter: int = 50) -> np.ndarray:
    """Минимальная реализация Ллойда без внешних зависимостей (sklearn не
    подключён — тот же принцип "не тащить инфраструктуру раньше, чем она
    реально нужна", что и с DINOv2/FAISS в describe/locate). Возвращает
    метку кластера (0..k-1) для каждой точки."""
    rng = np.random.default_rng(seed)
    k = max(1, min(k, len(points)))
    centroid_idx = rng.choice(len(points), size=k, replace=False)
    centroids = points[centroid_idx].copy()

    labels = np.zeros(len(points), dtype=int)
    for _ in range(max_iter):
        dists = np.linalg.norm(points[:, None, :] - centroids[None, :, :], axis=2)
        new_labels = dists.argmin(axis=1)
        if np.array_equal(new_labels, labels) and _ > 0:
            break
        labels = new_labels
        for c in range(k):
            members = points[labels == c]
            if len(members) > 0:
                centroids[c] = members.mean(axis=0)
    return labels


def cluster_islands(pieces: list[PieceRecord], seed: int = 0) -> dict[str, list[str]]:
    """Кластеризовать внутренние детали (без прямой стороны — kind=center)
    по эмбеддингу лица (цвет+форма, см. app/pipeline/embedding.py) на
    "острова" — каждый кластер соответствует лотку для раскладки.

    Число кластеров оценивается от целевого размера острова
    (no_reference.min_island_size), затем кластеры МЕНЬШЕ этого порога
    сливаются с ближайшим (по центроиду) более крупным кластером — лучше
    несколько крупных не идеально чистых островов, чем россыпь кластеров
    из одной-двух деталей, бесполезных как "лоток"."""
    cfg = get_config().section("no_reference")
    min_size = int(cfg["min_island_size"])

    interior = [p for p in pieces if p.kind == PieceKind.CENTER and p.embedding]
    if not interior:
        return {}

    points = np.array([p.embedding for p in interior], dtype=np.float64)
    k = max(1, round(len(interior) / max(min_size, 1)))
    labels = _kmeans(points, k, seed)

    clusters: dict[int, list[int]] = {}
    for idx, label in enumerate(labels):
        clusters.setdefault(int(label), []).append(idx)

    # Слить кластеры меньше min_size с ближайшим (по центроиду) кластером,
    # который после слияния не станет сам слишком маленьким чем-то другим -
    # проще и устойчивее: сливаем в порядке возрастания размера, каждый раз
    # пересчитывая центроиды, пока все оставшиеся кластеры не станут
    # достаточно большими или не останется один кластер.
    def centroid(idxs: list[int]) -> np.ndarray:
        return points[idxs].mean(axis=0)

    while len(clusters) > 1:
        smallest_label = min(clusters, key=lambda c: len(clusters[c]))
        if len(clusters[smallest_label]) >= min_size:
            break
        small_idxs = clusters.pop(smallest_label)
        small_centroid = centroid(small_idxs)
        target_label = min(
            clusters, key=lambda c: float(np.linalg.norm(centroid(clusters[c]) - small_centroid))
        )
        clusters[target_label].extend(small_idxs)

    return {
        f"island_{i}": [interior[idx].id for idx in idxs]
        for i, idxs in enumerate(clusters.values())
    }
