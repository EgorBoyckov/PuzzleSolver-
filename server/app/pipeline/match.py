"""Модуль 5: сопоставление сторон деталей.

Две физически соседние детали делят одну линию разреза: выступ одной —
ровно та же кривая, что и впадина другой, только пройденная в обратном
направлении (стороны деталей нормализованы против часовой стрелки каждая
вокруг своей детали) и с противоположным знаком отклонения (наружу от
одной детали = внутрь другой). Поэтому сравнение формы — это сравнение
кривой стороны A с ЗЕРКАЛЬНО ОТРАЖЁННОЙ (развёрнутой по порядку и по знаку
Y) кривой стороны B; сравнение цвета — то же самое без смены знака (цвет
непрерывен через шов, а не зеркален).

D_pos использует предсказанную позицию на образце (locate.location_candidates
+ поворот привязки): зная поворот, можно определить, в какую сторону света
(верх/право/низ/лево образца) смотрит каждая сторона детали, а значит — в
какой соседней клетке образца должна лежать деталь на другом конце шва.
"""
from __future__ import annotations

import math

import numpy as np

from app.core.config import get_config
from app.pipeline.schemas import EdgeMatch, PieceRecord, Side, SideKind

# Абсолютное направление стороны 0 (до поворота привязки) — "вверх" в
# системе координат образца; следующие индексы — по часовой стрелке
# (см. app/synth/piece_shapes.py: 0=top,1=right,2=bottom,3=left).
_DIR_DELTAS = {0: (-1, 0), 1: (0, 1), 2: (1, 0), 3: (0, -1)}


def _side_absolute_direction(side_index: int, rotation_deg: int) -> int:
    return (side_index + rotation_deg // 90) % 4


def _curve_y(side: Side) -> np.ndarray:
    return np.array([p.y for p in side.curve], dtype=np.float64)


def _color_lab(side: Side) -> np.ndarray:
    return np.array(side.color_strip_lab, dtype=np.float64)


def shape_distance_mm(side_a: Side, side_b: Side) -> float:
    """RMS-расстояние между профилем стороны A и зеркально отражённым
    (обратный порядок точек, знак Y инвертирован) профилем стороны B."""
    ya, yb = _curve_y(side_a), _curve_y(side_b)
    if len(ya) == 0 or len(ya) != len(yb):
        return math.inf
    yb_mirrored = -yb[::-1]
    return float(np.sqrt(np.mean((ya - yb_mirrored) ** 2)))


def color_distance_deltae(side_a: Side, side_b: Side) -> float:
    """Среднее евклидово расстояние в Lab между цветовыми полосами двух
    сторон (сопоставленными в обратном порядке, как и профиль формы —
    цвет по обе стороны шва непрерывен, знак менять не нужно)."""
    la, lb = _color_lab(side_a), _color_lab(side_b)
    if len(la) == 0 or len(la) != len(lb):
        return math.inf
    lb_reversed = lb[::-1]
    return float(np.mean(np.linalg.norm(la - lb_reversed, axis=1)))


def position_distance_cells(
    piece_a: PieceRecord, side_a_idx: int, piece_b: PieceRecord, side_b_idx: int
) -> float | None:
    """Расстояние (в клетках сетки образца) между клеткой, предсказанной для
    piece_b по позиции piece_a и повороту стороны side_a, и фактической
    лучшей клеткой piece_b. None, если у одной из деталей нет location_candidates
    (образец не задан или привязка не удалась) — позиционный сигнал просто
    недоступен, это не то же самое, что "плохое совпадение"."""
    if not piece_a.location_candidates or not piece_b.location_candidates:
        return None
    row_a, col_a, rot_a, _ = piece_a.location_candidates[0]
    row_b, col_b, rot_b, _ = piece_b.location_candidates[0]

    dir_a = _side_absolute_direction(side_a_idx, rot_a)
    dir_b = _side_absolute_direction(side_b_idx, rot_b)
    dr, dc = _DIR_DELTAS[dir_a]
    expected_row, expected_col = row_a + dr, col_a + dc

    dist = math.hypot(expected_row - row_b, expected_col - col_b)
    if dir_b != (dir_a + 2) % 4:
        # Стороны смотрят не друг на друга — явно не тот шов, штраф сверху
        # расстояния, а не замена его (сохраняет монотонность метрики).
        dist += 2.0
    return dist


def relative_rotation_deg(side_a_idx: int, side_b_idx: int) -> int:
    """На сколько градусов (по часовой стрелке) повернуть piece_b, оставляя
    piece_a в текущей ориентации, чтобы side_b встал напротив side_a."""
    return ((side_a_idx + 2 - side_b_idx) % 4) * 90


def score_side_pair(piece_a: PieceRecord, side_a: int, piece_b: PieceRecord, side_b: int) -> EdgeMatch:
    """S = w1*D_shape + w2*D_color + w3*D_pos, каждое расстояние нормировано
    своим *_scale_* в [0,1] перед взвешиванием (сырые единицы несравнимы
    между собой — мм, ΔE, клетки сетки). Выступ сравнивается только со
    впадиной (см. модуль docstring); прочие комбинации дают score=0."""
    cfg = get_config()
    mc = cfg.section("match")

    sa, sb = piece_a.sides[side_a], piece_b.sides[side_b]
    complementary = {sa.kind, sb.kind} == {SideKind.TAB, SideKind.BLANK}
    if not complementary:
        return EdgeMatch(
            piece_a=piece_a.id, side_a=side_a, piece_b=piece_b.id, side_b=side_b,
            score=0.0, d_shape=math.inf, d_color=math.inf, d_pos=math.inf,
        )

    d_shape = shape_distance_mm(sa, sb)
    d_color = color_distance_deltae(sa, sb)
    d_pos = position_distance_cells(piece_a, side_a, piece_b, side_b)

    n_shape = min(1.0, d_shape / float(mc["shape_scale_mm"])) if math.isfinite(d_shape) else 1.0
    n_color = min(1.0, d_color / float(mc["color_scale_deltae"])) if math.isfinite(d_color) else 1.0

    weights = mc["weights"]
    w_shape, w_color, w_pos = float(weights["shape"]), float(weights["color"]), float(weights["position"])

    if d_pos is None:
        # Позиционный сигнал недоступен -> перераспределяем его вес на
        # форму/цвет вместо того, чтобы считать это "плохим" совпадением.
        total = w_shape + w_color
        w_shape, w_color = w_shape / total, w_color / total
        cost = w_shape * n_shape + w_color * n_color
        d_pos_out = math.inf
    else:
        n_pos = min(1.0, d_pos / float(mc["pos_scale_cells"]))
        cost = w_shape * n_shape + w_color * n_color + w_pos * n_pos
        d_pos_out = d_pos

    score = max(0.0, 1.0 - cost)
    return EdgeMatch(
        piece_a=piece_a.id, side_a=side_a, piece_b=piece_b.id, side_b=side_b,
        score=score, d_shape=d_shape, d_color=d_color, d_pos=d_pos_out,
    )


def _matchable_sides(piece: PieceRecord) -> list[int]:
    return [s.index for s in piece.sides if s.kind in (SideKind.TAB, SideKind.BLANK)]


def find_side_candidates(
    piece: PieceRecord, side_idx: int, catalog: list[PieceRecord], top_k: int | None = None
) -> list[EdgeMatch]:
    """Топ-K кандидатов совместимости для одной стороны детали среди каталога.

    Основной путь — детали, чья предсказанная позиция на образце соседняя
    (в пределах match.neighbor_cell_radius); при недостатке уверенных
    позиционных кандидатов (типично для однотонных зон, где locate часто не
    может уверенно привязать деталь) добавляется запасной пул — top-K по
    одной лишь форме (faiss_top_k_shape) с усиленным весом формы
    (monochrome_shape_weight_boost), поскольку без позиции сравнение
    полагается только на геометрию и цвет шва."""
    cfg = get_config()
    mc = cfg.section("match")
    top_k = top_k if top_k is not None else int(mc["top_k_candidates_per_side"])
    kind = piece.sides[side_idx].kind
    if kind not in (SideKind.TAB, SideKind.BLANK):
        return []
    wanted_kind = SideKind.BLANK if kind == SideKind.TAB else SideKind.TAB

    others = [p for p in catalog if p.id != piece.id]

    neighbor_radius = int(mc["neighbor_cell_radius"])
    position_pool: list[tuple[PieceRecord, int]] = []
    if piece.location_candidates:
        for other in others:
            if not other.location_candidates:
                continue
            for s in other.sides:
                if s.kind != wanted_kind:
                    continue
                d_pos = position_distance_cells(piece, side_idx, other, s.index)
                if d_pos is not None and d_pos <= neighbor_radius:
                    position_pool.append((other, s.index))

    shape_boost = float(mc["monochrome_shape_weight_boost"])
    use_shape_fallback = len(position_pool) < top_k
    shape_pool: list[tuple[PieceRecord, int]] = []
    if use_shape_fallback:
        shape_top_k = int(mc["faiss_top_k_shape"])
        sa = piece.sides[side_idx]
        scored_sides = []
        for other in others:
            for s in other.sides:
                if s.kind != wanted_kind:
                    continue
                scored_sides.append((shape_distance_mm(sa, s), other, s.index))
        scored_sides.sort(key=lambda t: t[0])
        shape_pool = [(o, si) for _d, o, si in scored_sides[:shape_top_k]]

    pool: dict[tuple[str, int], tuple[PieceRecord, int]] = {}
    for other, si in position_pool + shape_pool:
        pool[(other.id, si)] = (other, si)

    scored: list[EdgeMatch] = []
    for other, si in pool.values():
        match = score_side_pair(piece, side_idx, other, si)
        if not math.isfinite(match.d_pos) and use_shape_fallback:
            # Без позиции полагаемся сильнее на форму — пересчитываем score
            # с усиленным весом формы, а не только с перераспределённым.
            n_shape = min(1.0, match.d_shape / float(mc["shape_scale_mm"])) if math.isfinite(match.d_shape) else 1.0
            n_color = min(1.0, match.d_color / float(mc["color_scale_deltae"])) if math.isfinite(match.d_color) else 1.0
            weights = mc["weights"]
            w_shape = float(weights["shape"]) * shape_boost
            w_color = float(weights["color"])
            total = w_shape + w_color
            cost = (w_shape / total) * n_shape + (w_color / total) * n_color
            match = match.model_copy(update={"score": max(0.0, 1.0 - cost)})
        scored.append(match)

    scored.sort(key=lambda m: -m.score)
    return scored[:top_k]


def build_edge_matches(pieces: list[PieceRecord], top_k: int | None = None) -> list[EdgeMatch]:
    """Топ-1 кандидат для каждой (не прямой) стороны каждой детали каталога —
    сырой пул рёбер-кандидатов для планировщика (plan.next_steps)."""
    edges: list[EdgeMatch] = []
    for piece in pieces:
        for side_idx in _matchable_sides(piece):
            candidates = find_side_candidates(piece, side_idx, pieces, top_k=top_k)
            if candidates:
                edges.append(candidates[0])
    return edges
