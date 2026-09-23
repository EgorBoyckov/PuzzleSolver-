"""Модуль 5: совместимость сторон деталей (форма замка + цвет шва + позиция).

Две соседние детали делят одну линию разреза: выступ одной — ровно впадина
другой. describe хранит каждую сторону как кривую в локальной системе (x —
вдоль стороны от угла k к углу k+1, y — наружу от детали, мм). Так как все
контуры обходятся в одной ориентации, общая линия у соседей проходится в
противоположных направлениях, и кривая партнёра переводится в систему
стороны A «зеркалированием»: обратный порядок точек, x -> L - x, y -> -y.
Цвет шва непрерывен, поэтому цветовые полосы сравниваются в обратном
порядке без смены знака.

Расстояние формы — не сравнение точка-в-точку: углы детали известны с
точностью ~0.5 мм, а граница маски сегментации у выступа и у впадины смещена
в противоположные стороны — одинаковые по форме кривые оказываются сдвинуты
вдоль себя и по нормали. На синтетике 2000 деталей сравнение профиля y(x)
или жёсткое совмещение Прокруста ставило выше истинной впадины ~8% случайных
впадин. Здесь партнёр жёстко совмещается ICP (соответствия ищутся в окне
вдоль кривой), а расстоянием служит СКО знакового отклонения по нормали —
оно не чувствительно к равномерному смещению границы: с субпиксельным
контуром (segment.refine_contour_subpixel) истинную впадину обходят лишь
~0.3% случайных.

Позиция (D_pos) берётся из итоговой раскладки глобальной сборки
(piece.placement, solve.py), а если её нет — из уверенного top-1
location_candidates. Направление стороны k детали, стоящей с поворотом r
(0..3, см. locate.canvas_side_index), в системе образца — (r + 3 - k) % 4
(0 — вверх, 1 — вправо, 2 — вниз, 3 — влево): стороны пронумерованы по ходу
контура ПРОТИВ часовой стрелки, а поворот r — по часовой.
"""
from __future__ import annotations

import math

import numpy as np

from app.core.config import get_config
from app.pipeline.schemas import EdgeMatch, PieceRecord, Side, SideKind

# Смещение соседней клетки образца по направлению (0 — вверх, по часовой).
_DIR_DELTAS = {0: (-1, 0), 1: (0, 1), 2: (1, 0), 3: (0, -1)}


# --- геометрия кривых и пакетный ICP ---------------------------------------


def side_curve(piece: PieceRecord, side: int) -> np.ndarray:
    return np.array([[p.x, p.y] for p in piece.sides[side].curve], dtype=np.float64)


def mate_curves(curves: np.ndarray, lengths: np.ndarray) -> np.ndarray:
    """(m, n, 2) кривые сторон -> кривые в системе координат стороны-партнёра."""
    out = curves[:, ::-1].copy()
    out[..., 0] = lengths[:, None] - out[..., 0]
    out[..., 1] = -out[..., 1]
    return out


def resample_curves(curves: np.ndarray, n: int) -> np.ndarray:
    """Равномерная по длине дуги передискретизация (m, k, 2) -> (m, n, 2)."""
    seg = np.linalg.norm(np.diff(curves, axis=1), axis=2)
    cum = np.concatenate([np.zeros((len(curves), 1)), np.cumsum(seg, axis=1)], axis=1)
    out = np.empty((len(curves), n, 2))
    for m in range(len(curves)):
        t = np.linspace(0.0, cum[m, -1], n)
        out[m, :, 0] = np.interp(t, cum[m], curves[m, :, 0])
        out[m, :, 1] = np.interp(t, cum[m], curves[m, :, 1])
    return out


def icp_shape_distance(a: np.ndarray, b_dense: np.ndarray, iters: int, window: int, chunk: int = 4000) -> np.ndarray:
    """Пакетное расстояние между кривыми a (m, n, 2) и плотно
    передискретизированными кривыми-партнёрами b_dense (m, d, 2), уже
    переведёнными в систему a (mate_curves). Возвращает (m,) — СКО знакового
    отклонения по нормали после жёсткого ICP-совмещения, мм."""
    out = np.empty(len(a))
    offsets = np.arange(-window, window + 1)
    for s in range(0, len(a), chunk):
        aa, bb = a[s : s + chunk], b_dense[s : s + chunk].copy()
        m, n = aa.shape[:2]
        dense = bb.shape[1]
        rows = np.arange(m)[:, None]
        # Начальные соответствия — по доле длины дуги (плотная кривая
        # партнёра равномерна по дуге, «своя» — не обязательно).
        seg = np.linalg.norm(np.diff(aa, axis=1), axis=2)
        cum = np.concatenate([np.zeros((m, 1)), np.cumsum(seg, axis=1)], axis=1)
        frac = cum / np.maximum(cum[:, -1:], 1e-9)
        idx = np.round(frac * (dense - 1)).astype(int)
        for _ in range(iters):
            corr = bb[rows, idx]
            ma, mb = aa.mean(axis=1, keepdims=True), corr.mean(axis=1, keepdims=True)
            h = np.einsum("mni,mnj->mij", corr - mb, aa - ma)
            u, _, vt = np.linalg.svd(h)
            d = np.sign(np.linalg.det(u @ vt))
            fix = np.stack([np.ones_like(d), d], axis=1)
            rot = u @ (fix[:, :, None] * vt)
            bb = np.einsum("mni,mij->mnj", bb - mb, rot) + ma
            cand = np.clip(idx[:, :, None] + offsets[None, None, :], 0, dense - 1)
            d2 = ((bb[rows[:, :, None], cand] - aa[:, :, None, :]) ** 2).sum(-1)
            idx = np.take_along_axis(cand, d2.argmin(axis=2)[..., None], axis=2)[..., 0]
        diff = aa - bb[rows, idx]
        tang = np.gradient(aa, axis=1)
        tang /= np.linalg.norm(tang, axis=2, keepdims=True) + 1e-9
        normal = np.stack([-tang[..., 1], tang[..., 0]], axis=-1)
        out[s : s + chunk] = np.sum(diff * normal, axis=-1).std(axis=1)
    return out


class SideShapeBank:
    """Предвычисленные кривые сторон всех деталей для быстрого пакетного
    сравнения: stride-подвыборка «своих» кривых и плотные зеркальные кривые
    «партнёров» считаются один раз на сторону, а не на пару."""

    def __init__(self, pieces: list[PieceRecord]):
        mc = get_config().section("match")
        self.iters = int(mc["icp_iterations"])
        self.window = int(mc["icp_window"])
        n_pts = max(2, max((len(s.curve) for p in pieces for s in p.sides), default=2))
        self.kinds = np.zeros((len(pieces), 4), dtype=np.int8)
        curves = np.zeros((len(pieces), 4, n_pts, 2))
        lengths = np.ones((len(pieces), 4))
        kind_code = {SideKind.STRAIGHT: 0, SideKind.TAB: 1, SideKind.BLANK: 2}
        for i, p in enumerate(pieces):
            for k, side in enumerate(p.sides[:4]):
                self.kinds[i, k] = kind_code[side.kind]
                c = side_curve(p, k)
                if len(c) < 2:
                    continue
                if len(c) != n_pts:
                    c = resample_curves(c[None], n_pts)[0]
                curves[i, k] = c
                lengths[i, k] = max(float(c[-1, 0]), 1e-6)
        stride = int(mc["icp_subsample"])
        self.own = curves[:, :, ::stride].copy()
        flat = mate_curves(curves.reshape(-1, n_pts, 2), lengths.reshape(-1))
        self.partner = resample_curves(flat, int(mc["icp_dense_points"])).reshape(len(pieces), 4, -1, 2)

    def complementary(self, p: np.ndarray, a: np.ndarray, q: np.ndarray, b: np.ndarray) -> np.ndarray:
        ka, kb = self.kinds[p, a], self.kinds[q, b]
        return ((ka == 1) & (kb == 2)) | ((ka == 2) & (kb == 1))

    def distance(self, p: np.ndarray, a: np.ndarray, q: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Расстояние формы (мм) для пакета пар сторон (деталь p, сторона a) — (деталь q, сторона b)."""
        return icp_shape_distance(self.own[p, a], self.partner[q, b], self.iters, self.window)


class _CatalogIndex:
    """Кэш по каталогу для find_side_candidates: банк кривых и сетка
    позиций. Пересобирается, если изменился состав каталога или позиции."""

    def __init__(self, catalog: list[PieceRecord], key: tuple):
        self.key = key
        self.pieces = catalog
        self.pos = {p.id: i for i, p in enumerate(catalog)}
        self.bank = SideShapeBank(catalog)
        self.by_cell: dict[tuple[int, int], list[int]] = {}
        for i, p in enumerate(catalog):
            position = _piece_position(p)
            if position is not None:
                self.by_cell.setdefault((position[0], position[1]), []).append(i)


_catalog_cache: _CatalogIndex | None = None


def _catalog_index(catalog: list[PieceRecord]) -> _CatalogIndex:
    global _catalog_cache
    key = tuple((p.id, _piece_position(p), len(p.sides)) for p in catalog)
    if _catalog_cache is None or _catalog_cache.key != key:
        _catalog_cache = _CatalogIndex(catalog, key)
    return _catalog_cache


# --- составляющие score ----------------------------------------------------


def side_direction(side_index: int, rotation_deg: int) -> int:
    """Направление (0 — вверх, 1 — вправо, 2 — вниз, 3 — влево) стороны
    side_index детали, стоящей на образце с поворотом rotation_deg — обратное
    к locate.canvas_side_index."""
    return (rotation_deg // 90 + 3 - side_index) % 4


def shape_distance_mm(side_a: Side, side_b: Side) -> float:
    """Расстояние формы (мм) между стороной A и зеркально отражённой стороной
    B — СКО нормального отклонения после ICP-совмещения (см. модуль)."""
    ca = np.array([[p.x, p.y] for p in side_a.curve], dtype=np.float64)
    cb = np.array([[p.x, p.y] for p in side_b.curve], dtype=np.float64)
    if len(ca) < 2 or len(cb) < 2:
        return math.inf
    mc = get_config().section("match")
    stride = int(mc["icp_subsample"])
    partner = resample_curves(mate_curves(cb[None], np.array([cb[-1, 0]])), int(mc["icp_dense_points"]))
    own = ca[::stride] if len(ca) >= 4 * stride else ca
    return float(icp_shape_distance(own[None], partner, int(mc["icp_iterations"]), int(mc["icp_window"]))[0])


def color_distance_deltae(side_a: Side, side_b: Side) -> float:
    """Среднее евклидово расстояние в Lab между цветовыми полосами двух
    сторон (сопоставленными в обратном порядке — цвет по обе стороны шва
    непрерывен, знак менять не нужно)."""
    la = np.array(side_a.color_strip_lab, dtype=np.float64)
    lb = np.array(side_b.color_strip_lab, dtype=np.float64)
    if len(la) == 0 or len(la) != len(lb):
        return math.inf
    return float(np.mean(np.linalg.norm(la - lb[::-1], axis=1)))


def _piece_position(piece: PieceRecord) -> tuple[int, int, int] | None:
    """(row, col, rotation_deg) детали на образце: итоговая раскладка
    глобальной сборки, а без неё — top-1 привязки, если он уверенный."""
    if piece.placement is not None:
        row, col, rot, _conf = piece.placement
        return int(row), int(col), int(rot)
    if piece.location_candidates:
        row, col, rot, conf = piece.location_candidates[0]
        if conf >= float(get_config().section("match")["position_min_confidence"]):
            return int(row), int(col), int(rot)
    return None


def position_distance_cells(
    piece_a: PieceRecord, side_a_idx: int, piece_b: PieceRecord, side_b_idx: int
) -> float | None:
    """Расстояние (в клетках сетки образца) между клеткой, предсказанной для
    piece_b по позиции piece_a и направлению стороны side_a, и фактической
    клеткой piece_b. None, если позиция одной из деталей неизвестна —
    позиционный сигнал недоступен, это не то же самое, что «плохое совпадение».

    Строгий top-1 (а не перебор топ-3 кандидатов привязки): на старом locate
    перебор 3x3 комбинаций измеримо ухудшал точность — топ-3 кандидатов
    географически близки, и чужие детали чаще давали правдоподобную пару."""
    pos_a, pos_b = _piece_position(piece_a), _piece_position(piece_b)
    if pos_a is None or pos_b is None:
        return None
    row_a, col_a, rot_a = pos_a
    row_b, col_b, rot_b = pos_b
    dir_a = side_direction(side_a_idx, rot_a)
    dir_b = side_direction(side_b_idx, rot_b)
    dr, dc = _DIR_DELTAS[dir_a]
    dist = math.hypot(row_a + dr - row_b, col_a + dc - col_b)
    if dir_b != (dir_a + 2) % 4:
        # Стороны смотрят не друг на друга — явно не тот шов, штраф сверху
        # расстояния, а не замена его (сохраняет монотонность метрики).
        dist += 2.0
    return dist


def relative_rotation_deg(side_a_idx: int, side_b_idx: int) -> int:
    """На сколько градусов по часовой стрелке повернуть piece_b относительно
    piece_a, чтобы side_b встал напротив side_a: из side_direction
    (r_b + 3 - b) = (r_a + 3 - a) + 2 следует r_b - r_a = b - a + 2."""
    return ((side_b_idx - side_a_idx + 2) % 4) * 90


def _score(
    piece_a: PieceRecord, side_a: int, piece_b: PieceRecord, side_b: int, d_shape: float, shape_boost: float = 1.0
) -> EdgeMatch:
    mc = get_config().section("match")
    d_color = color_distance_deltae(piece_a.sides[side_a], piece_b.sides[side_b])
    d_pos = position_distance_cells(piece_a, side_a, piece_b, side_b)

    n_shape = min(1.0, d_shape / float(mc["shape_scale_mm"])) if math.isfinite(d_shape) else 1.0
    n_color = min(1.0, d_color / float(mc["color_scale_deltae"])) if math.isfinite(d_color) else 1.0

    weights = mc["weights"]
    w_shape, w_color, w_pos = float(weights["shape"]), float(weights["color"]), float(weights["position"])
    if d_pos is None:
        # Позиционный сигнал недоступен -> его вес перераспределяется на
        # форму/цвет (без позиции — с усиленным весом формы), а не считается
        # «плохим» совпадением.
        w_shape *= shape_boost
        total = w_shape + w_color
        cost = (w_shape * n_shape + w_color * n_color) / total
        d_pos_out = math.inf
    else:
        n_pos = min(1.0, d_pos / float(mc["pos_scale_cells"]))
        cost = w_shape * n_shape + w_color * n_color + w_pos * n_pos
        d_pos_out = d_pos
    return EdgeMatch(
        piece_a=piece_a.id, side_a=side_a, piece_b=piece_b.id, side_b=side_b,
        score=max(0.0, 1.0 - cost), d_shape=d_shape, d_color=d_color, d_pos=d_pos_out,
    )


def score_side_pair(piece_a: PieceRecord, side_a: int, piece_b: PieceRecord, side_b: int) -> EdgeMatch:
    """S = 1 - (w1*n_shape + w2*n_color + w3*n_pos), каждое расстояние
    нормировано своим *_scale_* в [0,1]; 1.0 — идеальное совпадение. Выступ
    сравнивается только со впадиной; прочие комбинации дают score=0."""
    sa, sb = piece_a.sides[side_a], piece_b.sides[side_b]
    if {sa.kind, sb.kind} != {SideKind.TAB, SideKind.BLANK}:
        return EdgeMatch(
            piece_a=piece_a.id, side_a=side_a, piece_b=piece_b.id, side_b=side_b,
            score=0.0, d_shape=math.inf, d_color=math.inf, d_pos=math.inf,
        )
    return _score(piece_a, side_a, piece_b, side_b, shape_distance_mm(sa, sb))


def _matchable_sides(piece: PieceRecord) -> list[int]:
    return [s.index for s in piece.sides if s.kind in (SideKind.TAB, SideKind.BLANK)]


def find_side_candidates(
    piece: PieceRecord, side_idx: int, catalog: list[PieceRecord], top_k: int | None = None
) -> list[EdgeMatch]:
    """Топ-K кандидатов совместимости для одной стороны детали среди каталога.

    Основной путь — детали, чья позиция на образце соседняя по направлению
    этой стороны (в пределах match.neighbor_cell_radius; поиск по сетке, а не
    перебором каталога). При нехватке позиционных кандидатов добавляется
    запасной пул — faiss_top_k_shape лучших по форме среди всех сторон
    нужного типа (быстрый отсев по профилю y(x), затем точный ICP) с
    усиленным весом формы (monochrome_shape_weight_boost)."""
    mc = get_config().section("match")
    top_k = top_k if top_k is not None else int(mc["top_k_candidates_per_side"])
    kind = piece.sides[side_idx].kind
    if kind not in (SideKind.TAB, SideKind.BLANK):
        return []
    wanted = 2 if kind == SideKind.TAB else 1

    index = _catalog_index(catalog)
    bank = index.bank
    self_idx = index.pos.get(piece.id)
    if self_idx is None:
        # Деталь не из каталога — сравниваем через отдельный банк из неё одной.
        solo = SideShapeBank([piece])
        own = solo.own[0, side_idx]
    else:
        own = bank.own[self_idx, side_idx]

    pool: set[tuple[int, int]] = set()
    position = _piece_position(piece)
    if position is not None:
        row, col, rot = position
        dr, dc = _DIR_DELTAS[side_direction(side_idx, rot)]
        radius = int(mc["neighbor_cell_radius"])
        for rr in range(row + dr - radius, row + dr + radius + 1):
            for cc in range(col + dc - radius, col + dc + radius + 1):
                for q in index.by_cell.get((rr, cc), []):
                    if q == self_idx:
                        continue
                    for s in range(4):
                        if bank.kinds[q, s] != wanted:
                            continue
                        d_pos = position_distance_cells(piece, side_idx, index.pieces[q], s)
                        if d_pos is not None and d_pos <= radius:
                            pool.add((q, s))

    use_shape_fallback = len(pool) < top_k
    if use_shape_fallback:
        qs, ss = np.nonzero(bank.kinds == wanted)
        if self_idx is not None:
            keep = qs != self_idx
            qs, ss = qs[keep], ss[keep]
        # Пул не меньше запрошенного top_k: обходу рамки (no_reference) нужен
        # весь пул, иначе посещённые детали выбивают все места и обход встаёт.
        shape_top_k = max(int(mc["faiss_top_k_shape"]), top_k)
        if len(qs) > 3 * shape_top_k:
            # Быстрый отсев: RMS профиля y по индексу точки (партнёр уже
            # зеркалирован в систему стороны A).
            idx = np.round(np.linspace(0, bank.partner.shape[2] - 1, len(own))).astype(int)
            rough = np.sqrt(np.mean((bank.partner[qs, ss][:, idx, 1] - own[None, :, 1]) ** 2, axis=1))
            keep = np.argpartition(rough, 3 * shape_top_k)[: 3 * shape_top_k]
            qs, ss = qs[keep], ss[keep]
        if len(qs):
            d = icp_shape_distance(np.repeat(own[None], len(qs), axis=0), bank.partner[qs, ss], bank.iters, bank.window)
            for j in np.argsort(d)[:shape_top_k]:
                pool.add((int(qs[j]), int(ss[j])))

    if not pool:
        return []
    pairs = sorted(pool)
    qs = np.array([q for q, _ in pairs])
    ss = np.array([s for _, s in pairs])
    d_shape = icp_shape_distance(np.repeat(own[None], len(qs), axis=0), bank.partner[qs, ss], bank.iters, bank.window)
    boost = float(mc["monochrome_shape_weight_boost"]) if use_shape_fallback else 1.0
    scored = [
        _score(piece, side_idx, index.pieces[q], s, float(d), shape_boost=boost)
        for (q, s), d in zip(pairs, d_shape)
    ]
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
