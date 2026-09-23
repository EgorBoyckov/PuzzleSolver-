"""Модуль 5: совместимость сторон деталей (форма замка + цвет кромки).

Две соседние детали делят одну линию разреза: выступ одной — ровно впадина
другой. describe хранит каждую сторону как кривую в локальной системе (x —
вдоль стороны от угла k к углу k+1, y — наружу от детали, мм). Так как все
контуры обходятся в одной ориентации, общая линия у соседей проходится в
противоположных направлениях, и кривая партнёра переводится в систему
стороны A «зеркалированием»: обратный порядок точек, x -> L - x, y -> -y.

Почему не простое сравнение точка-в-точку по длине дуги: углы детали
известны с точностью ~0.5 мм, а граница маски сегментации у выступа и у
впадины смещена в противоположные стороны (выступ «худеет», впадина
«толстеет») — одинаковые по форме кривые оказываются сдвинуты вдоль себя и
по нормали. На синтетике 2000 деталей такое сравнение (с жёстким
совмещением Прокруста) ставило выше истинной впадины ~8% случайных впадин.
Здесь партнёр жёстко совмещается ICP (соответствия ищутся в окне вдоль
кривой), а расстоянием служит СКО знакового отклонения по нормали — оно не
чувствительно к равномерному смещению границы: ~2% случайных впадин лучше
истинной, т.е. форма в 4 раза информативнее.
"""
from __future__ import annotations

import numpy as np

from app.core.config import get_config
from app.pipeline.schemas import EdgeMatch, PieceRecord, SideKind


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
        idx = np.round(np.linspace(0, dense - 1, n)).astype(int)[None, :].repeat(m, axis=0)
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
        n_pts = max((len(s.curve) for p in pieces for s in p.sides), default=2)
        self.kinds = np.zeros((len(pieces), 4), dtype=np.int8)
        curves = np.zeros((len(pieces), 4, n_pts, 2))
        lengths = np.ones((len(pieces), 4))
        kind_code = {SideKind.STRAIGHT: 0, SideKind.TAB: 1, SideKind.BLANK: 2}
        for i, p in enumerate(pieces):
            for k, side in enumerate(p.sides[:4]):
                self.kinds[i, k] = kind_code[side.kind]
                c = side_curve(p, k)
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


def _color_strip_distance(piece_a: PieceRecord, side_a: int, piece_b: PieceRecord, side_b: int) -> float:
    """Средний ΔE (Lab) цветовых полос вдоль стороны; полоса партнёра идёт в обратном порядке."""
    sa = np.array(piece_a.sides[side_a].color_strip_lab, dtype=np.float64)
    sb = np.array(piece_b.sides[side_b].color_strip_lab, dtype=np.float64)[::-1]
    if len(sa) == 0 or len(sa) != len(sb):
        return float("nan")
    return float(np.linalg.norm(sa - sb, axis=1).mean())


def score_side_pair(piece_a: PieceRecord, side_a: int, piece_b: PieceRecord, side_b: int) -> EdgeMatch:
    """S = w_shape*D_shape + w_color*D_color (+ w_position*D_pos, если известны
    позиции на образце). Меньше — лучше. Выступ сравнивается только со
    впадиной; несовместимые по типу стороны получают score=inf."""
    mc = get_config().section("match")
    weights = mc["weights"]
    ka, kb = piece_a.sides[side_a].kind, piece_b.sides[side_b].kind
    if {ka, kb} != {SideKind.TAB, SideKind.BLANK}:
        return EdgeMatch(
            piece_a=piece_a.id, side_a=side_a, piece_b=piece_b.id, side_b=side_b,
            score=float("inf"), d_shape=float("inf"), d_color=float("nan"), d_pos=0.0,
        )
    bank = SideShapeBank([piece_a, piece_b])
    d_shape = float(bank.distance(np.array([0]), np.array([side_a]), np.array([1]), np.array([side_b]))[0])
    d_color = _color_strip_distance(piece_a, side_a, piece_b, side_b)
    d_pos = 0.0
    score = float(weights["shape"]) * d_shape / float(mc["shape_scale_mm"])
    if np.isfinite(d_color):
        score += float(weights["color"]) * d_color / float(mc["color_scale_delta_e"])
    return EdgeMatch(
        piece_a=piece_a.id, side_a=side_a, piece_b=piece_b.id, side_b=side_b,
        score=score, d_shape=d_shape, d_color=d_color, d_pos=d_pos,
    )
