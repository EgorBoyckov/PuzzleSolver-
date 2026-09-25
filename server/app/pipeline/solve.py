"""Модуль 6а: глобальная сборка — однозначная раскладка всех деталей по сетке.

Вход — тензор стоимостей «деталь x поворот x ячейка» от locate (сходство с
образцом) и формы сторон деталей (match). Выход — для каждой ячейки одна
деталь и её поворот (взаимно однозначно), с уверенностью.

Одного сходства с образцом недостаточно: в однотонных зонах (небо, вода)
соседние ячейки неразличимы по цвету — на синтетике 2000 деталей верная
ячейка первой по внешнему виду лишь у ~55% однотонных деталей. Зато у
соседних деталей ФОРМА общей линии разреза совпадает, а тип (выступ/впадина)
обязан быть противоположным. Поэтому сборка идёт так, как собирает человек:

1. Якоря: детали, у которых лучшая по образцу ячейка с большим отрывом
   лучше любой другой (текстурные зоны, рамка) — ставятся сразу. По парам
   соседних якорей оценивается, как выглядит верный стык: типичное
   расстояние формы sigma_S и его разброс spread_S (так масштаб не нужно
   подбирать под конкретный пазл).
2. Рост от якорей. Для пустой ячейки фронта кандидаты (деталь, поворот) —
   все свободные детали с типами сторон, комплементарными ВСЕМ стоящим
   соседям; M лучших по образцу сравниваются по форме. Оценка кандидата
       стоимость_образца / sigma_A + сумма ((d - sigma_S)+ / spread_S)^2
   (≈ -2 log правдоподобия; штраф стыка ограничен сверху — один «плохой»
   стык из-за ошибки измерения не должен топить верную деталь). Уверенность
   ячейки — вероятность лучшего кандидата (softmax по -оценка/2) с учётом
   «нулевой гипотезы» (верной детали среди кандидатов нет: потеряна при
   съёмке или уже ошибочно стоит в другом месте). Ставится самая уверенная
   ячейка; пороги уверенности снижаются ступенями (growth_min_prob), так что
   сомнительные ячейки ждут, пока у них появятся новые соседи.
3. Ремонт: детали с неправдоподобными стыками снимаются, запоминаются как
   запрещённые для этой ячейки, рост повторяется.
4. Локальный поиск по энергии раскладки (замены и обмены деталей вокруг
   ячеек с плохими стыками и пустых ячеек) — разрывает цепочки вида «деталь
   A стоит в ячейке B, а в ячейке A — чужая деталь».
5. Остаток — венгерским алгоритмом по сходству с образцом, но только если
   это правдоподобнее пустой ячейки: «чужая» деталь хуже честной пустоты
   (вероятнее всего, деталь потеряна при съёмке).

Прежде чем прийти к этой схеме, на синтетике проверены и отвергнуты: чистое
сравнение с образцом + венгерский алгоритм (однотон ~55%), жадный рост с
отрывом «лучший минус второй» (единственный кандидат получал бесконечную
уверенность и занимал «дыру» потерянной детали — 0.5% потерь давали 3%
ошибок) и жёсткий порог на каждый стык (один выброс измерения — каскад).
"""
from __future__ import annotations

import heapq
import logging
import time
from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment

from app.core.config import get_config
from app.pipeline.locate import INFEASIBLE_COST, LocateResult, canvas_side_index
from app.pipeline.match import SideShapeBank

logger = logging.getLogger(__name__)

# Смещение соседа по направлению стороны ячейки: 0 — верх, 1 — право, 2 — низ, 3 — лево.
_DIRS = ((-1, 0), (0, 1), (1, 0), (0, -1))


@dataclass
class Placement:
    piece_index: int       # индекс в LocateResult.pieces
    row: int
    col: int
    rotation: int          # 0..3, см. locate.canvas_side_index
    confidence: float
    source: str            # "anchor" | "growth" | "fallback"


@dataclass
class Layout:
    rows: int
    cols: int
    placements: list[Placement]

    def grid(self) -> np.ndarray:
        """(rows, cols) индекс детали в ячейке, -1 — пусто."""
        g = -np.ones((self.rows, self.cols), dtype=int)
        for pl in self.placements:
            g[pl.row, pl.col] = pl.piece_index
        return g


class _Assembler:
    def __init__(self, costs: np.ndarray, bank: SideShapeBank, rows: int, cols: int, cfg: dict):
        self.costs = costs
        self.per_cell = costs.min(axis=1)
        self.bank = bank
        self.rows, self.cols = rows, cols
        self.m_candidates = int(cfg["candidates_per_cell"])
        self.w_shape = float(cfg["shape_weight"])
        self.anchor_margin = float(cfg["anchor_margin"])
        self.neighbor_bonus = float(cfg["neighbor_bonus"])
        self.max_edge_sigma = float(cfg["max_edge_sigma"])
        self.repair_rounds = int(cfg["repair_rounds"])
        n_pieces, n_cells = self.per_cell.shape
        # kinds_rot[p, r, i] — тип стороны детали p, оказавшейся стороной i ячейки при повороте r.
        side_of = np.array([[canvas_side_index(i, r) for i in range(4)] for r in range(4)])
        self.kinds_rot = bank.kinds[:, side_of]                 # (n, 4, 4)
        self.feasible = costs < INFEASIBLE_COST / 10              # (n, 4, cells)
        self.tabu: set[tuple[int, int]] = set()
        self.refine_rounds = int(cfg["refine_rounds"])
        self.refine_candidates = int(cfg["refine_candidates"])
        self.refine_min_gain = float(cfg["refine_min_gain"])
        self.mismatch_energy = float(cfg["mismatch_energy"])
        self.empty_energy = float(cfg["empty_energy"])
        # Сходство с образцом (в единицах sigma_A), типичное на пороге для
        # верной детали — «нулевая гипотеза» и цена пустой ячейки.
        self.null_appearance = float(cfg["null_appearance"])
        self.null_edge_sigma = float(cfg["null_edge_sigma"])
        self.growth_min_prob = [float(x) for x in cfg["growth_min_prob"]]
        self.cell_piece = -np.ones(n_cells, dtype=int)
        self.cell_rot = np.zeros(n_cells, dtype=int)
        self.cell_conf = np.zeros(n_cells)
        self.cell_source = [""] * n_cells
        self.piece_cell = -np.ones(n_pieces, dtype=int)
        self.shape_cache: dict[tuple[int, int, int, int], float] = {}
        self.sigma_a = 1.0
        self.sigma_s = 1.0
        self.spread_s = 0.2

    # --- вспомогательное -------------------------------------------------
    def _neighbors(self, cell: int):
        r0, c0 = divmod(cell, self.cols)
        for i, (dr, dc) in enumerate(_DIRS):
            r, c = r0 + dr, c0 + dc
            if 0 <= r < self.rows and 0 <= c < self.cols:
                yield i, r * self.cols + c

    def _place(self, piece: int, cell: int, rot: int, conf: float, source: str) -> None:
        self.cell_piece[cell], self.cell_rot[cell] = piece, rot
        self.cell_conf[cell], self.cell_source[cell] = conf, source
        self.piece_cell[piece] = cell

    def _shape(self, keys: list[tuple[int, int, int, int]]) -> None:
        need = list(dict.fromkeys(k for k in keys if k not in self.shape_cache))
        if not need:
            return
        arr = np.array(need)
        dist = self.bank.distance(arr[:, 0], arr[:, 1], arr[:, 2], arr[:, 3])
        for (p, a, q, b), d in zip(need, dist):
            self.shape_cache[(p, a, q, b)] = float(d)
            self.shape_cache[(q, b, p, a)] = float(d)

    def _edge_term(self, d: float) -> float:
        """Штраф стыка: у верных пар расстояние формы кучно лежит около
        sigma_S (разброс spread_S ~ 0.17 sigma_S), у случайных — широко и
        заметно выше. Поэтому штрафуется только превышение над типичным для
        верной пары, в единицах разброса верных пар (≈ -2 log правдоподобия),
        с «потолком» на пороге правдоподобия (робастная функция потерь)."""
        z = max(0.0, min(d, self.edge_limit) - self.sigma_s) / self.spread_s
        return self.w_shape * z * z

    def _null_score(self, n_neighbors: int) -> float:
        """Оценка «нулевой гипотезы» — верной детали для ячейки нет среди
        кандидатов (потеряна или стоит в другом месте): так выглядела бы
        верная деталь с сомнительным (null_appearance) сходством с образцом и
        стыками на null_edge_sigma разбросов хуже типичного. Кандидат,
        который хуже этого, скорее чужой — ячейка остаётся ждать."""
        z = self.null_edge_sigma
        return self.null_appearance + n_neighbors * self.w_shape * z * z

    @property
    def edge_limit(self) -> float:
        return self.sigma_s + self.max_edge_sigma * self.spread_s

    # --- этапы -------------------------------------------------------------
    def place_anchors(self) -> int:
        per_cell = self.per_cell
        part = np.partition(per_cell, 1, axis=1)
        margin = part[:, 1] / np.maximum(part[:, 0], 1e-9)
        best = per_cell.argmin(axis=1)
        feasible = part[:, 0] < INFEASIBLE_COST / 10
        is_anchor = (margin > self.anchor_margin) & feasible
        self.sigma_a = float(np.median(part[is_anchor, 0])) if is_anchor.sum() >= 10 else float(np.median(part[feasible, 0]))
        self.sigma_a = max(self.sigma_a, 1e-6)
        placed = 0
        for p in np.argsort(-margin):
            if not is_anchor[p]:
                break
            cell = int(best[p])
            if self.cell_piece[cell] >= 0:
                continue
            self._place(int(p), cell, int(self.costs[p, :, cell].argmin()), 1.0 - 1.0 / margin[p], "anchor")
            placed += 1

        # Масштаб формы — по парам соседних якорей, прошедших тип-проверку.
        keys = []
        for cell in np.where(self.cell_piece >= 0)[0]:
            for i, nb in self._neighbors(int(cell)):
                if i in (1, 2) and self.cell_piece[nb] >= 0:
                    p, q = self.cell_piece[cell], self.cell_piece[nb]
                    a = canvas_side_index(i, self.cell_rot[cell])
                    b = canvas_side_index((i + 2) % 4, self.cell_rot[nb])
                    if self.bank.complementary(np.array([p]), np.array([a]), np.array([q]), np.array([b]))[0]:
                        keys.append((int(p), a, int(q), b))
        self._shape(keys)
        if len(keys) >= 10:
            d = np.array([self.shape_cache[k] for k in keys])
            self.sigma_s = max(float(np.median(d)), 1e-3)
            # Разброс — по хвосту (95-й перцентиль), а не только по MAD:
            # распределение верных пар тяжелохвостое (неточные углы,
            # локальные дефекты контура), и оценка по MAD отбраковывала бы
            # заметную долю верных стыков.
            mad = 1.4826 * float(np.median(np.abs(d - self.sigma_s)))
            tail = (float(np.percentile(d, 95)) - self.sigma_s) / 1.645
            self.spread_s = max(mad, tail, 0.1 * self.sigma_s)
        else:
            self.sigma_s = float(get_config().section("match")["shape_scale_mm"])
            self.spread_s = 0.2 * self.sigma_s
        logger.info(
            "solve: якорей %d, sigma_A=%.3f, форма верной пары %.3f ± %.3f мм (по %d парам)",
            placed, self.sigma_a, self.sigma_s, self.spread_s, len(keys),
        )
        return placed

    def _evaluate(self, cell: int):
        nbs = [(i, int(self.cell_piece[nb]), int(self.cell_rot[nb])) for i, nb in self._neighbors(cell) if self.cell_piece[nb] >= 0]
        if not nbs:
            return None
        # Кандидаты (деталь, поворот): свободные, допустимые по рамке и с
        # типами сторон, комплементарными ВСЕМ стоящим соседям (выступ <->
        # впадина) — фильтр векторный по всем деталям, поэтому глубина
        # поиска не ограничена «топ-M по цвету» (в однотонной зоне верная
        # деталь бывает и 100-й по сходству с образцом).
        ok = self.feasible[:, :, cell] & (self.piece_cell < 0)[:, None]
        for i, q, rq in nbs:
            kq = self.bank.kinds[q, canvas_side_index((i + 2) % 4, rq)]
            need = 2 if kq == 1 else 1 if kq == 2 else -1
            ok &= self.kinds_rot[:, :, i] == need
        pr = np.argwhere(ok)
        if len(pr) == 0:
            return None
        if self.tabu:
            keep = np.array([(cell, int(p)) not in self.tabu for p in pr[:, 0]])
            pr = pr[keep]
            if len(pr) == 0:
                return None
        cost = self.costs[pr[:, 0], pr[:, 1], cell]
        if len(pr) > self.m_candidates:
            sel = np.argpartition(cost, self.m_candidates)[: self.m_candidates]
            pr, cost = pr[sel], cost[sel]
        combos = []
        for (p, rot), c in zip(pr, cost):
            p, rot = int(p), int(rot)
            pairs = [(p, canvas_side_index(i, rot), q, canvas_side_index((i + 2) % 4, rq)) for i, q, rq in nbs]
            combos.append((p, rot, pairs, float(c)))
        self._shape([k for _, _, pairs, _ in combos for k in pairs])
        limit = self.edge_limit
        app_terms, shape_terms, kept = [], [], []
        for p, rot, pairs, c in combos:
            d = [self.shape_cache[k] for k in pairs]
            # Робастно: стык хуже порога штрафуется как стык на пороге, но
            # кандидат не отбрасывается — у верной детали изредка бывает
            # один «плохой» стык из-за ошибки измерения (угол, сегментация),
            # и три отличных стыка должны это перевесить.
            if sum(x <= limit for x in d) == 0 and len(d) > 1:
                continue
            kept.append((p, rot))
            app_terms.append(c / self.sigma_a)
            shape_terms.append(sum(self._edge_term(x) for x in d))
        if not kept:
            return None
        scores = np.array(app_terms) + np.array(shape_terms)
        order = np.argsort(scores)
        best_p, best_rot = kept[order[0]]
        # Уверенность — вероятность лучшего кандидата (softmax по -score/2),
        # где в знаменателе есть и «нулевая гипотеза»: верной детали среди
        # кандидатов нет (её потеряли при съёмке или она уже ошибочно стоит в
        # другом месте), а лучшая подходит лишь на пороге правдоподобия по
        # каждому стыку. Вероятность, а не отрыв от второго, правильно
        # учитывает глубину пула: 20 почти равных кандидатов — это не
        # уверенность, даже если второй чуть хуже первого.
        null_score = self._null_score(len(nbs))
        best_score = scores[order[0]]
        other = np.array([scores[j] for j in order[1:] if kept[j][0] != best_p])
        mass = np.exp(-(other - best_score) / 2.0).sum() + np.exp(-(null_score - best_score) / 2.0)
        prob = float(1.0 / (1.0 + mass))
        priority = prob * (1.0 + self.neighbor_bonus * (len(nbs) - 1))
        return priority, prob, best_p, best_rot

    def grow(self, min_prob: float = 0.0) -> int:
        """Рост от стоящих деталей: самая уверенная ячейка фронта ставится
        первой. Ячейки с вероятностью лучшего кандидата ниже min_prob ждут —
        их пересчитают, когда появятся новые соседи."""
        version = np.zeros(len(self.cell_piece), dtype=int)
        heap: list = []

        def push(cell: int) -> None:
            res = self._evaluate(cell)
            if res is not None and res[1] >= min_prob:
                priority, prob, p, rot = res
                heapq.heappush(heap, (-priority, int(version[cell]), cell, p, rot, prob))

        for cell in range(len(self.cell_piece)):
            if self.cell_piece[cell] < 0:
                push(cell)
        placed = 0
        while heap:
            _, ver, cell, p, rot, prob = heapq.heappop(heap)
            if ver != version[cell] or self.cell_piece[cell] >= 0:
                continue
            if self.piece_cell[p] >= 0:          # деталь уже заняла другую ячейку
                version[cell] += 1
                push(cell)
                continue
            self._place(p, cell, rot, float(prob), "growth")
            placed += 1
            for _, nb in self._neighbors(cell):
                if self.cell_piece[nb] < 0:
                    version[nb] += 1
                    push(nb)
        return placed

    def _bad_edges(self) -> list[int]:
        """Ячейки с деталями «роста», у которых хотя бы один стык с соседом
        неправдоподобен по форме (или по типу замка)."""
        limit = self.edge_limit
        keys, owners = [], []
        bad: set[int] = set()
        for cell in np.where(self.cell_piece >= 0)[0]:
            for i, nb in self._neighbors(int(cell)):
                if i not in (1, 2) or self.cell_piece[nb] < 0:
                    continue
                p, q = int(self.cell_piece[cell]), int(self.cell_piece[nb])
                a = canvas_side_index(i, int(self.cell_rot[cell]))
                b = canvas_side_index((i + 2) % 4, int(self.cell_rot[nb]))
                if not self.bank.complementary(np.array([p]), np.array([a]), np.array([q]), np.array([b]))[0]:
                    bad.update(self._blame(int(cell), nb))
                    continue
                keys.append((p, a, q, b))
                owners.append((int(cell), nb))
        self._shape(keys)
        for k, (c1, c2) in zip(keys, owners):
            if self.shape_cache[k] > limit:
                bad.update(self._blame(c1, c2))
        return sorted(bad)

    def _blame(self, c1: int, c2: int) -> list[int]:
        """Какую из двух ячеек плохого стыка освободить: деталь роста с
        меньшей уверенностью (якоря не снимаются)."""
        cand = [c for c in (c1, c2) if self.cell_source[c] != "anchor"]
        if not cand:
            return []
        return [min(cand, key=lambda c: self.cell_conf[c])]

    def repair(self) -> int:
        """Снять детали с неправдоподобными стыками, запретить им эти
        ячейки и дорастить заново — разрывает цепочки ошибок роста."""
        removed_total = 0
        for _ in range(self.repair_rounds):
            bad = self._bad_edges()
            if not bad:
                break
            for cell in bad:
                p = int(self.cell_piece[cell])
                self.tabu.add((cell, p))
                self.piece_cell[p] = -1
                self.cell_piece[cell] = -1
                self.cell_source[cell] = ""
            removed_total += len(bad)
            for min_prob in self.growth_min_prob:
                self.grow(min_prob)
        return removed_total

    # --- локальная оптимизация --------------------------------------------
    def _edge_energy(self, cell: int, i: int, nb: int) -> float:
        p, q = int(self.cell_piece[cell]), int(self.cell_piece[nb])
        a = canvas_side_index(i, int(self.cell_rot[cell]))
        b = canvas_side_index((i + 2) % 4, int(self.cell_rot[nb]))
        ka, kb = self.bank.kinds[p, a], self.bank.kinds[q, b]
        if not ((ka == 1 and kb == 2) or (ka == 2 and kb == 1)):
            return self.mismatch_energy
        key = (p, a, q, b)
        if key not in self.shape_cache:
            self._shape([key])
        return min(self._edge_term(self.shape_cache[key]), self.mismatch_energy)

    def _local_energy(self, cells: set[int]) -> float:
        """Энергия ячеек cells и всех стыков, которых они касаются (каждый
        стык один раз): стоимость образца / sigma_A + форма стыков; пустая
        ячейка стоит как «деталь на пороге правдоподобия по каждому стыку»."""
        energy = 0.0
        seen: set[tuple[int, int]] = set()
        for cell in cells:
            if self.cell_piece[cell] < 0:
                n_nb = sum(1 for _, nb in self._neighbors(cell) if self.cell_piece[nb] >= 0)
                energy += self.empty_energy + self._null_score(n_nb)
                continue
            energy += float(self.costs[self.cell_piece[cell], self.cell_rot[cell], cell]) / self.sigma_a
            for i, nb in self._neighbors(cell):
                if self.cell_piece[nb] < 0:
                    continue
                key = (min(cell, nb), max(cell, nb))
                if key in seen:
                    continue
                seen.add(key)
                energy += self._edge_energy(cell, i, nb)
        return energy

    def _suspects(self) -> list[int]:
        limit = self._edge_term(self.edge_limit)
        out = []
        for cell in np.where(self.cell_piece >= 0)[0]:
            cell = int(cell)
            if self.cell_source[cell] == "anchor":
                continue
            if any(self.cell_piece[nb] >= 0 and self._edge_energy(cell, i, nb) > limit for i, nb in self._neighbors(cell)):
                out.append(cell)
        # Пустые ячейки с соседями — тоже кандидаты на улучшение.
        for cell in np.where(self.cell_piece < 0)[0]:
            if any(self.cell_piece[nb] >= 0 for _, nb in self._neighbors(int(cell))):
                out.append(int(cell))
        return out

    def _best_rotation(self, cell: int, piece: int, region: set[int]) -> tuple[int, float]:
        best_rot, best_e = -1, np.inf
        old_p, old_r = int(self.cell_piece[cell]), int(self.cell_rot[cell])
        for rot in range(4):
            if not self.feasible[piece, rot, cell]:
                continue
            self.cell_piece[cell], self.cell_rot[cell] = piece, rot
            e = self._local_energy(region)
            if e < best_e:
                best_rot, best_e = rot, e
        self.cell_piece[cell], self.cell_rot[cell] = old_p, old_r
        return best_rot, best_e

    def _try_moves(self, cell: int) -> bool:
        """Лучшее улучшающее действие для ячейки: заменить её деталь свободной
        деталью, обменять с деталью другой ячейки (не якорем) или освободить."""
        nbs = [(i, int(self.cell_piece[nb]), int(self.cell_rot[nb])) for i, nb in self._neighbors(cell) if self.cell_piece[nb] >= 0]
        ok = self.feasible[:, :, cell].copy()
        for i, q, rq in nbs:
            kq = self.bank.kinds[q, canvas_side_index((i + 2) % 4, rq)]
            need = 2 if kq == 1 else 1 if kq == 2 else -1
            ok &= self.kinds_rot[:, :, i] == need
        cur = int(self.cell_piece[cell])
        if cur >= 0:
            ok[cur] = False
        placed_at = self.piece_cell
        anchor_cells = np.array([src == "anchor" for src in self.cell_source])
        anchor_piece = np.zeros(len(placed_at), dtype=bool)
        anchor_piece[self.cell_piece[anchor_cells]] = True
        ok &= ~anchor_piece[:, None]
        pr = np.argwhere(ok)
        if len(pr) == 0:
            return False
        cost = self.costs[pr[:, 0], pr[:, 1], cell]
        if len(pr) > self.refine_candidates:
            sel = np.argpartition(cost, self.refine_candidates)[: self.refine_candidates]
            pr = pr[sel]
        candidates = list(dict.fromkeys(int(p) for p in pr[:, 0]))

        best_delta, best_action = -self.refine_min_gain, None
        for q in candidates:
            other = int(placed_at[q])
            region = {cell} | ({other} if other >= 0 else set())
            before = self._local_energy(region)
            saved = [(c, int(self.cell_piece[c]), int(self.cell_rot[c])) for c in region]
            # q -> cell; текущая деталь cell -> other (обмен) или в запас.
            if other >= 0:
                self.cell_piece[other] = -1
            self.cell_piece[cell] = -1
            rot_q, _ = self._best_rotation(cell, q, region)
            if rot_q < 0:
                for c, pp, rr in saved:
                    self.cell_piece[c], self.cell_rot[c] = pp, rr
                continue
            self.cell_piece[cell], self.cell_rot[cell] = q, rot_q
            options = [(-1, -1)]
            if other >= 0 and cur >= 0:
                rot_p, _ = self._best_rotation(other, cur, region)
                if rot_p >= 0:
                    options.append((cur, rot_p))
            for piece_o, rot_o in options:
                if other >= 0:
                    self.cell_piece[other], self.cell_rot[other] = (piece_o, rot_o) if piece_o >= 0 else (-1, 0)
                delta = self._local_energy(region) - before
                if delta < best_delta:
                    best_delta, best_action = delta, (q, rot_q, other, piece_o, rot_o)
            for c, pp, rr in saved:
                self.cell_piece[c], self.cell_rot[c] = pp, rr
        if best_action is None:
            return False
        q, rot_q, other, piece_o, rot_o = best_action
        if cur >= 0:
            self.piece_cell[cur] = -1
        if other >= 0:
            self.cell_piece[other] = -1
            self.piece_cell[q] = -1
        self._place(q, cell, rot_q, 0.5, "refine")
        if other >= 0 and piece_o >= 0:
            self._place(piece_o, other, rot_o, 0.5, "refine")
        elif other >= 0:
            self.cell_source[other] = ""
        return True

    def refine(self) -> int:
        """Локальный поиск по энергии раскладки: обмены/замены вокруг ячеек
        с неправдоподобными стыками (и пустых ячеек) до сходимости."""
        moves = 0
        for _ in range(self.refine_rounds):
            changed = 0
            for cell in self._suspects():
                if self._try_moves(cell):
                    changed += 1
            moves += changed
            if changed == 0:
                break
        return moves

    def fill_rest(self) -> int:
        """Оставшиеся детали — в оставшиеся ячейки венгерским алгоритмом по
        сходству с образцом, но только если это правдоподобнее пустой ячейки
        (типичная для верной детали стоимость образца + стыки на пороге).
        Неправдоподобное оставляется пустым: деталь, скорее всего, потеряна
        при съёмке, а «чужая» деталь в ячейке хуже честной пустоты."""
        cells = np.where(self.cell_piece < 0)[0]
        pieces = np.where(self.piece_cell < 0)[0]
        if len(cells) == 0 or len(pieces) == 0:
            return 0
        sub = self.per_cell[np.ix_(pieces, cells)] / self.sigma_a
        ri, ci = linear_sum_assignment(sub)
        placed = 0
        for a, b in zip(ri, ci):
            p, cell = int(pieces[a]), int(cells[b])
            rot = int(self.costs[p, :, cell].argmin())
            if not self.feasible[p, rot, cell]:
                continue
            before = self._local_energy({cell})
            self.cell_piece[cell], self.cell_rot[cell] = p, rot
            after = self._local_energy({cell})
            self.cell_piece[cell] = -1
            if after < before:
                self._place(p, cell, rot, 0.0, "fallback")
                placed += 1
        return placed


def solve_layout(located: LocateResult) -> Layout:
    """Разложить детали LocateResult по сетке образца; заполнить piece.placement."""
    cfg = get_config().section("solve")
    index = located.index
    t0 = time.time()
    bank = SideShapeBank(located.pieces)
    asm = _Assembler(located.costs, bank, index.rows, index.cols, cfg)
    n_anchor = asm.place_anchors()
    n_grow = sum(asm.grow(min_prob) for min_prob in asm.growth_min_prob)
    n_repaired = asm.repair()
    n_refined = asm.refine()
    n_fallback = asm.fill_rest()
    logger.info(
        "solve: якоря %d, рост %d, снято при ремонте %d, улучшений локальным поиском %d, "
        "остаток (венгерский) %d; сравнений форм %d; %.1f с",
        n_anchor, n_grow, n_repaired, n_refined, n_fallback, len(asm.shape_cache) // 2, time.time() - t0,
    )

    placements = []
    for cell in np.where(asm.cell_piece >= 0)[0]:
        row, col = divmod(int(cell), index.cols)
        pl = Placement(int(asm.cell_piece[cell]), row, col, int(asm.cell_rot[cell]), float(asm.cell_conf[cell]), asm.cell_source[cell])
        placements.append(pl)
        located.pieces[pl.piece_index].placement = (row, col, pl.rotation * 90, round(pl.confidence, 4))
    return Layout(index.rows, index.cols, placements)
