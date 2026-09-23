"""Модуль 4: привязка деталей к образцу (картинке коробки).

Картинка коробки делится на сетку rows x cols по числу деталей. Для каждой
детали считается стоимость «деталь стоит в ячейке c с поворотом r» для ВСЕХ
ячеек и всех 4 поворотов сразу:

1. Выравнивание по углам. describe находит 4 угла детали с субпиксельной
   точностью — а углы детали в собранном пазле совпадают с узлами сетки.
   Поэтому деталь не «ищется» скользящим окном, а переносится аффинным
   преобразованием (по 4 углам, МНК) прямо в систему координат ячейки
   образца, в том же масштабе, с полем pad вокруг ячейки для выступов.
   4 поворота — 4 циклических сдвига соответствия углов.
2. Дескриптор — сетка grid x grid средних цветов (BGR, взвешенно по маске
   детали) по выровненному окну; для каждой ячейки образца — то же окно.
   Сравнение — взвешенная сумма квадратов разностей в Lab (яркость L с
   меньшим весом: она сильнее всего искажается освещением стола).
   Взвешенная разность раскладывается в три матричных произведения, так что
   полный перебор 5000 деталей x 4 поворота x 5000 ячеек — секунды, без
   ненадёжного предотбора кандидатов.
3. Ограничения рамки: прямые стороны детали обязаны смотреть наружу пазла, а
   непрямые — внутрь. Это отсекает недопустимые (ячейка, поворот) целиком и
   однозначно задаёт поворот краевых деталей.
4. Коррекция освещения: по уверенным деталям первого прохода (отрыв лучшей
   ячейки от второй > anchor_margin) для каждого кадра оценивается линейное
   по координатам кадра поле усиления BGR (градиент освещения стола,
   виньетирование, баланс белого) — и дескрипторы деталей кадра
   корректируются, после чего стоимости пересчитываются.

Прежняя реализация (эмбеддинг-предотбор top-K + TM_CCORR_NORMED в окне
поиска) теряла истинную ячейку уже на предотборе у ~5% деталей, масштаб
шаблона зависел от того, есть ли у детали выступы, и тратила ~3 с на деталь
(~4.5 ч на 5000 деталей).

Результат — `LocateResult` с тензором стоимостей (для глобальной сборки,
см. solve.py) и piece.location_candidates (топ-3 ячейки).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import cv2
import numpy as np

from app.core.config import get_config
from app.pipeline.schemas import PieceRecord, SideKind

logger = logging.getLogger(__name__)

# Стоимость недопустимой пары (ячейка, поворот) — например, прямая сторона
# детали смотрит внутрь пазла. Конечное число (а не inf), чтобы не ломать
# арифметику; всё, что >= INFEASIBLE_COST / 10, считается недопустимым.
INFEASIBLE_COST = 1e6


def canvas_side_index(canvas_side: int, rotation: int) -> int:
    """Какая сторона детали (индекс в piece.sides / piece.corners_px) при
    повороте rotation (0..3) оказывается стороной canvas_side ячейки
    (0 — верх, 1 — право, 2 — низ, 3 — лево).

    Углы детали хранятся по ходу контура, который describe нормирует к
    одной ориентации (против часовой стрелки на изображении); сторона k —
    от угла k к углу k+1. Ячейка обходится по часовой: TL, TR, BR, BL.
    """
    return (3 - (canvas_side - rotation)) % 4


@dataclass
class ReferenceGridIndex:
    reference_bgr: np.ndarray
    rows: int
    cols: int
    cell_w: float
    cell_h: float
    pad_x: int
    pad_y: int
    grid: int
    # (rows*cols, grid*grid, 3) — средний BGR по бинам окна ячейки (с полем pad)
    feat_bgr: np.ndarray = field(repr=False)
    # (rows*cols, grid*grid) — доля бина, лежащая внутри картинки образца
    valid: np.ndarray = field(repr=False)
    # Исходные (до цветовой модели образца) признаки — см. correct_reference_colors.
    feat_bgr_raw: np.ndarray | None = field(default=None, repr=False)

    @property
    def canvas_size(self) -> tuple[int, int]:
        return int(round(self.cell_w)) + 2 * self.pad_x, int(round(self.cell_h)) + 2 * self.pad_y

    def cell_bounds_px(self, row: int, col: int) -> tuple[int, int, int, int]:
        x0, y0 = int(round(col * self.cell_w)), int(round(row * self.cell_h))
        x1, y1 = int(round((col + 1) * self.cell_w)), int(round((row + 1) * self.cell_h))
        return x0, y0, x1, y1

    def cell_window(self, row: int, col: int) -> tuple[np.ndarray, np.ndarray]:
        """Окно ячейки с полем pad в координатах канваса детали: BGR и доля
        пикселя внутри картинки образца (0 за её краем)."""
        return _reference_window(self.reference_bgr, row, col, self.cell_w, self.cell_h, self.pad_x, self.pad_y)


_WINDOW_BORDER = 2


def _reference_window(
    reference_bgr: np.ndarray, row: int, col: int, cell_w: float, cell_h: float, pad_x: int, pad_y: int,
    padded: tuple[np.ndarray, np.ndarray] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    if padded is None:
        padded = _pad_reference(reference_bgr, pad_x, pad_y)
    canvas_w, canvas_h = int(round(cell_w)) + 2 * pad_x, int(round(cell_h)) + 2 * pad_y
    # Левый верхний угол ячейки в padded-координатах — (col*cw + pad + border, ...);
    # в канвасе он должен оказаться в (pad, pad).
    m = np.array([[1.0, 0.0, -(col * cell_w + _WINDOW_BORDER)], [0.0, 1.0, -(row * cell_h + _WINDOW_BORDER)]])
    win = cv2.warpAffine(padded[0], m, (canvas_w, canvas_h), flags=cv2.INTER_LINEAR)
    val = cv2.warpAffine(padded[1], m, (canvas_w, canvas_h), flags=cv2.INTER_LINEAR)
    return win, val


def _pad_reference(reference_bgr: np.ndarray, pad_x: int, pad_y: int) -> tuple[np.ndarray, np.ndarray]:
    h, w = reference_bgr.shape[:2]
    by, bx = pad_y + _WINDOW_BORDER, pad_x + _WINDOW_BORDER
    padded = cv2.copyMakeBorder(reference_bgr, by, by, bx, bx, cv2.BORDER_CONSTANT, value=0)
    inside = cv2.copyMakeBorder(np.ones((h, w), np.float32), by, by, bx, bx, cv2.BORDER_CONSTANT, value=0)
    return padded, inside


def _bin_means(bgr: np.ndarray, weight: np.ndarray, grid: int) -> tuple[np.ndarray, np.ndarray]:
    """Взвешенные средние BGR по бинам grid x grid и доля веса в каждом бине."""
    w = weight.astype(np.float32)
    num = cv2.resize(bgr.astype(np.float32) * w[..., None], (grid, grid), interpolation=cv2.INTER_AREA)
    den = cv2.resize(w, (grid, grid), interpolation=cv2.INTER_AREA)
    means = num / np.maximum(den[..., None], 1e-6)
    return means.reshape(-1, 3), den.reshape(-1)


def build_reference_index(reference_bgr: np.ndarray, rows: int, cols: int) -> ReferenceGridIndex:
    lc = get_config().section("locate")
    h, w = reference_bgr.shape[:2]
    cell_w, cell_h = w / cols, h / rows
    pad_frac = float(lc["pad_fraction"])
    pad_x, pad_y = int(round(pad_frac * cell_w)), int(round(pad_frac * cell_h))
    grid = int(lc["descriptor_grid"])
    padded = _pad_reference(reference_bgr, pad_x, pad_y)
    feats = np.zeros((rows * cols, grid * grid, 3), np.float32)
    valid = np.zeros((rows * cols, grid * grid), np.float32)
    for r in range(rows):
        for c in range(cols):
            win, val = _reference_window(reference_bgr, r, c, cell_w, cell_h, pad_x, pad_y, padded)
            feats[r * cols + c], valid[r * cols + c] = _bin_means(win, val, grid)
    return ReferenceGridIndex(reference_bgr, rows, cols, cell_w, cell_h, pad_x, pad_y, grid, feats, valid, feats.copy())


def _affine_lstsq(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    a = np.hstack([src, np.ones((len(src), 1))])
    x, *_ = np.linalg.lstsq(a, dst, rcond=None)
    return x.T


def piece_canvases(frame_bgr: np.ndarray, piece: PieceRecord, index: ReferenceGridIndex) -> list[tuple[np.ndarray, np.ndarray]]:
    """Деталь, перенесённая по 4 углам в систему координат окна ячейки
    образца, для каждого из 4 поворотов: [(bgr, mask)] * 4."""
    contour = np.array([[p.x, p.y] for p in piece.contour_px], dtype=np.float64)
    corners = np.array([[p.x, p.y] for p in piece.corners_px], dtype=np.float64)
    canvas_w, canvas_h = index.canvas_size
    px, py = index.pad_x, index.pad_y
    cw, ch = index.cell_w, index.cell_h
    # Углы по ходу контура (против часовой) -> по часовой, как TL, TR, BR, BL ячейки.
    cw_corners = corners[[0, 3, 2, 1]]
    dst = np.array([[px, py], [px + cw, py], [px + cw, py + ch], [px, py + ch]], dtype=np.float64)

    x0, y0 = np.floor(contour.min(axis=0)).astype(int) - 4
    x1, y1 = np.ceil(contour.max(axis=0)).astype(int) + 4
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(frame_bgr.shape[1], x1), min(frame_bgr.shape[0], y1)
    crop = frame_bgr[y0:y1, x0:x1]
    mask = np.zeros(crop.shape[:2], np.uint8)
    cv2.fillPoly(mask, [np.round(contour - [x0, y0]).astype(np.int32)], 255)

    side_px = float(np.mean(np.linalg.norm(np.roll(cw_corners, -1, axis=0) - cw_corners, axis=1)))
    scale = 0.5 * (cw + ch) / max(side_px, 1e-6)
    if scale < 1.0:
        # Антиалиасинг перед уменьшением (warpAffine не усредняет по площади).
        crop = cv2.GaussianBlur(crop, (0, 0), 0.5 / scale)

    out = []
    for rot in range(4):
        src = np.roll(cw_corners, rot, axis=0) - [x0, y0]
        m = _affine_lstsq(src, dst)
        out.append(
            (
                cv2.warpAffine(crop, m, (canvas_w, canvas_h), flags=cv2.INTER_LINEAR),
                cv2.warpAffine(mask, m, (canvas_w, canvas_h), flags=cv2.INTER_LINEAR),
            )
        )
    return out


@dataclass
class PieceAppearance:
    piece_id: str
    batch_number: int
    center_px: np.ndarray            # центр детали в кадре — для поля освещения
    frame_shape: tuple[int, int]
    feat_bgr: np.ndarray             # (4, grid*grid, 3) — по поворотам
    weight: np.ndarray               # (4, grid*grid)
    straight: np.ndarray             # (4,) прямые стороны детали (в порядке piece.sides)
    gain: np.ndarray = field(default_factory=lambda: np.ones(3, np.float32))
    # Уменьшенный канвас поворота 0 (BGRA) — для отладочного рендера сборки.
    preview: np.ndarray | None = field(default=None, repr=False)


def piece_appearance(piece: PieceRecord, frame_bgr: np.ndarray, index: ReferenceGridIndex) -> PieceAppearance | None:
    if len(piece.corners_px) != 4 or len(piece.sides) != 4:
        return None
    feats, weights = [], []
    canvases = piece_canvases(frame_bgr, piece, index)
    for bgr, mask in canvases:
        f, w = _bin_means(bgr, mask.astype(np.float32) / 255.0, index.grid)
        feats.append(f)
        weights.append(w)
    preview_side = int(get_config().section("locate")["preview_canvas_px"])
    preview = cv2.resize(np.dstack(canvases[0]), (preview_side, preview_side), interpolation=cv2.INTER_AREA)
    contour = np.array([[p.x, p.y] for p in piece.contour_px], dtype=np.float64)
    return PieceAppearance(
        piece_id=piece.id,
        batch_number=piece.batch_number,
        center_px=contour.mean(axis=0),
        frame_shape=frame_bgr.shape[:2],
        feat_bgr=np.array(feats, np.float32),
        weight=np.array(weights, np.float32),
        straight=np.array([s.kind == SideKind.STRAIGHT for s in piece.sides]),
        preview=preview,
    )


def save_appearances(path, apps: list[PieceAppearance]) -> None:
    """Сохранить признаки деталей (например, одной партии) в .npz — чтобы
    пересчёт привязки не перечитывал кадры всех партий с диска."""
    if not apps:
        return
    np.savez_compressed(
        path,
        ids=np.array([a.piece_id for a in apps]),
        batch=np.array([a.batch_number for a in apps]),
        center=np.array([a.center_px for a in apps]),
        frame_shape=np.array([a.frame_shape for a in apps]),
        feat=np.array([a.feat_bgr for a in apps]),
        weight=np.array([a.weight for a in apps]),
        straight=np.array([a.straight for a in apps]),
        preview=np.array([a.preview for a in apps]),
    )


def load_appearances(path) -> list[PieceAppearance]:
    data = np.load(path)
    return [
        PieceAppearance(
            piece_id=str(data["ids"][i]),
            batch_number=int(data["batch"][i]),
            center_px=data["center"][i],
            frame_shape=tuple(int(v) for v in data["frame_shape"][i]),
            feat_bgr=data["feat"][i],
            weight=data["weight"][i],
            straight=data["straight"][i],
            preview=data["preview"][i],
        )
        for i in range(len(data["ids"]))
    ]

def _bgr_to_lab(feat_bgr: np.ndarray) -> np.ndarray:
    """(..., 3) BGR 0..255 float -> Lab в шкале uint8-OpenCV (L 0..255, a/b со сдвигом 0)."""
    shape = feat_bgr.shape
    flat = np.clip(feat_bgr.reshape(-1, 1, 3) / 255.0, 0.0, 1.0).astype(np.float32)
    lab = cv2.cvtColor(flat, cv2.COLOR_BGR2LAB).reshape(shape)
    lab[..., 0] *= 255.0 / 100.0
    return lab


def _border_feasibility(straight: np.ndarray, rows: int, cols: int) -> np.ndarray:
    """(4 поворота, rows*cols) — допустима ли ячейка при данном повороте по
    прямым сторонам детали: прямая сторона ровно там, где край пазла."""
    rr, cc = np.divmod(np.arange(rows * cols), cols)
    on_border = np.stack([rr == 0, cc == cols - 1, rr == rows - 1, cc == 0], axis=1)
    feasible = np.zeros((4, rows * cols), dtype=bool)
    for rot in range(4):
        canvas_straight = np.array([straight[canvas_side_index(i, rot)] for i in range(4)])
        feasible[rot] = (on_border == canvas_straight[None, :]).all(axis=1)
    return feasible


def appearance_costs(apps: list[PieceAppearance], index: ReferenceGridIndex, chunk: int = 256) -> np.ndarray:
    """(n_pieces, 4, rows*cols) float32 — взвешенная средняя квадратичная
    разность Lab-дескрипторов детали и окна ячейки, с INFEASIBLE_COST для
    поворотов/ячеек, запрещённых прямыми сторонами."""
    lc = get_config().section("locate")
    lw = np.sqrt(np.array(lc["lab_weights"], np.float32))
    ref = _bgr_to_lab(index.feat_bgr) * lw                  # (C, B, 3)
    rv = index.valid                                         # (C, B)
    ref_sq = (rv * (ref**2).sum(-1)).T                       # (B, C)
    ref_w = (ref * rv[..., None]).reshape(len(ref), -1).T    # (B*3, C)
    n_cells = len(ref)
    out = np.empty((len(apps), 4, n_cells), np.float32)
    for s in range(0, len(apps), chunk):
        part = apps[s : s + chunk]
        feat = np.concatenate([a.feat_bgr * (1.0 / a.gain)[None, None, :] for a in part])   # (m*4, B, 3)
        w = np.concatenate([a.weight for a in part])                                         # (m*4, B)
        p = _bgr_to_lab(feat) * lw
        t1 = (w * (p**2).sum(-1)) @ rv.T
        t2 = w @ ref_sq
        t3 = (p * w[..., None]).reshape(len(p), -1) @ ref_w
        den = w @ rv.T
        cost = (t1 + t2 - 2.0 * t3) / np.maximum(den, 1e-6)
        out[s : s + len(part)] = np.maximum(cost, 0.0).reshape(len(part), 4, n_cells)
    for k, a in enumerate(apps):
        out[k][~_border_feasibility(a.straight, index.rows, index.cols)] = INFEASIBLE_COST
    return out


def _margins(costs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Лучшая ячейка каждой детали и отрыв (вторая лучшая ячейка / лучшая)."""
    per_cell = costs.min(axis=1)
    part = np.partition(per_cell, 1, axis=1)
    return per_cell.argmin(axis=1), part[:, 1] / np.maximum(part[:, 0], 1e-6)


def correct_lighting(apps: list[PieceAppearance], costs: np.ndarray, index: ReferenceGridIndex) -> int:
    """Оценить для каждого кадра линейное (по координатам кадра) поле
    усиления BGR по уверенным деталям и записать его в app.gain. Возвращает
    число кадров, для которых поле оценено."""
    lc = get_config().section("locate")
    best, margin = _margins(costs)
    anchor_margin = float(lc["anchor_margin"])
    samples: dict[int, list[tuple[np.ndarray, np.ndarray]]] = {}
    for k, a in enumerate(apps):
        if margin[k] <= anchor_margin or costs[k].min() >= INFEASIBLE_COST / 10:
            continue
        cell = int(best[k])
        rot = int(costs[k, :, cell].argmin())
        w = a.weight[rot] * index.valid[cell]
        if w.sum() < 1e-3:
            continue
        piece_mean = (a.feat_bgr[rot] * w[:, None]).sum(0) / w.sum()
        ref_mean = (index.feat_bgr[cell] * w[:, None]).sum(0) / w.sum()
        pos = a.center_px / np.array(a.frame_shape[::-1], np.float64)
        samples.setdefault(a.batch_number, []).append((pos, piece_mean / np.maximum(ref_mean, 1.0)))

    min_samples = int(lc["lighting_min_samples"])
    all_gains = [g for data in samples.values() for _, g in data]
    fallback = np.median(np.array(all_gains), axis=0) if all_gains else np.ones(3)
    fields = {}
    for batch, data in samples.items():
        if len(data) >= min_samples:
            x = np.array([[1.0, p[0], p[1]] for p, _ in data])
            y = np.array([g for _, g in data])
            fields[batch], *_ = np.linalg.lstsq(x, y, rcond=None)
    for a in apps:
        coef = fields.get(a.batch_number)
        if coef is None:
            gain = fallback
        else:
            pos = a.center_px / np.array(a.frame_shape[::-1], np.float64)
            gain = np.array([1.0, pos[0], pos[1]]) @ coef
        a.gain = np.clip(gain, 0.4, 2.5).astype(np.float32)
    logger.info("locate: поле освещения оценено для %d кадров по %d уверенным деталям", len(fields), len(all_gains))
    return len(fields)


def _color_model_features(bgr: np.ndarray, pos: np.ndarray) -> np.ndarray:
    """Признаки цветовой модели образца: полином 2-й степени по BGR (гамма,
    насыщенность, перекрёстные искажения печати/камеры) + линейный градиент
    по положению на образце (неравномерная засветка фото коробки)."""
    x = bgr / 255.0
    b, g, r = x[:, 0], x[:, 1], x[:, 2]
    return np.stack([np.ones_like(b), b, g, r, b * b, g * g, r * r, b * g, b * r, g * r, pos[:, 0], pos[:, 1]], axis=1)


def correct_reference_colors(apps: list[PieceAppearance], costs: np.ndarray, index: ReferenceGridIndex) -> bool:
    """Подогнать цвета образца под то, как выглядят детали на фото.

    Картинка коробки почти никогда не совпадает по цвету с напечатанными
    деталями: другая камера/гамма/баланс белого, засветка фото коробки. По
    уверенным деталям (якорям) собираются пары «бин образца -> тот же бин
    детали» (десятки тысяч точек) и методом наименьших квадратов (с одним
    проходом отсева выбросов) подгоняется модель _color_model_features ->
    BGR детали. Модель применяется к признакам всех ячеек образца."""
    lc = get_config().section("locate")
    best, margin = _margins(costs)
    anchors = [k for k in range(len(apps)) if margin[k] > float(lc["anchor_margin"]) and costs[k].min() < INFEASIBLE_COST / 10]
    if len(anchors) < int(lc["reference_color_min_anchors"]):
        return False
    rows_x, rows_y, weights = [], [], []
    for k in anchors:
        a = apps[k]
        cell = int(best[k])
        rot = int(costs[k, :, cell].argmin())
        w = a.weight[rot] * index.valid[cell]
        sel = w > 0.5
        if not sel.any():
            continue
        r0, c0 = divmod(cell, index.cols)
        pos = np.tile([(c0 + 0.5) / index.cols, (r0 + 0.5) / index.rows], (int(sel.sum()), 1))
        rows_x.append(_color_model_features(index.feat_bgr_raw[cell][sel].astype(np.float64), pos))
        rows_y.append((a.feat_bgr[rot][sel] / a.gain[None, :]).astype(np.float64))
        weights.append(w[sel])
    x, y, w = np.concatenate(rows_x), np.concatenate(rows_y), np.concatenate(weights)
    # Регуляризация к тождественному отображению (а не к нулю): при
    # хорошем образце модель не должна ничего портить.
    identity = np.zeros((x.shape[1], 3))
    identity[1, 0] = identity[2, 1] = identity[3, 2] = 255.0
    ridge = 1e-3 * w.sum() * np.eye(x.shape[1])
    target = y - x @ identity
    coef = identity
    keep = np.ones(len(x), dtype=bool)
    for _ in range(2):
        xw = x[keep] * w[keep, None]
        delta = np.linalg.solve(x[keep].T @ xw + ridge, xw.T @ target[keep])
        coef = identity + delta
        resid = np.linalg.norm(y - x @ coef, axis=1)
        keep = resid < 3.0 * np.median(resid[keep]) + 1e-6
    n_cells, n_bins = index.feat_bgr_raw.shape[:2]
    rr, cc = np.divmod(np.arange(n_cells), index.cols)
    pos = np.repeat(np.stack([(cc + 0.5) / index.cols, (rr + 0.5) / index.rows], axis=1), n_bins, axis=0)
    mapped = _color_model_features(index.feat_bgr_raw.reshape(-1, 3).astype(np.float64), pos) @ coef
    index.feat_bgr = np.clip(mapped, 0, 255).reshape(index.feat_bgr_raw.shape).astype(np.float32)
    logger.info("locate: цветовая модель образца по %d якорям (%d точек)", len(anchors), int(keep.sum()))
    return True


@dataclass
class LocateResult:
    apps: list[PieceAppearance]
    pieces: list[PieceRecord]        # детали, для которых посчитаны стоимости (порядок = apps)
    costs: np.ndarray                # (n, 4, rows*cols)
    index: ReferenceGridIndex


def locate_pieces(pieces: list[PieceRecord], frames: dict[int, np.ndarray], index: ReferenceGridIndex) -> LocateResult:
    """Удобная обёртка: признаки деталей по кадрам + locate_appearances."""
    used, apps = [], []
    for p in pieces:
        if p.batch_number not in frames:
            continue
        app = piece_appearance(p, frames[p.batch_number], index)
        if app is not None:
            used.append(p)
            apps.append(app)
    return locate_appearances(used, apps, index)


def locate_appearances(pieces: list[PieceRecord], apps: list[PieceAppearance], index: ReferenceGridIndex) -> LocateResult:
    """Посчитать стоимости всех (деталь, ячейка, поворот), скорректировать
    освещение, заполнить piece.location_candidates — топ-N ячеек
    (row, col, rotation_deg, confidence). pieces и apps — в одном порядке
    (признаки считаются piece_appearance по мере обработки кадров, чтобы
    не держать в памяти все кадры партий)."""
    lc = get_config().section("locate")
    if not apps:
        return LocateResult([], [], np.zeros((0, 4, index.rows * index.cols), np.float32), index)

    index.feat_bgr = index.feat_bgr_raw.copy()
    costs = appearance_costs(apps, index)
    # Цвет образца и освещение кадров оцениваются по якорям поочерёдно:
    # каждое уточнение даёт больше уверенных деталей для следующего.
    for _ in range(int(lc["color_rounds"])):
        if bool(lc["reference_color_correction"]) and correct_reference_colors(apps, costs, index):
            costs = appearance_costs(apps, index)
        if bool(lc["lighting_correction"]):
            correct_lighting(apps, costs, index)
            costs = appearance_costs(apps, index)

    _fill_candidates(pieces, costs, index, int(lc["top_k_result_cells"]))
    return LocateResult(apps, list(pieces), costs, index)


def _fill_candidates(pieces: list[PieceRecord], costs: np.ndarray, index: ReferenceGridIndex, top_n: int) -> None:
    per_cell = costs.min(axis=1)
    best_rot = costs.argmin(axis=1)
    _, margin = _margins(costs)
    anchor_margin = float(get_config().section("locate")["anchor_margin"])
    anchors = margin > anchor_margin
    # Масштаб «шума» стоимости — типичная стоимость верного совпадения
    # (медиана лучшей стоимости уверенных деталей): уверенность кандидата —
    # softmax по стоимостям с этой температурой.
    tau = float(np.median(per_cell[anchors].min(axis=1))) if anchors.sum() >= 10 else float(np.median(per_cell.min(axis=1)))
    tau = max(tau, 1e-3)
    for k, p in enumerate(pieces):
        order = np.argsort(per_cell[k])[: max(top_n, 10)]
        c = per_cell[k, order]
        feasible = c < INFEASIBLE_COST / 10
        if not feasible.any():
            p.location_candidates = []
            continue
        weights = np.exp(-(c - c[0]) / tau) * feasible
        conf = weights / weights.sum()
        p.location_candidates = [
            (int(cell // index.cols), int(cell % index.cols), int(best_rot[k, cell]) * 90, float(cf))
            for cell, cf, ok in zip(order[:top_n], conf[:top_n], feasible[:top_n])
            if ok
        ]


def render_layout(located: LocateResult, placements, cell_px: int | None = None) -> np.ndarray:
    """Картинка «собранного пазла»: каждая деталь (по превью канваса,
    повёрнутая на найденный поворот) в своей ячейке. Для визуальной проверки
    сборки: ошибка видна как «чужой» кусок картинки или разрыв линии."""
    index = located.index
    side = int(cell_px or get_config().section("locate")["preview_canvas_px"])
    pad = int(round(side * index.pad_x / index.canvas_size[0]))
    cell = side - 2 * pad
    h, w = index.rows * cell + 2 * pad, index.cols * cell + 2 * pad
    out = np.zeros((h, w, 3), np.float32)
    for pl in placements:
        prev = located.apps[pl.piece_index].preview
        if prev is None:
            continue
        # Канвас поворота r — это канвас поворота 0, повёрнутый на r*90° по часовой.
        img = np.ascontiguousarray(np.rot90(prev, k=-pl.rotation))
        img = cv2.resize(img, (side, side), interpolation=cv2.INTER_AREA).astype(np.float32)
        alpha = img[:, :, 3:4] / 255.0
        y0, x0 = pl.row * cell, pl.col * cell
        region = out[y0 : y0 + side, x0 : x0 + side]
        region[:] = region * (1 - alpha) + img[:, :, :3] * alpha
    return np.clip(out, 0, 255).astype(np.uint8)
