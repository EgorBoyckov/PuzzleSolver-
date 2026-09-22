"""Генератор синтетического датасета: раскладка партий деталей на столе.

Режет процедурную (или переданную) картинку "коробки" на детали классической
формы (piece_shapes.py), раскладывает их партиями по 60-120 шт. на однотонном
фоне со случайным поворотом, положением (без наложений), освещением и шумом,
рядом кладёт ArUco-маркер — и сохраняет точную разметку (contour/rotation/id)
для тестов всех последующих этапов конвейера.
"""
from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from app.synth.box_art import generate_box_art
from app.synth.piece_shapes import DEFAULT_TAB_PARAMS, PieceGeometry, PuzzleGrid, build_puzzle_grid

logger = logging.getLogger(__name__)

ARUCO_DICT_NAMES = {
    "DICT_4X4_50": cv2.aruco.DICT_4X4_50,
    "DICT_5X5_50": cv2.aruco.DICT_5X5_50,
    "DICT_6X6_50": cv2.aruco.DICT_6X6_50,
}


@dataclass
class SyntheticDatasetConfig:
    total_pieces: int
    out_dir: Path
    piece_size_mm: float = 25.0
    aspect_ratio: tuple[float, float] = (4.0, 3.0)
    source_image_path: Path | None = None

    batch_size_min: int = 60
    batch_size_max: int = 120

    min_gap_mm: float = 10.0
    max_gap_mm: float = 15.0

    box_px_per_mm: float = 3.0
    box_max_canvas_px: int = 4000
    table_px_per_mm: float = 6.0
    table_max_canvas_px: int = 6000
    table_packing_factor: float = 1.85
    table_margin_mm: float = 40.0

    aruco_dictionary: str = "DICT_4X4_50"
    aruco_marker_id: int = 0
    aruco_marker_size_mm: float = 100.0

    background_color_bgr: tuple[int, int, int] = (40, 34, 30)
    lighting_gradient_strength: float = 0.25
    noise_sigma: float = 4.0
    jpeg_quality: int = 92

    tab_params: dict = field(default_factory=lambda: dict(DEFAULT_TAB_PARAMS))
    seed: int = 42

    def grid_dims(self) -> tuple[int, int]:
        aw, ah = self.aspect_ratio
        cols = max(1, round(math.sqrt(self.total_pieces * aw / ah)))
        rows = max(1, round(self.total_pieces / cols))
        return rows, cols


def _rasterize_mask(local_contour_px: np.ndarray, shape_hw: tuple[int, int]) -> np.ndarray:
    mask = np.zeros(shape_hw, dtype=np.uint8)
    pts = np.round(local_contour_px).astype(np.int32)
    cv2.fillPoly(mask, [pts], 255)
    return mask


def _build_piece_patch(
    piece: PieceGeometry,
    box_art: np.ndarray,
    box_px_per_mm: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Вырезать RGBA-патч детали (rotation=0) из картинки коробки.

    Возвращает (patch_bgr, mask_uint8) в разрешении box_px_per_mm.
    """
    xmin, ymin, xmax, ymax = piece.bbox
    x0 = max(0, int(math.floor(xmin * box_px_per_mm)))
    y0 = max(0, int(math.floor(ymin * box_px_per_mm)))
    x1 = min(box_art.shape[1], int(math.ceil(xmax * box_px_per_mm)))
    y1 = min(box_art.shape[0], int(math.ceil(ymax * box_px_per_mm)))
    x1, y1 = max(x1, x0 + 1), max(y1, y0 + 1)
    patch = box_art[y0:y1, x0:x1].copy()

    local_px = (piece.local_contour()) * box_px_per_mm
    mask = _rasterize_mask(local_px, (patch.shape[0], patch.shape[1]))
    return patch, mask


def _rotate_rgba(rgba: np.ndarray, contour_px: np.ndarray, angle_deg: float) -> tuple[np.ndarray, np.ndarray]:
    """Повернуть RGBA-патч на angle_deg без обрезки краёв; вернуть новый патч
    и контур, пересчитанный той же аффинной матрицей (пиксельно точное совпадение)."""
    h0, w0 = rgba.shape[:2]
    center = (w0 / 2.0, h0 / 2.0)
    M = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
    cos, sin = abs(M[0, 0]), abs(M[0, 1])
    new_w = max(1, int(math.ceil(h0 * sin + w0 * cos)))
    new_h = max(1, int(math.ceil(h0 * cos + w0 * sin)))
    M[0, 2] += new_w / 2.0 - center[0]
    M[1, 2] += new_h / 2.0 - center[1]

    rotated = cv2.warpAffine(
        rgba, M, (new_w, new_h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0)
    )
    rotated_contour = contour_px @ M[:, :2].T + M[:, 2]
    return rotated, rotated_contour


def _alpha_composite(canvas: np.ndarray, patch_rgba: np.ndarray, top_left: tuple[int, int]) -> None:
    x0, y0 = top_left
    h, w = patch_rgba.shape[:2]
    cx0, cy0 = max(0, x0), max(0, y0)
    cx1, cy1 = min(canvas.shape[1], x0 + w), min(canvas.shape[0], y0 + h)
    if cx1 <= cx0 or cy1 <= cy0:
        return
    px0, py0 = cx0 - x0, cy0 - y0
    px1, py1 = px0 + (cx1 - cx0), py0 + (cy1 - cy0)

    region = canvas[cy0:cy1, cx0:cx1].astype(np.float32)
    patch = patch_rgba[py0:py1, px0:px1].astype(np.float32)
    alpha = (patch[:, :, 3:4] / 255.0) if patch.shape[2] == 4 else np.ones((*patch.shape[:2], 1), np.float32)
    blended = region * (1 - alpha) + patch[:, :, :3] * alpha
    canvas[cy0:cy1, cx0:cx1] = np.clip(blended, 0, 255).astype(np.uint8)


def _apply_lighting_and_noise(
    canvas: np.ndarray, rng: np.random.Generator, gradient_strength: float, noise_sigma: float
) -> np.ndarray:
    h, w = canvas.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    angle = rng.uniform(0, 2 * math.pi)
    gx, gy = math.cos(angle), math.sin(angle)
    gradient = (xx / w) * gx + (yy / h) * gy
    gradient = (gradient - gradient.min()) / (gradient.max() - gradient.min() + 1e-6)
    factor = 1.0 + gradient_strength * (gradient - 0.5)
    out = canvas.astype(np.float32) * factor[:, :, None]
    noise = rng.normal(0, noise_sigma, size=canvas.shape).astype(np.float32)
    out = np.clip(out + noise, 0, 255).astype(np.uint8)
    return out


def _place_aruco_marker(
    canvas: np.ndarray, config: SyntheticDatasetConfig, rng: np.random.Generator
) -> dict:
    dict_id = ARUCO_DICT_NAMES[config.aruco_dictionary]
    aruco_dict = cv2.aruco.getPredefinedDictionary(dict_id)
    marker_side_px = int(round(config.aruco_marker_size_mm * config.table_px_per_mm))
    marker_img = cv2.aruco.generateImageMarker(aruco_dict, config.aruco_marker_id, marker_side_px)

    border_px = int(round(marker_side_px * 0.25))
    sheet_side = marker_side_px + 2 * border_px
    sheet = np.full((sheet_side, sheet_side, 3), 255, dtype=np.uint8)
    marker_bgr = cv2.cvtColor(marker_img, cv2.COLOR_GRAY2BGR)
    sheet[border_px : border_px + marker_side_px, border_px : border_px + marker_side_px] = marker_bgr

    margin_px = int(round(config.table_margin_mm * config.table_px_per_mm * 0.5))
    x0, y0 = margin_px, margin_px
    canvas[y0 : y0 + sheet_side, x0 : x0 + sheet_side] = sheet

    corners_px = [
        [x0 + border_px, y0 + border_px],
        [x0 + border_px + marker_side_px, y0 + border_px],
        [x0 + border_px + marker_side_px, y0 + border_px + marker_side_px],
        [x0 + border_px, y0 + border_px + marker_side_px],
    ]
    exclusion_zone = (x0, y0, x0 + sheet_side, y0 + sheet_side)
    return {
        "marker_id": config.aruco_marker_id,
        "dictionary": config.aruco_dictionary,
        "marker_size_mm": config.aruco_marker_size_mm,
        "corners_px": corners_px,
    }, exclusion_zone


def _sample_placements(
    n: int,
    radii_px: list[float],
    canvas_w: int,
    canvas_h: int,
    margin_px: int,
    gap_px: float,
    exclusion_zone: tuple[int, int, int, int],
    rng: np.random.Generator,
    max_attempts: int = 800,
) -> list[tuple[float, float]]:
    placed: list[tuple[float, float]] = []
    placed_r: list[float] = []
    ex0, ey0, ex1, ey1 = exclusion_zone

    def collides(x: float, y: float, r: float) -> bool:
        if (ex0 - gap_px) < x < (ex1 + gap_px) and (ey0 - gap_px) < y < (ey1 + gap_px):
            return True
        if placed:
            arr = np.array(placed)
            rarr = np.array(placed_r)
            dist = np.hypot(arr[:, 0] - x, arr[:, 1] - y)
            if np.any(dist < (rarr + r + gap_px)):
                return True
        return False

    for i in range(n):
        r = radii_px[i]
        ok = False
        for _ in range(max_attempts):
            x = rng.uniform(margin_px + r, canvas_w - margin_px - r)
            y = rng.uniform(margin_px + r, canvas_h - margin_px - r)
            if not collides(x, y, r):
                placed.append((x, y))
                placed_r.append(r)
                ok = True
                break
        if not ok:
            # Резервный вариант: детерминированный скан по мелкой сетке кандидатов
            # (шаг меньше диаметра детали, чтобы не "перепрыгивать" узкие свободные
            # промежутки между уже размещёнными деталями) с проверкой коллизий —
            # гарантированно валидная (без наложений) позиция.
            step = max(r * 0.5 + gap_px * 0.5, 8.0)
            found = False
            y = margin_px + r
            while y <= canvas_h - margin_px - r and not found:
                x = margin_px + r
                while x <= canvas_w - margin_px - r and not found:
                    if not collides(x, y, r):
                        placed.append((x, y))
                        placed_r.append(r)
                        found = True
                    x += step
                y += step
            if not found:
                raise RuntimeError(
                    f"Не удалось разместить деталь {i}/{n} на холсте {canvas_w}x{canvas_h}px "
                    "без наложений — увеличьте table_packing_factor или table_max_canvas_px"
                )
            logger.warning("piece %d: rejection sampling failed, used deterministic grid scan fallback", i)
    return placed


def generate_catalog(config: SyntheticDatasetConfig, rng: np.random.Generator) -> tuple[PuzzleGrid, np.ndarray, dict]:
    """Построить сетку деталей и картинку коробки (реальную или процедурную)."""
    rows, cols = config.grid_dims()
    grid = build_puzzle_grid(rows, cols, config.piece_size_mm, rng, config.tab_params)

    box_px_per_mm = min(config.box_px_per_mm, config.box_max_canvas_px / max(grid.width_mm, grid.height_mm))
    box_w_px = max(1, int(round(grid.width_mm * box_px_per_mm)))
    box_h_px = max(1, int(round(grid.height_mm * box_px_per_mm)))

    if config.source_image_path is not None:
        src = cv2.imread(str(config.source_image_path))
        if src is None:
            raise FileNotFoundError(f"Не удалось прочитать source_image_path={config.source_image_path}")
        box_art = cv2.resize(src, (box_w_px, box_h_px), interpolation=cv2.INTER_AREA)
    else:
        box_art = generate_box_art(box_w_px, box_h_px, rng)

    meta = {
        "rows": rows,
        "cols": cols,
        "total_pieces": rows * cols,
        "piece_size_mm": config.piece_size_mm,
        "box_px_per_mm": box_px_per_mm,
        "box_width_px": box_w_px,
        "box_height_px": box_h_px,
    }
    return grid, box_art, meta


def _make_batches(piece_ids: list[tuple[int, int]], config: SyntheticDatasetConfig, rng: np.random.Generator) -> list[list[tuple[int, int]]]:
    """Разбить детали на партии, каждая строго в [batch_size_min, batch_size_max]
    (кроме случая, когда деталей меньше batch_size_min — тогда одна партия из всех)."""
    order = list(piece_ids)
    rng.shuffle(order)
    n = len(order)
    min_b, max_b = config.batch_size_min, config.batch_size_max

    if n <= max_b:
        return [order]

    num_batches = max(1, math.ceil(n / max_b))
    while num_batches > 1 and n / num_batches < min_b:
        num_batches -= 1

    base, extra = divmod(n, num_batches)
    sizes = [base + (1 if i < extra else 0) for i in range(num_batches)]
    rng.shuffle(sizes)

    batches: list[list[tuple[int, int]]] = []
    i = 0
    for size in sizes:
        batches.append(order[i : i + size])
        i += size
    return batches


def render_batch(
    batch_number: int,
    piece_keys: list[tuple[int, int]],
    grid: PuzzleGrid,
    box_art: np.ndarray,
    box_px_per_mm: float,
    config: SyntheticDatasetConfig,
    rng: np.random.Generator,
) -> tuple[np.ndarray, dict]:
    n = len(piece_keys)
    avg_diam_px = config.piece_size_mm * 1.5 * config.table_px_per_mm
    gap_px = config.min_gap_mm * config.table_px_per_mm
    canvas_side = int(
        min(
            config.table_max_canvas_px,
            max(600, math.sqrt(n) * (avg_diam_px + gap_px) * config.table_packing_factor)
            + 2 * config.table_margin_mm * config.table_px_per_mm,
        )
    )
    canvas = np.zeros((canvas_side, canvas_side, 3), dtype=np.uint8)
    canvas[:, :] = config.background_color_bgr

    marker_info, exclusion_zone = _place_aruco_marker(canvas, config, rng)
    margin_px = int(round(config.table_margin_mm * config.table_px_per_mm))

    rotated_patches = []
    radii = []
    for row, col in piece_keys:
        piece = grid.piece(row, col)
        patch_bgr, mask = _build_piece_patch(piece, box_art, box_px_per_mm)
        rgba = np.dstack([patch_bgr, mask])

        s = config.table_px_per_mm / box_px_per_mm
        w0 = max(1, int(round(rgba.shape[1] * s)))
        h0 = max(1, int(round(rgba.shape[0] * s)))
        rgba_scaled = cv2.resize(rgba, (w0, h0), interpolation=cv2.INTER_AREA)
        scaled_contour = (piece.local_contour()) * config.table_px_per_mm

        angle_deg = float(rng.uniform(0, 360))
        rotated_rgba, rotated_contour = _rotate_rgba(rgba_scaled, scaled_contour, angle_deg)

        center = np.array([rotated_rgba.shape[1] / 2.0, rotated_rgba.shape[0] / 2.0])
        radius = float(np.max(np.linalg.norm(rotated_contour - center, axis=1))) if len(rotated_contour) else max(w0, h0) / 2
        radii.append(radius)
        rotated_patches.append((row, col, rotated_rgba, rotated_contour, angle_deg))

    centers = _sample_placements(
        n, radii, canvas_side, canvas_side, margin_px, gap_px, exclusion_zone, rng
    )

    annotations = []
    for (row, col, rotated_rgba, rotated_contour, angle_deg), (cx, cy) in zip(rotated_patches, centers):
        piece = grid.piece(row, col)
        h, w = rotated_rgba.shape[:2]
        top_left = (int(round(cx - w / 2.0)), int(round(cy - h / 2.0)))
        _alpha_composite(canvas, rotated_rgba, top_left)

        final_contour = rotated_contour + np.array(top_left)
        xmin, ymin = final_contour.min(axis=0)
        xmax, ymax = final_contour.max(axis=0)

        batch_index = piece_keys.index((row, col)) + 1
        catalog_id = f"B{batch_number:02d}-{batch_index:03d}"

        annotations.append(
            {
                "id": catalog_id,
                "batch_number": batch_number,
                "number_in_batch": batch_index,
                "grid_row": row,
                "grid_col": col,
                "kind": piece.kind,
                "rotation_deg": angle_deg,
                "center_px": [float(cx), float(cy)],
                "contour_px": final_contour.round(2).tolist(),
                "bbox_px": [float(xmin), float(ymin), float(xmax), float(ymax)],
            }
        )

    canvas = _apply_lighting_and_noise(canvas, rng, config.lighting_gradient_strength, config.noise_sigma)

    batch_meta = {
        "batch_number": batch_number,
        "num_pieces": n,
        "canvas_px": canvas_side,
        "mm_per_pixel": 1.0 / config.table_px_per_mm,
        "marker": marker_info,
        "pieces": annotations,
    }
    return canvas, batch_meta


def generate_dataset(config: SyntheticDatasetConfig) -> dict:
    """Сгенерировать полный датасет: каталог + все партии + файлы на диск.

    Возвращает сводный dataset_meta (тот же, что сохраняется в JSON).
    """
    rng = np.random.default_rng(config.seed)
    out_dir = Path(config.out_dir)
    (out_dir / "catalog").mkdir(parents=True, exist_ok=True)
    (out_dir / "batches").mkdir(parents=True, exist_ok=True)

    grid, box_art, catalog_meta = generate_catalog(config, rng)
    box_path = out_dir / "catalog" / "box.jpg"
    cv2.imwrite(str(box_path), box_art, [cv2.IMWRITE_JPEG_QUALITY, config.jpeg_quality])

    piece_keys = list(grid.pieces.keys())
    batches = _make_batches(piece_keys, config, rng)

    batch_summaries = []
    for idx, keys in enumerate(batches, start=1):
        canvas, batch_meta = render_batch(
            idx, keys, grid, box_art, catalog_meta["box_px_per_mm"], config, rng
        )
        img_name = f"batch_{idx:02d}.jpg"
        json_name = f"batch_{idx:02d}.json"
        cv2.imwrite(str(out_dir / "batches" / img_name), canvas, [cv2.IMWRITE_JPEG_QUALITY, config.jpeg_quality])
        with (out_dir / "batches" / json_name).open("w", encoding="utf-8") as f:
            json.dump(batch_meta, f, ensure_ascii=False, indent=2)
        batch_summaries.append(
            {"batch_number": idx, "num_pieces": batch_meta["num_pieces"], "image": f"batches/{img_name}", "annotations": f"batches/{json_name}"}
        )

    catalog_pieces = [
        {"grid_row": r, "grid_col": c, "kind": grid.piece(r, c).kind}
        for (r, c) in piece_keys
    ]
    with (out_dir / "catalog" / "grid.json").open("w", encoding="utf-8") as f:
        json.dump({**catalog_meta, "pieces": catalog_pieces}, f, ensure_ascii=False, indent=2)

    dataset_meta = {
        "total_pieces_target": config.total_pieces,
        "total_pieces_actual": catalog_meta["total_pieces"],
        "rows": catalog_meta["rows"],
        "cols": catalog_meta["cols"],
        "piece_size_mm": config.piece_size_mm,
        "box_image": "catalog/box.jpg",
        "box_px_per_mm": catalog_meta["box_px_per_mm"],
        "table_px_per_mm": config.table_px_per_mm,
        "num_batches": len(batches),
        "batches": batch_summaries,
        "seed": config.seed,
    }
    with (out_dir / "dataset_meta.json").open("w", encoding="utf-8") as f:
        json.dump(dataset_meta, f, ensure_ascii=False, indent=2)

    logger.info(
        "synthetic dataset: %d pieces (%dx%d grid), %d batches -> %s",
        catalog_meta["total_pieces"],
        catalog_meta["rows"],
        catalog_meta["cols"],
        len(batches),
        out_dir,
    )
    return dataset_meta
