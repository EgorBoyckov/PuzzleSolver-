import json

import numpy as np
import pytest

from app.pipeline.describe import describe_piece
from app.pipeline.preprocess import preprocess_frame
from app.pipeline.schemas import SideKind
from app.pipeline.segment import segment_pieces
from app.synth.dataset import SyntheticDatasetConfig, generate_dataset

from .pipeline_test_utils import match_pieces_to_ground_truth


@pytest.fixture(scope="module")
def described_batch(tmp_path_factory):
    out_dir = tmp_path_factory.mktemp("describe_synth")
    config = SyntheticDatasetConfig(
        total_pieces=150,
        out_dir=out_dir,
        batch_size_min=75,
        batch_size_max=75,
        table_px_per_mm=6.0,
        seed=21,
    )
    meta = generate_dataset(config)
    batch = meta["batches"][0]
    with (out_dir / batch["annotations"]).open(encoding="utf-8") as f:
        gt_batch = json.load(f)

    photo = out_dir / batch["image"]
    frame, rectified, valid_mask = preprocess_frame(photo, "puz", batch["batch_number"])
    assert frame.accepted

    pieces = segment_pieces(
        rectified,
        frame.mm_per_pixel,
        "puz",
        batch["batch_number"],
        marker_bbox_px=frame.marker_bbox_px,
        valid_mask=valid_mask,
    )
    for p in pieces:
        describe_piece(p, rectified, frame.mm_per_pixel)

    tolerance_px = 0.5 * meta["piece_size_mm"] / frame.mm_per_pixel
    matches = match_pieces_to_ground_truth(
        pieces, gt_batch, frame.marker_bbox_px, meta["table_px_per_mm"], 1.0 / frame.mm_per_pixel, tolerance_px
    )
    return meta, gt_batch, pieces, matches


def test_kind_classification_at_least_99_9_percent(described_batch):
    _meta, gt_batch, _pieces, matches = described_batch
    gt_kind_by_id = {gt["id"]: gt["kind"] for gt in gt_batch["pieces"]}

    correct = 0
    for gt_id, (piece, _dist) in matches.items():
        detected_kind = piece.kind.value if piece.kind else None
        if detected_kind == gt_kind_by_id[gt_id]:
            correct += 1

    accuracy = correct / len(matches)
    assert accuracy >= 0.999, f"kind accuracy={accuracy:.4f} ({correct}/{len(matches)})"


def test_straight_side_count_matches_kind_for_every_piece(described_batch):
    _meta, _gt_batch, pieces, _matches = described_batch
    kind_to_straight = {"corner": 2, "edge": 1, "center": 0}
    for p in pieces:
        if p.kind is None or not p.sides:
            continue
        straight = sum(1 for s in p.sides if s.kind == SideKind.STRAIGHT)
        assert straight == kind_to_straight[p.kind.value], p.id


def test_each_described_piece_has_four_sides_with_curves_and_color(described_batch):
    _meta, _gt_batch, pieces, _matches = described_batch
    described = [p for p in pieces if p.sides]
    assert len(described) > 0
    for p in described:
        assert len(p.sides) == 4
        for side in p.sides:
            assert len(side.curve) >= 32
            assert len(side.color_strip_lab) == 32
            # cv2.cvtColor(..., COLOR_BGR2LAB) на 8-битных изображениях кодирует
            # L/a/b в диапазоне [0,255] (не "учебные" L:[0,100], a/b:[-128,127]).
            for L, a, b in side.color_strip_lab:
                assert 0.0 <= L <= 255.0
                assert 0.0 <= a <= 255.0
                assert 0.0 <= b <= 255.0


def test_embeddings_present_and_finite(described_batch):
    _meta, _gt_batch, pieces, _matches = described_batch
    for p in pieces:
        if p.kind is None:
            continue
        assert p.embedding is not None
        arr = np.array(p.embedding)
        assert np.all(np.isfinite(arr))


def test_corners_exact_on_generator_geometry():
    """Углы по точной геометрии генератора (без рендеринга): все 4 угла
    каждой детали должны совпасть с узлами сетки — включая детали, у
    которых обе стороны угла с выступами (там прежний поиск по остроте
    вершин выпуклой оболочки выбирал вершины головок замков)."""
    from app.core.config import get_config
    from app.pipeline.describe import _find_corners
    from app.synth.piece_shapes import DEFAULT_TAB_PARAMS, build_puzzle_grid

    px_per_mm = 6.0
    grid = build_puzzle_grid(6, 6, 25.0, np.random.default_rng(3), dict(DEFAULT_TAB_PARAMS))
    ds = get_config().section("describe")
    both_tabs_seen = 0
    for piece in grid:
        contour = piece.contour * px_per_mm
        # Плотная передискретизация (прямые стороны в геометрии — 2 точки).
        closed = np.vstack([contour, contour[:1]])
        seg = np.linalg.norm(np.diff(closed, axis=0), axis=1)
        cum = np.concatenate([[0.0], np.cumsum(seg)])
        t = np.arange(0.0, cum[-1], 1.0)
        dense = np.stack([np.interp(t, cum, closed[:, 0]), np.interp(t, cum, closed[:, 1])], axis=1)
        found = _find_corners(dense, ds)
        assert found is not None
        _, corners = found
        r, c, s = piece.row, piece.col, 25.0 * px_per_mm
        truth = np.array([[c * s, r * s], [(c + 1) * s, r * s], [(c + 1) * s, (r + 1) * s], [c * s, (r + 1) * s]])
        err = np.linalg.norm(corners[:, None, :] - truth[None, :, :], axis=2).min(axis=1)
        assert err.max() < 0.5 * px_per_mm, (piece.row, piece.col, err)
        kinds = [side.kind for side in piece.sides]
        both_tabs_seen += any(kinds[i] == "tab" and kinds[(i + 1) % 4] == "tab" for i in range(4))
    assert both_tabs_seen >= 5
