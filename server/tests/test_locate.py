import json
import math

import cv2
import numpy as np
import pytest

from app.pipeline.describe import describe_piece
from app.pipeline.locate import INFEASIBLE_COST, build_reference_index, canvas_side_index, locate_pieces
from app.pipeline.match import SideShapeBank, score_side_pair
from app.pipeline.preprocess import preprocess_frame
from app.pipeline.schemas import SideKind
from app.pipeline.segment import segment_pieces
from app.pipeline.solve import solve_layout
from app.synth.dataset import SyntheticDatasetConfig, generate_dataset

from .pipeline_test_utils import match_pieces_to_ground_truth, rotation_matches_gt


@pytest.fixture(scope="module")
def located_dataset(tmp_path_factory):
    out_dir = tmp_path_factory.mktemp("locate_synth")
    config = SyntheticDatasetConfig(
        total_pieces=150,
        out_dir=out_dir,
        batch_size_min=75,
        batch_size_max=75,
        table_px_per_mm=6.0,
        seed=21,
    )
    meta = generate_dataset(config)
    reference_bgr = cv2.imread(str(out_dir / "catalog" / "box.jpg"))
    ref_index = build_reference_index(reference_bgr, meta["rows"], meta["cols"])

    all_pieces = []
    frames = {}
    truth = {}  # piece.id -> gt-аннотация

    for batch_summary in meta["batches"]:
        batch_number = batch_summary["batch_number"]
        with (out_dir / batch_summary["annotations"]).open(encoding="utf-8") as f:
            gt_batch = json.load(f)
        frame, rectified, valid_mask = preprocess_frame(out_dir / batch_summary["image"], "puz", batch_number)
        assert frame.accepted

        pieces = segment_pieces(
            rectified, frame.mm_per_pixel, "puz", batch_number,
            marker_bbox_px=frame.marker_bbox_px, valid_mask=valid_mask,
        )
        for p in pieces:
            describe_piece(p, rectified, frame.mm_per_pixel)
        frames[batch_number] = rectified
        all_pieces.extend(pieces)

        tolerance_px = 0.5 * meta["piece_size_mm"] / frame.mm_per_pixel
        matches = match_pieces_to_ground_truth(
            pieces, gt_batch, frame.marker_bbox_px, meta["table_px_per_mm"], 1.0 / frame.mm_per_pixel, tolerance_px
        )
        gt_by_id = {gt["id"]: gt for gt in gt_batch["pieces"]}
        for gt_id, (piece, _dist) in matches.items():
            truth[piece.id] = gt_by_id[gt_id]

    located = locate_pieces(all_pieces, frames, ref_index)
    layout = solve_layout(located)
    return meta, ref_index, located, layout, truth


def _rotation_matches_gt(piece, rotation_deg: int, gt: dict) -> bool:
    return rotation_matches_gt(piece, rotation_deg, gt["rotation_deg"])


def test_top3_location_accuracy(located_dataset):
    _meta, _ref_index, located, _layout, truth = located_dataset
    total = top1 = top3 = 0
    for piece in located.pieces:
        if piece.id not in truth or not piece.location_candidates:
            continue
        total += 1
        true_rc = (truth[piece.id]["grid_row"], truth[piece.id]["grid_col"])
        cands = [(r, c) for r, c, _rot, _conf in piece.location_candidates]
        top1 += cands[0] == true_rc
        top3 += true_rc in cands
    assert total >= 140
    assert top1 / total >= 0.80, f"top1={top1 / total:.2%}"
    assert top3 / total >= 0.90, f"top3={top3 / total:.2%}"


def test_location_candidates_well_formed(located_dataset):
    _meta, ref_index, located, _layout, _truth = located_dataset
    for p in located.pieces:
        assert 1 <= len(p.location_candidates) <= 3
        for row, col, rot, conf in p.location_candidates:
            assert 0 <= row < ref_index.rows
            assert 0 <= col < ref_index.cols
            assert rot in (0, 90, 180, 270)
            assert 0.0 <= conf <= 1.0


def test_straight_sides_forbid_interior_cells(located_dataset):
    _meta, ref_index, located, _layout, _truth = located_dataset
    for k, p in enumerate(located.pieces):
        n_straight = sum(s.kind == SideKind.STRAIGHT for s in p.sides)
        feasible_cells = np.where(located.costs[k].min(axis=0) < INFEASIBLE_COST / 10)[0]
        rows, cols = np.divmod(feasible_cells, ref_index.cols)
        on_border = (rows == 0) | (rows == ref_index.rows - 1) | (cols == 0) | (cols == ref_index.cols - 1)
        if n_straight > 0:
            assert on_border.all(), p.id
        else:
            assert not on_border.any(), p.id


def test_global_layout_places_pieces_correctly(located_dataset):
    meta, ref_index, located, layout, truth = located_dataset
    placed_pieces = [located.pieces[pl.piece_index].id for pl in layout.placements]
    assert len(set(placed_pieces)) == len(placed_pieces), "деталь поставлена дважды"
    assert len({(pl.row, pl.col) for pl in layout.placements}) == len(layout.placements)

    total = cell_ok = rot_ok = 0
    for pl in layout.placements:
        piece = located.pieces[pl.piece_index]
        gt = truth.get(piece.id)
        if gt is None:
            continue
        total += 1
        if (gt["grid_row"], gt["grid_col"]) == (pl.row, pl.col):
            cell_ok += 1
            rot_ok += _rotation_matches_gt(piece, pl.rotation * 90, gt)
        assert piece.placement is not None and piece.placement[:2] == (pl.row, pl.col)
    assert total >= 140
    assert cell_ok / total >= 0.97, f"верная ячейка у {cell_ok / total:.2%}"
    assert rot_ok / total >= 0.97, f"верные ячейка и поворот у {rot_ok / total:.2%}"


def test_true_neighbor_side_matches_best(located_dataset):
    """Форма общей линии разреза: у истинного соседа расстояние меньше,
    чем у подавляющего большинства случайных впадин/выступов."""
    _meta, _ref_index, located, _layout, truth = located_dataset
    by_cell = {}
    for pl in _layout_true_cells(located, truth):
        by_cell[(pl[1], pl[2])] = pl
    bank = SideShapeBank(located.pieces)
    rng = np.random.default_rng(0)
    worse_fraction = []
    for (row, col), (k, _r, _c, rot) in by_cell.items():
        right = by_cell.get((row, col + 1))
        if right is None:
            continue
        a, b = canvas_side_index(1, rot), canvas_side_index(3, right[3])
        if not bank.complementary(np.array([k]), np.array([a]), np.array([right[0]]), np.array([b]))[0]:
            continue
        true_d = bank.distance(np.array([k]), np.array([a]), np.array([right[0]]), np.array([b]))[0]
        want = 2 if bank.kinds[k, a] == 1 else 1
        others = [(q, s) for q in range(len(located.pieces)) for s in range(4) if bank.kinds[q, s] == want and q != right[0]]
        pick = [others[i] for i in rng.choice(len(others), size=min(40, len(others)), replace=False)]
        d = bank.distance(np.full(len(pick), k), np.full(len(pick), a), np.array([q for q, _ in pick]), np.array([s for _, s in pick]))
        worse_fraction.append(float((d > true_d).mean()))
    assert len(worse_fraction) >= 30
    assert np.mean(worse_fraction) >= 0.9, np.mean(worse_fraction)

    # score_side_pair — тот же сигнал через публичный API, плюс тип-проверка.
    p0 = located.pieces[0]
    straight = [i for i, s in enumerate(p0.sides) if s.kind == SideKind.STRAIGHT]
    if straight:
        assert math.isinf(score_side_pair(p0, straight[0], located.pieces[1], 0).score)


def _layout_true_cells(located, truth):
    """(индекс детали, row, col, истинный поворот) по разметке — поворот
    определяется так же, как в test_global_layout_places_pieces_correctly."""
    out = []
    for k, piece in enumerate(located.pieces):
        gt = truth.get(piece.id)
        if gt is None:
            continue
        for r in range(4):
            if _rotation_matches_gt(piece, r * 90, gt):
                out.append((k, gt["grid_row"], gt["grid_col"], r))
                break
    return out
