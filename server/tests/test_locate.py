import json

import cv2
import pytest

from app.pipeline.describe import describe_piece
from app.pipeline.locate import build_reference_index, locate_piece, refit_reference_colors
from app.pipeline.preprocess import preprocess_frame
from app.pipeline.segment import segment_pieces
from app.synth.dataset import SyntheticDatasetConfig, generate_dataset

from .pipeline_test_utils import match_pieces_to_ground_truth


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
    all_gt_batches = []
    frame_lookup = {}
    frames = {}

    for batch_summary in meta["batches"]:
        batch_number = batch_summary["batch_number"]
        with (out_dir / batch_summary["annotations"]).open(encoding="utf-8") as f:
            gt_batch = json.load(f)
        photo = out_dir / batch_summary["image"]
        frame, rectified, valid_mask = preprocess_frame(photo, "puz", batch_number)
        assert frame.accepted

        pieces = segment_pieces(
            rectified, frame.mm_per_pixel, "puz", batch_number,
            marker_bbox_px=frame.marker_bbox_px, valid_mask=valid_mask,
        )
        for p in pieces:
            describe_piece(p, rectified, frame.mm_per_pixel)
            locate_piece(p, rectified, ref_index)

        frame_lookup[batch_number] = rectified
        frames[batch_number] = frame
        all_pieces.extend(pieces)
        all_gt_batches.append((frame, gt_batch))

    refitted = refit_reference_colors(
        ref_index, all_pieces, frame_lookup, confidence_threshold=0.85, min_samples=30
    )
    assert refitted is not None
    ref_index = refitted
    for p in all_pieces:
        if not p.location_candidates or p.location_candidates[0][3] < 0.85:
            locate_piece(p, frame_lookup[p.batch_number], ref_index)

    matches = {}
    for frame, gt_batch in all_gt_batches:
        tolerance_px = 0.5 * meta["piece_size_mm"] / frame.mm_per_pixel
        batch_pieces = [p for p in all_pieces if p.batch_number == frame.batch_number]
        batch_matches = match_pieces_to_ground_truth(
            batch_pieces, gt_batch, frame.marker_bbox_px, meta["table_px_per_mm"], 1.0 / frame.mm_per_pixel, tolerance_px
        )
        for gt_id, (piece, dist) in batch_matches.items():
            matches[(frame.batch_number, gt_id)] = (piece, dist, gt_batch)

    return meta, ref_index, all_pieces, matches


def _gt_rc(gt_batch: dict, gt_id: str) -> tuple[int, int]:
    for gt in gt_batch["pieces"]:
        if gt["id"] == gt_id:
            return gt["grid_row"], gt["grid_col"]
    raise KeyError(gt_id)


def test_top3_location_accuracy_reasonable(located_dataset):
    _meta, _ref_index, _pieces, matches = located_dataset
    total = 0
    top1 = top3 = 0
    for (_batch, gt_id), (piece, _dist, gt_batch) in matches.items():
        if not piece.location_candidates:
            continue
        total += 1
        true_rc = _gt_rc(gt_batch, gt_id)
        cands = [(r, c) for r, c, _rot, _conf in piece.location_candidates]
        top1 += cands[0] == true_rc
        top3 += true_rc in cands

    top1_rate, top3_rate = top1 / total, top3 / total
    assert top1_rate >= 0.65, f"top1={top1_rate:.2%}"
    assert top3_rate >= 0.80, f"top3={top3_rate:.2%} (полный критерий >=90% проверяется scripts/stage2_acceptance.py)"


def test_location_candidates_well_formed(located_dataset):
    meta, ref_index, pieces, _matches = located_dataset
    for p in pieces:
        if not p.location_candidates:
            continue
        assert len(p.location_candidates) <= 3
        for row, col, rot, conf in p.location_candidates:
            assert 0 <= row < ref_index.rows
            assert 0 <= col < ref_index.cols
            assert rot in (0, 90, 180, 270)
            assert 0.0 <= conf <= 1.0


def test_color_refit_rebuilds_a_valid_index(located_dataset):
    # located_dataset уже утверждает refitted is not None (т.е. уверенных
    # совпадений хватило и цветокоррекция реально отработала) — здесь лишь
    # проверяем, что пересобранный индекс структурно валиден.
    _meta, ref_index, _pieces, _matches = located_dataset
    assert ref_index.embeddings.shape[0] == ref_index.rows * ref_index.cols
    assert ref_index.reference_bgr.shape[0] > 0 and ref_index.reference_bgr.shape[1] > 0
