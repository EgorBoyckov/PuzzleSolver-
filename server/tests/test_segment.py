import json

import cv2
import numpy as np
import pytest

from app.pipeline.preprocess import preprocess_frame
from app.pipeline.segment import segment_pieces
from app.synth.dataset import SyntheticDatasetConfig, generate_dataset

from .pipeline_test_utils import match_pieces_to_ground_truth


@pytest.fixture(scope="module")
def dataset_and_frame(tmp_path_factory):
    out_dir = tmp_path_factory.mktemp("segment_synth")
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
    return meta, gt_batch, frame, pieces


def test_found_rate_at_least_99_5_percent(dataset_and_frame):
    meta, gt_batch, frame, pieces = dataset_and_frame
    tolerance_px = 0.5 * meta["piece_size_mm"] / frame.mm_per_pixel

    matches = match_pieces_to_ground_truth(
        pieces, gt_batch, frame.marker_bbox_px, meta["table_px_per_mm"], 1.0 / frame.mm_per_pixel, tolerance_px
    )
    found_rate = len(matches) / len(gt_batch["pieces"])
    assert found_rate >= 0.995, f"found_rate={found_rate:.4f} ({len(matches)}/{len(gt_batch['pieces'])})"


def test_no_spurious_giant_components(dataset_and_frame):
    """Регрессия: чёрная рамка вне валидной области или неисключённый маркер
    раньше засчитывались одной гигантской 'деталью'."""
    _meta, _gt_batch, _frame, pieces = dataset_and_frame
    for p in pieces:
        xs = [pt.x for pt in p.contour_px]
        ys = [pt.y for pt in p.contour_px]
        w, h = max(xs) - min(xs), max(ys) - min(ys)
        assert w < 500 and h < 500, f"{p.id}: подозрительно большой bbox {w}x{h}"


def test_suspect_pieces_are_minority(dataset_and_frame):
    _meta, _gt_batch, _frame, pieces = dataset_and_frame
    suspect_fraction = sum(1 for p in pieces if p.is_suspect) / len(pieces)
    assert suspect_fraction < 0.15


def test_piece_ids_are_unique_and_well_formed(dataset_and_frame):
    _meta, _gt_batch, _frame, pieces = dataset_and_frame
    ids = [p.id for p in pieces]
    assert len(ids) == len(set(ids))
    for p in pieces:
        assert p.id.startswith(f"B{p.batch_number:02d}-")
