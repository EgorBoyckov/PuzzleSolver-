import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from app.synth.dataset import SyntheticDatasetConfig, generate_dataset


@pytest.fixture(scope="module")
def dataset(tmp_path_factory):
    out_dir = tmp_path_factory.mktemp("synth_dataset")
    config = SyntheticDatasetConfig(
        total_pieces=48,
        out_dir=out_dir,
        batch_size_min=15,
        batch_size_max=25,
        table_px_per_mm=5.0,
        seed=11,
    )
    meta = generate_dataset(config)
    return out_dir, meta


def test_dataset_meta_files_exist(dataset):
    out_dir, meta = dataset
    assert (out_dir / "dataset_meta.json").exists()
    assert (out_dir / "catalog" / "box.jpg").exists()
    assert (out_dir / "catalog" / "grid.json").exists()
    assert meta["total_pieces_actual"] == meta["rows"] * meta["cols"]
    assert meta["total_pieces_actual"] >= meta["total_pieces_target"] * 0.8


def test_batch_files_and_sizes(dataset):
    out_dir, meta = dataset
    total_annotated = 0
    for b in meta["batches"]:
        img_path = out_dir / b["image"]
        json_path = out_dir / b["annotations"]
        assert img_path.exists()
        assert json_path.exists()
        with json_path.open(encoding="utf-8") as f:
            batch_data = json.load(f)
        assert batch_data["num_pieces"] == len(batch_data["pieces"])
        total_annotated += batch_data["num_pieces"]
    assert total_annotated == meta["total_pieces_actual"]


def test_all_pieces_covered_exactly_once(dataset):
    out_dir, meta = dataset
    seen = set()
    for b in meta["batches"]:
        with (out_dir / b["annotations"]).open(encoding="utf-8") as f:
            batch_data = json.load(f)
        for p in batch_data["pieces"]:
            key = (p["grid_row"], p["grid_col"])
            assert key not in seen, f"piece {key} annotated twice"
            seen.add(key)
    assert len(seen) == meta["total_pieces_actual"]
    for r in range(meta["rows"]):
        for c in range(meta["cols"]):
            assert (r, c) in seen


def test_aruco_marker_detectable_in_every_batch(dataset):
    out_dir, meta = dataset
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    detector = cv2.aruco.ArucoDetector(aruco_dict, cv2.aruco.DetectorParameters())
    for b in meta["batches"]:
        img = cv2.imread(str(out_dir / b["image"]))
        assert img is not None
        corners, ids, _ = detector.detectMarkers(img)
        assert ids is not None and len(ids) >= 1, f"marker not detected in {b['image']}"
        assert 0 in ids.flatten()


def test_pieces_do_not_overlap(dataset):
    out_dir, meta = dataset
    for b in meta["batches"]:
        with (out_dir / b["annotations"]).open(encoding="utf-8") as f:
            batch_data = json.load(f)
        side = batch_data["canvas_px"]
        count = np.zeros((side, side), dtype=np.uint8)
        for p in batch_data["pieces"]:
            m = np.zeros((side, side), dtype=np.uint8)
            pts = np.array(p["contour_px"], dtype=np.int32)
            cv2.fillPoly(m, [pts], 1)
            count += m
        overlap_fraction = float((count > 1).sum()) / max(float((count >= 1).sum()), 1.0)
        assert overlap_fraction < 0.01


def test_batch_sizes_within_configured_range(dataset):
    out_dir, meta = dataset
    n_batches = len(meta["batches"])
    if n_batches == 1:
        return
    for b in meta["batches"]:
        assert 15 <= b["num_pieces"] <= 25


def test_kind_distribution_matches_grid_geometry(dataset):
    out_dir, meta = dataset
    rows, cols = meta["rows"], meta["cols"]
    expected = {
        "corner": 4,
        "edge": 2 * (rows - 2) + 2 * (cols - 2),
        "center": (rows - 2) * (cols - 2),
    }
    actual = {"corner": 0, "edge": 0, "center": 0}
    for b in meta["batches"]:
        with (out_dir / b["annotations"]).open(encoding="utf-8") as f:
            batch_data = json.load(f)
        for p in batch_data["pieces"]:
            actual[p["kind"]] += 1
    assert actual == expected


def test_small_puzzle_single_batch_no_crash(tmp_path):
    config = SyntheticDatasetConfig(
        total_pieces=12,
        out_dir=tmp_path / "tiny",
        batch_size_min=60,
        batch_size_max=120,
        seed=5,
    )
    meta = generate_dataset(config)
    assert meta["num_batches"] == 1
    assert meta["batches"][0]["num_pieces"] == meta["total_pieces_actual"]


def test_full_scale_batch_120_pieces_renders_without_fallback(tmp_path, caplog):
    import logging

    config = SyntheticDatasetConfig(
        total_pieces=120,
        out_dir=tmp_path / "full_batch",
        batch_size_min=120,
        batch_size_max=120,
        seed=9,
    )
    with caplog.at_level(logging.WARNING, logger="app.synth.dataset"):
        meta = generate_dataset(config)
    assert meta["batches"][0]["num_pieces"] == meta["total_pieces_actual"]
    assert abs(meta["total_pieces_actual"] - 120) / 120 < 0.1
    fallback_warnings = [r for r in caplog.records if "fallback" in r.message]
    assert len(fallback_warnings) == 0
