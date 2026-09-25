import json

import pytest

from app.cli.pipeline_cli import run_pipeline_on_folder
from app.synth.dataset import SyntheticDatasetConfig, generate_dataset


@pytest.fixture(scope="module")
def synth_batches_dir(tmp_path_factory):
    out_dir = tmp_path_factory.mktemp("cli_synth")
    config = SyntheticDatasetConfig(
        total_pieces=40,
        out_dir=out_dir,
        batch_size_min=20,
        batch_size_max=20,
        table_px_per_mm=6.0,
        seed=5,
    )
    meta = generate_dataset(config)
    return out_dir, meta


def test_run_pipeline_on_folder_produces_summary_and_catalog(synth_batches_dir, tmp_path):
    out_dir_synth, meta = synth_batches_dir
    out_dir = tmp_path / "run"

    summary = run_pipeline_on_folder(out_dir_synth / "batches", "cli-test", out_dir)

    assert summary["frames_total"] == meta["num_batches"]
    assert summary["frames_accepted"] == meta["num_batches"]
    assert summary["pieces_found"] > 0
    assert summary["pieces_found"] <= meta["total_pieces_actual"] * 1.05

    assert (out_dir / "summary.json").exists()
    assert (out_dir / "catalog" / "pieces.json").exists()
    with (out_dir / "catalog" / "pieces.json").open(encoding="utf-8") as f:
        pieces = json.load(f)
    assert len(pieces) == summary["pieces_found"]
    assert all(p["id"].startswith("B") for p in pieces)


def test_run_pipeline_with_reference_fills_location_candidates(synth_batches_dir, tmp_path):
    out_dir_synth, meta = synth_batches_dir
    out_dir = tmp_path / "run_located"

    summary = run_pipeline_on_folder(
        out_dir_synth / "batches",
        "cli-test-located",
        out_dir,
        reference_path=out_dir_synth / "catalog" / "box.jpg",
        grid_rows=meta["rows"],
        grid_cols=meta["cols"],
    )

    assert summary["located"] == summary["pieces_found"]
    with (out_dir / "catalog" / "pieces.json").open(encoding="utf-8") as f:
        pieces = json.load(f)
    located_with_candidates = [p for p in pieces if p["location_candidates"]]
    assert len(located_with_candidates) == summary["pieces_found"]
    for p in located_with_candidates:
        assert 1 <= len(p["location_candidates"]) <= 3
    assert summary["layout"]["placed"] == summary["pieces_found"]
    assert (out_dir / "debug" / "assembled.jpg").exists()
    with (out_dir / "catalog" / "layout.json").open(encoding="utf-8") as f:
        layout = json.load(f)
    placed_ids = [cell["piece_id"] for row in layout["cells"] for cell in row if cell]
    assert len(placed_ids) == len(set(placed_ids)) == summary["pieces_found"]
    assert all(p["placement"] is not None for p in pieces)

    assert summary["edge_matches"] is not None
    assert summary["assembly_steps"] is not None
    assert (out_dir / "catalog" / "steps.json").exists()
    with (out_dir / "catalog" / "steps.json").open(encoding="utf-8") as f:
        steps = json.load(f)
    assert len(steps) == summary["assembly_steps"]
    for s in steps:
        assert s["rotation_deg"] in (0, 90, 180, 270)
        assert 0.0 <= s["confidence"] <= 1.0


def test_run_pipeline_skip_match_omits_steps(synth_batches_dir, tmp_path):
    out_dir_synth, _meta = synth_batches_dir
    out_dir = tmp_path / "run_skip_match"

    summary = run_pipeline_on_folder(out_dir_synth / "batches", "cli-test-skip", out_dir, skip_match=True)

    assert summary["edge_matches"] is None
    assert summary["assembly_steps"] is None
    assert not (out_dir / "catalog" / "steps.json").exists()


def test_run_pipeline_without_reference_produces_frame_chain_and_islands(synth_batches_dir, tmp_path):
    out_dir_synth, _meta = synth_batches_dir
    out_dir = tmp_path / "run_no_reference"

    summary = run_pipeline_on_folder(out_dir_synth / "batches", "cli-test-noref", out_dir)

    assert summary["edge_matches"] is None
    assert summary["assembly_steps"] is None
    assert summary["frame_chain_length"] is not None
    assert summary["island_count"] is not None
    assert (out_dir / "catalog" / "frame_chain.json").exists()
    assert (out_dir / "catalog" / "islands.json").exists()

    with (out_dir / "catalog" / "frame_chain.json").open(encoding="utf-8") as f:
        chain = json.load(f)
    assert len(chain) == summary["frame_chain_length"]

    with (out_dir / "catalog" / "islands.json").open(encoding="utf-8") as f:
        islands = json.load(f)
    assert len(islands) == summary["island_count"]
