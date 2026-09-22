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
    return out_dir / "batches", meta


def test_run_pipeline_on_folder_produces_summary_and_catalog(synth_batches_dir, tmp_path):
    batches_dir, meta = synth_batches_dir
    out_dir = tmp_path / "run"

    summary = run_pipeline_on_folder(batches_dir, "cli-test", out_dir)

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
