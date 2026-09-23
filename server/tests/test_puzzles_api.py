from pathlib import Path

import pytest
import yaml
from fastapi.testclient import TestClient

from app.main import app
from app.synth.dataset import SyntheticDatasetConfig, generate_dataset

REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture()
def api_client(tmp_path, monkeypatch):
    """Реальный config.yaml (нужны все секции пайплайна: preprocess/segment/
    describe/locate/match/plan), но server.data_dir/debug_dir переопределены
    во временную папку, чтобы тест не писал в рабочую копию репозитория."""
    from app.core import config as config_module

    real_config = yaml.safe_load((REPO_ROOT / "config" / "config.yaml").read_text(encoding="utf-8"))
    real_config["server"]["data_dir"] = str(tmp_path / "data")
    real_config["server"]["debug_dir"] = str(tmp_path / "debug")
    real_config["server"]["database_url"] = f"sqlite:///{tmp_path}/data/puzzlevision.db"
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(yaml.safe_dump(real_config), encoding="utf-8")

    monkeypatch.setenv("PUZZLEVISION_CONFIG", str(cfg_path))
    config_module.get_config.cache_clear()

    from app.api import puzzles as puzzles_module
    puzzles_module._ref_index_cache.clear()

    yield TestClient(app)

    puzzles_module._ref_index_cache.clear()
    config_module.get_config.cache_clear()


@pytest.fixture(scope="module")
def synth_dir(tmp_path_factory):
    out_dir = tmp_path_factory.mktemp("api_synth")
    config = SyntheticDatasetConfig(
        total_pieces=150, out_dir=out_dir,
        batch_size_min=75, batch_size_max=75, table_px_per_mm=6.0, seed=21,
    )
    meta = generate_dataset(config)
    return out_dir, meta


def test_full_end_to_end_flow(api_client, synth_dir):
    out_dir, meta = synth_dir

    with (out_dir / "catalog" / "box.jpg").open("rb") as ref_f:
        r = api_client.post(
            "/api/puzzles",
            data={"puzzle_id": "e2e-test", "grid_rows": meta["rows"], "grid_cols": meta["cols"]},
            files={"reference": ("box.jpg", ref_f, "image/jpeg")},
        )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["id"] == "e2e-test"
    assert body["has_reference"] is True

    # Повторное создание того же ID -> конфликт.
    r_dup = api_client.post("/api/puzzles", data={"puzzle_id": "e2e-test"})
    assert r_dup.status_code == 409

    total_found = 0
    for batch_summary in meta["batches"]:
        photo_path = out_dir / batch_summary["image"]
        with photo_path.open("rb") as f:
            r = api_client.post(
                f"/api/puzzles/e2e-test/batches", files={"file": (photo_path.name, f, "image/jpeg")}
            )
        assert r.status_code == 200, r.text
        batch_body = r.json()
        assert batch_body["accepted"] is True
        assert batch_body["pieces_found_in_batch"] > 0
        total_found += batch_body["pieces_found_in_batch"]

    r = api_client.get("/api/puzzles/e2e-test")
    assert r.status_code == 200
    status = r.json()
    assert status["total_pieces"] == total_found
    assert status["batches_uploaded"] == meta["num_batches"]
    assert status["located"] is not None and status["located"] > 0
    assert status["steps_total"] > 0

    r = api_client.get("/api/puzzles/e2e-test/steps", params={"limit": 5})
    assert r.status_code == 200
    steps = r.json()
    assert 0 < len(steps) <= 5
    for s in steps:
        assert s["status"] == "pending"
        assert s["rotation_deg"] in (0, 90, 180, 270)

    first_step = steps[0]
    r = api_client.post(
        f"/api/puzzles/e2e-test/steps/{first_step['step_number']}/feedback", json={"status": "done"}
    )
    assert r.status_code == 200
    remaining = r.json()
    assert all(s["step_number"] != first_step["step_number"] for s in remaining)

    r_bad_step = api_client.post("/api/puzzles/e2e-test/steps/999999/feedback", json={"status": "done"})
    assert r_bad_step.status_code == 404

    second_step = remaining[0] if remaining else None
    if second_step is not None:
        r = api_client.post(
            f"/api/puzzles/e2e-test/steps/{second_step['step_number']}/feedback", json={"status": "rejected"}
        )
        assert r.status_code == 200
        after_reject = r.json()
        # Отклонённое ребро не должно снова появиться тем же piece/side.
        assert not any(
            s["piece_a"] == second_step["piece_a"] and s["side_a"] == second_step["side_a"]
            and s["piece_b"] == second_step["piece_b"] and s["side_b"] == second_step["side_b"]
            for s in after_reject
        )

    first_photo = out_dir / meta["batches"][0]["image"]
    with first_photo.open("rb") as f:
        r = api_client.post("/api/puzzles/e2e-test/recognize", files={"file": (first_photo.name, f, "image/jpeg")})
    assert r.status_code == 200
    recognized = r.json()
    assert len(recognized) > 0
    for item in recognized:
        assert item["rotation_deg"] in (0, 90, 180, 270)


def test_get_unknown_puzzle_404(api_client):
    r = api_client.get("/api/puzzles/does-not-exist")
    assert r.status_code == 404


def test_create_reference_requires_grid_dims(api_client, synth_dir):
    out_dir, _meta = synth_dir
    with (out_dir / "catalog" / "box.jpg").open("rb") as ref_f:
        r = api_client.post(
            "/api/puzzles",
            data={"puzzle_id": "missing-grid"},
            files={"reference": ("box.jpg", ref_f, "image/jpeg")},
        )
    assert r.status_code == 400


def test_create_without_reference(api_client):
    r = api_client.post("/api/puzzles", data={"puzzle_id": "no-ref"})
    assert r.status_code == 200
    assert r.json()["has_reference"] is False


def test_no_reference_flow(api_client, synth_dir):
    out_dir, meta = synth_dir

    r = api_client.post("/api/puzzles", data={"puzzle_id": "noref-flow"})
    assert r.status_code == 200

    for batch_summary in meta["batches"]:
        photo_path = out_dir / batch_summary["image"]
        with photo_path.open("rb") as f:
            r = api_client.post("/api/puzzles/noref-flow/batches", files={"file": (photo_path.name, f, "image/jpeg")})
        assert r.status_code == 200, r.text

    r = api_client.get("/api/puzzles/noref-flow")
    assert r.status_code == 200
    status = r.json()
    assert status["located"] is None
    assert status["frame_chain_length"] is not None
    assert status["island_count"] is not None

    r = api_client.get("/api/puzzles/noref-flow/no_reference")
    assert r.status_code == 200
    body = r.json()
    assert len(body["frame_chain"]) == status["frame_chain_length"]
    assert len(body["islands"]) == status["island_count"]
    clustered_ids = {pid for members in body["islands"].values() for pid in members}
    assert len(clustered_ids) == sum(len(v) for v in body["islands"].values())  # без дублей

    # /steps и /no_reference принадлежат разным режимам пазла.
    r_wrong = api_client.get("/api/puzzles/noref-flow/steps")
    assert r_wrong.status_code == 200
    assert r_wrong.json() == []

    with (out_dir / "catalog" / "box.jpg").open("rb") as ref_f:
        r_ref = api_client.post(
            "/api/puzzles",
            data={"puzzle_id": "with-ref-for-guard-test", "grid_rows": meta["rows"], "grid_cols": meta["cols"]},
            files={"reference": ("box.jpg", ref_f, "image/jpeg")},
        )
    assert r_ref.status_code == 200
    r_guard = api_client.get("/api/puzzles/with-ref-for-guard-test/no_reference")
    assert r_guard.status_code == 400


def test_ar_frame_endpoint(api_client, synth_dir):
    out_dir, meta = synth_dir

    r = api_client.post("/api/puzzles", data={"puzzle_id": "ar-test"})
    assert r.status_code == 200

    first_photo = out_dir / meta["batches"][0]["image"]
    with first_photo.open("rb") as f:
        r = api_client.post("/api/puzzles/ar-test/batches", files={"file": (first_photo.name, f, "image/jpeg")})
    assert r.status_code == 200, r.text

    # AR-эндпоинт работает на СЫРОМ кадре (без ректификации) — используем
    # тот же исходный файл партии, что и для загрузки, а не выпрямленный.
    with first_photo.open("rb") as f:
        r = api_client.post("/api/puzzles/ar-test/ar_frame", files={"file": (first_photo.name, f, "image/jpeg")})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["marker_found"] is True
    assert body["image_width"] > 0 and body["image_height"] > 0
    assert len(body["pieces"]) > 0
    for p in body["pieces"]:
        assert p["rotation_deg"] in (0, 90, 180, 270)
        assert 0 <= p["x_px"] <= body["image_width"]
        assert 0 <= p["y_px"] <= body["image_height"]


def test_ar_frame_no_marker_returns_marker_not_found(api_client):
    import io

    from PIL import Image

    r = api_client.post("/api/puzzles", data={"puzzle_id": "ar-no-marker"})
    assert r.status_code == 200

    blank = Image.new("RGB", (400, 400), (128, 128, 128))
    buf = io.BytesIO()
    blank.save(buf, format="JPEG")
    buf.seek(0)

    r = api_client.post("/api/puzzles/ar-no-marker/ar_frame", files={"file": ("blank.jpg", buf, "image/jpeg")})
    assert r.status_code == 200
    body = r.json()
    assert body["marker_found"] is False
    assert body["pieces"] == []


def test_ar_frame_unknown_puzzle_404(api_client):
    import io

    r = api_client.post("/api/puzzles/does-not-exist/ar_frame", files={"file": ("x.jpg", io.BytesIO(b"\xff\xd8"), "image/jpeg")})
    assert r.status_code == 404
