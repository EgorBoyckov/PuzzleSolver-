from pathlib import Path

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health_ok():
    r = client.get("/api/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["service"] == "puzzlevision-server"


def test_debug_upload_roundtrip(tmp_path, monkeypatch):
    from app.core import config as config_module

    config_module.get_config.cache_clear()
    monkeypatch.setenv("PUZZLEVISION_CONFIG", str(_write_minimal_config(tmp_path)))
    config_module.get_config.cache_clear()

    files = {"file": ("frame.jpg", b"\xff\xd8\xff\xfake-jpeg-bytes", "image/jpeg")}
    r = client.post("/api/debug/upload", files=files)
    assert r.status_code == 200
    body = r.json()
    assert body["filename"] == "frame.jpg"
    assert body["size_bytes"] == len(b"\xff\xd8\xff\xfake-jpeg-bytes")
    saved = Path(body["saved_path"])
    assert saved.exists()
    assert saved.is_relative_to(tmp_path)

    config_module.get_config.cache_clear()


def _write_minimal_config(tmp_path) -> str:
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(
        f"""
server:
  debug_dir: "{tmp_path}/data/debug"
""".strip()
    )
    return str(cfg_path)
