import cv2
import pytest

from app.pipeline.ar import recognize_raw_frame
from app.pipeline.describe import describe_piece
from app.pipeline.preprocess import detect_marker_and_scale, preprocess_frame
from app.pipeline.segment import segment_pieces
from app.synth.dataset import SyntheticDatasetConfig, generate_dataset


@pytest.fixture(scope="module")
def raw_batch_and_catalog(tmp_path_factory):
    """Каталог строится как обычно (через preprocess -> выпрямленный кадр),
    но AR проверяется на СЫРОМ, невыпрямленном фото той же партии — реальный
    сценарий: каталог уже собран, а AR-режим потом узнаёт те же детали на
    новом (в данном случае том же, но НЕ ректифицированном) кадре камеры."""
    out_dir = tmp_path_factory.mktemp("ar_synth")
    config = SyntheticDatasetConfig(
        total_pieces=60, out_dir=out_dir,
        batch_size_min=60, batch_size_max=60, table_px_per_mm=6.0, seed=9,
    )
    meta = generate_dataset(config)
    batch_summary = meta["batches"][0]
    photo_path = out_dir / batch_summary["image"]

    frame, rectified, valid_mask = preprocess_frame(photo_path, "ar-test", batch_summary["batch_number"])
    assert frame.accepted

    pieces = segment_pieces(
        rectified, frame.mm_per_pixel, "ar-test", batch_summary["batch_number"],
        marker_bbox_px=frame.marker_bbox_px, valid_mask=valid_mask,
    )
    for p in pieces:
        describe_piece(p, rectified, frame.mm_per_pixel)
    catalog = [p for p in pieces if not p.is_suspect and len(p.sides) == 4]

    raw_image = cv2.imread(str(photo_path))
    return raw_image, catalog, meta


def test_detect_marker_and_scale_finds_marker(raw_batch_and_catalog):
    raw_image, _catalog, meta = raw_batch_and_catalog
    corners, mm_per_pixel = detect_marker_and_scale(raw_image, "DICT_4X4_50", 0, 100.0)
    assert corners is not None
    assert corners.shape == (4, 2)
    assert mm_per_pixel is not None
    assert mm_per_pixel > 0
    # Грубая проверка порядка величины: реальный масштаб исходного (не
    # выпрямленного) фото близок к table_px_per_mm генератора синтетики
    # (тот же кадр, без искажения перспективы съёмки прямо сверху).
    expected_mm_per_pixel = 1.0 / meta["table_px_per_mm"]
    assert mm_per_pixel == pytest.approx(expected_mm_per_pixel, rel=0.3)


def test_detect_marker_and_scale_none_without_marker():
    import numpy as np

    blank = np.zeros((200, 200, 3), dtype="uint8")
    corners, mm_per_pixel = detect_marker_and_scale(blank, "DICT_4X4_50", 0, 100.0)
    assert corners is None
    assert mm_per_pixel is None


def test_recognize_raw_frame_finds_marker_and_pieces(raw_batch_and_catalog):
    raw_image, catalog, _meta = raw_batch_and_catalog
    marker_found, results = recognize_raw_frame(raw_image, catalog, "ar-test")
    assert marker_found is True
    assert len(results) > 0
    catalog_ids = {p.id for p in catalog}
    assert set(results.keys()) <= catalog_ids
    for _pid, (x, y, rot) in results.items():
        assert 0 <= x <= raw_image.shape[1]
        assert 0 <= y <= raw_image.shape[0]
        assert rot in (0, 90, 180, 270)


def test_recognize_raw_frame_no_marker_returns_empty(raw_batch_and_catalog):
    import numpy as np

    _raw_image, catalog, _meta = raw_batch_and_catalog
    blank = np.full((400, 400, 3), 200, dtype="uint8")
    marker_found, results = recognize_raw_frame(blank, catalog, "ar-test")
    assert marker_found is False
    assert results == {}
