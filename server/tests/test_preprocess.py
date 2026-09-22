import cv2
import numpy as np
import pytest

from app.pipeline.preprocess import preprocess_frame
from app.synth.dataset import SyntheticDatasetConfig, generate_dataset


@pytest.fixture(scope="module")
def small_dataset(tmp_path_factory):
    out_dir = tmp_path_factory.mktemp("preprocess_synth")
    config = SyntheticDatasetConfig(
        total_pieces=20,
        out_dir=out_dir,
        batch_size_min=20,
        batch_size_max=20,
        table_px_per_mm=5.0,
        seed=3,
    )
    meta = generate_dataset(config)
    return out_dir, meta


def test_accepts_clean_frame_and_computes_scale(small_dataset, tmp_path):
    out_dir, meta = small_dataset
    photo = out_dir / meta["batches"][0]["image"]

    frame, rectified, valid_mask = preprocess_frame(photo, "puz", 1, debug_dir=tmp_path)

    assert frame.accepted
    assert frame.marker_found
    assert frame.rejection_reason is None
    assert rectified is not None
    assert valid_mask is not None
    assert abs(frame.mm_per_pixel - 1.0 / 6.0) < 0.01  # config.preprocess.output_px_per_mm = 6.0
    assert frame.marker_bbox_px is not None
    assert (tmp_path / "batch_01_rectified.jpg").exists()


def test_rejects_blurry_frame(small_dataset, tmp_path):
    out_dir, meta = small_dataset
    photo = out_dir / meta["batches"][0]["image"]

    img = cv2.imread(str(photo))
    blurred = cv2.GaussianBlur(img, (0, 0), sigmaX=25)
    blurry_path = tmp_path / "blurry.jpg"
    cv2.imwrite(str(blurry_path), blurred)

    frame, rectified, valid_mask = preprocess_frame(blurry_path, "puz", 1)
    assert not frame.accepted
    assert frame.rejection_reason == "blurry"
    assert rectified is None and valid_mask is None


def test_rejects_overexposed_frame(small_dataset, tmp_path):
    out_dir, meta = small_dataset
    photo = out_dir / meta["batches"][0]["image"]

    img = cv2.imread(str(photo))
    # Масштабируем яркость (а не сдвигаем на константу), чтобы не сгладить
    # градиенты до нуля — иначе кадр случайно попадёт под "blurry" раньше,
    # чем до проверки пересвета.
    blown = np.clip(img.astype(np.float32) * 3.0, 0, 255).astype(np.uint8)
    overexposed_path = tmp_path / "overexposed.jpg"
    cv2.imwrite(str(overexposed_path), blown)

    frame, rectified, valid_mask = preprocess_frame(overexposed_path, "puz", 1)
    assert not frame.accepted
    assert frame.rejection_reason == "overexposed"


def test_marker_white_sheet_does_not_trigger_overexposure_rejection(small_dataset):
    """Регрессия: раньше собственный белый лист маркера засчитывался в
    пересвет и кадр отбраковывался, хотя реального пересвета не было."""
    out_dir, meta = small_dataset
    photo = out_dir / meta["batches"][0]["image"]

    frame, rectified, _ = preprocess_frame(photo, "puz", 1)
    assert frame.accepted
    assert frame.rejection_reason is None


def test_rejects_frame_without_marker(tmp_path):
    # Лёгкий шум, а не плоская заливка — иначе кадр отбракуется как "blurry"
    # (нулевая дисперсия Лапласиана) раньше, чем дойдёт до поиска маркера,
    # и тест перестанет проверять то, что заявлен.
    blank = np.random.default_rng(1).integers(40, 80, (600, 600, 3), dtype=np.uint8)
    blank_path = tmp_path / "no_marker.jpg"
    cv2.imwrite(str(blank_path), blank)

    frame, rectified, valid_mask = preprocess_frame(blank_path, "puz", 1)
    assert not frame.accepted
    assert frame.rejection_reason == "marker_not_found"
    assert not frame.marker_found


def test_rectified_output_has_no_giant_border_artifact(small_dataset, tmp_path):
    """Регрессия: неверный расчёт размера холста оставлял широкую чёрную
    рамку вне проекции исходного кадра, которую сегментация принимала за
    одну гигантскую "деталь"."""
    out_dir, meta = small_dataset
    photo = out_dir / meta["batches"][0]["image"]

    frame, rectified, valid_mask = preprocess_frame(photo, "puz", 1)
    assert rectified is not None and valid_mask is not None

    # Вне маски валидных пикселей должно быть немного (только буферный
    # отступ по краям), а не значительная часть кадра.
    invalid_fraction = float((valid_mask == 0).mean())
    assert invalid_fraction < 0.35

    # Внутри валидной маски, за пределами узкого буфера, пикселей чистого
    # чёрного (0,0,0) быть не должно — это был бы признак утечки border-заливки.
    eroded_valid = cv2.erode(valid_mask, np.ones((21, 21), np.uint8))
    interior_pixels = rectified[eroded_valid > 0]
    pure_black_fraction = float(np.all(interior_pixels == 0, axis=1).mean())
    assert pure_black_fraction < 0.03
