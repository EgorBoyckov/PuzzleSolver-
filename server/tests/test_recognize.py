import pytest

from app.pipeline.describe import describe_piece
from app.pipeline.recognize import fingerprint_distance, recognize_pieces_on_frame
from app.pipeline.schemas import PieceRecord, Point2D, Side, SideKind
from app.pipeline.segment import segment_pieces
from app.pipeline.preprocess import preprocess_frame
from app.synth.dataset import SyntheticDatasetConfig, generate_dataset


def _side(index: int, kind: SideKind, ys: list[float]) -> Side:
    return Side(index=index, kind=kind, curve=[Point2D(x=float(i), y=float(y)) for i, y in enumerate(ys)])


def _piece(id_: str, kinds: list[SideKind], ys: list[float] | None = None) -> PieceRecord:
    ys = ys or [0.0, 1.0, 2.0, 1.0, 0.0]
    return PieceRecord(
        id=id_, puzzle_id="t", batch_number=0, number_in_batch=0,
        sides=[_side(i, k, ys if k != SideKind.STRAIGHT else [0.0] * len(ys)) for i, k in enumerate(kinds)],
    )


def test_fingerprint_distance_zero_for_identical_unrotated_piece():
    kinds = [SideKind.STRAIGHT, SideKind.TAB, SideKind.STRAIGHT, SideKind.BLANK]
    a = _piece("A", kinds)
    b = _piece("B", kinds)
    dist, rot = fingerprint_distance(a, b, penalty_mm=50.0)
    assert dist == pytest.approx(0.0, abs=1e-9)
    assert rot == 0


def test_fingerprint_distance_finds_correct_rotation():
    kinds = [SideKind.STRAIGHT, SideKind.TAB, SideKind.STRAIGHT, SideKind.BLANK]
    a = _piece("A", kinds)
    # candidate — та же деталь, но сфотографированная повёрнутой на 90°:
    # то, что было side0 у A, теперь оказывается на позиции side1 у B.
    rotated_kinds = [kinds[(i - 1) % 4] for i in range(4)]
    b = _piece("B", rotated_kinds)
    dist, rot = fingerprint_distance(a, b, penalty_mm=50.0)
    assert dist == pytest.approx(0.0, abs=1e-9)
    assert rot == 90


def test_fingerprint_distance_penalizes_kind_mismatch():
    a = _piece("A", [SideKind.STRAIGHT, SideKind.TAB, SideKind.STRAIGHT, SideKind.BLANK])
    b = _piece("B", [SideKind.STRAIGHT, SideKind.BLANK, SideKind.STRAIGHT, SideKind.TAB])
    dist, _rot = fingerprint_distance(a, b, penalty_mm=50.0)
    assert dist > 50.0  # минимум 2 несовпадающих типа стороны при любом повороте


@pytest.fixture(scope="module")
def catalog_and_frame(tmp_path_factory):
    out_dir = tmp_path_factory.mktemp("recognize_synth")
    config = SyntheticDatasetConfig(
        total_pieces=60, out_dir=out_dir,
        batch_size_min=60, batch_size_max=60, table_px_per_mm=6.0, seed=11,
    )
    meta = generate_dataset(config)
    batch_summary = meta["batches"][0]
    photo = out_dir / batch_summary["image"]
    frame, rectified, valid_mask = preprocess_frame(photo, "puz", batch_summary["batch_number"])
    assert frame.accepted

    pieces = segment_pieces(
        rectified, frame.mm_per_pixel, "puz", batch_summary["batch_number"],
        marker_bbox_px=frame.marker_bbox_px, valid_mask=valid_mask,
    )
    for p in pieces:
        describe_piece(p, rectified, frame.mm_per_pixel)
    catalog = [p for p in pieces if not p.is_suspect and len(p.sides) == 4]
    return catalog, rectified, frame.mm_per_pixel


def test_recognize_self_identification_high_accuracy(catalog_and_frame):
    """Узнавание того же кадра, что использовался для каталогизации — не
    настоящее "новое фото" (нет свежего шума съёмки/раскладки), но честно
    проверяет само ядро алгоритма (быстрый отбор по эмбеддингу + точный
    перебор поворотов по форме), не имея доступа к исходному сопоставлению
    id<->деталь: recognize должен САМ найти его заново по одной геометрии."""
    catalog, rectified, mm_per_pixel = catalog_and_frame
    results = recognize_pieces_on_frame(rectified, catalog, mm_per_pixel, puzzle_id="puz", batch_number=0)

    assert len(results) > 0
    recognized_ids = set(results.keys())
    catalog_ids = {p.id for p in catalog}
    correct = len(recognized_ids & catalog_ids)
    rate = correct / len(catalog_ids)
    assert rate >= 0.9, f"self-identification rate={rate:.2%}"

    for piece_id, (_x, _y, rot) in results.items():
        assert rot in (0, 90, 180, 270)
