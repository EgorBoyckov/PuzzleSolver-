import json
import math

import cv2
import pytest

from app.pipeline.describe import describe_piece
from app.pipeline.locate import build_reference_index, locate_piece, refit_reference_colors
from app.pipeline.match import (
    build_edge_matches,
    color_distance_deltae,
    find_side_candidates,
    position_distance_cells,
    relative_rotation_deg,
    score_side_pair,
    shape_distance_mm,
)
from app.pipeline.preprocess import preprocess_frame
from app.pipeline.schemas import PieceRecord, Point2D, Side, SideKind
from app.synth.dataset import SyntheticDatasetConfig, generate_dataset
from app.pipeline.segment import segment_pieces

from .pipeline_test_utils import match_pieces_to_ground_truth


def _piece(id_: str, sides: list[Side], location_candidates=None) -> PieceRecord:
    return PieceRecord(
        id=id_, puzzle_id="test", batch_number=0, number_in_batch=0,
        sides=sides, location_candidates=location_candidates or [],
    )


def _side(index: int, kind: SideKind, ys: list[float], lab: list[tuple[float, float, float]] | None = None) -> Side:
    n = len(ys)
    return Side(
        index=index, kind=kind,
        curve=[Point2D(x=float(i), y=float(y)) for i, y in enumerate(ys)],
        color_strip_lab=lab or [(50.0, 0.0, 0.0)] * n,
    )


def test_shape_distance_zero_for_perfectly_mirrored_curves():
    tab_ys = [0.0, 1.0, 3.0, 5.0, 3.0, 1.0, 0.0]
    blank_ys = [-y for y in reversed(tab_ys)]
    tab = _side(1, SideKind.TAB, tab_ys)
    blank = _side(3, SideKind.BLANK, blank_ys)
    assert shape_distance_mm(tab, blank) == pytest.approx(0.0, abs=1e-9)


def test_shape_distance_nonzero_for_mismatched_curves():
    tab = _side(1, SideKind.TAB, [0.0, 1.0, 3.0, 5.0, 3.0, 1.0, 0.0])
    wrong_blank = _side(3, SideKind.BLANK, [0.0, -1.0, -2.0, -3.0, -2.0, -1.0, 0.0])
    assert shape_distance_mm(tab, wrong_blank) > 0.5


def test_color_distance_zero_for_continuous_seam_color():
    lab = [(50.0, 10.0, -5.0), (52.0, 8.0, -4.0), (55.0, 6.0, -3.0)]
    side_a = _side(1, SideKind.TAB, [0.0, 1.0, 0.0], lab=lab)
    side_b = _side(3, SideKind.BLANK, [0.0, -1.0, 0.0], lab=list(reversed(lab)))
    assert color_distance_deltae(side_a, side_b) == pytest.approx(0.0, abs=1e-9)


def test_score_side_pair_rejects_incompatible_kinds():
    a = _piece("A", [_side(i, SideKind.STRAIGHT, [0.0] * 4) for i in range(4)])
    b = _piece("B", [_side(i, SideKind.STRAIGHT, [0.0] * 4) for i in range(4)])
    m = score_side_pair(a, 0, b, 0)
    assert m.score == 0.0
    assert math.isinf(m.d_shape)


def test_score_side_pair_high_for_good_match():
    tab_ys = [0.0, 2.0, 4.0, 2.0, 0.0]
    blank_ys = [-y for y in reversed(tab_ys)]
    lab = [(50.0, 0.0, 0.0)] * len(tab_ys)
    a_sides = [_side(0, SideKind.STRAIGHT, [0.0] * len(tab_ys)) for _ in range(4)]
    a_sides[1] = _side(1, SideKind.TAB, tab_ys, lab)
    b_sides = [_side(0, SideKind.STRAIGHT, [0.0] * len(tab_ys)) for _ in range(4)]
    b_sides[3] = _side(3, SideKind.BLANK, blank_ys, lab)
    a, b = _piece("A", a_sides), _piece("B", b_sides)
    m = score_side_pair(a, 1, b, 3)
    assert m.score > 0.9


def test_position_distance_cells_matches_true_neighbor():
    a = _piece("A", [], location_candidates=[(5, 5, 0, 0.9)])
    b_right = _piece("B", [], location_candidates=[(5, 6, 0, 0.9)])
    # side 1 (индекс) без поворота -> направление "право" (0,+1)
    d = position_distance_cells(a, 1, b_right, 3)
    assert d == pytest.approx(0.0)

    b_wrong = _piece("C", [], location_candidates=[(7, 7, 0, 0.9)])
    d_far = position_distance_cells(a, 1, b_wrong, 3)
    assert d_far > 1.0


def test_position_distance_cells_none_without_location():
    a = _piece("A", [])
    b = _piece("B", [], location_candidates=[(0, 0, 0, 0.9)])
    assert position_distance_cells(a, 0, b, 0) is None


def test_relative_rotation_deg_opposite_sides_zero_rotation():
    # Сторона 1 (право) детали A стыкуется со стороной 3 (лево) детали B
    # без дополнительного поворота B.
    assert relative_rotation_deg(1, 3) == 0
    # Сторона 1 (право) A стыкуется со стороной 1 (право) B -> B нужно
    # повернуть на 180°, чтобы его правая сторона "смотрела" на левую A.
    assert relative_rotation_deg(1, 1) == 180


@pytest.fixture(scope="module")
def matched_dataset(tmp_path_factory):
    out_dir = tmp_path_factory.mktemp("match_synth")
    config = SyntheticDatasetConfig(
        total_pieces=150, out_dir=out_dir,
        batch_size_min=75, batch_size_max=75, table_px_per_mm=6.0, seed=21,
    )
    meta = generate_dataset(config)
    reference_bgr = cv2.imread(str(out_dir / "catalog" / "box.jpg"))
    ref_index = build_reference_index(reference_bgr, meta["rows"], meta["cols"])

    all_pieces: list[PieceRecord] = []
    frame_lookup = {}
    gt_lookup: dict[str, tuple[int, int]] = {}

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

        tolerance_px = 0.5 * meta["piece_size_mm"] / frame.mm_per_pixel
        batch_matches = match_pieces_to_ground_truth(
            pieces, gt_batch, frame.marker_bbox_px, meta["table_px_per_mm"], 1.0 / frame.mm_per_pixel, tolerance_px
        )
        gt_rc = {gt["id"]: (gt["grid_row"], gt["grid_col"]) for gt in gt_batch["pieces"]}
        for gt_id, (piece, _dist) in batch_matches.items():
            gt_lookup[piece.id] = gt_rc[gt_id]

        frame_lookup[batch_number] = rectified
        all_pieces.extend(pieces)

    refitted = refit_reference_colors(ref_index, all_pieces, frame_lookup, confidence_threshold=0.85, min_samples=30)
    if refitted is not None:
        ref_index = refitted
        for p in all_pieces:
            if not p.location_candidates or p.location_candidates[0][3] < 0.85:
                locate_piece(p, frame_lookup[p.batch_number], ref_index)

    return all_pieces, gt_lookup


def test_build_edge_matches_well_formed(matched_dataset):
    pieces, _gt_lookup = matched_dataset
    edges = build_edge_matches(pieces)
    assert len(edges) > 0
    for e in edges:
        assert 0.0 <= e.score <= 1.0
        assert e.piece_a != e.piece_b


def test_edge_match_precision_against_true_neighbors(matched_dataset):
    """Честная метрика: доля найденных топ-1 совпадений сторон, чьи детали
    действительно физически соседние по эталонной раскладке (манхэттенское
    расстояние клеток сетки == 1) — не строгий критерий приёмки (его нет в
    доступной части спецификации для этапа match), а регрессионный порог.

    Диагностика (scratchpad/diag_match.py, scratchpad/diag_shape_sign.py)
    подтвердила, что сама математика зеркалирования формы/цвета верна на
    реальных Side из describe.py (у истинно смежных пар d_shape обычно
    0.3-1.5мм, у случайных заметно выше) — измеренная точность ~39% на
    этом датасете объясняется двумя унаследованными ограничениями, а не
    багом: (1) позиционный член наследует ошибку top-1 locate (78.57%
    top1 на этом прогоне; попытка расширить его до топ-3 кандидатов
    ИЗМЕРИМО УХУДШИЛА точность с 48.86% до 26.04% в подвыборке "оба
    верно локализованы" — топ-3 кандидатов locate часто географически
    близки друг к другу, поэтому расширение резко повышает шанс
    случайного позиционного совпадения чужих деталей; отклонено); (2)
    синтетические выступы/впадины — одна довольно однородная по форме
    "приподнятый косинус" семья кривых (варьируется в основном глубиной
    и положением центра), поэтому чистая форма сама по себе не всегда
    уникально отличает правильную пару от случайной — на это и рассчитан
    вес цвета/позиции в формуле, а не только формы."""
    pieces, gt_lookup = matched_dataset
    edges = build_edge_matches(pieces)

    checked = correct = 0
    for e in edges:
        rc_a, rc_b = gt_lookup.get(e.piece_a), gt_lookup.get(e.piece_b)
        if rc_a is None or rc_b is None:
            continue
        checked += 1
        manhattan = abs(rc_a[0] - rc_b[0]) + abs(rc_a[1] - rc_b[1])
        correct += manhattan == 1

    assert checked > 0
    precision = correct / checked
    assert precision >= 0.30, f"edge match precision={precision:.2%} (n={checked})"


def test_find_side_candidates_returns_ranked_list(matched_dataset):
    pieces, _gt_lookup = matched_dataset
    piece = next(p for p in pieces if any(s.kind == SideKind.TAB for s in p.sides))
    side_idx = next(s.index for s in piece.sides if s.kind == SideKind.TAB)
    candidates = find_side_candidates(piece, side_idx, pieces)
    assert len(candidates) <= 3
    scores = [c.score for c in candidates]
    assert scores == sorted(scores, reverse=True)
