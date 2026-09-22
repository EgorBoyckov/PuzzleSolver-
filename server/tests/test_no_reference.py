import json

import cv2
import pytest

from app.pipeline.describe import describe_piece
from app.pipeline.no_reference import _frame_relevant_sides, build_frame_chain, cluster_islands
from app.pipeline.preprocess import preprocess_frame
from app.pipeline.schemas import PieceKind, PieceRecord, Side, SideKind
from app.pipeline.segment import segment_pieces
from app.synth.dataset import SyntheticDatasetConfig, generate_dataset

from .pipeline_test_utils import match_pieces_to_ground_truth


def _piece(id_: str, kind: PieceKind, kinds: list[SideKind]) -> PieceRecord:
    return PieceRecord(
        id=id_, puzzle_id="t", batch_number=0, number_in_batch=0, kind=kind,
        sides=[Side(index=i, kind=k) for i, k in enumerate(kinds)],
    )


def test_frame_relevant_sides_corner():
    p = _piece("A", PieceKind.CORNER, [SideKind.STRAIGHT, SideKind.TAB, SideKind.BLANK, SideKind.STRAIGHT])
    assert sorted(_frame_relevant_sides(p)) == [1, 2]


def test_frame_relevant_sides_edge():
    p = _piece("A", PieceKind.EDGE, [SideKind.STRAIGHT, SideKind.TAB, SideKind.BLANK, SideKind.TAB])
    # прямая на индексе 0 -> вдоль рамки идут соседние 3 и 1, а 2 (напротив) смотрит внутрь
    assert sorted(_frame_relevant_sides(p)) == [1, 3]


def test_frame_relevant_sides_center_piece_empty():
    p = _piece("A", PieceKind.CENTER, [SideKind.TAB, SideKind.BLANK, SideKind.TAB, SideKind.BLANK])
    assert _frame_relevant_sides(p) == []


def test_build_frame_chain_empty_without_frame_pieces():
    assert build_frame_chain([]) == []
    # Только center-деталь (0 прямых сторон) -> не рамочная -> цепочка пуста.
    center_only = [_piece("A", PieceKind.CENTER, [SideKind.TAB, SideKind.BLANK, SideKind.TAB, SideKind.BLANK])]
    assert build_frame_chain(center_only) == []


def test_build_frame_chain_single_corner_no_other_frame_pieces():
    # Единственная рамочная деталь без кандидатов на соседей -> цепочка из неё одной.
    corner_only = [_piece("A", PieceKind.CORNER, [SideKind.STRAIGHT, SideKind.TAB, SideKind.BLANK, SideKind.STRAIGHT])]
    assert build_frame_chain(corner_only) == ["A"]


@pytest.fixture(scope="module")
def described_dataset(tmp_path_factory):
    """Датасет без привязки к образцу (locate НЕ вызывается) — именно
    сценарий этапа 5: только describe, никакой картинки коробки."""
    out_dir = tmp_path_factory.mktemp("no_ref_synth")
    config = SyntheticDatasetConfig(
        total_pieces=140, out_dir=out_dir,
        batch_size_min=140, batch_size_max=140, table_px_per_mm=6.0, seed=17,
    )
    meta = generate_dataset(config)

    batch_summary = meta["batches"][0]
    with (out_dir / batch_summary["annotations"]).open(encoding="utf-8") as f:
        gt_batch = json.load(f)
    photo = out_dir / batch_summary["image"]
    frame, rectified, valid_mask = preprocess_frame(photo, "noref", batch_summary["batch_number"])
    assert frame.accepted

    pieces = segment_pieces(
        rectified, frame.mm_per_pixel, "noref", batch_summary["batch_number"],
        marker_bbox_px=frame.marker_bbox_px, valid_mask=valid_mask,
    )
    for p in pieces:
        describe_piece(p, rectified, frame.mm_per_pixel)

    tolerance_px = 0.5 * meta["piece_size_mm"] / frame.mm_per_pixel
    matches = match_pieces_to_ground_truth(
        pieces, gt_batch, frame.marker_bbox_px, meta["table_px_per_mm"], 1.0 / frame.mm_per_pixel, tolerance_px
    )
    gt_rc = {gt["id"]: (gt["grid_row"], gt["grid_col"]) for gt in gt_batch["pieces"]}
    gt_lookup = {piece.id: gt_rc[gt_id] for gt_id, (piece, _dist) in matches.items()}

    return pieces, gt_lookup, meta


def test_build_frame_chain_mostly_true_perimeter_neighbors(described_dataset):
    """Честная метрика (без строгого критерия из ТЗ — недоступен в этой
    сессии, как и для этапа 3): доля последовательных пар в собранной
    цепочке, чьи детали действительно соседние по эталонной раскладке
    (манхэттенское расстояние ячеек сетки == 1)."""
    pieces, gt_lookup, _meta = described_dataset
    chain = build_frame_chain(pieces)
    assert len(chain) >= 4  # хотя бы несколько рамочных деталей нашлось и связалось

    checked = correct = 0
    for a, b in zip(chain, chain[1:]):
        rc_a, rc_b = gt_lookup.get(a), gt_lookup.get(b)
        if rc_a is None or rc_b is None:
            continue
        checked += 1
        manhattan = abs(rc_a[0] - rc_b[0]) + abs(rc_a[1] - rc_b[1])
        correct += manhattan == 1

    assert checked > 0
    precision = correct / checked
    assert precision >= 0.25, f"frame chain precision={precision:.2%} (n={checked})"


def test_cluster_islands_structure(described_dataset):
    pieces, _gt_lookup, _meta = described_dataset
    clusters = cluster_islands(pieces, seed=1)
    interior_ids = {p.id for p in pieces if p.kind == PieceKind.CENTER and p.embedding}
    if not interior_ids:
        assert clusters == {}
        return

    all_clustered = [pid for members in clusters.values() for pid in members]
    assert set(all_clustered) == interior_ids
    assert len(all_clustered) == len(set(all_clustered))  # каждая деталь ровно в одном кластере
    assert len(clusters) >= 1


def test_cluster_islands_deterministic_given_seed(described_dataset):
    pieces, _gt_lookup, _meta = described_dataset
    a = cluster_islands(pieces, seed=5)
    b = cluster_islands(pieces, seed=5)
    assert {k: sorted(v) for k, v in a.items()} == {k: sorted(v) for k, v in b.items()}
