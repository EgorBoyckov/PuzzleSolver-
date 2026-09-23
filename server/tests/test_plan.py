from app.pipeline.plan import apply_feedback, next_steps, next_steps_with_catalog
from app.pipeline.schemas import AssemblyStep, EdgeMatch, PieceKind, PieceRecord, StepStatus


def _edge(piece_a: str, side_a: int, piece_b: str, side_b: int, score: float, d_pos=float("inf")) -> EdgeMatch:
    return EdgeMatch(piece_a=piece_a, side_a=side_a, piece_b=piece_b, side_b=side_b, score=score, d_shape=0.0, d_color=0.0, d_pos=d_pos)


def _piece(id_: str, kind: PieceKind) -> PieceRecord:
    return PieceRecord(id=id_, puzzle_id="test", batch_number=0, number_in_batch=0, kind=kind)


def test_next_steps_respects_max_steps_and_side_reuse():
    edges = [
        _edge("A", 0, "B", 2, score=0.95),
        _edge("A", 0, "C", 2, score=0.90),  # переиспользует сторону A-0 -> должен быть пропущен
        _edge("B", 1, "D", 3, score=0.85),
        _edge("D", 0, "E", 2, score=0.80),
    ]
    steps = next_steps(edges, max_steps=2)
    assert len(steps) == 2
    used_sides = set()
    for s in steps:
        assert (s.piece_a, s.side_a) not in used_sides
        assert (s.piece_b, s.side_b) not in used_sides
        used_sides.add((s.piece_a, s.side_a))
        used_sides.add((s.piece_b, s.side_b))
    # Наивысший score выбран первым шагом.
    assert steps[0].piece_a == "A" and steps[0].piece_b == "B"


def test_next_steps_computes_rotation_deg():
    edges = [_edge("A", 1, "B", 3, score=0.9)]
    steps = next_steps(edges, max_steps=1)
    assert steps[0].rotation_deg == 0  # сторона 1 напротив стороны 3 без доп. поворота

    edges2 = [_edge("A", 1, "B", 1, score=0.9)]
    steps2 = next_steps(edges2, max_steps=1)
    assert steps2[0].rotation_deg == 180


def test_next_steps_with_catalog_prefers_frame_pieces_on_tie():
    catalog = [
        _piece("A", PieceKind.CORNER),
        _piece("B", PieceKind.CENTER),
        _piece("C", PieceKind.CENTER),
        _piece("D", PieceKind.CENTER),
    ]
    edges = [
        _edge("C", 0, "D", 2, score=0.8, d_pos=0.0),   # оба center, но с позицией -> "textured"
        _edge("A", 0, "B", 2, score=0.8, d_pos=0.0),   # A рамочная -> "frame", должен выиграть при равном score
    ]
    steps = next_steps_with_catalog(edges, catalog, max_steps=1)
    assert steps[0].piece_a == "A" and steps[0].sector == "frame"


def test_next_steps_with_catalog_monochrome_sector_when_no_position():
    catalog = [_piece("X", PieceKind.CENTER), _piece("Y", PieceKind.CENTER)]
    edges = [_edge("X", 0, "Y", 2, score=0.5, d_pos=float("inf"))]
    steps = next_steps_with_catalog(edges, catalog, max_steps=1)
    assert steps[0].sector == "monochrome"


def test_apply_feedback_sets_status():
    step = AssemblyStep(step_number=1, piece_a="A", piece_b="B", side_a=0, side_b=2, rotation_deg=0, confidence=0.9)
    assert step.status == StepStatus.PENDING
    apply_feedback(step, StepStatus.DONE)
    assert step.status == StepStatus.DONE
    apply_feedback(step, StepStatus.REJECTED)
    assert step.status == StepStatus.REJECTED
