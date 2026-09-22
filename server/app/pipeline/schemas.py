"""Общие типы данных конвейера PuzzleVision.

Каждый модуль конвейера (preprocess/segment/describe/locate/match/plan/
recognize) принимает и возвращает эти pydantic-модели — так интерфейс между
модулями фиксирован и типизирован независимо от того, на каком этапе
разработки реализовано тело функции. Реализация самих модулей появляется
по этапам 1-5; здесь заготовлены только контракты (Stage 0).
"""
from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class PieceKind(str, Enum):
    CORNER = "corner"   # 2 прямые стороны
    EDGE = "edge"        # 1 прямая сторона
    CENTER = "center"    # 0 прямых сторон


class SideKind(str, Enum):
    STRAIGHT = "straight"
    TAB = "tab"          # выступ
    BLANK = "blank"       # впадина


class Point2D(BaseModel):
    x: float
    y: float


class Side(BaseModel):
    """Одна из четырёх сторон детали, в порядке против часовой стрелки от угла 0."""

    index: int = Field(ge=0, le=3)
    kind: SideKind
    # Нормализованная кривая стороны (64-128 точек), от угла до угла.
    curve: list[Point2D] = []
    # Цветовая полоса вдоль края в Lab, N сэмплов (см. config.describe.color_strip_samples).
    color_strip_lab: list[tuple[float, float, float]] = []


class PieceRecord(BaseModel):
    """Деталь в каталоге пазла — результат этапов segment/describe/locate."""

    id: str                      # формат "B{batch:02d}-{number:03d}", напр. B07-134
    puzzle_id: str
    batch_number: int
    number_in_batch: int
    kind: PieceKind | None = None
    contour_px: list[Point2D] = []
    thumbnail_path: str | None = None
    sides: list[Side] = []
    embedding: list[float] | None = None
    # Топ-3 кандидата места на образце: (row, col, rotation_deg, confidence).
    location_candidates: list[tuple[int, int, int, float]] = []
    is_suspect: bool = False     # слипшаяся/обрезанная деталь, требует ручной проверки
    suspect_reason: str | None = None   # "merged" | "cut_by_frame" | "odd_area" | ...
    tray_label: str | None = None


class BatchFrame(BaseModel):
    """Один кадр партии деталей после preprocess."""

    puzzle_id: str
    batch_number: int
    image_path: str
    mm_per_pixel: float
    marker_found: bool
    accepted: bool               # прошёл ли кадр отбраковку (резкость/засветка)
    rejection_reason: str | None = None
    # Прямоугольник маркера в выпрямленном кадре (px) — сегментация исключает
    # эту зону, чтобы не принять маркер за деталь.
    marker_bbox_px: tuple[float, float, float, float] | None = None


class EdgeMatch(BaseModel):
    """Оценка совместимости двух сторон деталей (см. config.match)."""

    piece_a: str
    side_a: int
    piece_b: str
    side_b: int
    score: float
    d_shape: float
    d_color: float
    d_pos: float


class StepStatus(str, Enum):
    PENDING = "pending"
    DONE = "done"
    REJECTED = "rejected"
    NOT_FOUND = "not_found"


class AssemblyStep(BaseModel):
    """Один шаг инструкции сборки, отдаваемый клиенту."""

    step_number: int
    piece_a: str
    piece_b: str                 # либо ID детали, либо ID собранного фрагмента
    side_a: int
    side_b: int
    rotation_deg: int
    sector: str | None = None
    confidence: float
    status: StepStatus = StepStatus.PENDING


class PuzzleMeta(BaseModel):
    id: str
    total_pieces: int
    box_image_path: str | None = None
    grid_rows: int | None = None
    grid_cols: int | None = None
    has_reference: bool = True   # False -> режим "без образца" (раздел 7 ТЗ)
    created_at: str
