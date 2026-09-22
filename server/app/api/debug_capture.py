"""Отладочный эндпоинт этапа 0: принять фото с телефона и сохранить его.

Используется PWA-клиентом, чтобы на этапе каркаса подтвердить сквозной путь
"телефон снял кадр -> сервер принял и сохранил файл" ещё до появления
настоящего конвейера обработки (сегментация и т.д. приезжают на этапе 1).
"""
from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone

from fastapi import APIRouter, UploadFile
from pydantic import BaseModel

from app.core.config import get_config

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/debug", tags=["debug"])


class UploadResult(BaseModel):
    id: str
    filename: str
    content_type: str | None
    size_bytes: int
    saved_path: str


@router.post("/upload", response_model=UploadResult)
async def upload_frame(file: UploadFile) -> UploadResult:
    config = get_config()
    debug_dir = config.path(config.server.debug_dir) / "uploads"
    debug_dir.mkdir(parents=True, exist_ok=True)

    frame_id = uuid.uuid4().hex[:12]
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    suffix = "".join(ch for ch in (file.filename or "") if ch.isalnum() or ch in "._-")[-40:]
    out_name = f"{timestamp}_{frame_id}_{suffix or 'frame.jpg'}"
    out_path = debug_dir / out_name

    content = await file.read()
    out_path.write_bytes(content)
    logger.info("saved debug upload %s (%d bytes)", out_path, len(content))

    return UploadResult(
        id=frame_id,
        filename=file.filename or out_name,
        content_type=file.content_type,
        size_bytes=len(content),
        saved_path=str(out_path),
    )
