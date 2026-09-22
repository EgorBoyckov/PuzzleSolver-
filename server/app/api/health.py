"""Служебные эндпоинты: проверка живости и версия конвейера."""
from __future__ import annotations

from fastapi import APIRouter

from app import __about__

router = APIRouter(tags=["health"])


@router.get("/api/health")
def health() -> dict:
    return {"status": "ok", "service": "puzzlevision-server", "version": __about__.VERSION}
