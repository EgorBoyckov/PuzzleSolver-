"""Точка входа сервера PuzzleVision.

FastAPI отдаёт API (/api/...) и собранное PWA-клиентское приложение
(client/dist) статикой по тому же адресу и порту, чтобы телефону было
достаточно одного HTTPS-адреса в локальной сети.
"""
from __future__ import annotations

import logging
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.__about__ import VERSION
from app.api import debug_capture, health, puzzles
from app.core.config import get_config
from app.core.logging import setup_logging

setup_logging()
logger = logging.getLogger(__name__)

CLIENT_DIST = Path(__file__).resolve().parents[2] / "client" / "dist"


def create_app() -> FastAPI:
    config = get_config()

    app = FastAPI(title="PuzzleVision", version=VERSION)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=config.server.cors_allow_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(health.router)
    app.include_router(debug_capture.router)
    app.include_router(puzzles.router)

    if CLIENT_DIST.exists():
        app.mount("/", StaticFiles(directory=str(CLIENT_DIST), html=True), name="client")
        logger.info("serving PWA client from %s", CLIENT_DIST)
    else:
        logger.warning(
            "client/dist not found at %s — run `npm run build` in client/ before "
            "deploying; API-only mode for now",
            CLIENT_DIST,
        )

    return app


app = create_app()
