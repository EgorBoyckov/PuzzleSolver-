"""Загрузка config.yaml — единственного источника порогов и весов конвейера.

Любой модуль сервера получает свои настройки через get_config(), а не через
захардкоженные константы. Путь к config.yaml можно переопределить переменной
окружения PUZZLEVISION_CONFIG (используется в Docker и в тестах).
"""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG_PATH = REPO_ROOT / "config" / "config.yaml"


class ServerConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

    host: str = "0.0.0.0"
    port: int = 8443
    tls_certfile: str = "certs/server.crt"
    tls_keyfile: str = "certs/server.key"
    data_dir: str = "data/puzzles"
    debug_dir: str = "data/debug"
    debug_images: bool = True
    database_url: str = "sqlite:///data/puzzlevision.db"
    cors_allow_origins: list[str] = ["*"]


class AppConfig(BaseModel):
    """Типизированный доступ к известным разделам + сырые данные для остальных.

    Секции алгоритмов (segment/describe/locate/match/...) описываются своими
    pydantic-моделями по мере реализации соответствующих этапов; до этого
    они читаются как обычные dict через `raw`.
    """

    model_config = ConfigDict(extra="allow")

    server: ServerConfig = ServerConfig()
    raw: dict[str, Any] = {}

    def section(self, name: str) -> dict[str, Any]:
        """Вернуть сырой словарь раздела конфига (например, "aruco", "preprocess")."""
        return self.raw.get(name, {})

    def path(self, relative: str) -> Path:
        """Разрешить путь из конфига относительно корня репозитория."""
        p = Path(relative)
        return p if p.is_absolute() else REPO_ROOT / p


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


@lru_cache(maxsize=1)
def get_config() -> AppConfig:
    config_path = Path(os.environ.get("PUZZLEVISION_CONFIG", str(DEFAULT_CONFIG_PATH)))
    raw = _load_yaml(config_path)
    server_raw = raw.get("server", {})
    return AppConfig(server=ServerConfig(**server_raw), raw=raw)


def reload_config_for_tests(config_path: Path) -> AppConfig:
    """Сбросить кэш и загрузить конфиг из другого пути — используется в тестах."""
    get_config.cache_clear()
    os.environ["PUZZLEVISION_CONFIG"] = str(config_path)
    return get_config()
