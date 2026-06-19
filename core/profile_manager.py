"""Profile Manager - manages multiple TTS role profiles stored in a JSON file."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

from astrbot.api import logger


class ProfileManager:
    """Manages TTS role profiles.

    Each profile is a complete snapshot of: model, default_params, entry_storage.
    Profiles are stored in a JSON file in the plugin's data directory.
    """

    def __init__(self, data_dir: Path):
        self._file = data_dir / "profiles.json"
        self._profiles: dict[str, dict[str, Any]] = {}
        self._load()

    def _load(self) -> None:
        if self._file.exists():
            try:
                with self._file.open("r", encoding="utf-8") as f:
                    self._profiles = json.load(f)
                logger.info(f"已从 {self._file} 加载 {len(self._profiles)} 个角色")
            except Exception as e:
                logger.error(f"加载角色配置失败: {e}")
                self._profiles = {}
        else:
            self._profiles = {}

    def _save(self) -> None:
        self._file.parent.mkdir(parents=True, exist_ok=True)
        with self._file.open("w", encoding="utf-8") as f:
            json.dump(self._profiles, f, ensure_ascii=False, indent=2)

    def list_profiles(self) -> list[str]:
        """Return list of profile names."""
        return list(self._profiles.keys())

    def get_profile(self, name: str) -> dict[str, Any] | None:
        """Return a deep copy of the named profile, or None."""
        profile = self._profiles.get(name)
        return copy.deepcopy(profile) if profile else None

    def save_profile(
        self,
        name: str,
        model: dict[str, Any],
        default_params: dict[str, Any],
        entry_storage: list[dict[str, Any]],
    ) -> None:
        """Save current config as a named profile."""
        self._profiles[name] = {
            "model": copy.deepcopy(model),
            "default_params": copy.deepcopy(default_params),
            "entry_storage": copy.deepcopy(entry_storage),
        }
        self._save()
        logger.info(f"已保存角色: {name}")

    def delete_profile(self, name: str) -> bool:
        """Delete a named profile. Returns True if deleted."""
        if name in self._profiles:
            del self._profiles[name]
            self._save()
            logger.info(f"已删除角色: {name}")
            return True
        return False

    def exists(self, name: str) -> bool:
        return name in self._profiles