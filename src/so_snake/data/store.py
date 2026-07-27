"""The episode library: what is on disk, and how to get at it safely.

Thin on purpose. The store lists, loads, renames and deletes; it does not cache,
index or watch. An episode directory is the source of truth, so a run recorded
by a script on the command line shows up in the GUI without either of them
knowing about the other.
"""

from __future__ import annotations

import re
import shutil
from pathlib import Path

from .episode import (
    DEFAULT_EPISODE_ROOT,
    FRAMES_NAME,
    META_NAME,
    Episode,
    EpisodeMeta,
    read_episode,
    read_meta,
)

# Episode ids are generated, but they also arrive as URL query parameters, so
# they are validated on the way in rather than trusted. Anything outside this
# alphabet cannot name a directory.
_ID_PATTERN = re.compile(r"^[A-Za-z0-9_.-]{1,128}$")


class EpisodeStore:
    """The episodes under one root directory."""

    def __init__(self, root: Path = DEFAULT_EPISODE_ROOT) -> None:
        self.root = Path(root)

    def ensure_root(self) -> Path:
        self.root.mkdir(parents=True, exist_ok=True)
        return self.root

    def path_of(self, episode_id: str) -> Path:
        """Resolve an id to its directory, refusing anything that escapes the root."""
        if not _ID_PATTERN.match(episode_id) or episode_id in {".", ".."}:
            raise ValueError(f"invalid episode id: {episode_id!r}")
        path = (self.root / episode_id).resolve()
        if path.parent != self.root.resolve():
            raise ValueError(f"episode id escapes the store root: {episode_id!r}")
        return path

    def exists(self, episode_id: str) -> bool:
        try:
            path = self.path_of(episode_id)
        except ValueError:
            return False
        return (path / META_NAME).is_file() and (path / FRAMES_NAME).is_file()

    def list_meta(self) -> list[EpisodeMeta]:
        """Newest first. Directories that fail to parse are skipped, not fatal.

        A single corrupt episode must not make the library unreadable -- the
        other twenty takes from that session are still good.
        """
        if not self.root.is_dir():
            return []
        out: list[EpisodeMeta] = []
        for path in sorted(self.root.iterdir()):
            if not path.is_dir() or not (path / META_NAME).is_file():
                continue
            try:
                out.append(read_meta(path))
            except (ValueError, OSError, UnicodeDecodeError):
                continue
        out.sort(key=lambda m: (m.created_at, m.id), reverse=True)
        return out

    def load(self, episode_id: str) -> Episode:
        path = self.path_of(episode_id)
        if not (path / META_NAME).is_file():
            raise FileNotFoundError(f"no such episode: {episode_id}")
        return read_episode(path)

    def load_meta(self, episode_id: str) -> EpisodeMeta:
        return read_meta(self.path_of(episode_id))

    def delete(self, episode_id: str) -> bool:
        path = self.path_of(episode_id)
        if not path.is_dir():
            return False
        shutil.rmtree(path)
        return True

    def annotate(
        self,
        episode_id: str,
        *,
        name: str | None = None,
        task: str | None = None,
        notes: str | None = None,
    ) -> EpisodeMeta:
        """Edit the labels of an already-recorded episode.

        Only the labels. The frames and the config snapshot are what happened,
        and are not editable through here.
        """
        import json

        path = self.path_of(episode_id)
        meta = read_meta(path)
        if name is not None:
            meta.name = name
        if task is not None:
            meta.task = task
        if notes is not None:
            meta.notes = notes
        (path / META_NAME).write_text(
            json.dumps(meta.to_json(), indent=2, ensure_ascii=False), encoding="utf-8"
        )
        return meta

    def disk_usage_bytes(self, episode_id: str) -> int:
        path = self.path_of(episode_id)
        return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
