from __future__ import annotations

import os
import tempfile
from pathlib import Path


def _fsync_dir(dir_path: Path) -> None:
    """Best-effort directory fsync to persist rename/unlink metadata."""
    try:
        flags = os.O_RDONLY
        if hasattr(os, "O_DIRECTORY"):
            flags |= os.O_DIRECTORY
        fd = os.open(str(dir_path), flags)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    except OSError:
        # Some platforms/filesystems do not support directory fsync.
        pass


def atomic_write_text(path: Path, content: str, *, encoding: str = "utf-8") -> None:
    """Atomically write text to path and fsync file + parent directory."""
    path.parent.mkdir(parents=True, exist_ok=True)

    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding=encoding) as f:
            f.write(content)
            f.flush()
            os.fsync(f.fileno())

        os.replace(tmp_path, path)
        _fsync_dir(path.parent)
    finally:
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError:
                pass
