"""Publish ancillary outputs without touching historical research artifacts."""
from __future__ import annotations

from contextlib import contextmanager
import os
from pathlib import Path
import shutil
import tempfile

import pandas as pd

from .provenance import protect_output


@contextmanager
def new_output_directory(path: str | Path):
    """Make an entire new directory visible only after its writer succeeds."""
    target = protect_output(path)
    if target.exists():
        raise FileExistsError(f"Output already exists: {target}; choose a new directory")
    target.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{target.name}.staging-", dir=target.parent))
    try:
        yield stage
        protect_output(target)
        if target.exists():
            raise FileExistsError(f"Output appeared during generation: {target}")
        os.rename(stage, target)
    finally:
        if stage.exists():
            shutil.rmtree(stage)


def write_new_csv(frame: pd.DataFrame, path: str | Path) -> None:
    """Atomically create a CSV; an existing file can never be replaced."""
    target = protect_output(path)
    if target.exists():
        raise FileExistsError(f"Output already exists: {target}; choose a new file")
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=target.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8-sig", newline="") as stream:
            frame.to_csv(stream, index=False)
            stream.flush()
            os.fsync(stream.fileno())
        protect_output(target)
        # Linking a sibling temporary file is atomic and fails if target exists.
        os.link(temporary, target)
    finally:
        Path(temporary).unlink(missing_ok=True)
