from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator

from ace_atlas.data.manifest import DatasetEntry, DatasetManifest


def iter_texts(entry: DatasetEntry) -> Iterator[str]:
    path = Path(entry.path)
    if entry.format == "txt":
        yield path.read_text(encoding="utf-8")
        return
    if entry.format == "jsonl":
        if not entry.text_key:
            raise ValueError(f"jsonl entry requires text_key: {entry.path}")
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            value = record.get(entry.text_key)
            if value:
                yield str(value)
        return
    raise ValueError(f"Unsupported entry format: {entry.format}")


def iter_manifest_texts(manifest: DatasetManifest, split: str | None = None) -> Iterator[str]:
    for entry in manifest.entries:
        if split is not None and entry.split != split:
            continue
        yield from iter_texts(entry)

