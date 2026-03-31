from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path


SUPPORTED_FORMATS = {"txt", "jsonl"}


@dataclass(slots=True)
class DatasetEntry:
    path: str
    format: str
    split: str = "train"
    weight: float = 1.0
    text_key: str | None = None
    note: str | None = None


@dataclass(slots=True)
class DatasetManifest:
    name: str
    tokenizer: str
    sequence_length: int
    entries: list[DatasetEntry]

    def to_dict(self) -> dict:
        return asdict(self)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def from_dict(cls, data: dict) -> "DatasetManifest":
        entries = [DatasetEntry(**entry) for entry in data.get("entries", [])]
        return cls(
            name=data["name"],
            tokenizer=data["tokenizer"],
            sequence_length=data["sequence_length"],
            entries=entries,
        )

    @classmethod
    def load(cls, path: str | Path) -> "DatasetManifest":
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.from_dict(raw)


def infer_entry(path: Path, split: str = "train", text_key: str | None = None) -> DatasetEntry:
    suffix = path.suffix.lstrip(".")
    if suffix not in SUPPORTED_FORMATS:
        raise ValueError(f"Unsupported corpus format: {path}")
    return DatasetEntry(
        path=str(path),
        format=suffix,
        split=split,
        text_key=text_key if suffix == "jsonl" else None,
    )

