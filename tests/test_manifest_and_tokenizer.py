from pathlib import Path

from ace_atlas.data.manifest import DatasetManifest, infer_entry
from ace_atlas.tokenizer.byte_level import ByteTokenizer


def test_byte_tokenizer_roundtrip() -> None:
    tokenizer = ByteTokenizer()
    text = "hello"
    tokens = tokenizer.encode(text)
    assert tokens[-1] == tokenizer.eos_token_id
    assert tokenizer.decode(tokens) == text


def test_manifest_roundtrip(tmp_path: Path) -> None:
    text_path = tmp_path / "sample.txt"
    text_path.write_text("abc", encoding="utf-8")
    manifest = DatasetManifest(
        name="demo",
        tokenizer="byte",
        sequence_length=128,
        entries=[infer_entry(text_path)],
    )
    manifest_path = tmp_path / "manifest.json"
    manifest.save(manifest_path)
    loaded = DatasetManifest.load(manifest_path)
    assert loaded.name == "demo"
    assert loaded.entries[0].path == str(text_path)

