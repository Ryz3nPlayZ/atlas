from pathlib import Path

from ace_atlas.data.manifest import DatasetManifest, infer_entry
from ace_atlas.tokenizer.byte_level import ByteTokenizer
from ace_atlas.train.data import TokenizedJsonlDataset


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


def test_tokenized_jsonl_dataset_builds_lm_examples(tmp_path: Path) -> None:
    tokenized_path = tmp_path / "train_tokens.jsonl"
    tokenized_path.write_text(
        '{"tokens": [1, 2, 3, 4, 5]}\n{"tokens": [10, 11, 12, 13, 14, 15, 16, 17, 18]}\n',
        encoding="utf-8",
    )

    dataset = TokenizedJsonlDataset(tokenized_path, sequence_length=4)

    assert len(dataset) == 3
    first = dataset[0]
    second = dataset[1]
    third = dataset[2]
    assert first["input_ids"].tolist() == [1, 2, 3, 4]
    assert first["labels"].tolist() == [2, 3, 4, 5]
    assert second["input_ids"].tolist() == [10, 11, 12, 13]
    assert second["labels"].tolist() == [11, 12, 13, 14]
    assert third["input_ids"].tolist() == [14, 15, 16, 17]
    assert third["labels"].tolist() == [15, 16, 17, 18]


def test_tokenized_jsonl_dataset_applies_loss_mask(tmp_path: Path) -> None:
    tokenized_path = tmp_path / "masked_tokens.jsonl"
    tokenized_path.write_text(
        '{"tokens": [1, 2, 3, 4, 5], "loss_mask": [0, 0, 1, 1, 1]}\n',
        encoding="utf-8",
    )

    dataset = TokenizedJsonlDataset(tokenized_path, sequence_length=4)
    record = dataset[0]

    assert record["input_ids"].tolist() == [1, 2, 3, 4]
    assert record["labels"].tolist() == [-100, 3, 4, 5]
