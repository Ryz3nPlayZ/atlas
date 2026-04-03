from __future__ import annotations


MODE_GENERAL = 0
MODE_CODE = 1
MODE_ANSWER = 2

MODE_NAME_TO_ID = {
    "general": MODE_GENERAL,
    "code": MODE_CODE,
    "answer": MODE_ANSWER,
}


def resolve_mode_id(name: str) -> int:
    try:
        return MODE_NAME_TO_ID[name]
    except KeyError as exc:
        raise ValueError(f"Unsupported mode name: {name}") from exc
