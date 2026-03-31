from __future__ import annotations

import json
from pathlib import Path
import platform
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def check_torch() -> dict:
    try:
        import torch  # type: ignore
    except ModuleNotFoundError:
        return {
            "installed": False,
            "cuda_available": False,
            "version": None,
        }

    return {
        "installed": True,
        "cuda_available": bool(torch.cuda.is_available()),
        "version": torch.__version__,
        "device_count": int(torch.cuda.device_count()) if torch.cuda.is_available() else 0,
    }


def main() -> None:
    report = {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": check_torch(),
        "configs_present": sorted(str(path.relative_to(ROOT)) for path in (ROOT / "configs").glob("*.json")),
        "docs_present": sorted(str(path.relative_to(ROOT)) for path in (ROOT / "docs").glob("*.md")),
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

