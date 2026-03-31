from __future__ import annotations

from dataclasses import dataclass
import subprocess
import sys
import tempfile
from pathlib import Path


@dataclass(slots=True)
class ExecutionResult:
    returncode: int
    stdout: str
    stderr: str
    timed_out: bool


class PythonExecutor:
    """Simple bounded Python executor used for verifier-path scaffolding."""

    def __init__(self, timeout_seconds: float = 2.0) -> None:
        self.timeout_seconds = timeout_seconds

    def execute(self, code: str) -> ExecutionResult:
        with tempfile.TemporaryDirectory(prefix="ace_atlas_exec_") as tmp_dir:
            script_path = Path(tmp_dir) / "main.py"
            script_path.write_text(code, encoding="utf-8")
            try:
                result = subprocess.run(
                    [sys.executable, str(script_path)],
                    capture_output=True,
                    text=True,
                    timeout=self.timeout_seconds,
                    check=False,
                )
                return ExecutionResult(
                    returncode=result.returncode,
                    stdout=result.stdout,
                    stderr=result.stderr,
                    timed_out=False,
                )
            except subprocess.TimeoutExpired as exc:
                return ExecutionResult(
                    returncode=-1,
                    stdout=exc.stdout or "",
                    stderr=exc.stderr or "",
                    timed_out=True,
                )

