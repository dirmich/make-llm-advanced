"""로깅 (basic과 동일)."""
from __future__ import annotations
import sys, time
from typing import Any


class Logger:
    def __init__(self, name: str = "makellm-adv", verbose: bool = True):
        self.name = name
        self.verbose = verbose
        self._start = time.time()

    def _fmt(self, level: str, msg: str) -> str:
        elapsed = time.time() - self._start
        m, s = divmod(int(elapsed), 60)
        h, m = divmod(m, 60)
        return f"[{h:02d}:{m:02d}:{s:02d}] {level} | {msg}"

    def info(self, msg: Any) -> None:
        if self.verbose:
            print(self._fmt("INFO", str(msg)), flush=True)

    def warn(self, msg: Any) -> None:
        print(self._fmt("WARN", str(msg)), file=sys.stderr, flush=True)

    def error(self, msg: Any) -> None:
        print(self._fmt("ERR ", str(msg)), file=sys.stderr, flush=True)
