"""Public API symbols declared with `SD_API` in the shipped headers."""

from __future__ import annotations

import re
from pathlib import Path

_COMMENT = re.compile(r"//[^\n]*|/\*.*?\*/", re.S)
_PREPROCESSOR = re.compile(r"^\s*#.*$", re.M)
_DECLARATION = re.compile(r"\bSD_API\b([^;{]*?)(\w+)\s*(\(|\[)", re.S)


def api_symbols(*headers: Path) -> list[str]:
    """Returns the sorted function and data names the headers export."""
    symbols: set[str] = set()
    for header in headers:
        source = _PREPROCESSOR.sub("", _COMMENT.sub("", header.read_text()))
        symbols.update(m.group(2) for m in _DECLARATION.finditer(source))
    return sorted(symbols)
