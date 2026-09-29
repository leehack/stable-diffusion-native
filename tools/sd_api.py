"""Public API symbols declared with `SD_API` in `stable-diffusion.h`."""

from __future__ import annotations

import re
from pathlib import Path

_COMMENT = re.compile(r"//[^\n]*|/\*.*?\*/", re.S)
_PREPROCESSOR = re.compile(r"^\s*#.*$", re.M)
_DECLARATION = re.compile(r"\bSD_API\b([^;{]*?)(\w+)\s*(\(|\[)", re.S)


def api_symbols(header: Path) -> list[str]:
    """Returns the sorted function and data names the header exports."""
    source = _PREPROCESSOR.sub("", _COMMENT.sub("", header.read_text()))
    return sorted({m.group(2) for m in _DECLARATION.finditer(source)})
