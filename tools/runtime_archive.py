"""Portable, unambiguous runtime archive paths before inspecting their bytes."""
from pathlib import PurePosixPath
import tarfile


def validate_names(names) -> None:
    seen = set()
    for name in names:
        path = PurePosixPath(name)
        if (not path.parts or any(ord(char) < 32 or ord(char) == 127 for char in name)
                or path.is_absolute() or path.as_posix() != name
                or any(part in ('.', '..') or '\\' in part or ':' in part
                       or part.endswith((' ', '.')) for part in path.parts)):
            raise ValueError('runtime archive contains a noncanonical path')
        # Runtime archives are consumed on case-insensitive Windows/Apple hosts.
        key = name.casefold()
        if key in seen:
            raise ValueError('runtime archive contains duplicate normalized paths')
        seen.add(key)


def runtime_members(archive: tarfile.TarFile) -> dict[str, tarfile.TarInfo]:
    members = archive.getmembers()
    # Ordinary TAR writers may include directories with one trailing slash.
    # Normalize only that representation; every file name remains exact.
    names = [member.name[:-1] if member.isdir() and member.name.endswith('/')
             else member.name for member in members]
    validate_names(names)
    if any(not member.isfile() and not member.isdir() for member in members):
        raise ValueError('runtime archive contains a linked or nonregular member')
    return dict(zip(names, members))
