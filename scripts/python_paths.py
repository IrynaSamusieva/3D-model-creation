"""Resolve package directories after relocating the Python installation to D:."""
import os
import site
import sys
from pathlib import Path


def package_directories():
    prefix = Path(sys.prefix)
    candidates = [
        prefix / "Lib" / "site-packages",
        prefix / "site-packages",
        Path(sys.executable).parent / "Lib" / "site-packages",
        Path(sys.executable).parent / "site-packages",
        Path(r"D:\Python\Python314\site-packages"),
    ]
    result = []
    seen = set()
    for candidate in candidates:
        try:
            resolved = candidate.resolve()
        except OSError:
            continue
        if resolved.is_dir() and str(resolved).casefold() not in seen:
            seen.add(str(resolved).casefold())
            result.append(resolved)
    return result


def configure_python_paths():
    """Add relocated site-packages folders to imports in this process."""
    for directory in package_directories():
        site.addsitedir(str(directory))


def child_environment():
    """Pass relocated site-packages to child Python processes as well."""
    env = os.environ.copy()
    paths = [str(path) for path in package_directories()]
    existing = env.get("PYTHONPATH")
    if existing:
        paths.append(existing)
    if paths:
        env["PYTHONPATH"] = os.pathsep.join(paths)
    return env
