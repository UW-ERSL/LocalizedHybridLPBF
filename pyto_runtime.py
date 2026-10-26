"""Resolve one PyTO checkout and record the actual numerical environment."""
from pathlib import Path
import importlib.metadata
import os
import subprocess
import sys
import hashlib

PYTO_COMMIT = "1d4ce75e8092429ee4ec31fd09c0ef738773a7ea"
_SOURCE = None


def configure_pyto(source=None):
    global _SOURCE
    requested = source or os.environ.get("PYTO_SRC") or _SOURCE
    if requested is None:
        requested = Path(__file__).resolve().parent.parent / "PyTO" / "src"
    path = Path(requested).expanduser().resolve()
    required = ("linear_solvers", "bound_cond", "tet_transient_thermal",
                "tet_mesher", "mat_lib", "tet_thermal_fea", "stl_reader")
    missing = [name + ".py" for name in required if not (path / (name + ".py")).is_file()]
    if missing:
        raise ImportError(f"Invalid PyTO source directory {path}; missing {missing}. "
                          "Set PYTO_SRC or paths.pyto_source_directory to PyTO/src.")
    if _SOURCE is not None and path != _SOURCE:
        raise ImportError("PyTO is already configured from another checkout; start a fresh Python process.")
    for name in required + ("tet_thermal_examples",):
        module = sys.modules.get(name)
        if module is not None:
            actual = Path(module.__file__).resolve()
            if actual.parent != path:
                raise ImportError(f"{name} was imported from {actual}, not {path}; restart Python.")
    if str(path) in sys.path:
        sys.path.remove(str(path))
    sys.path.insert(0, str(path))
    _SOURCE = path
    return path


def environment_metadata():
    versions = {}
    for package in ("numpy", "scipy", "torch", "tetgen", "pyvista", "numba", "pyamg", "pypardiso"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    commit = None
    dirty = None
    if _SOURCE is not None:
        try:
            result = subprocess.run(["git", "-C", str(_SOURCE), "rev-parse", "HEAD"],
                                    capture_output=True, text=True, check=True)
            commit = result.stdout.strip()
            result = subprocess.run(["git", "-C", str(_SOURCE), "status", "--porcelain"],
                                    capture_output=True, text=True, check=True)
            dirty = bool(result.stdout.strip())
        except (OSError, subprocess.CalledProcessError):
            pass
    source_hashes = {}
    if _SOURCE is not None:
        for path in _SOURCE.glob("*.py"):
            source_hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    return {"python": sys.version, "packages": versions,
            "pyto_source_directory": str(_SOURCE), "pyto_commit": commit,
            "pyto_dirty": dirty, "pyto_source_sha256": source_hashes,
            "supported_pyto_commit": PYTO_COMMIT}
