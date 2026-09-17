"""Refresh generated executable files without re-copying verified model inputs."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import tempfile

OWNER = ".compatibility-spike-owned"


def _replace(source: Path, destination: Path) -> None:
    with tempfile.NamedTemporaryFile(prefix=f".{destination.name}.", dir=destination.parent, delete=False) as temporary:
        temporary_path = Path(temporary.name)
    try:
        shutil.copy2(source, temporary_path)
        os.replace(temporary_path, destination)
    finally:
        temporary_path.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if not (output / OWNER).is_file():
        raise SystemExit(f"refusing to refresh non-owned output directory: {output}")
    source = Path(__file__).parent
    # Model inputs, wheelhouse, manifest, and provenance are intentionally untouched.
    for name in ("spike.py", "rm_spike.py", "model_runtime.py", "input_bounds.py", "Dockerfile"):
        _replace(source / name, output / name)
    for relative in (
        "services/__init__.py", "services/llm/__init__.py",
        "services/llm/providers/__init__.py", "services/llm/providers/input_bounds.py",
        "services/llm/queue/__init__.py", "services/llm/queue/contracts.py",
        "services/llm/resource_manager/__init__.py",
        "services/llm/resource_manager/contracts.py",
        "services/llm/resource_manager/protocol.py",
        "services/llm/resource_manager/state.py",
        "services/llm/resource_manager/core.py",
    ):
        destination = output / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        _replace(Path(__file__).parents[2] / relative, destination)
    _replace(Path(__file__).parents[2] / "services/llm/provisioning/artifacts.py", output / "artifacts.py")
    print(f"refreshed executable context: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
