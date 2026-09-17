"""Prepare a hash-locked aiohttp wheelhouse outside the offline image build."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
from pathlib import Path
import subprocess
import sys
import zipfile


def _wheel_identity(path: Path) -> tuple[str, str]:
    with zipfile.ZipFile(path) as wheel:
        metadata = next(name for name in wheel.namelist() if name.endswith(".dist-info/METADATA"))
        fields = {}
        for line in wheel.read(metadata).decode("utf-8").splitlines():
            if ": " in line:
                key, value = line.split(": ", 1)
                if key in {"Name", "Version"}: fields[key] = value
        if set(fields) != {"Name", "Version"}: raise ValueError(f"invalid wheel metadata: {path.name}")
        return fields["Name"], fields["Version"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wheelhouse", type=Path, required=True)
    parser.add_argument("--lock", type=Path, required=True)
    args = parser.parse_args()
    args.wheelhouse.mkdir(parents=True, exist_ok=True)
    version = importlib.metadata.version("aiohttp")
    subprocess.run([sys.executable, "-m", "pip", "download", "--only-binary=:all:",
                    "--dest", str(args.wheelhouse), f"aiohttp=={version}"], check=True)
    entries = []
    for wheel in sorted(args.wheelhouse.glob("*.whl")):
        name, wheel_version = _wheel_identity(wheel)
        digest = hashlib.sha256(wheel.read_bytes()).hexdigest()
        entries.append((name.lower().replace("_", "-"), wheel_version, digest))
    if not entries: raise ValueError("no wheels downloaded")
    args.lock.parent.mkdir(parents=True, exist_ok=True)
    args.lock.write_text("".join(f"{name}=={version} --hash=sha256:{digest}\n"
                                 for name, version, digest in sorted(entries)), encoding="ascii")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
