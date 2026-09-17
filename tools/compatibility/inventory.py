"""CLI for explicit-root artifact inventory and manifest generation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from services.llm.provisioning.artifacts import inventory, manifest, spec  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smollm-root", required=True, type=Path)
    parser.add_argument("--coedit-root", required=True, type=Path)
    parser.add_argument("--gector-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--staged-root", type=Path, help="rewrite manifest roots for a staged image")
    args = parser.parse_args()
    entries = {model: inventory(spec(model, root)) for model, root in (
        ("SmolLM", args.smollm_root), ("CoEdIT", args.coedit_root), ("GECToR", args.gector_root))}
    result = manifest(entries)
    if args.staged_root:
        # Keep the selected relative layout while making the image manifest self-contained.
        for entry in result["models"].values():
            entry["root"] = str(args.staged_root / entry["model_id"])
        result = manifest(result["models"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
