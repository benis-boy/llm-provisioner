#!/usr/bin/env python3
"""Provision exact offline model artifacts into a read-only runtime volume."""
from __future__ import annotations

import argparse
from pathlib import Path

from services.llm.provisioning.volume import provision


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--model", action="append", required=True, metavar="MODEL=ROOT")
    args = parser.parse_args()
    roots = {}
    for value in args.model:
        model, separator, root = value.partition("=")
        if not separator or not model or not root or model in roots:
            parser.error("--model must be unique MODEL=ROOT")
        roots[model] = root
    document = provision(roots, args.output)
    print(f"digest={document['manifest_sha256']} models={len(document['models'])} output={args.output / 'current'}")
    for model in sorted(document["models"]):
        print(f"model={model} files={len(document['models'][model]['files'])} root=models/{model}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
