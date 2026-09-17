#!/bin/sh
set -eu
if [ "$#" -eq 1 ] && [ "$1" = "--help" ]; then
    exec /opt/venv/bin/python -m services.llm.bootstrap --help
fi
echo "refusing runtime serve: distinct llm/ollama supervision is not implemented" >&2
echo "this image foundation only supports import/help smoke checks" >&2
exit 78
