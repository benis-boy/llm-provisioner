#!/bin/sh
set -eu
exec /opt/venv/bin/python -m services.llm.bootstrap "$@"
