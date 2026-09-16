# Development container

The container installs OpenCode `1.18.31` and bind-mounts the host's OpenCode
configuration and account files so it uses the same authenticated accounts
without copying credentials into the image or repository.

It also provides Python 3, pip, and venv. Post-create setup creates a project-local
`.venv` and installs this project in editable mode into that environment.

Required host paths:

- `/mnt/d/AdeptusMechanicus/LLMs` is mounted at
  `/workspaces/llm-provider/LLMs`.
- `$HOME/.config/opencode` is mounted read-only at
  `/home/vscode/.config/opencode`.
- `$HOME/.local/share/opencode/auth.json` and `account.json` are mounted at the
  equivalent paths under `/home/vscode/.local/share/opencode`.

The account files remain writable so OpenCode can refresh authentication. Such
updates affect the host files. Session databases, logs, snapshots, and tool
output are not shared with the host, avoiding concurrent database access and
unnecessary exposure of host session history.

The post-create check fails if the mounted `auth.json` is absent or empty. The
configuration assumes Dev Containers is launched from the same WSL environment
where `/mnt/d/AdeptusMechanicus/LLMs` and `$HOME/.local/share/opencode` exist.
