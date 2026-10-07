# Development container

The container installs the pinned OpenCode v2 CLI `@opencode/cli@2.0.24`.
Configuration and mutable OpenCode state are isolated from the host. A read-only
credential-transfer directory is mounted without copying credentials into the
image or repository.

The npm package installs the `opencode` command (and its `opencode2` alias) and
selects the Linux native binary during post-install. Rebuild the container after
changing the pinned version (for example, **Dev Containers: Rebuild Container**)
so the image, rather than the currently running container, receives the upgrade.

It also provides Python 3, pip, and venv. Post-create setup creates a project-local
`.venv` and installs this project in editable mode into that environment.

Required host paths:

- `/mnt/d/AdeptusMechanicus/LLMs` is mounted at
  `/workspaces/llm-provider/LLMs`.
- The named Docker volume `llm-provider-opencode-data` is mounted at
  `/home/vscode/.local/share/opencode`.
- The Windows host data directory at
  `/mnt/c/Users/benja/.local/share/opencode` is mounted read-only at
  `/mnt/host-opencode`, solely to read a deliberately exported credential file.

The devcontainer is launched through WSL, where `${localEnv:HOME}` resolves to
`/home/ben`, not the Windows profile that contains the existing OpenCode
credential database. The Windows path above is explicit so the container can
read a user-created transfer file. If the Windows profile changes, update that
mount source.

OpenCode v2 stores active credentials in its SQLite database; `auth.json` is
only a legacy, one-time import source. Do not use the Windows data directory as
the container's active data directory: Docker exposes this bind as `v9fs`, and
the OpenCode service's SQLite bootstrap fails with `disk I/O error` there. The
Docker volume provides native Linux filesystem semantics for the database, WAL,
SHM, sessions, logs, snapshots, and tool output. The active configuration is
also container-owned: the host configuration contains service connection state
and is read-only over `v9fs`. Container login state, service connection state,
and history are therefore isolated from the host.

After rebuilding, transfer the existing host authentication with OpenCode's
supported export/import flow. Run the following in host PowerShell. It exports
only GitHub Copilot, without displaying the sensitive JSON; `cmd` redirection
preserves UTF-8 bytes even on Windows PowerShell 5.1:

```powershell
cmd.exe /d /c 'opencode auth export github-copilot > "%USERPROFILE%\.local\share\opencode\devcontainer-auth.json"'
```

Then, from the rebuilt container, import the file and verify only the integration
list:

```sh
opencode auth import /mnt/host-opencode/devcontainer-auth.json
opencode auth list
```

After confirming the import, delete the transfer file in host PowerShell:

```powershell
Remove-Item "$env:USERPROFILE\.local\share\opencode\devcontainer-auth.json"
```

Treat it as a credential-bearing file: do not display, commit, copy into
the repository, or leave it in shared storage. Rebuild the container after
changing this configuration so the updated mounts and post-create check apply.

The repository configuration uses the v2 `permissions` rule array and `agents`
map. The project Markdown agents use ordered `permissions` rules; v1 `bash` and
`task` action names are migrated to v2 `shell` and `subagent`. Wildcards remain
before specific exceptions, so the design agent can load `goal-oriented-design`,
the reviewer can inspect Git without shell access generally, and the tester
edit exceptions remain limited to test files. The design default, subagent
depth, skill rules, and disabled built-in agents are preserved. OpenCode v2 uses
the same standard config and data locations. Host configuration contents may
still require their own v2 migration; this isolated container does not consume
them.

Run OpenCode from the repository root so it can discover `.opencode/agents`.
Version checks alone do not prove effective agent permissions, and a model's
self-reported permissions are not a security-policy verification.

Legacy per-agent temperatures are retained under `request.body.temperature`,
but v2 currently does not send agent request overlays to providers. They are
not effective sampling settings until upstream supports them; provider/model
settings must be used if a specific temperature is required.

The repository's `.opencode/package.json` still pins the v1
`@opencode-ai/plugin` dependency. It was intentionally not changed: this
upgrade does not establish that a project plugin is active or that a compatible
v2 plugin release is required. If the plugin is used, verify its compatibility
separately before upgrading it.

The post-create check verifies that the mounted data directory exists and is
readable and writable; it does not require `auth.json`. The configuration
assumes Dev Containers is launched from WSL with access to
`/mnt/d/AdeptusMechanicus/LLMs` and the Windows-profile paths listed above.
