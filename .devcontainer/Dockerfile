FROM mcr.microsoft.com/devcontainers/base:ubuntu-24.04

ARG OPENCODE_VERSION=1.18.31

RUN apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        ca-certificates \
        curl \
        git \
        nodejs \
        npm \
        python3 \
        python3-pip \
        python3-venv \
    && npm install --global "opencode-ai@${OPENCODE_VERSION}" \
    && npm cache clean --force \
    && rm -rf /var/lib/apt/lists/*

RUN mkdir -p \
        /home/vscode/.config/opencode \
        /home/vscode/.local/share/opencode \
    && chown -R vscode:vscode \
        /home/vscode/.config \
        /home/vscode/.local
