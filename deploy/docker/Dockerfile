# Foundation candidate only: CUDA and Ollama are pinned, OS package snapshot is not.
ARG CUDA_IMAGE=nvidia/cuda:12.8.1-runtime-ubuntu24.04@sha256:ebef3c171eeef0298e4eb2e4be843105edf3b8b0ac45e0b43acee358e8046867
FROM ${CUDA_IMAGE} AS python-build
ENV DEBIAN_FRONTEND=noninteractive PIP_DISABLE_PIP_VERSION_CHECK=1
RUN apt-get update && apt-get install -y --no-install-recommends python3 python3-venv ca-certificates libgomp1 \
    && rm -rf /var/lib/apt/lists/*
COPY deploy/docker/.inputs/requirements.lock /tmp/requirements.lock
COPY deploy/docker/.inputs/wheelhouse/ /tmp/wheelhouse/
RUN python3 -m venv /opt/venv \
    && /opt/venv/bin/python -m pip install --no-index --no-cache-dir --require-hashes \
       --find-links=/tmp/wheelhouse -r /tmp/requirements.lock \
    && rm -rf /tmp/wheelhouse /tmp/requirements.lock

FROM ${CUDA_IMAGE} AS ollama
COPY deploy/docker/.inputs/ollama-linux-amd64.tgz /tmp/ollama.tgz
RUN echo "fa9608a428c7d6bd46fce69f7016242feedf60fba1a57d3ee767257b071216f6  /tmp/ollama.tgz" | sha256sum -c - \
    && mkdir /out && tar -xzf /tmp/ollama.tgz -C /out \
    && test -x /out/bin/ollama \
    && test -d /out/lib/ollama \
    && rm /tmp/ollama.tgz

FROM ${CUDA_IMAGE}
ENV HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 PYTHONUNBUFFERED=1 \
    OLLAMA_HOST=127.0.0.1:11434 OLLAMA_MODELS=/var/lib/ollama/models
RUN apt-get update && apt-get install -y --no-install-recommends python3 ca-certificates libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --shell /usr/sbin/nologin llm \
    && useradd --create-home --shell /usr/sbin/nologin ollama \
    && mkdir -p /srv/llm/state /srv/llm/results /srv/llm/ollama-home /var/lib/ollama/models \
    && chown -R llm:llm /srv/llm/state /srv/llm/results \
    && chown -R ollama:ollama /srv/llm/ollama-home /var/lib/ollama
COPY --from=python-build /opt/venv /opt/venv
COPY --from=ollama /out/ /usr/local/
COPY services/ /opt/llm/services/
COPY pyproject.toml /opt/llm/pyproject.toml
COPY deploy/docker/entrypoint.sh /usr/local/bin/llm-foundation
RUN chmod 0555 /usr/local/bin/llm-foundation \
    && chown -R root:root /opt/venv /opt/llm \
    && chmod -R a-w /opt/venv /opt/llm
VOLUME ["/srv/llm/ollama-home", "/var/lib/ollama/models"]
USER llm
WORKDIR /opt/llm
ENTRYPOINT ["/usr/local/bin/llm-foundation"]
CMD ["--help"]
