# Candidate hypothesis, not a production acceptance pin.
ARG CUDA_RUNTIME
FROM ${CUDA_RUNTIME}
ENV DEBIAN_FRONTEND=noninteractive HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
    HF_DATASETS_OFFLINE=1 CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 \
    OLLAMA_HOST=127.0.0.1:11434 OLLAMA_MODELS=/var/lib/ollama/models
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-venv ca-certificates curl procps \
    libgomp1 && rm -rf /var/lib/apt/lists/*
WORKDIR /opt/llm
COPY wheelhouse/ /opt/llm/wheelhouse/
COPY requirements-candidate.txt /opt/llm/
COPY requirements.lock /opt/llm/
RUN python3 -c 'import sys; assert sys.version_info >= (3, 12), sys.version' \
    && python3 -m venv /opt/venv && /opt/venv/bin/python -m pip install \
    --no-index --no-cache-dir --require-hashes --find-links=/opt/llm/wheelhouse -r /opt/llm/requirements.lock
COPY ollama-linux-amd64.tgz /tmp/ollama.tgz
RUN echo "fa9608a428c7d6bd46fce69f7016242feedf60fba1a57d3ee767257b071216f6  /tmp/ollama.tgz" | sha256sum -c - \
    && tar -xzf /tmp/ollama.tgz -C /usr --strip-components=1 \
    && rm /tmp/ollama.tgz
COPY spike.py /opt/llm/spike.py
COPY rm_spike.py /opt/llm/rm_spike.py
COPY model_runtime.py /opt/llm/model_runtime.py
COPY input_bounds.py /opt/llm/input_bounds.py
COPY artifacts.py /opt/llm/artifacts.py
COPY services/ /opt/llm/services/
COPY models/ /opt/llm/models/
COPY manifest.json /opt/llm/manifest.json
ENTRYPOINT ["/opt/venv/bin/python", "/opt/llm/spike.py", "--all", "--manifest", "/opt/llm/manifest.json", "--models-root", "/opt/llm/models"]
