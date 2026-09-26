# Production image: vLLM OpenAI-compatible server with our model + chat template baked in.
#   docker build -f serve/Dockerfile -t toolace-qwen3-8b:fp8 --build-arg MODEL_DIR=models/lora_r16-fp8 .
#   docker run --gpus all -p 8000:8000 toolace-qwen3-8b:fp8
FROM vllm/vllm-openai:v0.30.0
ARG MODEL_DIR
COPY ${MODEL_DIR} /models/toolace
COPY serve/chat_template.jinja /app/chat_template.jinja
ENV HF_HUB_OFFLINE=1
HEALTHCHECK --interval=30s --timeout=5s --start-period=300s CMD curl -sf http://localhost:8000/health || exit 1
ENTRYPOINT ["vllm", "serve", "/models/toolace", \
  "--served-model-name", "toolace-qwen3-8b", \
  "--host", "0.0.0.0", "--port", "8000", \
  "--max-model-len", "32768", "--max-num-seqs", "64", "--gpu-memory-utilization", "0.90", \
  "--enable-prefix-caching", "--enable-chunked-prefill", \
  "--chat-template", "/app/chat_template.jinja", \
  "--enable-auto-tool-choice", "--tool-call-parser", "pythonic", \
  "--generation-config", "vllm"]
