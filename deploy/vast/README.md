# Vast.ai vLLM deployment

This directory contains thin launch scripts for independent vLLM servers, one
per GPU. The Python app only calls OpenAI-compatible endpoints; it does not
manage model server processes.

Install vLLM on the node, then start the servers:

```bash
cd deploy/vast
cp env.example .env
set -a
. ./.env
set +a
./start-vllm.sh
```

`MODEL_COUNT` controls how many one-GPU model copies to launch. The default
endpoints match `config.yaml`:

```text
http://127.0.0.1:8001/v1
http://127.0.0.1:8002/v1
http://127.0.0.1:8003/v1
http://127.0.0.1:8004/v1
```

For fewer GPUs, set both the deployment count and agent endpoints. For two GPUs:

```bash
MODEL_COUNT=2 ./start-vllm.sh
```

```yaml
retrieval:
  local_model_count: 2
  local_base_urls:
    - http://127.0.0.1:8001/v1
    - http://127.0.0.1:8002/v1
```

The scripts map model copy `N` to GPU `N` and port `START_PORT + N`. Do not set
vLLM tensor parallelism here; this project expects one full model copy per GPU.

Check all configured instances:

```bash
./healthcheck.sh
```

Run the agent against local retrieval:

```bash
grad-agent --schools input/schools.json --cv input/cv.md \
  --retrieval-backend local_qwen_vllm
```

Set `VLLM_API_KEY` only if you start vLLM with API-key enforcement.
