# Vast.ai vLLM deployment

This directory contains thin launch scripts for four independent vLLM servers,
one per GPU. The Python app only calls OpenAI-compatible endpoints; it does not
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

The default endpoints match `config.yaml`:

```text
http://127.0.0.1:8001/v1
http://127.0.0.1:8002/v1
http://127.0.0.1:8003/v1
http://127.0.0.1:8004/v1
```

Check all four instances:

```bash
./healthcheck.sh
```

Run the agent against local retrieval:

```bash
grad-agent --schools input/schools.json --cv input/cv.md \
  --retrieval-backend local_qwen_vllm --max-parallel 4
```

Set `VLLM_API_KEY` only if you start vLLM with API-key enforcement.
