# vLLM deployment

This directory contains launch scripts for independent vLLM servers, one per
GPU. The Python app only calls OpenAI-compatible endpoints; it does not manage
model server processes.

All non-secret deployment values come from the repository `config.yaml`.
The scripts derive the model from `models.local_retrieval`, the instance count
and ports from `retrieval.local_model_count` and `retrieval.local_base_urls`,
and vLLM launch settings from `deploy`.

On a fresh GPU node, run:

```bash
deploy/setup-node.sh
```

Then start the servers:

```bash
deploy/start-vllm.sh
```

The default endpoints are:

```text
http://127.0.0.1:8001/v1
http://127.0.0.1:8002/v1
http://127.0.0.1:8003/v1
http://127.0.0.1:8004/v1
```

For fewer GPUs, edit `config.yaml` so the model count and endpoints match the
node. For two GPUs:

```yaml
retrieval:
  local_model_count: 2
  local_base_urls:
    - http://127.0.0.1:8001/v1
    - http://127.0.0.1:8002/v1
```

The scripts map endpoint `N` to GPU `N`. Do not set vLLM tensor parallelism
here; this project expects one full model copy per GPU.

Check all configured instances:

```bash
deploy/healthcheck.sh
```

Run the agent against local retrieval:

```bash
~/.local/bin/micromamba run -n graduate-school-agent grad-agent \
    --schools input/schools.json \
    --cv input/cv.md \
    --context input/context.md \
    --retrieval-backend local_qwen_vllm
```

Set `ANTHROPIC_API_KEY` and `BRAVE_API_KEY` in the shell or root `.env` before
running the agent. Set `VLLM_API_KEY` only if you start vLLM with API-key
enforcement.
