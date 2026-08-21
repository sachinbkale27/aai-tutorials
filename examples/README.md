# Runnable examples

One standalone, runnable example per tutorial. Each file is **self-contained** (it does NOT import
the On-Call Copilot project) so you learn the component in isolation. Where a live API key or optional
dependency is missing, the example **degrades gracefully** (prints the structure / a clear message)
so it still runs offline.

## Run
Use the project venv (it already has most deps), or make your own:
```bash
# option A — reuse the project venv
PY=~/projects/nvidia-aai/.venv/bin/python
$PY examples/05_langgraph.py

# option B — fresh venv
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r examples/requirements.txt
python examples/05_langgraph.py
```

Each script's top docstring lists its exact `pip install` deps, whether it needs `OPENAI_API_KEY`, and
how to run it.

## Index
| Tutorial | Example |
|---|---|
| 01 Prompt engineering | `01_prompt_engineering.py` |
| 02 Tool & function calling | `02_tool_and_function_calling.py` |
| 03 MCP | `03_mcp_server.py` (+ `03_mcp_client.py`) |
| 04 Agentic RAG | `04_agentic_rag.py` |
| 05 LangGraph | `05_langgraph.py` |
| 06 Orchestrator–Worker | `06_orchestrator_worker.py` |
| 07 Human-in-the-loop | `07_human_in_the_loop.py` |
| 08 NeMo Guardrails | `08_nemo_guardrails.py` (+ `08_config/`) |
| 09 Resilience | `09_resilience.py` |
| 10 OpenTelemetry | `10_opentelemetry.py` |
| 11 Observability stack | `11_observability/` (docker-compose + emitter) |
| 12 Evaluation & regression | `12_evaluation.py` |
| 13 Config-driven design | `13_config_driven.py` (+ `13_sample.yaml`) |
| 14 SSE streaming | `14_sse_server.py` (+ curl/client note) |
| 15 RAGAS (RAG eval) | `15_ragas.py` (real run needs pinned ragas + a key; offline demo otherwise) |
| 16 LangGraph + Redis semantic cache | `16_semantic_cache/` (run `demo.py`; best with Redis Stack, in-memory fallback otherwise) |
