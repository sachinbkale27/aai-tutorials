# Agentic AI — Component Tutorials

A hands-on curriculum to master each building block of a production-style agentic system,
**one component at a time**. Every tutorial is standalone (its own minimal runnable example),
then shows how the component is used in the real **On-Call Copilot** project
(`~/projects/nvidia-aai`) so theory and practice connect.

## How to use this
Work top to bottom (each builds intuition for the next), or jump to any component you want to
drill. Each tutorial has: the mental model → a tiny runnable example → how the project uses it →
build-up variations → gotchas → **exercises to master it** → links to related components.

## The curriculum (foundations → production)

| # | Component | You'll master |
|---|---|---|
| 01 | [Prompt engineering](01-prompt-engineering.md) | System prompts, few-shot, structured/role prompts — config-driven |
| 02 | [Tool & function calling](02-tool-and-function-calling.md) | OpenAI tool schemas, the tool-call loop, structured outputs |
| 03 | [Model Context Protocol (MCP)](03-model-context-protocol.md) | Standard tool servers/clients; stdio transport |
| 04 | [Agentic RAG](04-agentic-rag.md) | Vector stores, retrieval, agent-driven "whether/what to retrieve" |
| 05 | [LangGraph](05-langgraph.md) | StateGraph, nodes/edges, shared state, interrupts |
| 06 | [Orchestrator–Worker multi-agent](06-orchestrator-worker-multi-agent.md) | Decomposition, fan-out workers, synthesis |
| 07 | [Human-in-the-loop (HITL)](07-human-in-the-loop.md) | Approval gates before irreversible actions; pause/resume |
| 08 | [NeMo Guardrails](08-nemo-guardrails.md) | Input/output/execution rails; config-driven safety |
| 09 | [Resilience](09-resilience.md) | Retries + backoff, circuit breaker, fallback, graceful degrade |
| 10 | [OpenTelemetry](10-opentelemetry.md) | Traces, spans, metrics, OTLP export |
| 11 | [Observability stack](11-observability-stack.md) | OTel Collector → Prometheus + Tempo → Grafana |
| 12 | [Evaluation & regression](12-evaluation-and-regression.md) | Golden sets, trajectory metrics, precision/recall/FPR, CI gates |
| 13 | [Config-driven design](13-config-driven-design.md) | Behavior in YAML, hot-reload, the loader pattern |
| 14 | [SSE streaming (glass-box)](14-sse-streaming.md) | Server-Sent Events, an event contract, live UIs |
| 15 | [RAGAS (RAG evaluation)](15-ragas.md) | Faithfulness, answer relevancy, context precision/recall; wiring into CI |
| 16 | [LangGraph + Redis (semantic caching)](16-langgraph-redis-semantic-caching.md) | Vector-similarity cache as a graph edge; thresholds, namespaces, TTLs |

## General best practices (project-agnostic)
See **[BEST_PRACTICES.md](BEST_PRACTICES.md)** — the cross-cutting principles for any agentic/LLM
system (start simple, eval before scale, defense-in-depth safety, reliability, observability & cost,
config-driven design, honesty). The per-tutorial `## ✅ Best Practices` sections cover the
component-specific patterns; this is the universal layer.

## Prerequisites
- Python 3.11, Node (for the SSE/UI tutorial), Docker (for the observability stack and the Redis semantic-cache tutorial).
- Reference project: `~/projects/nvidia-aai` (the On-Call Copilot). Each tutorial cites its real files.

## Suggested pace
One component per sitting. Do the exercises before moving on — they're what turn "I read it" into
"I can build it and defend it."

## Best practices for using this curriculum
- **Run the matching example, then break it.** Read the tutorial, run `examples/NN_*.py`, then change one thing and predict the result — that's where understanding sticks.
- **Read the cited project files.** Each tutorial points at real code in `~/projects/nvidia-aai`; open it. Concepts + real usage together beat either alone.
- **Do the exercises, especially the "wire it into the copilot" ones** — several advance the actual project (real tool-calling, LangGraph-as-engine, RAGAS scoring).
- **Only claim what you can defend.** Treat every tutorial as prep for a "tell me more about X" interview probe; if a topic still feels thin after the exercises, revisit before listing it as a skill.
- **Follow the Gotchas → Best Practices split** in each file: the *pitfalls* are what breaks; the *best practices* are the senior-engineer defaults to adopt.
