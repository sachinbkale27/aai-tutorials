# 10 · OpenTelemetry

> After this you can stand up a `TracerProvider` + `MeterProvider`, emit spans and counters/histograms, export them to the console or an OTLP collector via one env var, and read the exact GenAI-semconv instrumentation the On-Call Copilot uses to turn each incident into a span tree plus latency/token/cost metrics.

## 1. Mental model — signals, spans + context, API vs SDK, OTLP

**OpenTelemetry (OTel)** is one vendor-neutral standard for producing telemetry. It has three **signals**:

- **Traces** — a *trace* is one end-to-end operation (one incident, one request). It's a tree of **spans**; each span has a name, start/end time, attributes (key/value tags), events, and a parent. The parent/child links are carried in **context** — an implicit "who's my parent right now" that propagates as you nest work.
- **Metrics** — numeric aggregates over time. The two you'll use constantly: a **counter** (monotonic, "add N", e.g. tokens used, conversations) and a **histogram** (records a distribution you later slice into percentiles, e.g. latency ms).
- **Logs** — structured log records (correlatable to a trace). The Copilot uses traces + metrics; logs are the same shape if you need them.

The single most important design point: **API vs SDK split.**

- The **API** (`opentelemetry-api`, imported as `from opentelemetry import trace, metrics`) is what *library* code calls. On its own it's a **no-op** — `get_tracer(...).start_span(...)` does nothing until an SDK is installed and configured.
- The **SDK** (`opentelemetry-sdk`) is what the *application* wires up once at startup: it creates a `TracerProvider`/`MeterProvider`, attaches **processors/readers** and **exporters**, and registers them as global. From then on every API call routes through your SDK.

This is why NeMo Guardrails can emit spans without ever depending on a backend: the library uses the API, and *your app* supplies the SDK. See the docstring in `app/observability.py:1-18` — "NeMo Guardrails' OpenTelemetry tracing adapter and metrics use only the OTel *API*; the application must configure the *SDK*."

**OTLP** (OpenTelemetry Protocol) is the wire format — a gRPC (or HTTP) protocol that any OTel-instrumented app speaks and any collector/backend (Grafana Tempo, Prometheus, Arize Phoenix, Langfuse, Jaeger) understands. You export OTLP; you don't couple to a vendor SDK. The gRPC default port is **4317**.

Pipeline shape:

```
your code (API)  →  Provider  →  Processor/Reader  →  Exporter  →  Console | OTLP:4317 → Collector → Prometheus/Grafana/Tempo
   spans/metrics     (SDK)        batch vs simple      OTLP/console
```

## 2. Smallest working example

Standalone, no backend required. Set up a tracer + a meter, emit one span and one counter, and print to the console. Then flip one env var to send OTLP instead.

```bash
pip install opentelemetry-sdk opentelemetry-exporter-otlp
python otel_hello.py
```

```python
# otel_hello.py — a TracerProvider + span and a MeterProvider + counter, console by default
import os, time

from opentelemetry import trace, metrics
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor, ConsoleSpanExporter
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import (
    PeriodicExportingMetricReader, ConsoleMetricExporter,
)

# One Resource describes WHO is producing telemetry; it tags every span + metric.
resource = Resource.create({"service.name": "otel-hello"})

endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT")   # unset → console; set → OTLP

# ---- TRACES: pick an exporter, wrap it in a processor, register the provider ----
if endpoint:
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
    span_exporter = OTLPSpanExporter(endpoint=endpoint)   # honours OTEL_EXPORTER_OTLP_INSECURE
else:
    span_exporter = ConsoleSpanExporter()

tp = TracerProvider(resource=resource)
tp.add_span_processor(SimpleSpanProcessor(span_exporter))  # simple = export each span now
trace.set_tracer_provider(tp)                              # register GLOBAL — API now routes here

# ---- METRICS: a reader pulls the SDK on an interval and pushes to an exporter ----
if endpoint:
    from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter
    metric_exporter = OTLPMetricExporter(endpoint=endpoint)
else:
    metric_exporter = ConsoleMetricExporter()

reader = PeriodicExportingMetricReader(metric_exporter, export_interval_millis=5000)
metrics.set_meter_provider(MeterProvider(resource=resource, metric_readers=[reader]))

# ---- USE the API (this is what library/app code writes) ----
tracer = trace.get_tracer("otel-hello")
meter = metrics.get_meter("otel-hello")

latency = meter.create_histogram("demo.op.duration", unit="ms")
runs = meter.create_counter("demo.runs")

with tracer.start_as_current_span("do-work") as span:   # opens a span + sets context
    span.set_attribute("work.kind", "demo")
    t0 = time.time()
    time.sleep(0.05)
    with tracer.start_as_current_span("sub-step"):      # nested → child of do-work
        time.sleep(0.02)
    latency.record((time.time() - t0) * 1000, {"work.kind": "demo"})
    runs.add(1, {"work.kind": "demo"})

# metrics export on the reader's interval — force one final flush before exit
metrics.get_meter_provider().force_flush()
trace.get_tracer_provider().force_flush()
print("done — a parent span, a child span, one histogram sample, one counter tick")
```

Run it and you'll see JSON spans (`do-work` with a nested `sub-step`) and a metric block on the console. Now switch to OTLP without touching the code:

```bash
# Any OTLP collector on 4317 (e.g. `docker run ... otel/opentelemetry-collector`)
export OTEL_EXPORTER_OTLP_ENDPOINT=localhost:4317
export OTEL_EXPORTER_OTLP_INSECURE=true    # REQUIRED for plaintext gRPC to a local collector
python otel_hello.py
```

Same code, different destination. That env-var swap is the whole point of OTLP.

## 3. How the On-Call Copilot uses it

Three files split the job cleanly: **config → env → SDK setup → API emit.**

### 3a. Config drives the endpoint — `config/observability.yaml` + `app/obs_config.py`

`config/observability.yaml` says *where* telemetry goes, so switching backends is a config edit, not a code change:

```yaml
service_name: oncall-copilot
otel_backend:
  endpoint: "localhost:4317"   # OTel Collector → Prometheus → Grafana
  insecure: true               # local collector has no TLS
```

`app/obs_config.py:apply()` translates that YAML into the `OTEL_*` env vars the SDK reads — and crucially only if you haven't already set them explicitly (`obs_config.py:29-34`):

```python
if endpoint and not os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT"):
    os.environ["OTEL_EXPORTER_OTLP_ENDPOINT"] = endpoint
if backend.get("insecure") and not os.getenv("OTEL_EXPORTER_OTLP_INSECURE"):
    os.environ["OTEL_EXPORTER_OTLP_INSECURE"] = "true"   # plaintext gRPC to a local collector
```

Blank the `endpoint` and the whole thing falls back to the console exporter. `apply()` must run **before** `setup_observability()`, because setup reads those env vars.

### 3b. SDK setup — `app/observability.py`

`setup_observability()` (`observability.py:29-52`) is the app-side SDK wiring. It's a **safe no-op** if the SDK isn't installed, **idempotent** (guarded by `_DONE`), and honours `OTEL_DISABLED`. It builds one `Resource` with `service.name` + `deployment.environment` (`observability.py:48`) and hands it to both traces and metrics.

The dev-vs-prod branching is the interview-relevant part:

```python
# _setup_traces — observability.py:55-75
provider = TracerProvider(resource=resource)
if prod or endpoint:
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
    ep = endpoint or "http://localhost:4317"
    exp = OTLPSpanExporter(endpoint=ep)
    provider.add_span_processor(
        BatchSpanProcessor(exp) if prod else SimpleSpanProcessor(exp))   # prod batches
else:
    provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))
trace.set_tracer_provider(provider)
```

- **Exporter switched by `OTEL_EXPORTER_OTLP_ENDPOINT`**: no endpoint (and not prod) → `ConsoleSpanExporter`; endpoint set → `OTLPSpanExporter`.
- **Processor switched by env**: prod → `BatchSpanProcessor` (buffers spans, exports in the background — the right call under load); dev → `SimpleSpanProcessor` (exports each span synchronously — easy to read, slower).

Metrics setup (`observability.py:78-94`) always uses a `PeriodicExportingMetricReader` — OTLP in prod/endpoint, else `ConsoleMetricExporter` with `export_interval_millis=60000` (60s). `MeterProvider` is registered global with that reader.

`get_state()` reports the active config (`console` vs `OTLP <ep> (batched/dev)`) for the health/verification dashboard.

### 3c. API emit — `app/metrics.py`

The app never touches the SDK again; it just uses the API. `_otel()` (`metrics.py:24-44`) lazily grabs the global tracer + meter and creates the instruments **once**, in **GenAI semantic-convention** style:

```python
inst = {
    "latency": meter.create_histogram("gen_ai.client.operation.duration", unit="ms"),
    "ttft":    meter.create_histogram("gen_ai.server.time_to_first_token", unit="ms"),
    "tokens":  meter.create_counter("gen_ai.client.token.usage", unit="token"),
    "cost":    meter.create_counter("gen_ai.client.cost", unit="USD"),
    "convos":  meter.create_counter("support.conversations"),
    "blocks":  meter.create_counter("support.guardrail.blocks"),
}
```

Using the standard `gen_ai.*` names means any OTel-aware GenAI dashboard understands them without custom config.

**The span tree per incident.** `TraceRecorder` taps one conversation's SSE event stream, accumulating `self._spans` as `(name, start_ts, end_ts)` tuples (`metrics.py:69`). On `finalize()` → `_emit_otel()` (`metrics.py:120-154`) it builds a real waterfall:

```python
span = tracer.start_span("support.conversation", start_time=ns(self.t0))  # ROOT per incident
span.set_attribute("gen_ai.usage.input_tokens", self.tin)
span.set_attribute("gen_ai.usage.output_tokens", self.tout)
span.set_attribute("gen_ai.cost.usd", self.cost)
span.set_attribute("gen_ai.request.model", self.model or "unknown")
span.set_attribute("guardrail.blocked", self.blocked)
ctx = trace.set_span_in_context(span)                       # make it the parent context
for nm, ss, ee in self._spans:                              # child span per stage
    cs = tracer.start_span(nm or "step", context=ctx, start_time=ns(ss))
    cs.end(end_time=ns(ee))                                 # retrieve.* / mcp.* / guardrail.*
span.end(end_time=ns(end))
```

So each incident is one root `support.conversation` span with child spans for retrieval, MCP tool calls, and guardrail stages — a latency waterfall you can open in Tempo/Phoenix. Note it uses **explicit timestamps** (`start_time`/`end_time` from recorded event times) rather than wall-clock `start_as_current_span`, because it's replaying an already-finished stream.

Then it records the instruments (`metrics.py:143-152`): latency + ttft histograms, token counter split by `gen_ai.token.type` input/output, cost by model, convo count, and a guardrail-block tick. Attributes on a metric become **dimensions** you can group by later. All of this is wrapped in try/except so a telemetry failure never breaks the request.

## 4. Build it up

**Variation A — nested spans / context propagation.** `start_as_current_span` makes the span the current context, so any span opened inside it becomes a child automatically:

```python
with tracer.start_as_current_span("incident") as root:
    with tracer.start_as_current_span("retrieve-runbook"):   # child of incident
        ...
    with tracer.start_as_current_span("call-tool") as tool:
        with tracer.start_as_current_span("mcp.restart"):    # grandchild
            ...
```

When you *can't* use the `with`-based current context (e.g. replaying finished events like `metrics.py` does), pass `context=` explicitly: `tracer.start_span(name, context=ctx)`.

**Variation B — span attributes + events.** Attributes are dimensions; events are timestamped marks:

```python
with tracer.start_as_current_span("generate") as span:
    span.set_attribute("gen_ai.request.model", "gpt-4o-mini")
    span.set_attribute("gen_ai.usage.input_tokens", 812)
    span.add_event("first-token", attributes={"ms": 240})
    # on failure, record it so the span shows an error status:
    # span.record_exception(err); span.set_status(Status(StatusCode.ERROR))
```

Prefer standard `gen_ai.*` keys over ad-hoc names — that's what makes the Copilot's spans render in generic GenAI tooling.

**Variation C — histograms + percentiles.** A histogram records raw samples; the backend computes p50/p95/p99. The Copilot records latency per conversation with a `kind` dimension (`metrics.py:144`):

```python
latency = meter.create_histogram("gen_ai.client.operation.duration", unit="ms")
latency.record(elapsed_ms, {"kind": "incident"})   # group percentiles by kind
```

(For in-process display the Copilot also computes its own percentiles in `_pct()` at `metrics.py:157-161` — a plain nearest-rank calc — but the *exported* histogram lets Prometheus/Grafana do it across instances.)

**Variation D — resource attributes / `service.name` + force_flush.** The `Resource` tags every signal with identity so a backend can separate services and environments:

```python
resource = Resource.create({
    "service.name": "oncall-copilot",          # REQUIRED — how backends name the service
    "deployment.environment": "production",
    "service.version": "1.4.0",
})
```

And because periodic/batch export is asynchronous, short-lived scripts must flush before exit or lose the tail:

```python
trace.get_tracer_provider().force_flush()
metrics.get_meter_provider().force_flush()
```

## 5. Gotchas & pitfalls

- **`OTEL_EXPORTER_OTLP_INSECURE=true` for local gRPC.** The gRPC OTLP exporter assumes TLS by default. A local collector on `localhost:4317` speaks plaintext, so without `insecure=true` the export silently fails to connect. `config/observability.yaml` sets `insecure: true` and `obs_config.py` maps it to the env var (`obs_config.py:31-32`). This is the #1 "why do no traces show up locally" bug.
- **Token/cost read 0 until a real LLM (M1).** The instruments are wired, but `self.tin/tout/cost` come from `metrics` events on the stream (`metrics.py:91-95`). In demo/fallback mode there's no real model call, so `gen_ai.usage.*` and `gen_ai.client.cost` legitimately export **0** until a real LLM key is attached — the plumbing works; there's just nothing to count yet.
- **Metrics export every 60s.** `PeriodicExportingMetricReader` (console path) uses `export_interval_millis=60000` (`observability.py:89`). So after emitting a counter you may wait up to a minute before it appears. In a test/script call `force_flush()` to push immediately.
- **Counters are monotonic / cumulative.** `counter.add(n)` only ever increases the running total; you never see a rate directly — the backend derives rate-of-change. Don't try to "reset" a counter to represent a current value; use a gauge/histogram or compute deltas downstream.
- **Dev vs prod span processors.** `SimpleSpanProcessor` exports synchronously per span — great for reading console output, bad under load. `BatchSpanProcessor` buffers and exports in a background thread — use it in prod (`observability.py:67`). Don't ship Simple to production.
- **Register the provider exactly once.** `set_tracer_provider`/`set_meter_provider` are global and effectively one-shot; a second call is ignored with a warning. The Copilot guards with `_DONE` (`observability.py:31-33`) to stay idempotent across reloads.
- **API without SDK is a silent no-op.** If `opentelemetry-sdk` isn't installed, every `start_span`/`record` does nothing (no error). Great for optional instrumentation, confusing when you expect output — check `get_state()`.
- **Wrap emit in try/except.** Telemetry must never break the request path. `_emit_otel` catches everything and prints a skip note (`metrics.py:153-154`).

## ✅ Best Practices

- **Instrument the SDK once, use the API everywhere.** Configure providers, exporters, and processors in one bootstrap module, then call only the OpenTelemetry API from application code so instrumentation stays decoupled from wiring.
- **Follow the GenAI semantic conventions.** Name LLM attributes with the standard `gen_ai.*` keys (`gen_ai.system`, `gen_ai.request.model`, `gen_ai.usage.*`) so any conformant backend can parse and aggregate your spans.
- **Always set resource attributes.** Populate `service.name`, `service.version`, and `deployment.environment` on the Resource so traces and metrics are attributable and filterable per service and env.
- **Batch in prod, Simple in dev.** Use `BatchSpanProcessor` in production for async, buffered export and reserve `SimpleSpanProcessor` for local debugging where synchronous console output is what you want.
- **Keep the backend swappable via OTLP.** Export over OTLP to a local collector rather than coding to a vendor SDK, so you can redirect telemetry to Jaeger, Tempo, or a SaaS by changing config, not code.
- **Enrich LLM spans with token and cost attributes.** Attach `gen_ai.usage.input_tokens`, `gen_ai.usage.output_tokens`, and a derived cost attribute to model spans so you can slice spend and latency by model and route.
- **Sample high-volume traces.** Apply a parent-based or ratio sampler on hot paths to cap trace volume and cost while still preserving complete traces for the requests you keep.
- **Record real usage, never fabricate metrics.** Emit counters and histograms only from actual measured events (real token counts, real durations) so dashboards reflect ground truth instead of placeholder values.

## 6. Exercises

1. **Add a new metric.** In `app/metrics.py` `_otel()`, add a counter `gen_ai.client.requests` and `.add(1, {"model": ...})` inside `_emit_otel`. Confirm it shows on the console exporter after a `force_flush`.
2. **Add span attributes.** Add `span.set_attribute("hitl.decision", self.decision or "none")` to the root `support.conversation` span and verify it appears in the exported JSON. Which existing field feeds it?
3. **Lower the export interval.** Change the console `PeriodicExportingMetricReader` in `observability.py:89` from `60000` to `5000` ms and observe metrics flushing every 5s instead of 60. What's the trade-off in prod?
4. **Instrument a function.** Wrap `snapshot()` in `metrics.py` with `with tracer.start_as_current_span("observability.snapshot"):` and record its duration in a histogram. Where would this span land in the trace tree (hint: it has no parent context)?
5. **Force console vs OTLP.** Run the app with `OTEL_EXPORTER_OTLP_ENDPOINT` unset, then set to `localhost:4317` + `OTEL_EXPORTER_OTLP_INSECURE=true`. Compare `get_state()["traces"]` in each case.
6. **Break INSECURE on purpose.** Point at a local collector but *omit* `OTEL_EXPORTER_OTLP_INSECURE`. Watch spans stop arriving, then add it back. This is the muscle memory that saves an hour.

## 7. Connections

- **[11-observability-stack.md]** — where these OTLP signals land: the Collector → Prometheus → Grafana/Tempo pipeline, plus Phoenix/Langfuse for LLM traces. The Copilot's `snapshot()`/`profile()` dashboards read the same records this file emits.
- **[09-resilience.md]** — the try/except-wrapped emit and the safe-no-op SDK setup are resilience patterns: telemetry degrades silently rather than taking down the request path.
- **[12-evaluation-and-regression.md]** — the eval rates (`block_rate`, `refusal_rate`, `hitl_rate`) and `ragas` scoring ride on the same trace/metric records; span attributes become the labels you slice regressions by.

## 8. Further reading

- OpenTelemetry Python docs — Traces, Metrics, and the SDK/exporter guide: <https://opentelemetry.io/docs/languages/python/>
- OTel GenAI semantic conventions (`gen_ai.*` spans + metrics): <https://opentelemetry.io/docs/specs/semconv/gen-ai/>
- OTLP protocol + exporter env vars (`OTEL_EXPORTER_OTLP_ENDPOINT`, `_INSECURE`): <https://opentelemetry.io/docs/specs/otel/protocol/exporter/>
- NeMo Guardrails observability (API-only, app supplies SDK): <https://docs.nvidia.com/nemo/guardrails/latest/user-guides/observability.html>
