# 11 · Observability Stack (Collector · Prometheus · Tempo · Grafana)

> Run the industry-standard OSS observability stack locally — an app pushes OTLP to a Collector, which fans out **metrics → Prometheus** and **traces → Tempo**, both surfaced in one **Grafana**.

## 1. Mental model — push vs pull, and why you need a Collector

Tutorial 10 gave you an app that *emits* OpenTelemetry signals. This tutorial is about where
those signals **go**. The hard part is that the two dominant metric systems disagree about
who initiates the transfer:

- **Push (OTLP).** Your app opens a connection and *sends* spans and metrics out to a
  collector endpoint (gRPC on `:4317`, HTTP on `:4318`). The app decides when to flush.
  This is how OpenTelemetry SDKs work by default.
- **Pull (Prometheus scrape).** Prometheus does the opposite: it periodically **scrapes** an
  HTTP `/metrics` endpoint that your target exposes and *pulls* the current counter values.
  Nothing is pushed to Prometheus; it reaches out on its own `scrape_interval`.

These two models are incompatible on the wire. An OTLP-exporting app cannot talk to Prometheus
directly — the app wants to push, Prometheus wants to pull. **That mismatch is the entire reason
the Collector exists in this stack.**

The **OpenTelemetry Collector** is a vendor-neutral pipe. It **receives** OTLP (push) on one
side and **re-exports** on the other. Crucially it can expose a `prometheus` exporter — an HTTP
`/metrics` page (here `:8889`) — that Prometheus then scrapes. So the Collector is a
**push→pull bridge**: OTLP in, Prometheus scrape-target out. For traces there's no impedance
mismatch: the Collector just forwards OTLP straight to Tempo.

Roles of each service in the stack:

| Service | Role | Port(s) |
|---|---|---|
| **OTel Collector** | Receives OTLP; fans out metrics→Prometheus exporter, traces→Tempo | `4317` gRPC, `4318` HTTP, `8889` scrape |
| **Prometheus** | Time-series DB; **scrapes** the Collector, stores metrics, answers PromQL | `9090` |
| **Tempo** | Trace store; receives OTLP traces, answers trace-by-ID / search | `3200` |
| **Grafana** | UI over both datasources; dashboards + Explore for traces | `3000` |
| **grafana-image-renderer** | Headless Chromium that renders panels to PNG | `8081` |

The shape to hold in your head:

```
  app  --OTLP push :4317-->  Collector  --:8889 /metrics-->  Prometheus (pull/scrape)
                                 \                                     \
                                  --OTLP push-->  Tempo                 Grafana <-- PromQL + TraceQL
```

Why a Collector at all, rather than pointing the app straight at each backend? Decoupling:
you can swap Prometheus for Grafana Cloud, add Tempo for traces, batch/retry/redact centrally,
and change backends **without touching app code** — the app only ever knows about one OTLP
endpoint. That is the payoff §6 has you exploit when repointing at Datadog.

## 2. Smallest working example — Collector + Prometheus + Grafana

Start with the metrics half only (add Tempo in §4). Three services, one `docker-compose.yml`:

```yaml
# compose.min.yml
services:
  otel-collector:
    image: otel/opentelemetry-collector-contrib:latest
    command: ["--config=/etc/otelcol/config.yaml"]
    volumes: ["./otel-collector-config.yaml:/etc/otelcol/config.yaml"]
    ports: ["4317:4317", "4318:4318", "8889:8889"]
  prometheus:
    image: prom/prometheus:latest
    volumes: ["./prometheus.yml:/etc/prometheus/prometheus.yml"]
    ports: ["9090:9090"]
  grafana:
    image: grafana/grafana:latest
    environment: ["GF_SECURITY_ADMIN_PASSWORD=admin"]
    ports: ["3000:3000"]
```

The Collector config — OTLP in, Prometheus exporter out:

```yaml
# otel-collector-config.yaml
receivers:
  otlp:
    protocols:
      grpc: { endpoint: 0.0.0.0:4317 }
      http: { endpoint: 0.0.0.0:4318 }
processors:
  batch: {}
exporters:
  prometheus:                     # Prometheus scrapes this at :8889/metrics
    endpoint: 0.0.0.0:8889
    resource_to_telemetry_conversion: { enabled: true }
service:
  pipelines:
    metrics:
      receivers: [otlp]
      processors: [batch]
      exporters: [prometheus]
```

And tell Prometheus to scrape the Collector (not your app):

```yaml
# prometheus.yml
global: { scrape_interval: 5s }
scrape_configs:
  - job_name: 'otel-collector'
    static_configs:
      - targets: ['otel-collector:8889']   # the collector's exporter, over the compose network
```

Run it and point any OTLP app at `localhost:4317`:

```bash
docker compose -f compose.min.yml up -d

# minimal exporter env for a local Python app (see tutorial 10):
export OTEL_EXPORTER_OTLP_ENDPOINT=localhost:4317
export OTEL_EXPORTER_OTLP_INSECURE=true      # plaintext gRPC, no TLS — see §5
python your_app_that_emits_metrics.py
```

Verify the bridge end-to-end:

1. Collector is exposing metrics: `curl -s localhost:8889/metrics | head`.
2. Prometheus sees the target UP: open `http://localhost:9090/targets`.
3. Query in Prometheus: `http://localhost:9090/graph`, run `support_conversations_total`.
4. In Grafana (`http://localhost:3000`, admin/admin) add a Prometheus datasource
   `http://prometheus:9090` and graph the same metric.

If step 1 shows your metric but step 2 doesn't, it's almost always a scrape-target name or
network issue — Prometheus must reach `otel-collector:8889` **inside** the compose network,
not `localhost`.

## 3. How the On-Call Copilot uses it

The project's real stack lives in `deploy/observability/`. Bring it up with
`cd deploy/observability && docker compose up -d`.

**Compose topology** — `deploy/observability/docker-compose.yml`. Five services wired exactly
like §1: `otel-collector` (ports `4317/4318/8889`), `prometheus`, `tempo` (`3200`),
`renderer` (grafana-image-renderer on `8081`), and `grafana` (`3000`). Note the `depends_on`
chain: collector→tempo, prometheus→collector, grafana→{prometheus,tempo,renderer}. Grafana
mounts `./grafana/provisioning` so datasources and dashboards come up pre-wired.

**Collector pipelines** — `deploy/observability/otel-collector-config.yaml`. One OTLP receiver
feeds **two** pipelines:

- `metrics: [otlp] → batch → [prometheus]` — the push→pull bridge. `resource_to_telemetry_conversion`
  turns OTel resource attributes into Prometheus labels so they're queryable.
- `traces: [otlp] → batch → [otlp/tempo, debug]` — forwards traces to `tempo:4317` over OTLP
  gRPC (`tls.insecure: true`), **and** to the `debug` exporter which prints span summaries to the
  Collector's own stdout. The `debug` exporter is your first stop when traces "disappear":
  `docker compose logs otel-collector` tells you whether spans even arrived.

**Prometheus** — `deploy/observability/prometheus.yml`. A single scrape job targeting
`otel-collector:8889` at a 5s interval. Prometheus never talks to the app; it only knows about
the Collector's exporter.

**Tempo** — `deploy/observability/tempo.yaml`. A deliberately minimal single-binary Tempo:
`server` (http on 3200), a `distributor` with an OTLP/gRPC receiver on `:4317` (where the
Collector pushes), and `storage.trace` on the local filesystem (`wal` + `blocks`). No
ingester/compactor/microservices config — see the gotcha in §5.

**Grafana provisioning** — `deploy/observability/grafana/provisioning/`:

- `datasources/prometheus.yml` — Prometheus at `http://prometheus:9090`, **`uid: prometheus`**,
  `isDefault: true`.
- `datasources/tempo.yml` — Tempo at `http://tempo:3200`, **`uid: tempo`**.
- `dashboards/provider.yml` — a file provider that loads every dashboard JSON in the
  dashboards directory.
- `dashboards/oncall.json` — the "On-Call Copilot" dashboard (uid `oncall-copilot`). Its panels
  reference the datasources **by uid** (`"datasource": {"type":"prometheus","uid":"prometheus"}`),
  which is exactly why the uids in the datasource files must be fixed (§5).

**Traffic generator** — `deploy/observability/gen_traffic.py`. Because the demo has no live
users, this drives synthetic incidents so signals actually flow. It:

- Sets `OTEL_EXPORTER_OTLP_ENDPOINT=localhost:4317` and `OTEL_EXPORTER_OTLP_INSECURE=true`
  (lines 14–15) before importing the app, so the app's OTel SDK exports to your local Collector.
- Runs incidents through `incident_events` / `resume_events`, taping each into a
  `M.TraceRecorder` (`app/metrics.py`) that emits a real OTel **span tree** plus metrics
  (`support.conversations`, `gen_ai.client.operation.duration`, tokens, cost, guardrail blocks).
- **Force-flushes** both the meter and tracer providers after each burst
  (`force_flush()`, lines 48–49) so metrics land in Prometheus within ~5s instead of waiting for
  the 60s periodic export. That single call is the difference between "the dashboard is empty"
  and "the dashboard moves" during a demo.

```bash
python deploy/observability/gen_traffic.py          # one burst (~10 incidents), flush, exit
python deploy/observability/gen_traffic.py --loop    # keep emitting every ~3s
```

Then open Grafana → the On-Call Copilot dashboard and watch the panels fill in.

## 4. Build it up

### 4a. Add Tempo for traces

Add the Tempo service and give the Collector a traces pipeline. Tempo config (this is the whole
file — keep it minimal, see §5):

```yaml
# tempo.yaml
server: { http_listen_port: 3200 }
distributor:
  receivers:
    otlp:
      protocols:
        grpc: { endpoint: 0.0.0.0:4317 }   # collector pushes here as tempo:4317
storage:
  trace:
    backend: local
    wal:   { path: /var/tempo/wal }
    local: { path: /var/tempo/blocks }
```

Collector traces pipeline (add to `exporters` + `service.pipelines`):

```yaml
exporters:
  otlp/tempo:
    endpoint: tempo:4317
    tls: { insecure: true }        # plaintext inside the compose network
  debug: { verbosity: normal }
# ...
  traces:
    receivers: [otlp]
    processors: [batch]
    exporters: [otlp/tempo, debug]
```

Both Tempo and the Collector listen on `4317` — that's fine, they're **different containers**.
The app→Collector hop uses the *host-published* `4317`; the Collector→Tempo hop uses Tempo's
container port over the internal network (`tempo:4317`).

### 4b. Provision a datasource + dashboard as code

Instead of clicking through the UI, drop YAML/JSON under `provisioning/` and mount it. Datasources
(`datasources/*.yml`) and a dashboard **provider** (`dashboards/provider.yml`) that auto-loads
JSON. This is the difference between a demo you can `git clone && docker compose up` and one you
have to hand-configure. Pin every datasource `uid` (§5).

### 4c. A dual-axis panel (rates vs latency in one graph)

Panel 7 of `oncall.json` plots four series on one timeseries — three rates and a p95 latency.
Latency is in milliseconds and dwarfs per-second rates, so it gets its **own right-hand axis**
via a field override matched by regex:

```json
"overrides": [
  { "matcher": { "id": "byRegexp", "options": ".*latency.*" },
    "properties": [
      { "id": "custom.axisPlacement", "value": "right" },
      { "id": "unit", "value": "ms" } ] } ]
```

The p95 target is standard Prometheus histogram math:
`histogram_quantile(0.95, sum(rate(gen_ai_client_operation_duration_milliseconds_bucket[5m])) by (le))`.
Note the metric name: OTel's `gen_ai.client.operation.duration` (unit `ms`) becomes
`gen_ai_client_operation_duration_milliseconds_bucket` after the Prometheus exporter sanitizes
dots→underscores and appends the unit + `_bucket` suffix.

### 4d. Screenshot panels with the image-renderer

Grafana can't rasterize panels by itself; it delegates to the `grafana-image-renderer` sidecar.
Wire them with a **shared token** — Grafana's `GF_RENDERING_RENDERER_TOKEN` must equal the
renderer's `AUTH_TOKEN` (`oncall-render-token` in this project), plus `GF_RENDERING_SERVER_URL`
pointing at `http://renderer:8081/render`. Then any panel's share menu → "Direct link rendered
image", or hit the render API:

```bash
curl -o panel.png "http://localhost:3000/render/d-solo/oncall-copilot/?panelId=7&width=1000&height=500&from=now-1h&to=now"
```

Useful for putting a dashboard PNG in a Slack incident channel or a nightly report.

## 5. Gotchas & pitfalls

- **Prometheus is pull-based — you need the Collector's `prometheus` exporter.** An OTLP app
  cannot push to Prometheus. The Collector's `prometheus` exporter exposes `:8889/metrics` and
  Prometheus scrapes *that*. Forget the exporter (or forget the scrape job) and metrics silently
  never appear. Debug from the bottom up: `curl :8889/metrics` first, then Prometheus `/targets`.
- **`OTEL_EXPORTER_OTLP_INSECURE=true` for local gRPC.** The OTLP gRPC exporter defaults to TLS.
  Locally the Collector listens plaintext, so without this env var the app fails to connect (often
  a confusing handshake error). `gen_traffic.py` sets it at line 15. Same idea inside the
  Collector→Tempo hop, expressed as `tls: { insecure: true }`.
- **Grafana datasources need a fixed `uid`, and you must *recreate* (not restart) Grafana after
  changing it.** Provisioned dashboards reference datasources by uid
  (`"uid": "prometheus"` / `"tempo"`). If you let Grafana auto-generate a random uid, dashboard
  provisioning fails to bind and panels show "datasource not found." After editing a uid,
  `docker compose up -d --force-recreate grafana` — a plain `restart` can keep the stale uid in
  Grafana's SQLite state.
- **Tempo's strict config rejects unknown keys — go minimal.** Copy-pasting a full production
  Tempo config with `ingester:`, `compactor:`, or `usage_report:` blocks makes single-binary
  Tempo refuse to start with a field-validation error. The `tempo.yaml` here is deliberately just
  `server` + `distributor` + `storage`; add nothing you don't need.
- **Tempo has NO UI.** There is no `localhost:3200` dashboard. View traces through **Grafana →
  Explore → Tempo datasource** (search or paste a trace ID). `:3200` is a query API, not a page.
- **grafana-image-renderer needs matching tokens.** `GF_RENDERING_RENDERER_TOKEN` (on Grafana)
  must equal `AUTH_TOKEN` (on the renderer). Mismatch → renders fail with 401 and panels export
  blank. Also set `GF_RENDERING_SERVER_URL` and `GF_RENDERING_CALLBACK_URL`.
- **Guard PromQL that may have no data: `... or vector(0)`.** Token/cost panels
  (`sum(gen_ai_client_token_usage_token_total) or vector(0)`) show `0` before any real LLM traffic
  exists. Without the `or vector(0)`, an empty result renders as "No data" instead of a clean `0` —
  ugly on a stat panel.
- **Force-flush in short-lived scripts.** OTel batches and exports periodically (~60s). A script
  that emits and exits will lose everything unless it calls `force_flush()` on both providers
  (`gen_traffic.py` lines 48–49). Long-running servers don't need this.
- **Metric names get mangled by the Prometheus exporter.** Dots→underscores, unit appended,
  `_total`/`_bucket`/`_count`/`_sum` suffixes added. `gen_ai.client.operation.duration` (ms)
  → `gen_ai_client_operation_duration_milliseconds_bucket`. Always confirm the exact name via
  `curl :8889/metrics` before writing PromQL.
- **`localhost` vs service names.** From your shell, targets are `localhost:<published-port>`.
  Between containers, use the compose **service name** (`otel-collector:8889`, `tempo:4317`,
  `prometheus:9090`). Mixing these up is the #1 "why is my target down" cause.

## ✅ Best Practices

- **Always front your apps with an OTel Collector.** Point app code at a single OTLP endpoint and let the Collector fan out to backends, so you can swap Prometheus for Grafana Cloud or add Datadog without redeploying the app.
- **Provision datasources and dashboards as code with fixed UIDs.** Commit `provisioning/*.yml` and dashboard JSON with pinned `uid`s so a clean `git clone && docker compose up` reproduces the exact same wired stack every time.
- **Run separate pipelines for metrics, traces, and logs.** Keep one OTLP receiver feeding distinct `service.pipelines` per signal type, so a misbehaving exporter on one signal can't stall the others.
- **Tune retention and scrape intervals to the workload, not the defaults.** Set Prometheus `scrape_interval` and Tempo/Prometheus retention deliberately — tight enough for demo responsiveness, loose enough to control disk and cardinality in production.
- **Use dual-axis panels for mixed-scale series.** Put per-second rates and millisecond latencies on the same timeseries via a right-hand `axisPlacement` override so neither series flattens the other.
- **Alert on SLOs, not raw metrics.** Build alerts on derived objectives (p95 latency, error-rate budgets, block rate) rather than individual counters, so pages reflect user-facing impact instead of noise.
- **Secure and scope every exposed endpoint.** Lock down Grafana admin credentials, restrict the Collector's OTLP ports and Prometheus/Tempo APIs to trusted networks, and use matched tokens between Grafana and the image-renderer.
- **Keep the whole stack in docker-compose for reproducibility.** Define the Collector, Prometheus, Tempo, Grafana, and renderer together with explicit `depends_on` ordering so anyone can bring up an identical observability environment in one command.

## 6. Exercises

1. **Add a stat panel.** Add an "Approval rate" panel to `oncall.json` using a PromQL ratio of
   approved vs total HITL decisions. Reload Grafana and confirm it provisions. Guard it with
   `or vector(0)` and verify it shows `0` before traffic, then moves under `--loop`.
2. **Prove the bridge.** Stop Prometheus but keep the Collector. Run `gen_traffic.py`, then
   `curl :8889/metrics` — metrics are there. Restart Prometheus and watch it back-scrape. Explain
   in one paragraph why the app never noticed Prometheus was down (decoupling via the Collector).
3. **Point at Grafana Cloud / Datadog via OTLP.** Add a second metrics exporter to the Collector
   (`otlphttp` with your vendor's OTLP endpoint + API-key header) alongside the local `prometheus`
   exporter — fan out to both. Confirm data appears in the SaaS **without changing app code**.
   This is the Collector's whole value proposition.
4. **Add Loki for logs.** Add a `loki` service, a `loki` datasource (fixed uid), and a Collector
   `loki` exporter fed by a logs pipeline. Emit a few log records and view them in Grafana Explore.
   Then wire a trace→logs correlation so a Tempo span links to its logs.
5. **Break a uid on purpose.** Change the Prometheus datasource uid to something random, `restart`
   (don't recreate) Grafana, and observe the "datasource not found" failure on provisioned panels.
   Fix it with `--force-recreate` and note the difference.
6. **Add TTFT.** The app records `gen_ai.server.time_to_first_token`. Add a p95 TTFT timeseries
   panel and compare its shape to end-to-end latency (panel 5). What does the gap tell you about
   where time goes — retrieval/tool time before first token vs generation streaming?

## 7. Connections

- **[10-opentelemetry.md](10-opentelemetry.md)** — the app-side instrumentation that *produces*
  the OTLP this stack consumes; this tutorial is where those spans and metrics land.
- **[09-resilience.md](09-resilience.md)** — you can't set timeouts, retries, and circuit-breaker
  thresholds intelligently without the latency percentiles and error rates these dashboards expose;
  observability closes the resilience loop.
- **[12-evaluation-and-regression.md](12-evaluation-and-regression.md)** — the same metrics
  (block rate, latency p95, cost/convo) feed offline eval and regression gates; the dashboard is
  the live view, eval is the batch/CI view of the same signals.

## 8. Further reading

- OpenTelemetry Collector — https://opentelemetry.io/docs/collector/
- Collector `prometheus` exporter — https://github.com/open-telemetry/opentelemetry-collector-contrib/tree/main/exporter/prometheusexporter
- Prometheus docs (scraping, PromQL, `histogram_quantile`) — https://prometheus.io/docs/
- Grafana Tempo (single binary, TraceQL) — https://grafana.com/docs/tempo/latest/
- Grafana provisioning (datasources & dashboards as code) — https://grafana.com/docs/grafana/latest/administration/provisioning/
- Grafana image renderer — https://grafana.com/docs/grafana/latest/setup-grafana/image-rendering/
- OTLP exporter env vars (`OTEL_EXPORTER_OTLP_*`) — https://opentelemetry.io/docs/specs/otel/protocol/exporter/
