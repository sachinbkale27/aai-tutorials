# 11 · Minimal Observability Stack (metrics-only)

A standalone, metrics-only slice of tutorial 11 §2: an **OTel Collector** receives
OTLP metrics (push, gRPC `:4317`), re-exposes them on a Prometheus exporter
(`:8889/metrics`), **Prometheus** scrapes that, and **Grafana** graphs it. No Tempo,
no traces — add those from the tutorial when you want them.

```
emit.py --OTLP push :4317--> Collector --:8889/metrics--> Prometheus --> Grafana
```

## Run

From this folder:

```bash
docker compose up -d           # collector + prometheus + grafana
python emit.py                 # push a few metrics, force-flush, exit
```

`emit.py` also works from the repo root: `python examples/11_observability/emit.py`.
It needs the OpenTelemetry SDK + OTLP gRPC exporter installed. If the Collector
isn't up it degrades gracefully — it prints a hint and exits, it does not crash.

## Verify the bridge (bottom-up)

1. Collector is exposing metrics: `curl -s localhost:8889/metrics | grep support_conversations`
2. Prometheus sees the target UP: <http://localhost:9090/targets>
3. Query it: <http://localhost:9090/graph>, run `support_conversations_total`
4. Grafana at <http://localhost:3000> (admin/admin) — the Prometheus datasource is
   pre-provisioned (uid `prometheus`); use Explore to graph the same metric.

## Gotchas (the ones that actually bite)

- **Prometheus is pull-based.** An OTLP app cannot push to it. Metrics only appear
  because the Collector's `prometheus` exporter (`:8889`) gives Prometheus something
  to scrape. Debug bottom-up: `curl :8889/metrics` first, then `/targets`.
- **`OTEL_EXPORTER_OTLP_INSECURE=true` for local gRPC.** The OTLP gRPC exporter
  defaults to TLS; the local Collector is plaintext. Without it you get a confusing
  handshake error. `emit.py` sets it (and passes `insecure=True`).
- **Grafana datasource needs a fixed `uid`.** Provisioned dashboards bind datasources
  by uid, so `grafana-provisioning/datasources/prometheus.yml` pins `uid: prometheus`.
  If you change a uid, **recreate** Grafana (`docker compose up -d --force-recreate
  grafana`) — a plain restart can keep the stale uid in Grafana's SQLite state.
- **Force-flush in short scripts.** OTel exports periodically (~60s); `emit.py` calls
  `force_flush()` so metrics land in ~5s instead of being lost on exit.
- **`localhost` vs service names.** From your shell use `localhost:<port>`; between
  containers use the compose service name (`otel-collector:8889`, `prometheus:9090`).

## Teardown

```bash
docker compose down
```
