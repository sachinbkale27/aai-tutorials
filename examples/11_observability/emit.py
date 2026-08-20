#!/usr/bin/env python3
"""Emit a few OTLP metrics to a local OTel Collector, then force-flush and exit.

Run the stack first (from this folder):
    docker compose up -d
Then:
    python examples/11_observability/emit.py

This is standalone — it does NOT import the project's `app`. It only needs the
OpenTelemetry SDK + OTLP gRPC exporter.

Gotchas baked in below:
  - The OTLP gRPC exporter defaults to TLS. The local Collector listens plaintext,
    so we set OTEL_EXPORTER_OTLP_INSECURE=true (or pass insecure=True) or the app
    fails to connect with a confusing handshake error.
  - OTel batches/exports periodically (~60s). A short-lived script that emits and
    exits loses everything unless it calls force_flush(). We do, below.
  - If the Collector isn't reachable the OTLP exporter just logs a warning and
    retries in the background — it does NOT raise. So this script degrades
    gracefully: it prints a hint and exits 0 rather than crashing.
"""
import os

# Set BEFORE creating the exporter so the SDK picks it up. Plaintext gRPC for local dev.
os.environ.setdefault("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4317")
os.environ.setdefault("OTEL_EXPORTER_OTLP_INSECURE", "true")

from opentelemetry import metrics
from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
from opentelemetry.sdk.resources import Resource, SERVICE_NAME


def build_provider() -> MeterProvider:
    # insecure=True mirrors OTEL_EXPORTER_OTLP_INSECURE=true — plaintext gRPC to :4317.
    exporter = OTLPMetricExporter(endpoint="localhost:4317", insecure=True)
    reader = PeriodicExportingMetricReader(exporter)
    resource = Resource.create({SERVICE_NAME: "obs-example-emitter"})
    return MeterProvider(metric_readers=[reader], resource=resource)


def main() -> int:
    provider = build_provider()
    metrics.set_meter_provider(provider)
    meter = metrics.get_meter("obs.example")

    # A counter and a histogram — enough to see the push->pull bridge work.
    conversations = meter.create_counter(
        "support.conversations",
        unit="1",
        description="support conversations handled",
    )
    latency = meter.create_histogram(
        "gen_ai.client.operation.duration",
        unit="ms",
        description="model call latency",
    )

    for i in range(10):
        conversations.add(1, {"outcome": "resolved" if i % 3 else "escalated"})
        latency.record(120 + i * 15, {"model": "demo"})

    # Force-flush so metrics land in Prometheus within ~5s instead of waiting for the
    # periodic export. Without this a short script exits before anything is sent.
    try:
        provider.force_flush(timeout_millis=5000)
        print("Flushed metrics to Collector at localhost:4317.")
        print("Verify: curl -s localhost:8889/metrics | grep support_conversations")
    except Exception as exc:  # exporter down / unreachable — don't crash
        print(f"Flush hit an issue (collector likely not up yet): {exc}")
        print("Start the stack with `docker compose up -d` in this folder and retry.")
    finally:
        # shutdown() also flushes and closes the exporter cleanly.
        try:
            provider.shutdown()
        except Exception:
            pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
