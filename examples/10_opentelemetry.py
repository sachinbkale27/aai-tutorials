"""Smallest working OpenTelemetry example — traces + metrics to the console.

Demonstrates the API-vs-SDK split from tutorial 10, Section 2:
  * A TracerProvider with a nested span (parent `do-work` + child `sub-step`)
    carrying attributes.
  * A MeterProvider with a counter and a histogram.
  * Both export to the CONSOLE exporter — no external collector needed.
  * A force_flush before exit so short-lived scripts don't lose the tail
    (periodic/batch export is asynchronous).

Deps:
    pip install opentelemetry-sdk opentelemetry-exporter-otlp

Run:
    python examples/10_opentelemetry.py

You'll see JSON for the span tree (`do-work` -> `sub-step`) and a metric block
printed to the console.

Switch to a real OTLP collector WITHOUT touching the code by setting two env
vars (see the block near the exporter setup below):
    export OTEL_EXPORTER_OTLP_ENDPOINT=localhost:4317
    export OTEL_EXPORTER_OTLP_INSECURE=true   # REQUIRED for plaintext local gRPC
    python examples/10_opentelemetry.py
"""

import os
import time

from opentelemetry import trace, metrics
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor, ConsoleSpanExporter
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import (
    PeriodicExportingMetricReader,
    ConsoleMetricExporter,
)

# One Resource describes WHO is producing telemetry; it tags every span + metric.
resource = Resource.create({"service.name": "otel-hello", "service.version": "1.0.0"})

# unset -> console exporters; set -> OTLP exporters (see swap below).
endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT")

# ---- TRACES: pick an exporter, wrap it in a processor, register the provider ----
if endpoint:
    # To switch to OTLP, the ONLY change is the exporter — the API code below is
    # identical. OTLPSpanExporter honours OTEL_EXPORTER_OTLP_INSECURE (needed for
    # plaintext gRPC to a local collector on localhost:4317).
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter

    span_exporter = OTLPSpanExporter(endpoint=endpoint)
else:
    span_exporter = ConsoleSpanExporter()

provider = TracerProvider(resource=resource)
# SimpleSpanProcessor exports each span synchronously — easy to read on console.
# (Prod would use BatchSpanProcessor to buffer + export in the background.)
provider.add_span_processor(SimpleSpanProcessor(span_exporter))
trace.set_tracer_provider(provider)  # register GLOBAL — the API now routes here

# ---- METRICS: a reader pulls the SDK on an interval and pushes to an exporter ----
if endpoint:
    from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter

    metric_exporter = OTLPMetricExporter(endpoint=endpoint)
else:
    metric_exporter = ConsoleMetricExporter()

# Short interval so the demo is snappy; we also force_flush at the end regardless.
reader = PeriodicExportingMetricReader(metric_exporter, export_interval_millis=5000)
metrics.set_meter_provider(MeterProvider(resource=resource, metric_readers=[reader]))

# ---- USE the API (this is what library/app code writes) ----
tracer = trace.get_tracer("otel-hello")
meter = metrics.get_meter("otel-hello")

# A histogram records a distribution (backend derives p50/p95/p99); a counter is
# monotonic ("add N").
latency = meter.create_histogram("demo.op.duration", unit="ms")
runs = meter.create_counter("demo.runs")

# start_as_current_span opens a span AND sets it as the current context, so any
# span opened inside becomes its child automatically.
with tracer.start_as_current_span("do-work") as span:
    span.set_attribute("work.kind", "demo")  # attributes are dimensions/tags
    span.set_attribute("work.item", 42)
    t0 = time.time()
    time.sleep(0.05)
    with tracer.start_as_current_span("sub-step") as child:  # nested -> child of do-work
        child.set_attribute("step.name", "compute")
        time.sleep(0.02)
    elapsed_ms = (time.time() - t0) * 1000
    latency.record(elapsed_ms, {"work.kind": "demo"})  # attrs -> metric dimensions
    runs.add(1, {"work.kind": "demo"})

# Export is asynchronous, so a short-lived script MUST flush before exit or lose
# the tail (unexported spans/metrics).
trace.get_tracer_provider().force_flush()
metrics.get_meter_provider().force_flush()

print("done - parent span 'do-work', child span 'sub-step', 1 histogram sample, 1 counter tick")
