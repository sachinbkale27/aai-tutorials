"""Standalone SSE(Sever Sent Events) streaming demo — a typed event contract over Server-Sent Events.

A tiny FastAPI app whose /stream endpoint StreamingResponses an async generator of
SSE `data:` frames. The contract is three event types carried inside the JSON payload:

    step.start     -> a pipeline stage begins   (fields: name)
    token          -> one word of the reply      (fields: text)
    message.final  -> the turn is complete        (fields: text)

Every frame ends in a blank line (`\\n\\n`) — that double-newline is the SSE frame
terminator; miss it and clients buffer the event forever.

Dependencies:
    pip install fastapi uvicorn httpx      # httpx is needed by TestClient (self-test)

Run the SELF-TEST (no server needed — consumes its own stream in-process):
    python examples/14_sse_server.py

Run LIVE and consume with curl (`-N` disables curl's buffering so frames print live):
    uvicorn examples.14_sse_server:app --port 8010      # module path has no dashes? see note
    # (rename to sse_server.py if you hit the dashed-module import issue, or use --factory)
    curl -N -X POST http://localhost:8010/stream
    # data: {"type": "step.start", "name": "answering"}
    # data: {"type": "token", "text": "the "}
    # ...
    # data: {"type": "message.final", "text": ""}
"""

import asyncio
import json

from fastapi import FastAPI
from fastapi.responses import StreamingResponse

app = FastAPI()


def sse(event: dict) -> str:
    """Format one event dict as an SSE `data:` frame. Trailing \\n\\n closes the frame."""
    return f"data: {json.dumps(event)}\n\n"


async def event_stream():
    """Async generator that *is* the agent turn: emit typed events over time."""
    # 1) announce the stage so a UI can light up a "step" box
    yield sse({"type": "step.start", "name": "answering"})

    # 2) stream the reply word-by-word as `token` events (simulated latency)
    for word in "the checkout api is throwing 5xx".split():
        yield sse({"type": "token", "text": word + " "})
        await asyncio.sleep(0.05)  # yields the event loop -> natural backpressure

    # 3) exactly one terminator so the client can seal the bubble / stop spinning
    yield sse({"type": "message.final", "text": ""})


@app.post("/stream")
async def stream():
    # StreamingResponse + media_type text/event-stream is all FastAPI needs; it
    # consumes the generator and flushes each yield as it is produced.
    return StreamingResponse(event_stream(), media_type="text/event-stream")


def _parse_sse(raw: str):
    """Parse a raw SSE body into event dicts (split on \\n\\n, read `data:` lines)."""
    events = []
    for block in raw.split("\n\n"):
        block = block.strip()
        if not block:
            continue
        for line in block.split("\n"):
            if line.startswith("data:"):
                events.append(json.loads(line[len("data:"):].strip()))
    return events


def _self_test():
    """Consume our own stream in-process with TestClient — verifiable without uvicorn."""
    from fastapi.testclient import TestClient  # needs httpx installed

    client = TestClient(app)
    resp = client.post("/stream")
    assert resp.status_code == 200, resp.status_code
    ctype = resp.headers["content-type"]
    assert ctype.startswith("text/event-stream"), ctype

    events = _parse_sse(resp.text)
    print(f"content-type: {ctype}")
    print(f"parsed {len(events)} events:")
    for ev in events:
        print("  ", ev)

    # sanity: the contract's bookends must be present exactly as agreed
    types = [e["type"] for e in events]
    assert types[0] == "step.start", types
    assert types[-1] == "message.final", types
    assert types.count("message.final") == 1, "exactly one terminator"
    assert "token" in types, types
    print("OK: contract satisfied (step.start ... token ... message.final)")


if __name__ == "__main__":
    _self_test()
