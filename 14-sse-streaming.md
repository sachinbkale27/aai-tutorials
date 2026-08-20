# 14 · SSE Streaming (Glass-Box)

> Stream a typed event contract from an async Python generator over Server-Sent Events, and reduce it in the browser into a live "glass-box" trace of everything the agent is doing.

---

## 1. Mental model — SSE vs WebSockets vs polling

You have a backend that produces events *over time* — token, then a tool call, then a
result, then a final message — and a browser that wants to render each one the instant it
happens. Three transports can do this:

| Transport | Direction | Framing | Reconnect | Fit here |
|-----------|-----------|---------|-----------|----------|
| **Polling** | client pulls | request/response | manual, wasteful | ❌ latency + no partial output |
| **WebSockets** | full duplex | binary/text frames | you build it | overkill; we only stream *down* |
| **SSE** | server → client | `text/event-stream` over plain HTTP | **built-in** (`Last-Event-ID`) | ✅ one-way token/event stream |

An agent turn is **one-way after the request**: the client POSTs an alert, then only
listens. That's exactly SSE's shape — a long-lived HTTP response whose body never ends,
carrying newline-delimited text frames. No new protocol, no upgrade handshake, works
through ordinary HTTP infrastructure, and it's just `fetch` on the client.

### The `text/event-stream` wire format

An SSE stream is UTF-8 text. Each **event** is a block of `field: value` lines, and events
are separated by a **blank line** (`\n\n`). The fields that matter:

```
data: {"type":"token","text":"Rolling "}\n
\n
```

- `data:` — the payload. Multiple `data:` lines in one event are concatenated with `\n`.
- `event:` — an optional *named* event type (drives `addEventListener('foo', …)` in `EventSource`).
- `id:` — an optional id the browser remembers and replays as `Last-Event-ID` on reconnect.
- `retry:` — reconnect backoff in ms.
- A line starting with `:` is a comment (handy as a keep-alive heartbeat).

The frame terminator is the **double newline**. Miss it and the client buffers your event
forever, waiting for the block to close.

### Why a *typed event contract*

You *could* just stream text tokens. But this app streams **structured events with a
`type` field** — `step.start`, `tool.call`, `rail.fired`, `hitl.required`, `token`,
`message.final` — so the same channel carries *what the agent did*, not just *what it
said*. That contract is the whole product: the frontend reducer switches on `type` and
paints a live trace — steps lighting up, tools resolving, a guardrail firing, an approval
gate pausing the run. This is the "glass-box": you watch the agent think, not just its
final answer. The contract is the API between backend and frontend; both sides only have to
agree on the shape of each event `type`.

---

## 2. Smallest working example — standalone runnable

A tiny FastAPI app that streams an async generator, plus two consumers.

```bash
pip install fastapi uvicorn
```

`mini_sse.py`:

```python
import asyncio, json
from fastapi import FastAPI
from fastapi.responses import StreamingResponse

app = FastAPI()

def sse(event: dict) -> str:
    # a dict -> one SSE `data:` frame. The trailing \n\n is the frame terminator.
    return f"data: {json.dumps(event)}\n\n"

async def gen():
    yield sse({"type": "step.start", "name": "answering"})
    for word in "the checkout api is throwing 5xx".split():
        yield sse({"type": "token", "text": word + " "})
        await asyncio.sleep(0.1)          # simulate token latency
    yield sse({"type": "message.final", "text": ""})

@app.post("/stream")
async def stream():
    return StreamingResponse(gen(), media_type="text/event-stream")
```

Run it:

```bash
uvicorn mini_sse:app --port 8010
```

**Consume with curl** (`-N` disables curl's own buffering so frames print as they arrive):

```bash
curl -N -X POST http://localhost:8010/stream
# data: {"type": "step.start", "name": "answering"}
# data: {"type": "token", "text": "the "}
# data: {"type": "token", "text": "checkout "}
# ...
# data: {"type": "message.final", "text": ""}
```

**Consume in the browser.** Note: the native `EventSource` API is GET-only, so because we
POST a body we read the stream manually with `fetch` + a reader (exactly what this app
does):

```html
<script type="module">
const res = await fetch('http://localhost:8010/stream', { method: 'POST' });
const reader = res.body.getReader();
const dec = new TextDecoder();
let buf = '';
for (;;) {
  const { value, done } = await reader.read();
  if (done) break;
  buf += dec.decode(value, { stream: true });
  const parts = buf.split('\n\n');   // split on frame terminator
  buf = parts.pop();                 // keep the trailing partial frame
  for (const p of parts) {
    const line = p.split('\n').find((l) => l.startsWith('data:'));
    if (line) console.log(JSON.parse(line.slice(5).trim()));
  }
}
</script>
```

That's the entire mechanism. Everything below is this same loop with a richer event
contract and a reducer.

---

## 3. How the On-Call Copilot uses it

The real app is this pattern at production scale. Trace one incident through the five
files.

### The formatter — `app/sse.py`

```python
def sse(event: dict) -> str:
    """Format an event dict as a Server-Sent Events `data:` frame."""
    return f"data: {json.dumps(event)}\n\n"

async def say(text: str, delay: float = 0.04):
    """Stream a reply as word-by-word token events."""
    for word in text.split(" "):
        yield {"type": "token", "text": word + " "}
        await asyncio.sleep(delay)
```

Two primitives. `sse()` is the *wire* layer — dict → `data:` frame. `say()` is a
*generator of token events* reused everywhere the agent "speaks". Note `say()` yields
**dicts**, not frames — framing happens once, at the edge.

### The edge — `app/main.py`

The `/api/incident` route is the seam where events become bytes:

```python
@app.post("/api/incident")
async def incident(req: Request):
    body = await req.json()
    conv_id = body.get("conversationId", "anon")
    alert = body.get("message", "")
    rec = M.TraceRecorder(conv_id, "incident")

    async def gen():
        async for ev in incident_events(conv_id, alert):   # the agent, as a generator
            rec.record(ev)                                   # tap: tee every event to metrics
            yield sse(ev)                                    # frame it and push it down the wire
        rec.finalize()

    return StreamingResponse(gen(), media_type="text/event-stream")
```

Three things worth internalizing:

1. **`StreamingResponse` + `media_type="text/event-stream"`** is all FastAPI needs. It
   consumes the async generator and flushes each yield as it's produced.
2. **The tap.** `rec.record(ev)` tees every event into `TraceRecorder` *as it streams* — so
   `/api/observability` can later reconstruct the whole trace (see
   [10-opentelemetry.md](10-opentelemetry.md)). The stream is the single source of truth for
   both the UI and the metrics.
3. **`/api/incidents/{conv_id}/resume`** is a *second* stream on the same contract. After a
   human-in-the-loop pause, the frontend opens a fresh SSE stream that picks up where the
   first one stopped — same event types, new generator (`resume_events`).

### The contract — `app/incident.py`

`incident_events()` is an async generator that *is* the agent. Every stage `yield`s events
from the shared vocabulary. The full contract emitted:

| Event `type` | Emitted when | Key fields |
|--------------|--------------|-----------|
| `step.start` | a pipeline stage begins | `id`, `name` (e.g. `guardrail.input`, `orchestrator.plan`, `mcp.<tool>`) |
| `step.end` | a stage finishes | `id`, `ok`, `ms` |
| `tool.call` | a worker invokes a tool | `tool`, `args` |
| `tool.result` | the tool returns | `tool`, `ok`, `summary` |
| `rail.fired` | a guardrail triggers | `rail_type`, `name`, `stop`, `reason` |
| `token` | a word of the reply | `text` |
| `hitl.required` | a fix needs human approval | `action`, `args`, `draft`, `reason` |
| `hitl.resolved` | the human decided (resume stream) | `decision` |
| `message.final` | the turn is complete | `text`, `citations` |

(`resilience` events — retries / circuit-breaker — also ride the stream via
`guarded_tool()`; see [06-orchestrator-worker-multi-agent.md](06-orchestrator-worker-multi-agent.md).)

The pipeline reads like a story of yields:

```python
async def incident_events(conv_id, alert):
    # 1) INPUT RAILS — wrap firings in one step so the UI shows a "guardrail" box
    events, blocked, refusal = run_input_rails(alert)
    async for ev in _gate(events):        # yields step.start → rail events → step.end
        yield ev
    if blocked:
        async for ev in say(refusal): yield ev
        yield {"type": "message.final", "text": "", "citations": []}
        return

    # 2) ORCHESTRATOR plan
    yield {"type": "step.start", "id": "s2", "name": "orchestrator.plan"}
    yield {"type": "step.end", "id": "s2", "ok": True, "ms": 100}

    # 3) WORKERS — each streams tool.call → (resilience events) → tool.result
    for i, name in enumerate(G.plan_workers(alert), start=3):
        async for ev in _worker(f"s{i}", AC.worker(name), alert):
            yield ev

    # 4) SYNTHESIS — stream the root-cause reply through output rails (token events)
    async for ev in guarded_reply(...): yield ev

    # 5) HITL GATE — the fix mutates prod, so pause
    needs_approval, reason, rail = execution_gate(fix["action"], fix["args"])
    if needs_approval:
        yield {"type": "rail.fired", "rail_type": "execution", "name": rail, "stop": True, "reason": reason}
        PENDING[conv_id] = {**fix, "citations": citations}   # stash the fix
        yield {"type": "hitl.required", "action": fix["action"], "args": fix["args"],
               "draft": fix["draft"], "reason": reason}
        return                            # STOP — the generator ends; the stream closes
```

The `return` mid-generator is the elegant part: the stream simply **ends** at the approval
gate. No fix runs. The pending fix is stashed in `PENDING[conv_id]`. When the human clicks
approve, `resume_events()` runs on the *resume* route — it emits `hitl.resolved`, pops the
stashed fix, and either runs `_remediate(...)` (which streams `mcp.rollback_deploy` +
tokens + `message.final`) or says "won't run that" and finalizes. See
[07-human-in-the-loop.md](07-human-in-the-loop.md).

### The client reader — `src/api.js`

```js
async function readSSE(res, onEvent) {
  const reader = res.body.getReader();
  const dec = new TextDecoder();
  let buf = '';
  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    buf += dec.decode(value, { stream: true });
    const parts = buf.split('\n\n');   // frame terminator
    buf = parts.pop();                 // last chunk may be a partial frame — hold it
    for (const p of parts) {
      const line = p.split('\n').find((l) => l.startsWith('data:'));
      if (line) onEvent(JSON.parse(line.slice(5).trim()));  // slice off "data:"
    }
  }
}
```

The `buf = parts.pop()` line is the one people get wrong: a network chunk can split a frame
in half, so you **keep the trailing partial** and prepend it to the next chunk. `onEvent`
is called once per fully-received event.

### The reducer — `src/useConversation.js` (the glass-box)

Every event flows into a `switch (e.type)` that mutates React state immutably. This is
where the typed contract pays off — each `type` renders a different piece of UI:

```js
switch (e.type) {
  case 'step.start':   u.steps = [...st.steps, { id: e.id, name: e.name, status: 'running' }]; break;
  case 'step.end':     u.steps = st.steps.map(x => x.id === e.id ? {...x, status: e.ok?'ok':'fail', ms: e.ms} : x); break;
  case 'token': {      // append to the last streaming assistant bubble, or start one
    const last = m[m.length-1];
    if (last?.role === 'assistant' && !last.final) m[m.length-1] = {...last, text: last.text + e.text};
    else m.push({ role:'assistant', text: e.text, final:false });
    break;
  }
  case 'tool.call':    u.tools = [...st.tools, { tool: e.tool, args: e.args, status: 'running' }]; break;
  case 'tool.result':  u.tools = st.tools.map(t => t.tool===e.tool && t.status==='running'
                          ? {...t, status: e.ok?'ok':'fail', summary: e.summary} : t); break;
  case 'rail.fired':   u.rails = [...st.rails, { type: e.rail_type, name: e.name, stop: e.stop, reason: e.reason }]; break;
  case 'hitl.required': u.hitl = e; u.messages = [...st.messages, { role:'system', text:'👀 A specialist is reviewing this…' }]; break;
  case 'hitl.resolved': u.hitl = null; break;
  case 'message.final': // mark the last assistant bubble final + attach citations, clear busy
    if (last?.role === 'assistant') m[m.length-1] = {...last, final:true, citations: e.citations};
    u.busy = false; break;
}
```

Map each event to its render:

- `step.start`/`step.end` → the **timeline** of stages, each flipping running → ok/fail.
- `token` → **live typing** into a single assistant bubble (concatenated, not new bubbles).
- `tool.call`/`tool.result` → a **tool panel**; the result matches the running call and flips its status.
- `rail.fired` → a **guardrail badge** (blocked / gated, with a reason).
- `hitl.required` → renders the **approval card** (`s.hitl`) and drops a "specialist is reviewing" line; `hitl.resolved` clears it.
- `message.final` → seals the bubble, attaches **citations**, and sets `busy = false` so the input re-enables.

`send()` opens the stream via `sendMessage`; `resolve(decision)` opens the resume stream via
`resumeHitl`. Both pass `apply` (the reducer) as the `onEvent` callback. **One contract,
two streams, one reducer.**

---

## 4. Build it up

### Variation A — token streaming (the `say()` pattern)

The core trick: any text reply becomes an async generator of `token` events, and the
reducer concatenates them into one bubble. The key reducer invariant is *"append if the
last message is a non-final assistant bubble, else start a new one"* — that's what turns a
stream of words into typing, and it's why `message.final` must flip `final: true` (so the
*next* turn starts a fresh bubble instead of appending).

```python
async def say(text, delay=0.04):
    for word in text.split(" "):
        yield {"type": "token", "text": word + " "}
        await asyncio.sleep(delay)
```

### Variation B — a reducer for state (why not just append text?)

A naive client does `output += chunk`. That can't render a *trace*. By making events typed
and reducing them into **separate slices** (`steps`, `tools`, `rails`, `hitl`, `messages`),
the same stream drives five independent UI regions. Adding a new visual is "add a `case`",
not "reparse a text blob". This is the single most interview-relevant idea here:
**structured events + a reducer = a glass-box UI for free.**

### Variation C — reconnection with `id:` / `Last-Event-ID`

The native `EventSource` reconnects automatically and replays the last `id:` it saw as a
`Last-Event-ID` request header. To support it, tag events with an id and resume from it:

```python
async def gen(request: Request):
    start = int(request.headers.get("last-event-id", 0))
    for i, ev in enumerate(all_events()):
        if i < start:            # skip what the client already got
            continue
        yield f"id: {i}\n" + sse(ev)
```

```js
const es = new EventSource('/stream');            // GET only
es.onmessage = (m) => onEvent(JSON.parse(m.data));
// on drop, the browser reconnects and sends Last-Event-ID automatically
```

The copilot doesn't use `EventSource` (it needs POST bodies, so it uses `fetch` + reader),
and its "resume" is *semantic* (a HITL decision reopens a stream), not transport-level
replay. Know both: transport reconnect is for dropped sockets; the copilot's resume is a
deliberate second turn. If you need POST *and* auto-reconnect, add your own retry loop
around `readSSE` that re-POSTs with a `Last-Event-ID` header you track client-side.

### Variation D — backpressure

An async generator is naturally back-pressured: `yield` doesn't advance until the ASGI
server has handed the chunk to the socket. If the client is slow, `await`ing the write
slows the generator — memory stays flat because you never build the whole response. The
`await asyncio.sleep(delay)` in `say()` also yields the event loop so other requests
progress. **Never** build a giant list and stream it at the end; that defeats streaming and
spikes memory. Stream lazily, one `yield` at a time.

---

## 5. Gotchas & pitfalls

- **`\n\n` framing is load-bearing.** Every frame ends with a blank line. `sse()` bakes it
  in (`...\n\n`). Forget it and the client's `buf.split('\n\n')` never splits — the event
  hangs unrendered.
- **Buffering / flush.** `StreamingResponse` flushes per `yield`, but reverse proxies buffer
  by default. For **nginx** set `X-Accel-Buffering: no` (or `proxy_buffering off`). Disable
  gzip on the stream — compression buffers. For `curl`, use `-N`. Some CDNs buffer
  `text/event-stream` regardless; test through your real proxy.
- **Exactly one `message.final`.** The reducer treats it as the terminator: it seals the
  bubble and sets `busy = false`. Emit it on **every** path — blocked-by-rails, reject,
  success — or the UI spins forever. In `incident.py` note the `return` after each
  `message.final`; every branch that stops the turn finalizes first.
- **`data:` only, JSON inside.** This app puts the type *inside* the JSON payload
  (`{"type": ...}`) and uses the default `message` event, rather than SSE `event:` names.
  That keeps parsing to one path (`fetch` reader) and one `switch`. Pick one convention and
  hold it.
- **Newlines in payloads.** A raw `\n` inside a `data:` value would start a new SSE field.
  `json.dumps` escapes newlines to `\\n`, so JSON payloads are safe — but never hand-format
  multiline strings into `data:`.
- **Heartbeats.** Idle proxies kill connections. Send `: keep-alive\n\n` comment frames
  every ~15s on long-idle streams; clients ignore comment lines.
- **Errors.** A crash mid-generator just closes the socket — the client sees `done` with no
  `message.final` and hangs. Wrap the generator body and emit a typed
  `{"type":"error","message":...}` (plus a `message.final`) so the UI can show a failure and
  unblock. Then re-raise/log server-side.
- **Client partial-frame handling.** Keep `buf = parts.pop()`. Network chunks don't align to
  frames; the last piece is usually incomplete.
- **Content-Type must be `text/event-stream`.** Anything else and browsers won't treat it as
  a stream (and `EventSource` refuses it outright).

---

## ✅ Best Practices

- **Define a typed event contract first.** Agree on a small vocabulary of `type`d events (`step.start`, `tool.call`, `token`, `message.final`) up front so backend and frontend evolve independently and adding UI is just another `case`.
- **Always end with a terminal event.** Emit exactly one `message.final` on every path — success, rejection, or rail-blocked — so the client can seal the bubble, unset `busy`, and never spin forever.
- **Emit typed error events, not silent socket closes.** On any failure, `yield {"type":"error", ...}` followed by `message.final` so the UI shows a real error and re-enables input, then log/re-raise server-side.
- **Flush promptly and disable proxy buffering.** Stream one `yield` at a time and set `X-Accel-Buffering: no` (nginx `proxy_buffering off`) with gzip off, so events reach the browser the instant they're produced.
- **Send heartbeats and support `Last-Event-ID` reconnection.** Push `: keep-alive\n\n` comment frames on idle streams to survive proxy timeouts, and tag events with `id:` so a reconnecting client resumes from where it left off.
- **Keep the client reducer pure.** Map each event to immutable state slices with no side effects, so the same stream can drive multiple UI regions and replays are deterministic.
- **Use SSE for one-way streaming; reach for WebSockets only when bidirectional.** An agent turn only streams down after the request, so prefer plain-HTTP SSE and add WebSockets solely when the client must push mid-stream.
- **Cap payload sizes per event.** Keep each frame small (stream token-by-token, summarize tool results) so a slow client back-pressures naturally and memory stays flat.

## 6. Exercises

1. **Trace the framing.** Run `mini_sse.py` and `curl -N` it. Then remove the second `\n`
   from `sse()` (make it `\n`) and watch the browser consumer hang. Explain in one sentence
   why curl still prints but the JS reader doesn't.
2. **Add a new event type end-to-end.** Add `{"type": "note", "text": ...}` in
   `incident.py` (e.g. after synthesis), and a `case 'note':` in `useConversation.js` that
   pushes it into a new `notes` slice rendered as a sidebar. Confirm no other file needs to
   change — that's the contract paying off.
3. **Add a heartbeat.** Make `say()` (or the `gen()` in `main.py`) emit a `: ping\n\n`
   comment frame every 5 events. Verify the client's `readSSE` silently ignores it (it looks
   for `data:` lines, so comments are dropped for free).
4. **Add reconnection.** Wrap `readSSE` in `api.js` with a retry loop that re-POSTs on a
   dropped connection, tracking a client-side `lastId`. On the server, add `id: <n>\n`
   before each frame in `main.py`'s `gen()` and skip already-sent events using the
   `Last-Event-ID` header.
5. **Stream real LLM tokens.** Replace `say()` with an Anthropic streaming call: iterate the
   SDK's streamed text deltas and `yield {"type": "token", "text": delta}` for each. The
   reducer needs zero changes — it already concatenates `token` events. (Load the
   `claude-api` skill for the exact streaming API and model ids before you wire it.)
6. **Force a failure.** Make a tool raise inside `_worker`, and add an `error` event +
   `message.final` on the exception path so the UI unblocks. Confirm `busy` returns to
   `false` and the timeline shows a failed step.

---

## 7. Connections

- **[07-human-in-the-loop.md](07-human-in-the-loop.md)** — `hitl.required` ends the stream
  at the approval gate; the resume route opens a second stream (`hitl.resolved` → remediate).
  SSE is the transport that makes the pause/resume feel live.
- **[10-opentelemetry.md](10-opentelemetry.md)** — `main.py` taps every streamed event into
  `TraceRecorder` (`rec.record(ev)`), so the same event stream feeds both the glass-box UI
  and the observability dashboard.
- **[06-orchestrator-worker-multi-agent.md](06-orchestrator-worker-multi-agent.md)** — the
  `step.*` / `tool.*` / resilience events are emitted per worker as the orchestrator fans
  out; SSE is how the multi-agent structure becomes visible in real time.

---

## 8. Further reading

- **MDN — Using server-sent events**: https://developer.mozilla.org/en-US/docs/Web/API/Server-sent_events/Using_server-sent_events
- **MDN — `EventSource`**: https://developer.mozilla.org/en-US/docs/Web/API/EventSource
- **WHATWG HTML spec — the event stream format**: https://html.spec.whatwg.org/multipage/server-sent-events.html
- **FastAPI — `StreamingResponse`**: https://fastapi.tiangolo.com/advanced/custom-response/#streamingresponse
- **Starlette responses (what FastAPI builds on)**: https://www.starlette.io/responses/#streamingresponse
- **Streams API — `ReadableStream.getReader()`** (the browser-side reader): https://developer.mozilla.org/en-US/docs/Web/API/ReadableStream/getReader
