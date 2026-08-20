# 03 · Model Context Protocol (MCP)

> Expose your tools once behind a standard protocol so any client — Claude Desktop, an MCP inspector, or a LangGraph agent — can call them without bespoke glue.

## 1. Mental model — what MCP is, why a standard tool protocol matters

In tutorial 02 you learned the *tool-call loop*: you hand the model JSON schemas, it emits a
tool call, you execute a Python function, you feed the result back. That works, but the wiring
between "the model wants to call `query_logs`" and "the Python function that runs it" is **bespoke
to your app**. Every app reinvents: how tools are discovered, how arguments are shaped, how the
function is actually invoked, how errors come back.

**MCP (Model Context Protocol)** is a standard that factors that wiring out into a protocol.
Think of it as **"LSP for LLM tools"** (the Language Server Protocol that lets any editor talk to
any language's tooling), or **"USB for models"** — one connector, many devices.

The vocabulary:

- **Server** — a process that *exposes capabilities*. Three kinds:
  - **Tools** — functions the model can call (`query_logs`, `restart_service`). Side-effecting.
  - **Resources** — read-only data addressable by URI (`file://…`, `runbook://checkout-api`).
    Think GET; no side effects.
  - **Prompts** — reusable prompt templates the client can surface to the user.
- **Client** — a process that *connects to servers* and makes their capabilities available to a
  model. Claude Desktop is a client; so is the MCP inspector; so is a LangGraph agent wired up
  with `langchain-mcp-adapters`.
- **Host** — the app the human uses (Claude Desktop, your IDE, your agent). One host runs one or
  more clients, each talking to one server.
- **Transport** — how bytes move between client and server:
  - **stdio** — the client *launches the server as a subprocess* and talks over stdin/stdout.
    Local, zero network, one client per process. This is what our project uses.
  - **Streamable HTTP / SSE** — the server is a long-lived HTTP endpoint; many clients connect
    over the network. For remote/shared servers.
- The wire format underneath is **JSON-RPC 2.0**. You rarely touch it directly.

**Why a standard matters.** Without MCP, exposing your SRE tools to Claude Desktop *and* to your
own agent *and* to a colleague's script means three integrations. With MCP you write the server
**once** and all three clients speak the same protocol. That is the whole pitch: tools become a
reusable, swappable unit instead of app-internal plumbing.

`FastMCP` is the high-level Python API in the official `mcp` SDK. You decorate a plain function
with `@mcp.tool()` and it auto-generates the JSON schema from the function's **signature and
docstring** — the same schema you hand-wrote in tutorial 02, now derived for you.

## 2. Smallest working example — a tiny FastMCP server + client

Install the SDK (the `cli` extra gives you the `mcp` command-line inspector):

```bash
pip install "mcp[cli]"
```

### 2a. A server with two tools

```python
# tiny_server.py
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("demo")            # server name shown to clients

@mcp.tool()
def add(a: int, b: int) -> int:
    """Add two integers."""     # docstring becomes the tool description
    return a + b

@mcp.tool()
def greet(name: str = "world") -> str:
    """Say hello to someone."""
    return f"Hello, {name}!"

if __name__ == "__main__":
    mcp.run()                    # stdio transport by default
```

Note what you did *not* write: no JSON schema, no `"parameters": {...}` block. FastMCP reads the
type hints (`a: int`) and the docstring and generates the tool schema for you. Defaults
(`name: str = "world"`) become optional parameters.

### 2b. Inspect it (no client code needed)

```bash
# Interactive web UI to browse + call tools, resources, prompts:
mcp dev tiny_server.py
# or the standalone inspector:
npx @modelcontextprotocol/inspector python tiny_server.py
```

The inspector launches your server over stdio, lists the two tools with their generated schemas,
and lets you invoke them with a form. This is your first debugging move for any MCP server.

### 2c. A programmatic stdio client

A client is what a *host* uses under the hood. Here it is by hand, so you see the protocol:

```python
# tiny_client.py
import asyncio
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

async def main():
    # Client LAUNCHES the server as a subprocess and talks over its stdin/stdout.
    params = StdioServerParameters(command="python", args=["tiny_server.py"])
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()                 # protocol handshake

            tools = await session.list_tools()         # discovery
            print("tools:", [t.name for t in tools.tools])   # ['add', 'greet']

            result = await session.call_tool("add", {"a": 2, "b": 3})
            print("add(2,3) =", result.content[0].text)      # '5'

asyncio.run(main())
```

```bash
python tiny_client.py
```

Two things to internalize: (1) the client **spawns** the server — with stdio there is no port,
no server to start separately; (2) `list_tools()` is *discovery* — the client learns the schemas
at runtime, it doesn't hard-code them. That runtime discovery is exactly what makes tools swappable.

## 3. How the On-Call Copilot uses it

The On-Call Copilot (`~/projects/nvidia-aai`) has a set of SRE tools — diagnostic ones
(`query_logs`, `fetch_metrics`, `list_deploys`, …) and mutating ones (`restart_service`,
`rollback_deploy`, `scale_service`). MCP is how those tools are exposed to the outside world.

### The tool functions — plain Python, one contract

The tools live in [`mcp_server/tools.py`](../nvidia-aai/mcp_server/tools.py). They are ordinary
functions returning a short human-readable string — no MCP-specific code at all:

```python
# mcp_server/tools.py
def query_logs(service="checkout-api", pattern=None, **extra):
    logs = _incident(service).get("logs", [])
    if pattern:
        logs = [l for l in logs if pattern.lower() in l.lower()]
    if not logs:
        return f"No logs found for {service}."
    return f"{len(logs)} log lines for {service}; first error: {logs[0]}"

def restart_service(service="checkout-api", **extra):
    return f"Restarted {service} (rolling)."
```

Keeping the tool bodies MCP-agnostic is deliberate: the same functions are consumed **two ways**
(next two subsections), and swapping the mock backends for real log/metric APIs means editing only
these bodies.

### The server — FastMCP registering the functions

[`mcp_server/server.py`](../nvidia-aai/mcp_server/server.py) turns those functions into an MCP
server. Instead of decorating each with `@mcp.tool()`, it registers them in a loop — the decorator
is just a function, so `mcp.tool()(fn)` is equivalent to `@mcp.tool()` on `fn`:

```python
# mcp_server/server.py
from mcp.server.fastmcp import FastMCP
from . import tools

mcp = FastMCP("sre")

for _name in ("query_logs", "fetch_metrics", "list_deploys", "search_code", "recent_diffs",
              "search_runbooks", "restart_service", "rollback_deploy", "scale_service"):
    mcp.tool()(getattr(tools, _name))

if __name__ == "__main__":
    mcp.run()                    # stdio transport by default
```

Run and inspect it exactly like the tiny example:

```bash
python -m mcp_server.server
npx @modelcontextprotocol/inspector python -m mcp_server.server
```

### The transport config — stdio, declared in YAML

[`config/tools.yaml`](../nvidia-aai/config/tools.yaml) declares the server and its transport
(config-driven design, tutorial 13). A client reads this to know *how to launch* the server:

```yaml
# config/tools.yaml
servers:
  sre:
    transport: stdio            # local stdio MCP server (mcp_server/server.py)
    command: ["python", "-m", "mcp_server.server"]   # named mcp_server/ to avoid shadowing the `mcp` SDK
```

The `command` is precisely the `StdioServerParameters` from §2c: a client will run
`python -m mcp_server.server` as a subprocess and speak MCP over its stdin/stdout.

### In-process vs over-the-wire — the key distinction

Right now the app does **not** go through MCP. [`app/tools.py`](../nvidia-aai/app/tools.py) imports
the tool functions directly and calls them **in-process** — a plain Python function call, no
protocol, no subprocess:

```python
# app/tools.py
from mcp_server import tools as _t

def call_raw(name, args=None):
    """Invoke a tool; RAISES on failure so the resilience layer can retry/trip."""
    if name in FAILING:
        raise RuntimeError(f"{name} is unavailable")
    fn = getattr(_t, name, None)
    if not fn:
        raise ValueError(f"unknown tool: {name}")
    return fn(**(args or {}))
```

This is a real and reasonable design choice. **In-process** is faster (no serialization, no
subprocess), simpler to debug, and dodges an extra failure mode — good for the app's own hot path,
which also wraps calls in retries + a circuit breaker (`app/resilience.py`, tutorial 09).
**Over-the-wire via MCP** is what you need for *external* clients that can't just `import` your
Python: Claude Desktop, the inspector, a teammate's script.

The crucial point for interviews: **both call the identical functions.** `mcp_server/server.py`
registers them; `app/tools.py` imports them. There is one source of truth for tool behavior, exposed
two ways. Wiring the app's own LangGraph agent to the server over stdio via
**`langchain-mcp-adapters`** is the **M1** milestone — so the agent goes through the protocol too,
proving the tools work end-to-end as an MCP server (§6, exercise 5).

### The two real gotchas this project hit

**Gotcha 1 — never name a directory `mcp/`.** The obvious name for the server package is `mcp/`.
Do that and `from mcp.server.fastmcp import FastMCP` resolves to *your* directory, shadowing the
installed `mcp` SDK — you get `ImportError: cannot import name 'server'` or similar, and it looks
like the SDK is broken. The fix (visible in the config comment above) is to name the package
**`mcp_server/`**. General rule: never name a local package after a third-party import you depend on.

**Gotcha 2 — FastMCP rejects `**_` parameter names.** The tool functions all take `**extra` so
callers can pass surplus args without breaking. The natural throwaway name is `**_`, but FastMCP
introspects the signature to build the schema and **rejects `_` as a parameter name** (it validates
parameter names and a bare underscore isn't accepted). Name it `**extra` (or anything real) and it
works. That is why every function in `tools.py` ends with `**extra`, not `**_`.

## 4. Build it up — variations

### 4a. Richer tool schemas with type hints + docstrings

FastMCP generates the schema from the signature. Give it more and clients see more. `Field`
descriptions and `Literal` enums flow straight into the JSON schema:

```python
from typing import Literal
from pydantic import Field
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("sre")

@mcp.tool()
def scale_service(
    service: str = "checkout-api",
    replicas: int = Field(1, ge=1, le=50, description="Target replica count"),
    strategy: Literal["rolling", "recreate"] = "rolling",
) -> str:
    """Scale a service's replica count."""
    return f"Scaled {service} to {replicas} replicas via {strategy}."
```

The `ge`/`le` bounds and the `Literal` enum become validation the client can enforce *before*
calling — a free guardrail (compare tutorial 08).

### 4b. A reusable stdio client over the project server

Point the §2c client at the real server to confirm it works over the wire:

```python
import asyncio
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

async def main():
    params = StdioServerParameters(command="python", args=["-m", "mcp_server.server"])
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = await session.list_tools()
            print("exposed:", [t.name for t in tools.tools])
            out = await session.call_tool("fetch_metrics", {"service": "checkout-api"})
            print(out.content[0].text)

asyncio.run(main())
```

Run from the repo root (so `-m mcp_server.server` resolves) and you'll see all nine tools plus a
real metrics summary — proof the server is client-ready.

### 4c. Add a resource (read-only, URI-addressed)

Tools *do* things; **resources** *are* things. Expose runbooks as resources so a client can read
them without a side-effecting tool call:

```python
@mcp.resource("runbook://{service}")
def runbook(service: str) -> str:
    """The runbook for a given service."""
    path = RUNBOOKS / f"{service}.md"
    return path.read_text() if path.exists() else f"No runbook for {service}."
```

The client lists `runbook://checkout-api` and fetches it via `session.read_resource(...)`. Use a
resource when the answer is *data to read*, a tool when it's an *action to run*. (The project's
`search_runbooks` is a *tool* because it does retrieval/ranking — that's computation, not a plain
GET; contrast with tutorial 04.)

### 4d. Connect it to Claude Desktop

Claude Desktop is a ready-made MCP host. Add the server to its config
(`~/Library/Application Support/Claude/claude_desktop_config.json` on macOS):

```json
{
  "mcpServers": {
    "sre": {
      "command": "python",
      "args": ["-m", "mcp_server.server"],
      "cwd": "/Users/sachinkale/projects/nvidia-aai"
    }
  }
}
```

Restart Claude Desktop; the SRE tools appear behind the tools icon and Claude can call
`query_logs`, `fetch_metrics`, etc. **This is the payoff of the standard**: zero new code — the
same `command`/`args` from `tools.yaml`, now driving a different host. (`cwd` matters so
`-m mcp_server.server` resolves and the mock data under `data/` is found.)

## 5. Gotchas & pitfalls

- **[REAL] Don't name a local dir `mcp/`.** It shadows the installed `mcp` SDK and every
  `from mcp… import` breaks in confusing ways. The project uses **`mcp_server/`**. Rule: never name
  a package after a dependency you import.
- **[REAL] FastMCP rejects `**_`.** It introspects parameter names to build the schema; a bare
  underscore is rejected. Use **`**extra`** (as every function in `tools.py` does) for "swallow
  surplus kwargs."
- **Type-hint everything.** FastMCP derives the schema from hints + docstring. An un-hinted arg
  becomes an untyped/`any` parameter and clients lose validation. The docstring *is* the tool
  description the model reads — write it for the model.
- **Tools do, resources read.** Model side-effecting actions as tools; read-only data as resources.
  Mixing them up gives clients the wrong affordances (e.g. a "tool" the model calls just to read a
  file wastes a round trip and reads as an action).
- **stdio = one client, one subprocess, local.** For many concurrent/remote clients use Streamable
  HTTP transport instead. Don't reach for HTTP when a local subprocess suffices — stdio is simpler
  and has no port/auth surface.
- **In-process is a legitimate choice for your own hot path.** MCP earns its keep at process
  boundaries (external clients, language mismatch, isolation). The project calls tools in-process in
  `app/tools.py` for speed and only goes over MCP for *external* consumers — don't add a subprocess
  hop you don't need.
- **stdout belongs to the protocol.** With stdio, anything you `print()` to stdout corrupts the
  JSON-RPC stream. Log to **stderr** (or a file), never stdout, inside a stdio server.
- **Keep tool bodies transport-agnostic.** The project's functions know nothing about MCP, which is
  why the same code serves both the in-process and over-the-wire paths. Don't leak protocol details
  into business logic.

## ✅ Best Practices

- **One server per capability domain.** Group related tools (SRE diagnostics + mutations) into a single focused server rather than a monolith spanning unrelated domains — it keeps discovery, permissions, and deploys coherent.
- **Name tools like an API, describe them for the model.** Give each tool a clear verb-noun name (`restart_service`, not `do2`) and a docstring that reads as instructions to the model — the description *is* the contract the model plans against.
- **Separate read-only tools from mutating ones.** Keep diagnostics (`query_logs`, `fetch_metrics`) distinct from side-effecting actions (`restart_service`, `rollback_deploy`) so hosts can gate, confirm, or sandbox the dangerous set independently.
- **Version the tool contract.** Treat tool names, argument shapes, and enums as a public API; add fields rather than repurpose them, and bump a version when you must break a signature so existing clients don't silently misfire.
- **Prefer in-process for your hot path, expose over stdio for reuse.** Call tools directly in-process where latency matters, and stand up the same functions behind an stdio MCP server for external clients — one source of truth, two access paths.
- **Scope and secure server access.** Run the server with least-privilege credentials, keep secrets out of tool arguments and logs, and don't expose a mutating server over the network without auth on the transport.
- **Test every server with the MCP inspector.** Before wiring a client, launch `mcp dev` / the inspector, confirm each tool's generated schema and behavior, and use it as your first debugging move whenever a tool misbehaves.
- **Never name a local package after the `mcp` SDK.** Name your package `mcp_server/` (or similar) so `from mcp… import` always resolves to the installed SDK, avoiding import shadowing that looks like an SDK bug.

## 6. Exercises

1. **Stand it up and inspect.** From the repo root run
   `npx @modelcontextprotocol/inspector python -m mcp_server.server`. List all nine tools, then call
   `list_deploys` and `fetch_metrics` for `checkout-api` from the inspector UI. Confirm the
   generated schemas match the `parameters` blocks in `config/tools.yaml`.
2. **Reproduce Gotcha 1.** Copy `mcp_server/` to a dir literally named `mcp/`, try to import
   `FastMCP` from inside it, and read the error. Then reproduce Gotcha 2: change one tool's `**extra`
   to `**_`, start the server, and capture FastMCP's rejection. Write down both error messages —
   interviewers love a war story you can explain.
3. **Write a stdio client (§4b).** Discover the tool list programmatically and call `query_logs`
   with a `pattern` argument. Verify you get the filtered summary string back over the wire.
4. **Add a resource.** Implement `runbook://{service}` (§4c) exposing files under `data/runbooks/`,
   then read one from the inspector. Argue in two sentences why this is a resource, not a tool.
5. **[M1] Wire a LangGraph agent to the server via `langchain-mcp-adapters`.** Install
   `langchain-mcp-adapters`, use `MultiServerMCPClient` to launch `python -m mcp_server.server` over
   stdio, load its tools with `client.get_tools()`, and bind them to a LangGraph agent (tutorial 05).
   Have the agent diagnose the `checkout-api` incident using the MCP tools instead of the in-process
   `app/tools.py` path. This is the real M1 milestone.
6. **Connect to Claude Desktop (§4d).** Register the server, restart, and get Claude to call
   `fetch_metrics` and `list_deploys` unprompted-by-code. Note what changed vs the in-process path:
   nothing but the host.
7. **Stretch:** compare in-process (`app/tools.py`) vs MCP latency for one tool call in a tight
   loop. Quantify the subprocess/serialization cost so you can defend *when* MCP is worth it.

## 7. Connections

- **[02-tool-and-function-calling.md](02-tool-and-function-calling.md)** — MCP standardizes the very
  tool-call loop you built by hand there; `@mcp.tool()` auto-generates the JSON schemas you wrote
  manually.
- **[04-agentic-rag.md](04-agentic-rag.md)** — `search_runbooks` is an MCP tool whose backend is
  vector retrieval (Chroma); MCP is the interface, RAG is the implementation.
- **[05-langgraph.md](05-langgraph.md)** — the M1 step binds the MCP server's tools to a LangGraph
  agent via `langchain-mcp-adapters`, so the agent calls tools over the protocol.

## 8. Further reading

- MCP specification — https://modelcontextprotocol.io/specification
- MCP intro & concepts (tools, resources, prompts, transports) — https://modelcontextprotocol.io
- Python SDK (`mcp`, FastMCP) — https://github.com/modelcontextprotocol/python-sdk
- MCP Inspector — https://github.com/modelcontextprotocol/inspector
- `langchain-mcp-adapters` — https://github.com/langchain-ai/langchain-mcp-adapters
- Claude Desktop MCP setup — https://modelcontextprotocol.io/quickstart/user
- Example servers — https://github.com/modelcontextprotocol/servers
