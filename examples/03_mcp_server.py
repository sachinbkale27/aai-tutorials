"""Standalone FastMCP server — exposes two tools over the MCP protocol.

This is the "expose your tools once behind a standard protocol" idea from tutorial 03,
section 2. `FastMCP` reads each function's type hints + docstring and auto-generates the
JSON schema a client discovers at runtime — no handwritten `"parameters": {...}` block.

Two REAL gotchas this file deliberately respects:
  1. The file is named 03_mcp_server.py, NOT mcp.py, and lives outside any `mcp/` dir.
     A local `mcp.py`/`mcp/` would shadow the installed `mcp` SDK and break every import.
  2. Never write `**_` to swallow surplus kwargs. FastMCP introspects parameter names to
     build the schema and REJECTS a bare underscore outright:
         InvalidSignature: Parameter _ of <fn> cannot start with '_'
     Use a real name like `**extra` (see the `demo_extra_arg` tool below). Note: on newer
     mcp SDKs `**extra` is surfaced to clients as a REQUIRED `extra` object param, so the
     tools you actually call in the client keep a plain signature.

Also: with stdio transport, stdout belongs to the JSON-RPC stream. Never print() to
stdout inside the server — it corrupts the protocol. (We don't print anything here.)

Dependencies:
    pip install mcp

Run STANDALONE (spawns the stdio server; it waits for a client on stdin, so it will just
block — that's expected, Ctrl-C to quit). This is exactly how a client launches it:
    python -m examples.03_mcp_server        # run from the aai-tutorials repo root
    # or:  python examples/03_mcp_server.py

Normally you don't run this by hand — the client (03_mcp_client.py) spawns it for you,
or an inspector does:  npx @modelcontextprotocol/inspector python examples/03_mcp_server.py
"""

from mcp.server.fastmcp import FastMCP

# The server name is what clients see during the handshake.
mcp = FastMCP("demo")


@mcp.tool()
def add(a: int, b: int) -> int:
    """Add two integers and return their sum."""
    return a + b


@mcp.tool()
def greet(name: str = "world") -> str:
    """Say hello to someone. The default makes `name` an optional parameter."""
    return f"Hello, {name}!"


@mcp.tool()
def demo_extra_arg(service: str = "checkout-api", **extra) -> str:
    """Demonstrates the gotcha: `**extra` is a REAL name FastMCP accepts (`**_` is not).
    This registers fine, whereas `**_` would raise InvalidSignature at decoration time.
    On newer SDKs the client must supply `extra` (an object), so `add`/`greet` above are
    what the client calls in the happy path; this tool exists to prove `**extra` works.
    """
    return f"ok: {service} (surplus kwargs captured: {sorted(extra)})"


if __name__ == "__main__":
    # stdio transport by default: the process talks MCP over its own stdin/stdout.
    # There is no port and no separate server to start — a client subprocess-spawns this.
    mcp.run()
