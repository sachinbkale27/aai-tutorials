"""Standalone MCP stdio client — launches the server, discovers tools, calls one.

The companion to 03_mcp_server.py and the payoff of tutorial 03, section 2c: a client is
what a *host* (Claude Desktop, an agent) uses under the hood. Here it is by hand so you see
the protocol. Two things to internalize:

  1. With stdio there is no port. The client SUBPROCESS-SPAWNS the server and talks over its
     stdin/stdout — running this client runs the server too, no separate start needed.
  2. `list_tools()` is runtime DISCOVERY. The client learns the schemas at connect time; it
     does not hard-code them. That runtime discovery is what makes tools swappable.

Dependencies:
    pip install mcp

Run (spawns the server, lists tools, calls `add`, prints results, exits 0):
    python examples/03_mcp_client.py        # run from the aai-tutorials repo root
"""

import asyncio
import os
import sys

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

# Absolute path to the server file that sits next to this client. Passing the resolved path
# (rather than a bare "03_mcp_server.py") means the spawn works no matter the current dir.
SERVER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "03_mcp_server.py")


async def main() -> None:
    # StdioServerParameters IS the launch command: the client will run
    # `python <path>/03_mcp_server.py` as a subprocess and speak MCP over its stdio.
    # Use the SAME interpreter running this client (sys.executable) so the venv matches.
    params = StdioServerParameters(command=sys.executable, args=[SERVER])

    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()  # JSON-RPC protocol handshake

            # --- discovery: ask the server what it exposes ---
            tools = await session.list_tools()
            print("tools:", [t.name for t in tools.tools])  # -> ['add', 'greet', 'demo_extra_arg']

            # --- call a tool by name with a dict of args ---
            result = await session.call_tool("add", {"a": 2, "b": 3})
            # Tool results come back as a list of content blocks; text lives in .text.
            print("add(2, 3) =", result.content[0].text)  # -> '5'

            # --- call the second tool, exercising an optional/defaulted param ---
            hello = await session.call_tool("greet", {"name": "MCP"})
            print("greet('MCP') =", hello.content[0].text)  # -> 'Hello, MCP!'


if __name__ == "__main__":
    asyncio.run(main())
