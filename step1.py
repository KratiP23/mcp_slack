# connect to mcp server and list tools

import asyncio
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

async def main():
    server = StdioServerParameters(
        command="python",
        args=["server.py"]
    )

    async with stdio_client(server) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            print("✅ Connected!")

            tools = await session.list_tools()
            for tool in tools.tools:
                print(f"  Tool found: {tool.name}")

asyncio.run(main())
