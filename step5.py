# Give AI a real Slack task. See it return a tool_call (not text). We won't execute it yet — just see what AI picks.

import asyncio
import os
from dotenv import load_dotenv
from groq import Groq
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

CHANNEL_ID = "C0AK1769FL0"   

async def main():
    server = StdioServerParameters(command="python", args=["server.py"])

    async with stdio_client(server) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            mcp_tools = await session.list_tools()
            groq_tools = []
            for tool in mcp_tools.tools:
                groq_tools.append({
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": tool.inputSchema
                    }
                })

            # Give AI a real task this time
            response = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {"role": "user", "content": f"Read the last 3 messages from channel {CHANNEL_ID}"}
                ],
                tools=groq_tools
            )

            msg = response.choices[0].message
            print("🤖 AI text reply:", msg.content)
            print("🔧 AI wants to call:", msg.tool_calls)

            # Inspect what the AI picked
            if msg.tool_calls:
                for tc in msg.tool_calls:
                    print(f"\n  Tool name : {tc.function.name}")
                    print(f"  Arguments : {tc.function.arguments}")

asyncio.run(main())