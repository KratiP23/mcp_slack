import asyncio
import json
import os
from dotenv import load_dotenv
from groq import Groq
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

load_dotenv()

groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))


# ════════════════════════════════════════════════════════
# HELPER 1 — Convert MCP tools → Groq format
# ════════════════════════════════════════════════════════

def mcp_tools_to_groq_tools(mcp_tools) -> list[dict]:

    print("\n[HELPER 1] Converting MCP tools to Groq format...")

    tools = []
    for tool in mcp_tools:
        print(f"  [HELPER 1] Processing tool: '{tool.name}'")
        tools.append({
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description or "",
                "parameters": tool.inputSchema,
            },
        })

    print(f"  [HELPER 1] Done. Total tools converted: {len(tools)}")
    return tools


# ════════════════════════════════════════════════════════
# HELPER 2 — Actually run a tool via MCP
# ════════════════════════════════════════════════════════

async def call_mcp_tool(session: ClientSession, tool_name: str, tool_input: dict):

    print(f"\n  [HELPER 2] Calling MCP tool: '{tool_name}'")
    print(f"  [HELPER 2] With input: {tool_input}")

    result = await session.call_tool(tool_name, tool_input)

    print(f"  [HELPER 2] Raw result from MCP: {result}")

    texts = []
    for block in result.content:
        if hasattr(block, "text") and block.text:
            texts.append(block.text)

    final_output = "\n".join(texts) if texts else "No response"

    print(f"  [HELPER 2] Cleaned output:\n{final_output}")

    return final_output


# ════════════════════════════════════════════════════════
# AGENT LOOP — The brain of the whole system
# ════════════════════════════════════════════════════════

async def run_agent_turn(session, groq_tools, messages):

    print("\n[AGENT LOOP] Starting agent turn...")
    print(f"[AGENT LOOP] Total messages in history: {len(messages)}")

    loop_count = 0

    while True:

        loop_count += 1
        print(f"\n[AGENT LOOP] ── Loop #{loop_count} ──────────────────────")

        # ── BREAKPOINT 1: What are we sending to AI? ──
        print(f"[BP1] Sending {len(messages)} messages to Groq AI...")
        print(f"[BP1] Last message role : '{messages[-1]['role']}'")
        print(f"[BP1] Last message content: '{str(messages[-1].get('content',''))[:100]}'")

        # ── API CALL ──
        response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=messages,
            tools=groq_tools,
            tool_choice="auto",
            max_tokens=1024,
        )

        # ── BREAKPOINT 2: What did AI respond with? ──
        msg = response.choices[0].message
        print(f"\n[BP2] AI responded.")
        print(f"[BP2] msg.content    : {msg.content}")
        print(f"[BP2] msg.tool_calls : {msg.tool_calls}")

        tool_calls = msg.tool_calls or []

        # ── BREAKPOINT 3: Did AI want to call a tool? ──
        if tool_calls:

            print(f"\n[BP3] AI wants to call {len(tool_calls)} tool(s). Entering tool execution...")

            # Save AI's decision into message history
            messages.append({
                "role": "assistant",
                "content": msg.content or "",
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments
                        }
                    }
                    for tc in tool_calls
                ]
            })
            print(f"[BP3] Saved AI decision to message history.")

            # ── BREAKPOINT 4: Loop through each tool call ──
            for i, tool_call in enumerate(tool_calls):

                tool_name  = tool_call.function.name
                tool_input = json.loads(tool_call.function.arguments)

                print(f"\n[BP4] Executing tool {i+1}/{len(tool_calls)}:")
                print(f"[BP4]   tool_call.id : {tool_call.id}")
                print(f"[BP4]   tool_name    : {tool_name}")
                print(f"[BP4]   tool_input   : {tool_input}")

                # ── Run the actual tool via MCP ──
                output = await call_mcp_tool(session, tool_name, tool_input)

                # ── BREAKPOINT 5: Tool ran, what did we get back? ──
                print(f"\n[BP5] Tool '{tool_name}' finished.")
                print(f"[BP5] Output (first 200 chars): {output[:200]}")

                # Save tool result to history so AI can read it
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": output,
                })
                print(f"[BP5] Saved tool result to message history.")

            # ── BREAKPOINT 6: All tools done, looping back to AI ──
            print(f"\n[BP6] All tools executed. Looping back to AI with results...")

        else:

            # ── BREAKPOINT 7: AI gave a plain text reply — we are done ──
            print(f"\n[BP7] No tool calls. AI gave a final text reply. Breaking loop.")

            reply = msg.content or ""

            print(f"[BP7] Final reply (full):\n{reply}")

            messages.append({
                "role": "assistant",
                "content": reply
            })
            print(f"[BP7] Saved final reply to message history.")
            print(f"[BP7] Total loops taken: {loop_count}")

            # Print clean reply for the user
            print(f"\n🤖 {reply}\n")

            print("\n[HISTORY] Messages saved so far:")
            for i, m in enumerate(messages):
                print(f"  {i}. [{m['role']}] {str(m.get('content',''))[:100]}")
        
            
            break  # exit the while loop


# ════════════════════════════════════════════════════════
# MAIN — Startup + chat loop
# ════════════════════════════════════════════════════════

async def main():

    # ── BREAKPOINT 8: App starting ──
    print("\n[BP8] App starting...")

    print("=" * 55)
    print("   Slack AI Assistant  (Groq + LLaMA 3.3)")
    print("=" * 55)
    print("Examples:  read last 5 messages from C08XXXXXX")
    print("           send 'Hello!' to C08XXXXXX")
    print("           summarize messages from C08XXXXXX")
    print("Type 'exit' to quit.\n")

    # ── BREAKPOINT 9: Starting MCP server ──
    print("[BP9] Setting up MCP server connection...")

    server = StdioServerParameters(
        command="d:/slack-mcp/venv/Scripts/python.exe",
        args=["d:/slack-mcp/server.py"],
    )

    print("[BP9] StdioServerParameters created. Starting server.py as subprocess...")

    async with stdio_client(server) as (read, write):

        print("[BP9] stdio_client opened. read/write pipes established.")

        async with ClientSession(read, write) as session:

            # ── BREAKPOINT 10: Handshake with MCP ──
            print("\n[BP10] ClientSession created. Sending initialize handshake...")
            await session.initialize()
            print("[BP10] ✅ MCP server initialized and ready!")

            # ── BREAKPOINT 11: Getting tool list ──
            print("\n[BP11] Asking MCP server for list of tools...")
            mcp_tools_response = await session.list_tools()
            print(f"[BP11] MCP returned {len(mcp_tools_response.tools)} tool(s):")
            for t in mcp_tools_response.tools:
                print(f"  [BP11]  - {t.name}: {t.description}")

            # ── BREAKPOINT 12: Converting tools ──
            print("\n[BP12] Converting tools to Groq format...")
            groq_tools = mcp_tools_to_groq_tools(mcp_tools_response.tools)
            print(f"[BP12] ✅ Groq tools ready: {[t['function']['name'] for t in groq_tools]}")

            # ── BREAKPOINT 13: Building system prompt ──
            print("\n[BP13] Building initial message history with system prompt...")
            messages = [
                {
                    "role": "system",
                    "content": (
                        "You are a concise Slack assistant. "
                        "Use tools when needed. "
                        "Keep replies short and clear. "
                        "Extract channel IDs directly from the user message."
                    ),
                }
            ]
            print(f"[BP13] System prompt set. Message history initialized with {len(messages)} message.")

            print("\n✅ Ready!\n")

            # ── CHAT LOOP ──
            while True:

                try:
                    user_input = input("You: ").strip()
                except (EOFError, KeyboardInterrupt):
                    print("\n👋 Goodbye!")
                    break

                if not user_input:
                    print("[CHAT] Empty input, skipping...")
                    continue

                if user_input.lower() in ("exit", "quit"):
                    print("👋 Goodbye!")
                    break

                # ── BREAKPOINT 14: User typed something ──
                print(f"\n[BP14] User input received: '{user_input}'")
                print(f"[BP14] Adding to message history...")

                messages.append({
                    "role": "user",
                    "content": user_input
                })

                print(f"[BP14] Message history now has {len(messages)} messages.")
                print(f"[BP14] Handing off to agent loop...")

                await run_agent_turn(session, groq_tools, messages)

                # ── BREAKPOINT 15: Agent turn complete ──
                print(f"[BP15] Agent turn complete. Message history now has {len(messages)} messages.")
                print(f"[BP15] Waiting for next user input...\n")


if __name__ == "__main__":
    asyncio.run(main())