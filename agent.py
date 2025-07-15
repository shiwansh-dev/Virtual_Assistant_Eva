import os
from dotenv import load_dotenv
from livekit.agents import JobContext, WorkerOptions, cli
from livekit.agents.llm import ChatChunk
from livekit.agents.voice import Agent, AgentSession
from livekit.plugins import deepgram, cartesia, openai, silero  

from mcp_client import MCPServerSse
from mcp_client.agent_tools import MCPToolsIntegration
from prompts import AGENT_INSTRUCTION, SESSION_INSTRUCTION

load_dotenv()

class FunctionAgent(Agent):
    """A LiveKit agent that uses MCP tools from one or more MCP servers."""

    def __init__(self):
        # Initialize tool call message attribute
        self._tool_call_message = None

        super().__init__(
            instructions=AGENT_INSTRUCTION,
            stt=deepgram.STT(
                model="nova-2",
                language="hi"
            ),
            llm=openai.LLM(model="gpt-4o"),  # ✅ LLM remains OpenAI GPT-4o
            tts=cartesia.TTS(
                model="sonic-2",
                voice="56e35e2d-6eb6-4226-ab8b-9776515a7094",
                language="hi"  # ✅ Cartesia TTS with Hindi
                # Optional: add voice={"mode": "id", "id": "your_voice_id"} here
            ),
            vad=silero.VAD.load(),
            allow_interruptions=True
        )

    async def llm_node(self, chat_ctx, tools, model_settings):
        """Override the llm_node to say a message when a tool call is detected."""
        tool_call_detected = False

        async for chunk in super().llm_node(chat_ctx, tools, model_settings):
            if isinstance(chunk, ChatChunk) and chunk.delta and chunk.delta.tool_calls and not tool_call_detected:
                tool_call_detected = True
                self._tool_call_message = "मैं अभी जांच करती हूँ।"
            yield chunk

    async def on_tool_call(self, session, tool_name, args):
        """Called when a tool is about to be executed."""
        if self._tool_call_message:
            await session.say(self._tool_call_message)
            self._tool_call_message = None

        return await super().on_tool_call(session, tool_name, args)

async def entrypoint(ctx: JobContext):
    """Main entrypoint for the LiveKit agent application."""
    mcp_server = MCPServerSse(
        params={"url": os.environ.get("ZAPIER_MCP_URL")},
        cache_tools_list=True,
        name="SSE MCP Server"
    )

    agent = await MCPToolsIntegration.create_agent_with_tools(
        agent_class=FunctionAgent,
        mcp_servers=[mcp_server]
    )

    await ctx.connect()

    session = AgentSession()
    await session.start(agent=agent, room=ctx.room)

    # ✅ Greet the user on session start
    await session.generate_reply(
        instructions=SESSION_INSTRUCTION,
    )

if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
