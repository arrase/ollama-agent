import asyncio
import time
from contextlib import AsyncExitStack

from ollama_agent.settings import SubAgentSettings, ModelSettings
from ollama_agent.agent.subagents import build_subagents


async def test_performance():
    # Setup mocks
    model_settings = ModelSettings(name="test", base_url="http://localhost:11434")

    settings = []
    for i in range(10):
        # Mute some settings to mock properly without issues or create a valid one
        sa = SubAgentSettings(
            name=f"test{i}",
            description="test",
            mcp_servers={"test": {"command": "sleep", "args": ["0.1"]}},
        )
        settings.append(sa)

    # Note: load_subagent_mcp_tools does the waiting, let's patch it
    import ollama_agent.agent.subagents

    original_load = ollama_agent.agent.subagents.load_subagent_mcp_tools

    async def mock_load(*args, **kwargs):
        await asyncio.sleep(0.1)
        return [{"name": "test_tool"}]

    ollama_agent.agent.subagents.load_subagent_mcp_tools = mock_load

    async with AsyncExitStack() as exit_stack:
        start = time.time()
        await build_subagents(
            settings, model_settings=model_settings, exit_stack=exit_stack
        )
        end = time.time()

    ollama_agent.agent.subagents.load_subagent_mcp_tools = original_load
    return end - start


if __name__ == "__main__":
    t = asyncio.run(test_performance())
    print(f"Time taken: {t:.4f} seconds")
