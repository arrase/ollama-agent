"""Main entry point of the application."""

import argparse
import asyncio
from .config import Config
from .agent import ALLOWED_REASONING_EFFORTS, OllamaAgent
from .tui import ChatInterface


def parse_arguments():
    """
    Parses command line arguments.
    
    Returns:
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Ollama Agent - AI agent to interact with local models"
    )
    parser.add_argument(
        "--model",
        type=str,
        help="Specify the AI model to use (default: gpt-oss:20b)"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        help="Non-interactive mode, provide a prompt directly from the command line"
    )
    parser.add_argument(
        "--effort",
        type=str,
        choices=list(ALLOWED_REASONING_EFFORTS),
        help="Set reasoning effort level (low, medium, high)"
    )
    return parser.parse_args()


async def run_non_interactive(agent: OllamaAgent, prompt: str):
    """
    Runs the agent in non-interactive mode.
    
    Args:
        agent: The AI agent.
        prompt: The user's prompt.
    """
    print(f"User: {prompt}")
    print("Agent: thinking...")
    response = await agent.run_async(prompt)
    print(f"Agent: {response}")


def main():
    """Main entry point."""
    args = parse_arguments()
    
    # Load configuration
    config = Config()
    config_data = config.load()
    
    # Determine the model to use
    model = args.model if args.model else config_data.get("model", "gpt-oss:20b")
    base_url = config_data.get("base_url", "http://localhost:11434/v1/")
    api_key = config_data.get("api_key", "ollama")
    reasoning_effort = args.effort if args.effort else config_data.get("reasoning_effort", "medium")
    
    # Create the agent
    agent = OllamaAgent(model=model, base_url=base_url, api_key=api_key, reasoning_effort=reasoning_effort)
    
    # Non-interactive or interactive mode
    if args.prompt:
        # Non-interactive mode
        asyncio.run(run_non_interactive(agent, args.prompt))
    else:
        # Interactive mode with TUI
        app = ChatInterface(agent)
        app.run()


if __name__ == "__main__":
    main()
