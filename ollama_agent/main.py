"""Main entry point of the application."""

import argparse
import asyncio
from . import config
from .agent import ALLOWED_REASONING_EFFORTS, OllamaAgent
from .tui import ChatInterface


def parse_arguments():
    """Parses command line arguments."""
    parser = argparse.ArgumentParser(
        description="Ollama Agent - AI agent to interact with local models"
    )
    parser.add_argument(
        "--model",
        type=str,
        help="Specify the AI model to use"
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
    """Runs the agent in non-interactive mode."""
    print(f"User: {prompt}")
    print("Agent: thinking...")
    response = await agent.run_async(prompt)
    print(f"Agent: {response}")


def main():
    """Main entry point."""
    args = parse_arguments()
    
    # CLI args override config values
    agent = OllamaAgent(
        model=args.model or config.get("model"),
        base_url=config.get("base_url"),
        api_key=config.get("api_key"),
        reasoning_effort=args.effort or config.get("reasoning_effort")
    )
    
    if args.prompt:
        asyncio.run(run_non_interactive(agent, args.prompt))
    else:
        ChatInterface(agent).run()


if __name__ == "__main__":
    main()
