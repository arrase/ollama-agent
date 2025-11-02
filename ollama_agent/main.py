"""Main entry point of the application."""

import argparse
import asyncio

from . import config
from .agent import ALLOWED_REASONING_EFFORTS, OllamaAgent
from .tui import ChatInterface


def create_argument_parser() -> argparse.ArgumentParser:
    """
    Create and configure argument parser.
    
    Returns:
        Configured ArgumentParser instance.
    """
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
    return parser


async def run_non_interactive(agent: OllamaAgent, prompt: str) -> None:
    """
    Run the agent in non-interactive mode.
    
    Args:
        agent: The OllamaAgent instance.
        prompt: User prompt to process.
    """
    print(f"User: {prompt}")
    print("Agent: thinking...")
    response = await agent.run_async(prompt)
    print(f"Agent: {response}")


def create_agent_from_args(args: argparse.Namespace) -> OllamaAgent:
    """
    Create OllamaAgent instance with CLI args overriding config.
    
    Args:
        args: Parsed command line arguments.
        
    Returns:
        Configured OllamaAgent instance.
    """
    cfg = config.get_config()
    
    return OllamaAgent(
        model=args.model or cfg.model,
        base_url=cfg.base_url,
        api_key=cfg.api_key,
        reasoning_effort=args.effort or cfg.reasoning_effort,
        database_path=cfg.database_path
    )


def main() -> None:
    """Main entry point."""
    parser = create_argument_parser()
    args = parser.parse_args()
    agent = create_agent_from_args(args)
    
    if args.prompt:
        asyncio.run(run_non_interactive(agent, args.prompt))
    else:
        ChatInterface(agent).run()


if __name__ == "__main__":
    main()
