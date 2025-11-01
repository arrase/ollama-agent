"""Punto de entrada principal de la aplicación."""

import argparse
import asyncio
from .config import Config
from .agent import OllamaAgent
from .tui import ChatInterface


def parse_arguments():
    """
    Analiza los argumentos de la línea de comandos.
    
    Returns:
        Argumentos parseados.
    """
    parser = argparse.ArgumentParser(
        description="Ollama Agent - Agente de IA para interactuar con modelos locales"
    )
    parser.add_argument(
        "--model",
        type=str,
        help="Especifica el modelo de IA a utilizar (por defecto: gpt-oss:20b)"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        help="Modo no interactivo, proporciona un prompt directamente desde la línea de comandos"
    )
    return parser.parse_args()


async def run_non_interactive(agent: OllamaAgent, prompt: str):
    """
    Ejecuta el agente en modo no interactivo.
    
    Args:
        agent: El agente de IA.
        prompt: El prompt del usuario.
    """
    print(f"Usuario: {prompt}")
    print("Agente: pensando...")
    response = await agent.run_async(prompt)
    print(f"Agente: {response}")


def main():
    """Punto de entrada principal."""
    args = parse_arguments()
    
    # Cargar configuración
    config = Config()
    config_data = config.load()
    
    # Determinar el modelo a usar
    model = args.model if args.model else config_data.get("model", "gpt-oss:20b")
    base_url = config_data.get("base_url", "http://localhost:11434/v1/")
    api_key = config_data.get("api_key", "ollama")
    
    # Crear el agente
    agent = OllamaAgent(model=model, base_url=base_url, api_key=api_key)
    
    # Modo no interactivo o interactivo
    if args.prompt:
        # Modo no interactivo
        asyncio.run(run_non_interactive(agent, args.prompt))
    else:
        # Modo interactivo con TUI
        app = ChatInterface(agent)
        app.run()


if __name__ == "__main__":
    main()
