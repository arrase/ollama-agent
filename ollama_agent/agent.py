"""Agente de IA usando openai-agents y Ollama."""

from openai import AsyncOpenAI
from agents import Agent, Runner, function_tool, set_default_openai_client, set_tracing_disabled, set_default_openai_api


@function_tool
def execute_command(command: str) -> dict:
    """
    Ejecuta un comando del sistema operativo local.
    
    Args:
        command: El comando a ejecutar en el shell del sistema.
        
    Returns:
        Un diccionario con el resultado de la ejecución:
        - success: True si el comando se ejecutó correctamente
        - stdout: La salida estándar del comando
        - stderr: La salida de error del comando
        - exit_code: El código de salida del comando
    """
    import subprocess
    
    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=30
        )
        
        return {
            "success": result.returncode == 0,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "exit_code": result.returncode
        }
    except subprocess.TimeoutExpired:
        return {
            "success": False,
            "stdout": "",
            "stderr": "Error: El comando excedió el tiempo límite de 30 segundos",
            "exit_code": -1
        }
    except Exception as e:
        return {
            "success": False,
            "stdout": "",
            "stderr": f"Error al ejecutar el comando: {str(e)}",
            "exit_code": -1
        }


class OllamaAgent:
    """Agente de IA que se conecta a Ollama."""
    
    def __init__(self, model: str, base_url: str = "http://localhost:11434/v1/", api_key: str = "ollama"):
        """
        Inicializa el agente.
        
        Args:
            model: Nombre del modelo a utilizar.
            base_url: URL base del servidor Ollama.
            api_key: API key (requerida pero ignorada por Ollama).
        """
        self.model = model
        self.base_url = base_url
        self.api_key = api_key
        
        # Deshabilitar tracing para evitar intentos de conectarse a OpenAI
        set_tracing_disabled(True)
        
        # Configurar para usar chat_completions en lugar de responses
        set_default_openai_api("chat_completions")
        
        # Crear cliente OpenAI compatible con Ollama
        self.client = AsyncOpenAI(
            base_url=base_url,
            api_key=api_key
        )
        
        # Configurar el cliente globalmente para el agente
        set_default_openai_client(self.client, use_for_tracing=False)
        
        # Crear el agente con la herramienta de ejecución de comandos
        self.agent = Agent(
            name="Ollama Assistant",
            instructions="""Eres un asistente de IA útil que puede ayudar con diversas tareas.
Tienes acceso a una herramienta que te permite ejecutar comandos del sistema operativo.""",
            model=model,
            tools=[execute_command]
        )
    
    def run(self, prompt: str) -> str:
        """
        Ejecuta el agente con un prompt (versión síncrona fallback).
        
        Args:
            prompt: El prompt del usuario.
            
        Returns:
            La respuesta del agente.
        """
        import asyncio
        return asyncio.run(self.run_async(prompt))
    
    async def run_async(self, prompt: str) -> str:
        """
        Ejecuta el agente de forma asíncrona.
        
        Args:
            prompt: El prompt del usuario.
            
        Returns:
            La respuesta del agente.
        """
        try:
            result = await Runner.run(self.agent, input=prompt)
            return str(result.final_output)
        except Exception as e:
            return f"Error: {str(e)}"

