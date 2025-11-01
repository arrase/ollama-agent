"""Herramientas built-in para el agente."""

import subprocess
from typing import Any


def execute_command(command: str) -> dict[str, Any]:
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
