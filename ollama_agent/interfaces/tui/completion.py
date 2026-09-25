"""Autocompletion helpers and slash command definitions for the TUI."""

from __future__ import annotations

import re
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import ollama
from rich.text import Text

from ...core.models import get_model_thinking_config
from ...i18n import _

# @-mention regex: matches @"quoted", @'quoted', or @bare at word boundaries.
_AT_MENTION_RE = re.compile(r"""(?:^|[\s\(\[\{<])@(?:"([^"]*)|'([^']*)|([^\s"'\(\[\{<>,;]*))$""")

CMD_MODEL = "/model"
CMD_EFFORT = "/effort"
CMD_CONTEXT = "/context"
CMD_PARAMS = "/params"
CMD_SESSION = "/session"
CMD_TASK = "/task"
CMD_SKILL = "/skill"
CMD_RAG = "/rag"
CMD_MCP = "/mcp"
CMD_AGENTS = "/agents"
CMD_QUEUE = "/queue"
CMD_YOLO = "/yolo"
CMD_STEALTH = "/stealth"
CMD_NEW = "/new"
CMD_CLEAR = "/clear"
CMD_EXIT = "/exit"
CMD_QUIT = "/quit"
_CHAT_SCROLL_ID = "#chat-scroll"
_AUTOCOMPLETE_LIST_ID = "#autocomplete-list"


def _list_models_sync(base_url: str) -> list[Any]:
    """Fetch the list of available Ollama models synchronously."""
    client = ollama.Client(host=base_url)
    response = client.list()
    return list(response.models)


def _get_root_commands() -> list[tuple[str, str]]:
    return [
        (CMD_MODEL, _("Manage models")),
        (CMD_EFFORT, _("Show or set reasoning/thinking effort")),
        (CMD_CONTEXT, _("Show or set context window size (num_ctx)")),
        (CMD_PARAMS, _("Manage model sampling parameters")),
        (CMD_SESSION, _("Manage chat sessions")),
        (CMD_TASK, _("Manage saved tasks")),
        (CMD_SKILL, _("Manage skills")),
        (CMD_RAG, _("Manage RAG databases")),
        (CMD_MCP, _("Manage and check MCP servers")),
        (CMD_AGENTS, _("Manage configured subagents")),
        (CMD_QUEUE, _("Show or clear the prompt queue")),
        (CMD_YOLO, _("Toggle YOLO mode or set it explicitly (on/off)")),
        (CMD_STEALTH, _("Toggle stealth mode or set it explicitly (on/off)")),
        (CMD_NEW, _("Start a new chat session and clear the screen")),
        (CMD_CLEAR, _("Start a new chat session and clear the screen (alias for /new)")),
        (CMD_EXIT, _("Exit the REPL")),
        (CMD_QUIT, _("Exit the REPL (alias for /exit)")),
    ]


def _get_subcommands() -> dict[str, list[tuple[str, str]]]:
    queue_rm_desc = _("Remove a prompt from the queue")
    return {
        CMD_MODEL: [
            ("list", _("List available Ollama models")),
            ("set", _("Switch to a different model")),
        ],
        CMD_CONTEXT: [
            ("set", _("Set context window size (e.g. 8192, 16384, max)")),
        ],
        CMD_PARAMS: [
            ("list", _("Show active model parameters and resolution sources")),
            ("set", _("Set a parameter value (e.g. /params set temperature 0.7)")),
        ],
        CMD_SESSION: [
            ("list", _("List all past sessions")),
            ("search", _("Search past sessions by keyword")),
            ("resume", _("Resume a previous session")),
            ("switch", _("Switch to a previous session (alias for resume)")),
            ("new", _("Start a new session")),
            ("export", _("Export session to Markdown")),
            ("delete", _("Delete a session from history")),
        ],
        CMD_TASK: [
            ("list", _("List all saved tasks")),
            ("create", _("Create a task with agent guidance")),
            ("run", _("Run a saved task prompt")),
            ("delete", _("Delete a saved task")),
        ],
        CMD_SKILL: [
            ("list", _("List all available skills")),
            ("show", _("Show skill details and instructions")),
            ("create", _("Create a skill with agent guidance")),
            ("delete", _("Delete a skill")),
        ],
        CMD_RAG: [
            ("status", _("Show current RAG database status")),
            ("list", _("List all RAG databases")),
            ("create", _("Create a new RAG database")),
            ("delete", _("Delete a RAG database")),
            ("load", _("Load a RAG database")),
            ("unload", _("Unload active RAG database")),
            ("add", _("Add file or directory to RAG")),
        ],
        CMD_MCP: [
            ("list", _("List configured MCP servers and their status")),
            ("reload", _("Reload MCP servers and rebuild tool graph")),
        ],
        CMD_AGENTS: [
            ("list", _("List configured subagents and their properties")),
        ],
        CMD_QUEUE: [
            ("clear", _("Clear all queued prompts")),
            ("rm", queue_rm_desc),
            ("remove", queue_rm_desc),
            ("delete", queue_rm_desc),
        ],
        CMD_YOLO: [
            ("on", _("Enable YOLO mode (bypasses confirmations)")),
            ("off", _("Disable YOLO mode")),
        ],
        CMD_STEALTH: [
            ("on", _("Enable stealth mode (no SQLite history)")),
            ("off", _("Disable stealth mode")),
        ],
    }


def _complete_root_commands(token: str) -> list[tuple[str, Text]]:
    return [
        (
            cmd,
            Text.from_markup(f"[bold #38bdf8]{cmd:<12}[/bold #38bdf8] [dim #8b949e]{desc}[/dim #8b949e]"),
        )
        for cmd, desc in _get_root_commands()
        if cmd.startswith(token)
    ]


def _format_thinking_value(v: Any) -> str:
    if v is False:
        return "false"
    if v is True:
        return "true"
    return str(v)


def _complete_effort(sub_token: str, runtime_model: Any) -> list[tuple[str, Text]]:
    sub_token_lower = sub_token.lower()
    thinking_cfg = get_model_thinking_config(runtime_model.show_info) if runtime_model else None
    results = []
    if "default".startswith(sub_token_lower):
        results.append(
            (
                f"{CMD_EFFORT} default",
                Text.from_markup(
                    f"[bold #38bdf8]default     [/bold #38bdf8] [dim #8b949e]{_('(model default)')}[/dim #8b949e]"
                ),
            )
        )
    if not (thinking_cfg and "values" in thinking_cfg):
        return results

    default = thinking_cfg.get("default")
    for v in thinking_cfg["values"]:
        v_str = _format_thinking_value(v)
        if v_str.lower().startswith(sub_token_lower):
            def_label = _("(default)") if v == default else ""
            results.append(
                (
                    f"{CMD_EFFORT} {v_str}",
                    Text.from_markup(
                        f"[bold #38bdf8]{v_str:<12}[/bold #38bdf8] [dim #8b949e]{def_label}[/dim #8b949e]"
                    ),
                )
            )
    return results


def _complete_subcommands(root_cmd: str, sub_token: str) -> list[tuple[str, Text]]:
    subcommands = _get_subcommands()
    if root_cmd not in subcommands:
        return []
    return [
        (
            f"{root_cmd} {sub}",
            Text.from_markup(f"[bold #38bdf8]{sub:<12}[/bold #38bdf8] [dim #8b949e]{desc}[/dim #8b949e]"),
        )
        for sub, desc in subcommands[root_cmd]
        if sub.startswith(sub_token)
    ]


def _collect_dir_candidates(
    dirs: list[str],
    root_path: Path,
    cwd: Path,
    prefix_lower: str,
    show_hidden: bool,
) -> list[tuple[str, str]]:
    candidate_dirs = []
    for dirname in sorted(dirs):
        if not show_hidden and dirname.startswith("."):
            continue
        rel = (root_path / dirname).relative_to(cwd).as_posix() + "/"
        rel_lower = rel.lower()
        if prefix_lower.startswith(rel_lower) or rel_lower.startswith(prefix_lower):
            candidate_dirs.append((dirname, rel))
    return candidate_dirs


def _iter_dir_completions(candidate_dirs: list[tuple[str, str]], prefix_lower: str) -> Iterator[tuple[str, str]]:
    for _dirname, rel in candidate_dirs:
        rel_lower = rel.lower()
        if rel_lower != prefix_lower and rel_lower.startswith(prefix_lower):
            yield rel, _("dir")


def _filter_file_candidate(
    filename: str,
    root_path: Path,
    cwd: Path,
    prefix_lower: str,
    show_hidden: bool,
) -> tuple[str, str] | None:
    if not show_hidden and filename.startswith("."):
        return None
    rel = (root_path / filename).relative_to(cwd).as_posix()
    if not rel.lower().startswith(prefix_lower):
        return None
    try:
        size_kb = (root_path / filename).stat().st_size / 1024
        meta = f"{size_kb:.1f} KB"
    except OSError:
        meta = _("file")
    return rel, meta


def _iter_file_completions(
    files: list[str],
    root_path: Path,
    cwd: Path,
    prefix_lower: str,
    show_hidden: bool,
) -> Iterator[tuple[str, str]]:
    for filename in sorted(files):
        item = _filter_file_candidate(filename, root_path, cwd, prefix_lower, show_hidden)
        if item is not None:
            yield item
