"""Autocompletion helpers and slash command definitions for the TUI."""

from __future__ import annotations

import os
import re
import threading
import time
from collections.abc import Iterator, Sequence
from itertools import islice
from pathlib import Path
from typing import Any

import ollama
from rich.markup import escape
from rich.text import Text

from ...agent.episodic_memory import HistoryError
from ...core.models import format_thinking_value, get_model_thinking_config
from ...i18n import _
from ...rag import RAGError
from ...skills import SkillError
from ...tasks import TaskError
from ..commands.models import format_model_size
from ..commands.sessions import get_available_sessions
from .widgets.system import format_queue_preview

# @-mention regex: matches @"quoted", @'quoted', or @bare at word boundaries.
_AT_MENTION_RE = re.compile(r"""(?:^|[\s\(\[\{<])@(?:"([^"]*)|'([^']*)|([^\s"'\(\[\{<>,;]*))$""")

CMD_MODEL = "/model"
CMD_EFFORT = "/effort"
CMD_CONTEXT = "/context"
CMD_PARAMS = "/params"
CMD_AUDIO = "/audio"
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
NO_ARG_COMMANDS: frozenset[str] = frozenset({CMD_CLEAR, CMD_NEW, CMD_EXIT, CMD_QUIT})
_CHAT_SCROLL_ID = "#chat-scroll"
_AUTOCOMPLETE_LIST_ID = "#autocomplete-list"
_MODEL_CACHE_TTL = 10.0
_model_cache: dict[tuple[str, Any], tuple[float, list[Any]]] = {}
_model_refresh_inflight: set[tuple[str, Any]] = set()


def _list_models_sync(base_url: str) -> list[Any]:
    """Fetch the list of available Ollama models synchronously."""
    client = ollama.Client(host=base_url, timeout=2.0)
    response = client.list()
    return list(response.models)


def _refresh_models_in_background(base_url: str, *, wait: bool) -> None:
    fn = _list_models_sync
    key = (base_url, fn)
    if key in _model_refresh_inflight:
        return
    _model_refresh_inflight.add(key)

    def _worker() -> None:
        try:
            models = fn(base_url)
        except Exception:  # noqa: BLE001
            models = []
        finally:
            _model_refresh_inflight.discard(key)
        _model_cache[key] = (time.monotonic(), models)

    thread = threading.Thread(target=_worker, daemon=True)
    thread.start()
    if wait:
        thread.join(timeout=0.05)


def _get_cached_models(base_url: str) -> list[Any]:
    now = time.monotonic()
    key = (base_url, _list_models_sync)
    if key in _model_cache:
        ts, models = _model_cache[key]
        if now - ts >= _MODEL_CACHE_TTL:
            _refresh_models_in_background(base_url, wait=False)
        return models
    _refresh_models_in_background(base_url, wait=True)
    return _model_cache[key][1] if key in _model_cache else []


def _get_root_commands() -> list[tuple[str, str]]:
    return [
        (CMD_MODEL, _("Manage models")),
        (CMD_EFFORT, _("Show or set reasoning/thinking effort")),
        (CMD_CONTEXT, _("Show or set context window size (num_ctx)")),
        (CMD_PARAMS, _("Manage model sampling parameters")),
        (CMD_AUDIO, _("Show or set audio recording max duration")),
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
        CMD_AUDIO: [
            ("set", _("Set maximum audio recording duration in seconds (e.g. 10, 15, 30)")),
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


_format_thinking_value = format_thinking_value


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
                        f"[bold #38bdf8]{escape(v_str):<12}[/bold #38bdf8] [dim #8b949e]{def_label}[/dim #8b949e]"
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


def _complete_model_args(root_cmd: str, sub_cmd: str, arg_token: str, base_url: str) -> list[tuple[str, Text]]:
    models = _get_cached_models(base_url)
    results = []
    for m in models:
        if not (m.model and m.model.startswith(arg_token)):
            continue
        size_str = format_model_size(m.size)
        markup = f"[bold #e6edf3]{escape(m.model):<30}[/bold #e6edf3] [dim #8b949e]{size_str}[/dim #8b949e]"
        results.append((f"{root_cmd} {sub_cmd} {m.model}", Text.from_markup(markup)))
    return results


def _complete_context_args(root_cmd: str, sub_cmd: str, arg_token: str) -> list[tuple[str, Text]]:
    presets = ["4096", "8192", "16384", "32768", "65536", "131072", "max"]
    tok_label = _("tokens")
    return [
        (
            f"{root_cmd} {sub_cmd} {p}",
            Text.from_markup(f"[bold #e6edf3]{p:<10}[/bold #e6edf3] [dim #8b949e]{tok_label}[/dim #8b949e]"),
        )
        for p in presets
        if p.startswith(arg_token)
    ]


def _complete_audio_args(root_cmd: str, sub_cmd: str, arg_token: str) -> list[tuple[str, Text]]:
    presets = ["5", "10", "15", "30", "60"]
    sec_label = _("seconds")
    return [
        (
            f"{root_cmd} {sub_cmd} {p}",
            Text.from_markup(f"[bold #e6edf3]{p:<10}[/bold #e6edf3] [dim #8b949e]{sec_label}[/dim #8b949e]"),
        )
        for p in presets
        if p.startswith(arg_token)
    ]


def _complete_task_args(root_cmd: str, sub_cmd: str, arg_token: str, repl: Any) -> list[tuple[str, Text]]:
    try:
        tasks = repl._task_ctx.task_manager.list_all()
    except (OSError, ValueError, TaskError):
        return []
    return [
        (
            f"{root_cmd} {sub_cmd} {tid}",
            Text.from_markup(
                f"[bold #e6edf3]{escape(tid):<20}[/bold #e6edf3] [dim #8b949e]{escape(t.title)}[/dim #8b949e]"
            ),
        )
        for tid, t in tasks
        if tid.startswith(arg_token)
    ]


def _complete_skill_args(root_cmd: str, sub_cmd: str, arg_token: str, repl: Any) -> list[tuple[str, Text]]:
    try:
        skills = repl._skills_ctx.skill_manager.list_all()
    except (OSError, ValueError, SkillError):
        return []
    return [
        (
            f"{root_cmd} {sub_cmd} {sid}",
            Text.from_markup(
                f"[bold #e6edf3]{escape(sid):<20}[/bold #e6edf3] [dim #8b949e]{escape(s.name)}[/dim #8b949e]"
            ),
        )
        for sid, s in skills
        if sid.startswith(arg_token)
    ]


def _complete_session_args(root_cmd: str, sub_cmd: str, arg_token: str) -> list[tuple[str, Text]]:
    try:
        sessions = get_available_sessions()
    except (OSError, HistoryError):
        # HistoryError is a RuntimeError, not an OSError.
        return []
    steps_label = _("steps")
    return [
        (
            f"{root_cmd} {sub_cmd} {s['thread_id']}",
            Text.from_markup(
                f"[bold #e6edf3]{escape(s['thread_id'][:8]):<10}[/bold #e6edf3] "
                f"[dim #8b949e]{s['steps']} {steps_label}[/dim #8b949e]"
            ),
        )
        for s in sessions
        if s["thread_id"].startswith(arg_token)
    ]


def _complete_rag_args(root_cmd: str, sub_cmd: str, arg_token: str, repl: Any) -> list[tuple[str, Text]]:
    try:
        names = repl._get_rag_ctx().rag_manager.list_database_names()
    except (RAGError, OSError):
        return []
    return [
        (
            f"{root_cmd} {sub_cmd} {name}",
            Text.from_markup(f"[bold #e6edf3]{escape(name):<20}[/bold #e6edf3]"),
        )
        for name in names
        if name.startswith(arg_token)
    ]


def _complete_queue_args(
    root_cmd: str, sub_cmd: str, arg_token: str, prompt_queue: Sequence[Any]
) -> list[tuple[str, Text]]:
    clean_arg = arg_token.lstrip("#")
    items = []
    for idx, item in enumerate(prompt_queue, 1):
        if clean_arg and not str(idx).startswith(clean_arg):
            continue
        preview = format_queue_preview(item)
        items.append(
            (
                f"{root_cmd} {sub_cmd} {idx}",
                Text.from_markup(f"[bold #e6edf3]#{idx}[/bold #e6edf3] [dim #8b949e]{preview}[/dim #8b949e]"),
            )
        )
    return items


def _complete_arguments(
    root_cmd: str, sub_cmd: str, arg_token: str, repl: Any, prompt_queue: Sequence[Any]
) -> list[tuple[str, Text]]:
    if root_cmd == CMD_MODEL and sub_cmd == "set":
        return _complete_model_args(root_cmd, sub_cmd, arg_token, repl.runtime.settings.model.base_url)
    if root_cmd == CMD_CONTEXT and sub_cmd == "set":
        return _complete_context_args(root_cmd, sub_cmd, arg_token)
    if root_cmd == CMD_AUDIO and sub_cmd == "set":
        return _complete_audio_args(root_cmd, sub_cmd, arg_token)
    if root_cmd == CMD_TASK and sub_cmd in ("run", "delete"):
        return _complete_task_args(root_cmd, sub_cmd, arg_token, repl)
    if root_cmd == CMD_SKILL and sub_cmd in ("show", "delete"):
        return _complete_skill_args(root_cmd, sub_cmd, arg_token, repl)
    if root_cmd == CMD_SESSION and sub_cmd in ("resume", "switch", "delete"):
        return _complete_session_args(root_cmd, sub_cmd, arg_token)
    if root_cmd == CMD_RAG and sub_cmd in ("load", "delete"):
        return _complete_rag_args(root_cmd, sub_cmd, arg_token, repl)
    if root_cmd == CMD_QUEUE and sub_cmd in ("rm", "remove", "delete"):
        return _complete_queue_args(root_cmd, sub_cmd, arg_token, prompt_queue)
    return []


def slash_completions(text: str, repl: Any, prompt_queue: Sequence[Any]) -> list[tuple[str, Text]]:
    """Build slash-command autocomplete candidates for the current input *text*."""
    parts = text.split(" ")
    num_parts = len(parts)

    # Level 0: Root commands (e.g., "/" or "/mo")
    if num_parts == 1:
        return _complete_root_commands(parts[0])

    root_cmd = parts[0]
    if root_cmd == CMD_EFFORT and num_parts == 2:
        return _complete_effort(parts[1], repl.runtime.model)

    # Level 1: Subcommands (e.g., "/task " or "/task r")
    if num_parts == 2:
        if root_cmd == CMD_MODEL:
            base_url = repl.runtime.settings.model.base_url
            key = (base_url, _list_models_sync)
            if key not in _model_cache or time.monotonic() - _model_cache[key][0] >= _MODEL_CACHE_TTL:
                _refresh_models_in_background(base_url, wait=False)
        return _complete_subcommands(root_cmd, parts[1])

    # Level 2: Arguments / Dynamic entities (e.g., "/task run ")
    if num_parts == 3:
        return _complete_arguments(root_cmd, parts[1], parts[2], repl, prompt_queue)

    return []


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
        if (
            dirname.lower().startswith(prefix_lower)
            or prefix_lower.startswith(rel_lower)
            or rel_lower.startswith(prefix_lower)
        ):
            candidate_dirs.append((dirname, rel))
    return candidate_dirs


def _iter_dir_completions(candidate_dirs: list[tuple[str, str]], prefix_lower: str) -> Iterator[tuple[str, str]]:
    for dirname, rel in candidate_dirs:
        rel_lower = rel.lower()
        if rel_lower != prefix_lower and (
            dirname.lower().startswith(prefix_lower) or rel_lower.startswith(prefix_lower)
        ):
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
    if not (filename.lower().startswith(prefix_lower) or rel.lower().startswith(prefix_lower)):
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


def _iter_walk_completions(
    search_dir: Path,
    cwd: Path,
    prefix_lower: str,
    show_hidden: bool,
) -> Iterator[tuple[str, str]]:
    for root, dirs, files in os.walk(search_dir):
        root_path = Path(root)
        candidate_dirs = _collect_dir_candidates(dirs, root_path, cwd, prefix_lower, show_hidden)
        dirs[:] = [d[0] for d in candidate_dirs]
        yield from _iter_dir_completions(candidate_dirs, prefix_lower)
        yield from _iter_file_completions(files, root_path, cwd, prefix_lower, show_hidden)


def file_completions(token: str, max_completions: int = 15) -> Iterator[tuple[str, Text]]:
    """Yield `@`-mention file and directory completions matching *token*."""
    cwd = Path.cwd()
    token_path = Path(token)
    if token.endswith(("/", "\\")):
        search_dir = (cwd / token_path).resolve()
        prefix = ""
    else:
        search_dir = (cwd / token_path.parent).resolve()
        prefix = token_path.name

    try:
        rel_dir = search_dir.relative_to(cwd)
    except ValueError:
        return

    if not search_dir.is_dir():
        return

    show_hidden = prefix.startswith(".")
    prefix_lower = ("" if rel_dir == Path(".") else f"{rel_dir.as_posix().lower()}/") + prefix.lower()
    matches = _iter_walk_completions(search_dir, cwd, prefix_lower, show_hidden)

    for rel, meta in islice(matches, max_completions):
        escaped_rel = escape(rel)
        yield (
            rel,
            Text.from_markup(f"[bold #38bdf8]{escaped_rel:<30}[/bold #38bdf8] [dim #8b949e]{meta}[/dim #8b949e]"),
        )
