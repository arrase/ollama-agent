"""Interop tests: compaction helpers against the real deepagents SummarizationMiddleware.

``ollama_agent.agent.compaction`` delegates its pure state arithmetic to a live
deepagents engine (private methods ``_apply_event_to_messages``,
``_compute_state_cutoff``, ``_build_new_messages_with_path``). These tests
verify that the delegation is wired correctly and that our own pieces
(``is_summary_message``, ``offload_history``, state-key constants) still agree
with deepagents' behaviour. A failure here means deepagents changed its private
contract and the coupling documented in compaction.py must be re-examined.
"""

from __future__ import annotations

import inspect
import tempfile
import unittest
from pathlib import Path

from deepagents.backends import FilesystemBackend
from deepagents.middleware.summarization import SummarizationMiddleware
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, HumanMessage

from ollama_agent.agent.compaction import (
    SUMMARIZATION_SESSION_ID_KEY,
    SUMMARIZATION_STATE_KEY,
    apply_summarization_event,
    build_summary_message,
    compute_state_cutoff,
    is_summary_message,
    offload_history,
)


def _make_middleware(root: Path) -> tuple[SummarizationMiddleware, FilesystemBackend]:
    backend = FilesystemBackend(root_dir=root, virtual_mode=True)
    model = GenericFakeChatModel(messages=iter([AIMessage(content="ok")]))
    return SummarizationMiddleware(model=model, backend=backend), backend


class TestSummarizationInterop(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.tmpdir = tempfile.TemporaryDirectory()
        self.middleware, self.backend = _make_middleware(Path(self.tmpdir.name))

    def tearDown(self) -> None:
        self.tmpdir.cleanup()

    def test_delegated_private_methods_exist_on_engine(self) -> None:
        # Loud canary: an upstream rename must break here, not at runtime.
        for name in ("_is_summary_message", "_apply_event_to_messages", "_compute_state_cutoff", "_build_new_messages_with_path"):
            self.assertTrue(hasattr(self.middleware, name), f"deepagents engine lost {name}")

    def test_state_key_contract_matches_deepagents(self) -> None:
        # deepagents' engine source must reference exactly the state keys we use.
        source = inspect.getsource(type(self.middleware))
        self.assertIn(SUMMARIZATION_STATE_KEY, source)
        self.assertIn(SUMMARIZATION_SESSION_ID_KEY, source)

    def test_is_summary_message_agrees_with_deepagents(self) -> None:
        msgs = [
            HumanMessage(content="hi"),
            AIMessage(content="hello"),
            # Summary message produced by deepagents itself:
            self.middleware._build_new_messages_with_path("s", None)[0],
            # Summary message produced by our delegated helper:
            build_summary_message(self.middleware, "s2", "/conversation_history/x.md"),
        ]
        for msg in msgs:
            self.assertEqual(is_summary_message(msg), self.middleware._is_summary_message(msg))
        # Both must agree that deepagents' own summaries are summaries.
        self.assertTrue(is_summary_message(msgs[2]))
        self.assertTrue(self.middleware._is_summary_message(msgs[3]))

    def test_apply_event_and_state_cutoff_match_deepagents(self) -> None:
        msgs = ["old1", "old2", HumanMessage(content="new")]
        event: dict = {
            "cutoff_index": 2,
            "summary_message": HumanMessage(content="s"),
            "file_path": None,
        }
        self.assertEqual(
            apply_summarization_event(self.middleware, msgs, event),
            self.middleware._apply_event_to_messages(msgs, event),
        )
        self.assertEqual(
            apply_summarization_event(self.middleware, msgs, None),
            self.middleware._apply_event_to_messages(msgs, None),
        )
        self.assertEqual(
            compute_state_cutoff(self.middleware, None, 3),
            self.middleware._compute_state_cutoff(None, 3),
        )
        self.assertEqual(
            compute_state_cutoff(self.middleware, event, 3),
            self.middleware._compute_state_cutoff(event, 3),
        )

    def test_out_of_bounds_cutoff_matches_deepagents(self) -> None:
        msgs = [HumanMessage(content="only")]
        event: dict = {"cutoff_index": 10, "summary_message": HumanMessage(content="s"), "file_path": None}
        self.assertEqual(
            apply_summarization_event(self.middleware, msgs, event),
            self.middleware._apply_event_to_messages(msgs, event),
        )

    def test_summary_message_format_matches_deepagents(self) -> None:
        ours = build_summary_message(self.middleware, "the summary", "/conversation_history/s.md")
        theirs = self.middleware._build_new_messages_with_path("the summary", "/conversation_history/s.md")[0]
        self.assertEqual(ours.content, theirs.content)
        self.assertEqual(ours.additional_kwargs, theirs.additional_kwargs)

    async def test_offload_history_appends_after_deepagents_offload(self) -> None:
        msgs = [HumanMessage(content="turn one"), AIMessage(content="answer one")]
        path = self.middleware._offload_to_backend(self.backend, msgs, "interop_sess")
        self.assertIsNotNone(path)

        appended = await offload_history(self.backend, [HumanMessage(content="turn two")], path)
        self.assertEqual(appended, path)

        responses = self.backend.download_files([path])
        content = responses[0].content.decode("utf-8")
        self.assertEqual(content.count("## Summarized at"), 2)
        self.assertIn("turn one", content)
        self.assertIn("turn two", content)

    async def test_offload_history_reads_file_written_by_deepagents(self) -> None:
        msgs = [HumanMessage(content="original turn")]
        path = self.middleware._offload_to_backend(self.backend, msgs, "interop_sess2")

        appended = await offload_history(self.backend, [], path)
        self.assertEqual(appended, path)

        responses = self.backend.download_files([path])
        content = responses[0].content.decode("utf-8")
        self.assertIn("original turn", content)
        self.assertEqual(content.count("## Summarized at"), 2)


if __name__ == "__main__":
    unittest.main()
