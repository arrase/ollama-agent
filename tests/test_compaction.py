from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from deepagents.backends import FilesystemBackend
from deepagents.backends.protocol import FILE_NOT_FOUND, EditResult, FileDownloadResponse, WriteResult
from deepagents.middleware.summarization import SummarizationMiddleware
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from ollama_agent.agent.compaction import (
    HistoryOffloadError,
    apply_summarization_event,
    build_summary_message,
    compute_state_cutoff,
    find_safe_cutoff,
    is_summary_message,
    offload_history,
)


def make_summarization_engine() -> SummarizationMiddleware:
    """Build a real (unused-model) deepagents engine for pure-helper tests."""
    tmp = tempfile.TemporaryDirectory()
    backend = FilesystemBackend(root_dir=Path(tmp.name), virtual_mode=True)
    model = GenericFakeChatModel(messages=iter([AIMessage(content="ok")]))
    engine = SummarizationMiddleware(model=model, backend=backend)
    tmp.cleanup()
    return engine


def _tool_call_msg(call_id: str = "call-1") -> AIMessage:
    return AIMessage(content="", tool_calls=[{"name": "run", "args": {}, "id": call_id}])


class TestApplySummarizationEvent(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.engine = make_summarization_engine()

    def test_none_event_returns_full_list(self) -> None:
        msgs = [HumanMessage(content="a"), AIMessage(content="b")]
        self.assertEqual(apply_summarization_event(self.engine, msgs, None), msgs)

    def test_event_slices_messages(self) -> None:
        summary = HumanMessage(content="s")
        msgs = ["old1", "old2", HumanMessage(content="new")]
        event = {"cutoff_index": 2, "summary_message": summary, "file_path": None}
        self.assertEqual(
            apply_summarization_event(self.engine, msgs, event),
            [summary, msgs[2]],
        )

    def test_malformed_event_raises_value_error(self) -> None:
        with self.assertRaises(ValueError):
            apply_summarization_event(self.engine, [HumanMessage(content="a")], {"bogus": 1})

    def test_out_of_bounds_cutoff_returns_summary_only(self) -> None:
        summary = HumanMessage(content="s")
        event = {"cutoff_index": 10, "summary_message": summary, "file_path": None}
        self.assertEqual(apply_summarization_event(self.engine, ["a"], event), [summary])


class TestCutoff(unittest.TestCase):
    def test_not_enough_messages(self) -> None:
        self.assertEqual(find_safe_cutoff([HumanMessage(content="a")], keep=2), 0)
        self.assertEqual(find_safe_cutoff([], keep=2), 0)

    def test_simple_cut(self) -> None:
        msgs = [HumanMessage("1"), AIMessage("2"), HumanMessage("3"), AIMessage("4")]
        self.assertEqual(find_safe_cutoff(msgs, keep=2), 2)

    def test_cut_never_orphans_tool_messages(self) -> None:
        # target=2 lands on the ToolMessage; cutoff must move back to the AIMessage
        msgs = [
            HumanMessage("q"),
            _tool_call_msg("call-1"),
            ToolMessage(content="out", tool_call_id="call-1"),
            AIMessage("done"),
            HumanMessage("next"),
        ]
        self.assertEqual(find_safe_cutoff(msgs, keep=3), 1)

    def test_cut_advances_past_orphan_tool_messages(self) -> None:
        # ToolMessage at target with no matching AIMessage in range
        msgs = [HumanMessage("q"), ToolMessage(content="out", tool_call_id="x"), AIMessage("a")]
        self.assertEqual(find_safe_cutoff(msgs, keep=2), 2)


class TestStateCutoff(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.engine = make_summarization_engine()

    def test_no_prior_event(self) -> None:
        self.assertEqual(compute_state_cutoff(self.engine, None, 3), 3)

    def test_prior_event_adjusts_for_summary_slot(self) -> None:
        prior = {"cutoff_index": 2, "summary_message": HumanMessage("s"), "file_path": None}
        self.assertEqual(compute_state_cutoff(self.engine, prior, 3), 4)

    def test_malformed_prior_event_raises_value_error(self) -> None:
        with self.assertRaises(ValueError):
            compute_state_cutoff(self.engine, {"nope": 1}, 3)


class TestSummaryMessage(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.engine = make_summarization_engine()

    def test_with_file_path(self) -> None:
        msg = build_summary_message(self.engine, "the summary", "/conversation_history/s.md")
        self.assertTrue(is_summary_message(msg))
        self.assertIn("/conversation_history/s.md", msg.content)
        self.assertIn("the summary", msg.content)


class _FakeBackend:
    """Minimal backend honouring the deepagents response contracts."""

    def __init__(
        self,
        files: dict[str, str] | None = None,
        download_error: str | None = None,
        read_exception: Exception | None = None,
        write_error: str | None = None,
    ) -> None:
        self.files: dict[str, str] = dict(files or {})
        self.download_error = download_error
        self.read_exception = read_exception
        self.write_error = write_error

    async def adownload_files(self, paths: list[str]) -> list[FileDownloadResponse]:
        if self.read_exception is not None:
            raise self.read_exception
        responses = []
        for p in paths:
            if self.download_error is not None:
                responses.append(FileDownloadResponse(path=p, error=self.download_error))
            elif p in self.files:
                responses.append(FileDownloadResponse(path=p, content=self.files[p].encode("utf-8")))
            else:
                responses.append(FileDownloadResponse(path=p, error=FILE_NOT_FOUND))
        return responses

    async def awrite(self, path: str, content: str) -> WriteResult:
        if self.write_error is not None:
            return WriteResult(error=self.write_error)
        self.files[path] = content
        return WriteResult(path=path)

    async def aedit(self, path: str, old: str, new: str) -> EditResult:
        if self.write_error is not None:
            return EditResult(error=self.write_error)
        self.files[path] = new
        return EditResult(path=path, occurrences=1)


class TestOffloadHistory(unittest.IsolatedAsyncioTestCase):
    async def test_creates_file_when_missing(self) -> None:
        backend = _FakeBackend()
        path = await offload_history(backend, [HumanMessage(content="hello")], "/conversation_history/s.md")
        self.assertEqual(path, "/conversation_history/s.md")
        self.assertIn("hello", backend.files[path])
        self.assertIn("## Summarized at", backend.files[path])

    async def test_appends_to_existing_history_without_overwrite(self) -> None:
        backend = _FakeBackend()
        await offload_history(backend, [HumanMessage(content="first turn")], "/h.md")
        await offload_history(backend, [HumanMessage(content="second turn")], "/h.md")
        content = backend.files["/h.md"]
        self.assertIn("first turn", content)
        self.assertIn("second turn", content)
        self.assertEqual(content.count("## Summarized at"), 2)

    async def test_filters_previous_summary_messages(self) -> None:
        backend = _FakeBackend()
        summary = build_summary_message(make_summarization_engine(), "condensed", "/h.md")
        await offload_history(backend, [HumanMessage(content="question"), summary], "/h.md")
        self.assertNotIn("<summary>", backend.files["/h.md"])

    async def test_unexpected_download_exception_propagates(self) -> None:
        backend = _FakeBackend(read_exception=RuntimeError("backend down"))
        with self.assertRaises(RuntimeError):
            await offload_history(backend, [HumanMessage(content="x")], "/h.md")

    async def test_backend_read_error_other_than_not_found_raises(self) -> None:
        backend = _FakeBackend(download_error="permission_denied")
        with self.assertRaises(HistoryOffloadError):
            await offload_history(backend, [HumanMessage(content="x")], "/h.md")

    async def test_write_failure_raises_history_offload_error(self) -> None:
        backend = _FakeBackend(write_error="disk_full")
        with self.assertRaises(HistoryOffloadError):
            await offload_history(backend, [HumanMessage(content="x")], "/h.md")


if __name__ == "__main__":
    unittest.main()
