from __future__ import annotations

import unittest

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from ollama_agent.agent.compaction import (
    apply_summarization_event,
    build_summary_message,
    compute_state_cutoff,
    find_safe_cutoff,
    is_summary_message,
)


def _tool_call_msg(call_id: str = "call-1") -> AIMessage:
    return AIMessage(content="", tool_calls=[{"name": "run", "args": {}, "id": call_id}])


class TestApplySummarizationEvent(unittest.TestCase):
    def test_none_event_returns_full_list(self) -> None:
        msgs = [HumanMessage(content="a"), AIMessage(content="b")]
        self.assertEqual(apply_summarization_event(msgs, None), msgs)

    def test_event_slices_messages(self) -> None:
        summary = HumanMessage(content="s")
        msgs = ["old1", "old2", HumanMessage(content="new")]
        event = {"cutoff_index": 2, "summary_message": summary, "file_path": None}
        self.assertEqual(
            apply_summarization_event(msgs, event),
            [summary, msgs[2]],
        )

    def test_malformed_event_falls_back_to_full_list(self) -> None:
        msgs = [HumanMessage(content="a")]
        self.assertEqual(apply_summarization_event(msgs, {"bogus": 1}), msgs)

    def test_out_of_bounds_cutoff_returns_summary_only(self) -> None:
        summary = HumanMessage(content="s")
        event = {"cutoff_index": 10, "summary_message": summary, "file_path": None}
        self.assertEqual(apply_summarization_event(["a"], event), [summary])


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
    def test_no_prior_event(self) -> None:
        self.assertEqual(compute_state_cutoff(None, 3), 3)

    def test_prior_event_adjusts_for_summary_slot(self) -> None:
        prior = {"cutoff_index": 2, "summary_message": HumanMessage("s"), "file_path": None}
        self.assertEqual(compute_state_cutoff(prior, 3), 4)

    def test_malformed_prior_event(self) -> None:
        self.assertEqual(compute_state_cutoff({"nope": 1}, 3), 3)


class TestSummaryMessage(unittest.TestCase):
    def test_with_file_path(self) -> None:
        msg = build_summary_message("the summary", "/conversation_history/s.md")
        self.assertTrue(is_summary_message(msg))
        self.assertIn("/conversation_history/s.md", msg.content)
        self.assertIn("the summary", msg.content)

    def test_without_file_path(self) -> None:
        msg = build_summary_message("the summary", None)
        self.assertIn("the summary", msg.content)
        self.assertNotIn("<summary>", msg.content)


if __name__ == "__main__":
    unittest.main()
