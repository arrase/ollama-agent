from __future__ import annotations

import unittest
from unittest.mock import AsyncMock, patch

from ollama_agent.core.models import (
    ModelCapabilityError,
    ModelContextWindowError,
    _model_context_length,
    _parse_num_ctx,
    ensure_model_supports_tools,
    get_model_capabilities,
    model_supports_tools,
    resolve_context_window,
    resolve_ollama_reasoning,
    validate_reasoning_effort,
)


class TestModelsLogic(unittest.IsolatedAsyncioTestCase):
    """Unit tests for model capabilities and configuration helpers."""

    def test_parse_num_ctx_valid_formats(self) -> None:
        self.assertEqual(_parse_num_ctx("PARAMETER num_ctx 8192"), 8192)
        self.assertEqual(_parse_num_ctx("num_ctx 4096"), 4096)
        self.assertEqual(_parse_num_ctx("  num_ctx   16384  "), 16384)

    def test_parse_num_ctx_invalid_or_none(self) -> None:
        self.assertIsNone(_parse_num_ctx(None))
        self.assertIsNone(_parse_num_ctx(""))
        self.assertIsNone(_parse_num_ctx("temperature 0.7"))

    def test_model_context_length_extractor(self) -> None:
        info = {
            "llama.context_length": "8192",
            "general.architecture": "llama",
            "qwen.context_length": "32768",
        }
        self.assertEqual(_model_context_length(info), 32768)

    def test_model_context_length_empty(self) -> None:
        self.assertIsNone(_model_context_length({}))

    def test_validate_reasoning_effort_valid(self) -> None:
        self.assertEqual(validate_reasoning_effort("high"), "high")
        self.assertEqual(validate_reasoning_effort("low"), "low")
        self.assertEqual(validate_reasoning_effort("disabled"), "disabled")

    def test_validate_reasoning_effort_invalid_raises(self) -> None:
        with self.assertRaises(ValueError):
            validate_reasoning_effort("invalid_val")

    @patch("ollama_agent.core.models._show_model")
    async def test_get_model_capabilities(self, mock_show: AsyncMock) -> None:
        mock_show.return_value = type("Resp", (), {"capabilities": ["tools", "thinking"]})()
        caps = await get_model_capabilities("test-model", "http://localhost:11434")
        self.assertEqual(caps, {"tools", "thinking"})

    @patch("ollama_agent.core.models.get_model_capabilities")
    async def test_model_supports_tools(self, mock_caps: AsyncMock) -> None:
        mock_caps.return_value = {"tools"}
        self.assertTrue(await model_supports_tools("test-model", "http://localhost:11434"))

        mock_caps.return_value = set()
        self.assertFalse(await model_supports_tools("test-model", "http://localhost:11434"))

    @patch("ollama_agent.core.models.model_supports_tools")
    async def test_ensure_model_supports_tools_raises(self, mock_supports: AsyncMock) -> None:
        mock_supports.return_value = False
        with self.assertRaises(ModelCapabilityError):
            await ensure_model_supports_tools("test-model", "http://localhost:11434")

    async def test_resolve_context_window_explicit_value(self) -> None:
        resolved = await resolve_context_window("test-model", 4096, "http://localhost:11434")
        self.assertEqual(resolved, 4096)

    async def test_resolve_context_window_invalid_explicit_value_raises(self) -> None:
        with self.assertRaises(ModelContextWindowError):
            await resolve_context_window("test-model", 0, "http://localhost:11434")


if __name__ == "__main__":
    unittest.main()
