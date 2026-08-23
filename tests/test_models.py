from __future__ import annotations

import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from ollama_agent.core.models import (
    ModelCapabilityError,
    ModelContextWindowError,
    OLLAMA_PARAM_DEFAULTS,
    _model_context_length,
    _parse_modelfile_param,
    _parse_num_ctx,
    create_ollama_chat_model,
    ensure_model_supports_tools,
    get_model_capabilities,
    model_supports_thinking,
    model_supports_tools,
    resolve_context_window,
    resolve_model_parameters,
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
        self.assertEqual(validate_reasoning_effort("xhigh"), "xhigh")
        self.assertEqual(validate_reasoning_effort("low"), "low")
        self.assertEqual(validate_reasoning_effort("disabled"), "disabled")

    def test_validate_reasoning_effort_invalid_raises(self) -> None:
        with self.assertRaises(ValueError):
            validate_reasoning_effort("invalid_val")

    @patch("ollama_agent.core.models._show_model")
    async def test_get_model_capabilities(self, mock_show: AsyncMock) -> None:
        mock_show.return_value = MagicMock(capabilities=["tools", "thinking"])
        caps = await get_model_capabilities("test-model", "http://localhost:11434")
        self.assertEqual(caps, {"tools", "thinking"})

    @patch("ollama_agent.core.models.get_model_capabilities")
    async def test_model_supports_tools(self, mock_caps: AsyncMock) -> None:
        mock_caps.return_value = {"tools"}
        self.assertTrue(await model_supports_tools("test-model", "http://localhost:11434"))

        mock_caps.return_value = set()
        self.assertFalse(await model_supports_tools("test-model", "http://localhost:11434"))

    @patch("ollama_agent.core.models.get_model_capabilities")
    async def test_model_supports_thinking(self, mock_caps: AsyncMock) -> None:
        mock_caps.return_value = {"thinking"}
        self.assertTrue(await model_supports_thinking("test-model", "http://localhost:11434"))

    @patch("ollama_agent.core.models.ensure_model_supports_tools")
    @patch("ollama_agent.core.models.get_model_capabilities")
    async def test_resolve_ollama_reasoning(self, mock_caps: AsyncMock, mock_ensure: AsyncMock) -> None:
        mock_caps.return_value = {"thinking"}
        res = await resolve_ollama_reasoning("qwen:32b", "high", "http://localhost:11434")
        self.assertTrue(res)

        # Non-thinking model
        mock_caps.return_value = set()
        res = await resolve_ollama_reasoning("llama3:8b", "high", "http://localhost:11434")
        self.assertIsNone(res)

        # qwen3.8 model with official reasoning_effort support
        res_qwen_xhigh = await resolve_ollama_reasoning("qwen3.8:27b", "xhigh", "http://localhost:11434")
        self.assertEqual(res_qwen_xhigh, "xhigh")
        res_qwen_med = await resolve_ollama_reasoning("qwen3.8:27b", "medium", "http://localhost:11434")
        self.assertEqual(res_qwen_med, "medium")
        res_qwen_low = await resolve_ollama_reasoning("qwen3.8:27b", "low", "http://localhost:11434")
        self.assertEqual(res_qwen_low, "low")
        res_qwen_enabled = await resolve_ollama_reasoning("qwen3.8:27b", "enabled", "http://localhost:11434")
        self.assertEqual(res_qwen_enabled, "xhigh")
        res_qwen_hide = await resolve_ollama_reasoning("qwen3.8:27b", "hide", "http://localhost:11434")
        self.assertTrue(res_qwen_hide)
        res_qwen_disabled = await resolve_ollama_reasoning("qwen3.8:27b", "disabled", "http://localhost:11434")
        self.assertFalse(res_qwen_disabled)

        # gpt-oss special model
        res = await resolve_ollama_reasoning("gpt-oss:latest", "high", "http://localhost:11434")
        self.assertEqual(res, "high")
        res_xhigh = await resolve_ollama_reasoning("gpt-oss:latest", "xhigh", "http://localhost:11434")
        self.assertEqual(res_xhigh, "xhigh")
        res_enabled = await resolve_ollama_reasoning("gpt-oss:latest", "enabled", "http://localhost:11434")
        self.assertEqual(res_enabled, "medium")
        res_hide = await resolve_ollama_reasoning("gpt-oss:latest", "hide", "http://localhost:11434")
        self.assertIsNone(res_hide)
        res_disabled = await resolve_ollama_reasoning("gpt-oss:latest", "disabled", "http://localhost:11434")
        self.assertIsNone(res_disabled)

        # Thinking-capable models
        mock_caps.return_value = {"thinking"}
        for effort_level in ("low", "medium", "high", "xhigh", "enabled", "hide"):
            self.assertTrue(
                await resolve_ollama_reasoning("any-thinking-model", effort_level, "http://localhost:11434")
            )
        self.assertFalse(
            await resolve_ollama_reasoning("any-thinking-model", "disabled", "http://localhost:11434")
        )

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

    @patch("ollama_agent.core.models.resolve_context_window", AsyncMock(return_value=8192))
    @patch("ollama_agent.core.models.resolve_ollama_reasoning", AsyncMock(return_value=True))
    async def test_create_ollama_chat_model(self) -> None:
        model = await create_ollama_chat_model(
            model="gemma4:26b",
            base_url="http://localhost:11434",
            context_window=8192,
            reasoning_effort="high",
            temperature=0.7,
            top_p=0.95,
            top_k=50,
            min_p=0.05,
            presence_penalty=0.5,
            repeat_penalty=1.2,
        )
        self.assertIsNotNone(model)
        self.assertEqual(model.model, "gemma4:26b")
        self.assertEqual(model.temperature, 0.7)
        self.assertEqual(model.top_p, 0.95)
        self.assertEqual(model.top_k, 50)
        self.assertEqual(model.repeat_penalty, 1.2)
        self.assertEqual(model.min_p, 0.05)
        self.assertEqual(model.presence_penalty, 0.5)

        params = model._chat_params([])
        options = params["options"]
        self.assertEqual(options["temperature"], 0.7)
        self.assertEqual(options["top_p"], 0.95)
        self.assertEqual(options["top_k"], 50)
        self.assertEqual(options["repeat_penalty"], 1.2)
        self.assertEqual(options["min_p"], 0.05)
        self.assertEqual(options["presence_penalty"], 0.5)

    def test_parse_modelfile_param(self) -> None:
        text = "PARAMETER temperature 0.65\nPARAMETER top_p 0.85\nPARAMETER top_k 30\nrepeat_penalty 1.15"
        self.assertEqual(_parse_modelfile_param(text, "temperature"), "0.65")
        self.assertEqual(_parse_modelfile_param(text, "top_p"), "0.85")
        self.assertEqual(_parse_modelfile_param(text, "top_k"), "30")
        self.assertEqual(_parse_modelfile_param(text, "repeat_penalty"), "1.15")
        self.assertIsNone(_parse_modelfile_param(text, "min_p"))
        self.assertIsNone(_parse_modelfile_param(None, "temperature"))

    @patch("ollama_agent.core.models._show_model")
    async def test_resolve_model_parameters_precedence(self, mock_show: AsyncMock) -> None:
        mock_show.return_value = MagicMock(
            parameters="temperature 0.65\ntop_p 0.85\nrepetition_penalty 1.25",
            modelfile=None,
        )

        # 1. User overrides temperature; top_p and repeat_penalty resolve from metadata; others from defaults
        resolved = await resolve_model_parameters(
            "test-model",
            "http://localhost:11434",
            temperature=0.3,
            top_p=None,
            top_k=None,
            min_p=None,
            presence_penalty=None,
            repeat_penalty=None,
        )

        # User value
        self.assertEqual(resolved["temperature"], (0.3, "user"))
        # Modelfile values
        self.assertEqual(resolved["top_p"], (0.85, "modelfile"))
        self.assertEqual(resolved["repeat_penalty"], (1.25, "modelfile"))
        # Ollama Default values
        self.assertEqual(resolved["top_k"], (OLLAMA_PARAM_DEFAULTS["top_k"], "default"))
        self.assertEqual(resolved["min_p"], (OLLAMA_PARAM_DEFAULTS["min_p"], "default"))
        self.assertEqual(resolved["presence_penalty"], (OLLAMA_PARAM_DEFAULTS["presence_penalty"], "default"))

    @patch("ollama_agent.core.models._show_model")
    @patch("ollama_agent.core.models.resolve_context_window", AsyncMock(return_value=4096))
    @patch("ollama_agent.core.models.resolve_ollama_reasoning", AsyncMock(return_value=None))
    async def test_create_ollama_chat_model_resolves_defaults(self, mock_show: AsyncMock) -> None:
        mock_show.return_value = MagicMock(
            parameters="",
            modelfile="",
            capabilities=["tools"],
        )
        model = await create_ollama_chat_model(
            model="default-model",
            base_url="http://localhost:11434",
            context_window=4096,
            reasoning_effort="disabled",
        )
        self.assertEqual(model.temperature, 0.8)
        self.assertEqual(model.top_p, 0.9)
        self.assertEqual(model.top_k, 40)
        self.assertEqual(model.min_p, 0.0)
        self.assertEqual(model.presence_penalty, 0.0)
        self.assertEqual(model.repeat_penalty, 1.1)
        self.assertIn("temperature", model.effective_params)
        self.assertEqual(model.effective_params["temperature"][1], "default")


if __name__ == "__main__":
    unittest.main()

