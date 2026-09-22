from __future__ import annotations

import httpx
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from ollama_agent.core.common import DEFAULT_REASONING_EFFORT
from ollama_agent.core.models import (
    MIN_OLLAMA_VERSION,
    ExtendedShowResponse,
    ModelCapabilityError,
    ModelContextWindowError,
    OllamaVersionError,
    _get_model_info,
    _model_context_length,
    _parse_modelfile_param,
    _parse_num_ctx,
    check_ollama_version,
    create_ollama_chat_model,
    ensure_model_supports_tools,
    get_model_capabilities,
    get_model_thinking_config,
    get_ollama_version,
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

    def test_parse_num_ctx_invalid(self) -> None:
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

    def test_validate_reasoning_effort(self) -> None:
        self.assertEqual(validate_reasoning_effort("high"), "high")
        self.assertEqual(validate_reasoning_effort("  HIGH  "), "HIGH")
        self.assertEqual(validate_reasoning_effort("Medium"), "Medium")
        self.assertEqual(validate_reasoning_effort("max"), "max")
        self.assertEqual(validate_reasoning_effort("low"), "low")
        self.assertEqual(validate_reasoning_effort("default"), "default")
        self.assertEqual(validate_reasoning_effort(True), "true")
        self.assertEqual(validate_reasoning_effort(False), "false")
        self.assertEqual(validate_reasoning_effort(123), "123")
        with self.assertRaises(ValueError):
            validate_reasoning_effort("")
        with self.assertRaises(ValueError):
            validate_reasoning_effort("   ")

    @patch("ollama_agent.core.models._show_model")
    async def test_get_model_capabilities(self, mock_show: AsyncMock) -> None:
        mock_show.return_value = MagicMock(capabilities=["tools", "thinking"])
        caps = await get_model_capabilities("test-model", "http://localhost:11434")
        self.assertEqual(caps, {"tools", "thinking"})

    @patch("ollama_agent.core.models._show_model")
    async def test_get_model_capabilities_unknown_shape_raises(self, mock_show: AsyncMock) -> None:
        for bad_caps in ("tools", None, 42):
            with self.subTest(caps=bad_caps):
                mock_show.return_value = MagicMock(capabilities=bad_caps)
                with self.assertRaises(ModelCapabilityError):
                    await get_model_capabilities("test-model", "http://localhost:11434")

    @patch("ollama_agent.core.models._show_model")
    async def test_get_model_capabilities_dict_shape(self, mock_show: AsyncMock) -> None:
        mock_show.return_value = MagicMock(capabilities={"capabilities": ["tools"]})
        caps = await get_model_capabilities("test-model", "http://localhost:11434")
        self.assertEqual(caps, {"tools"})

    def test_get_model_info(self) -> None:
        response = MagicMock(model_info={"llama.context_length": 4096}, modelinfo=None)
        self.assertEqual(_get_model_info(response), {"llama.context_length": 4096})

        response_sdk = MagicMock(model_info=None, modelinfo={"gemma.context_length": 8192})
        self.assertEqual(_get_model_info(response_sdk), {"gemma.context_length": 8192})

        self.assertIsNone(_get_model_info(MagicMock(model_info="nope", modelinfo=None)))

    @patch("ollama_agent.core.models.get_model_capabilities")
    async def test_model_supports_tools(self, mock_caps: AsyncMock) -> None:
        mock_caps.return_value = {"tools"}
        self.assertTrue(await model_supports_tools("test-model", "http://localhost:11434"))

        mock_caps.return_value = set()
        self.assertFalse(await model_supports_tools("test-model", "http://localhost:11434"))

    def test_get_model_thinking_config(self) -> None:
        cfg = {"values": ["low", "high"], "default": "high"}
        self.assertEqual(get_model_thinking_config({"thinking": cfg}), cfg)
        self.assertEqual(get_model_thinking_config(MagicMock(thinking=cfg)), cfg)
        self.assertIsNone(get_model_thinking_config({"thinking": None}))
        self.assertIsNone(get_model_thinking_config({}))
        self.assertIsNone(get_model_thinking_config(MagicMock(thinking=None)))
        self.assertIsNone(get_model_thinking_config({"thinking": "not-a-dict"}))
        self.assertIsNone(get_model_thinking_config(None))

    @patch("ollama_agent.core.models._show_model")
    @patch("ollama_agent.core.models.get_model_capabilities")
    async def test_model_supports_thinking(self, mock_caps: AsyncMock, mock_show: AsyncMock) -> None:
        mock_show.return_value = MagicMock(thinking=None, capabilities=["thinking"])
        mock_caps.return_value = {"thinking"}
        self.assertTrue(await model_supports_thinking("test-model", "http://localhost:11434"))

        mock_show.return_value = MagicMock(thinking={"values": ["low", "high"], "default": "high"}, capabilities=[])
        self.assertTrue(await model_supports_thinking("test-model", "http://localhost:11434"))

        mock_show.return_value = MagicMock(thinking=None, capabilities=["tools"])
        mock_caps.return_value = {"tools"}
        self.assertFalse(await model_supports_thinking("test-model", "http://localhost:11434"))

    @patch("ollama_agent.core.models.ensure_model_supports_tools")
    @patch("ollama_agent.core.models.get_model_capabilities")
    async def test_resolve_ollama_reasoning(self, mock_caps: AsyncMock, mock_ensure: AsyncMock) -> None:
        warnings_log: list[str] = []
        warn = warnings_log.append

        # 1. Non-thinking model
        mock_caps.return_value = set()
        res_non = await resolve_ollama_reasoning(
            "llama3:8b", "high", "http://localhost:11434", warn, show_info=MagicMock(thinking=None, capabilities=[])
        )
        self.assertIsNone(res_non)

        # 2. Model with API-advertised thinking levels: glm-5.3-flash:cloud
        glm_show = MagicMock(
            thinking={"values": ["low", "high", "max"], "default": "max"},
            capabilities=["tools", "thinking"],
        )
        # Exact level match
        self.assertEqual(
            await resolve_ollama_reasoning("glm-5.3-flash:cloud", "max", "http://localhost:11434", warn, show_info=glm_show),
            "max",
        )
        self.assertEqual(
            await resolve_ollama_reasoning("glm-5.3-flash:cloud", "low", "http://localhost:11434", warn, show_info=glm_show),
            "low",
        )
        # Empty effort or "default" uses advertised default without warnings
        warnings_log.clear()
        self.assertEqual(
            await resolve_ollama_reasoning("glm-5.3-flash:cloud", "", "http://localhost:11434", warn, show_info=glm_show),
            "max",
        )
        self.assertEqual(
            await resolve_ollama_reasoning("glm-5.3-flash:cloud", "default", "http://localhost:11434", warn, show_info=glm_show),
            "max",
        )
        self.assertEqual(
            await resolve_ollama_reasoning("glm-5.3-flash:cloud", DEFAULT_REASONING_EFFORT, "http://localhost:11434", warn, show_info=glm_show),
            "max",
        )
        self.assertEqual(
            await resolve_ollama_reasoning("glm-5.3-flash:cloud", None, "http://localhost:11434", warn, show_info=glm_show),
            "max",
        )
        self.assertEqual(len(warnings_log), 0)

        # Activation coercion on string levels (no booleans in values): true/1/enabled/on returns default without warnings
        for activate_val in ("true", "1", "enabled", "on", True):
            warnings_log.clear()
            self.assertEqual(
                await resolve_ollama_reasoning("glm-5.3-flash:cloud", activate_val, "http://localhost:11434", warn, show_info=glm_show),
                "max",
            )
            self.assertEqual(len(warnings_log), 0)

        # Model with values=None handled safely
        null_values_show = MagicMock(
            thinking={"values": None, "default": "low"},
            capabilities=["tools", "thinking"],
        )
        self.assertEqual(
            await resolve_ollama_reasoning("null-val-model", "default", "http://localhost:11434", warn, show_info=null_values_show),
            "low",
        )

        # Thinking-only model: attempting "false" or "disabled" warns and falls back to default
        warnings_log.clear()
        res_glm_dis = await resolve_ollama_reasoning(
            "glm-5.3-flash:cloud", "false", "http://localhost:11434", warn, show_info=glm_show
        )
        self.assertEqual(res_glm_dis, "max")
        self.assertEqual(len(warnings_log), 1)
        self.assertIn("thinking-only", warnings_log[0])

        warnings_log.clear()
        res_glm_disabled = await resolve_ollama_reasoning(
            "glm-5.3-flash:cloud", "disabled", "http://localhost:11434", warn, show_info=glm_show
        )
        self.assertEqual(res_glm_disabled, "max")
        self.assertEqual(len(warnings_log), 1)
        self.assertIn("thinking-only", warnings_log[0])

        # Unsupported level warns and falls back to default
        warnings_log.clear()
        res_glm_unsupp = await resolve_ollama_reasoning(
            "glm-5.3-flash:cloud", "medium", "http://localhost:11434", warn, show_info=glm_show
        )
        self.assertEqual(res_glm_unsupp, "max")
        self.assertEqual(len(warnings_log), 1)
        self.assertIn("Allowed values", warnings_log[0])

        # 3. Model with [False, 'low', 'medium', 'xhigh']: qwen3.8:27b
        qwen_show = MagicMock(
            thinking={"values": [False, "low", "medium", "xhigh"], "default": "medium"},
            capabilities=["tools", "thinking"],
        )
        self.assertEqual(
            await resolve_ollama_reasoning("qwen3.8:27b", "xhigh", "http://localhost:11434", warn, show_info=qwen_show),
            "xhigh",
        )
        self.assertEqual(
            await resolve_ollama_reasoning("qwen3.8:27b", "medium", "http://localhost:11434", warn, show_info=qwen_show),
            "medium",
        )
        self.assertFalse(
            await resolve_ollama_reasoning("qwen3.8:27b", "false", "http://localhost:11434", warn, show_info=qwen_show)
        )
        self.assertFalse(
            await resolve_ollama_reasoning("qwen3.8:27b", "disabled", "http://localhost:11434", warn, show_info=qwen_show)
        )
        self.assertEqual(
            await resolve_ollama_reasoning("qwen3.8:27b", "default", "http://localhost:11434", warn, show_info=qwen_show),
            "medium",
        )
        self.assertEqual(
            await resolve_ollama_reasoning("qwen3.8:27b", "", "http://localhost:11434", warn, show_info=qwen_show),
            "medium",
        )

        # 4. Boolean model: gemma4 with [False, True]
        gemma_show = MagicMock(
            thinking={"values": [False, True], "default": True},
            capabilities=["tools", "thinking"],
        )
        self.assertFalse(
            await resolve_ollama_reasoning("gemma4", "false", "http://localhost:11434", warn, show_info=gemma_show)
        )
        self.assertFalse(
            await resolve_ollama_reasoning("gemma4", "disabled", "http://localhost:11434", warn, show_info=gemma_show)
        )
        self.assertTrue(
            await resolve_ollama_reasoning("gemma4", "true", "http://localhost:11434", warn, show_info=gemma_show)
        )
        self.assertTrue(
            await resolve_ollama_reasoning("gemma4", "default", "http://localhost:11434", warn, show_info=gemma_show)
        )
        self.assertTrue(
            await resolve_ollama_reasoning("gemma4", "", "http://localhost:11434", warn, show_info=gemma_show)
        )

        # 5. Legacy/custom model without thinking dict, but "thinking" capability
        mock_caps.return_value = {"thinking"}
        legacy_show = MagicMock(thinking=None, capabilities=["thinking"])
        self.assertEqual(
            await resolve_ollama_reasoning("custom-model", "high", "http://localhost:11434", warn, show_info=legacy_show),
            "high",
        )
        self.assertFalse(
            await resolve_ollama_reasoning("custom-model", "false", "http://localhost:11434", warn, show_info=legacy_show)
        )
        self.assertFalse(
            await resolve_ollama_reasoning("custom-model", "disabled", "http://localhost:11434", warn, show_info=legacy_show)
        )
        self.assertTrue(
            await resolve_ollama_reasoning("custom-model", "default", "http://localhost:11434", warn, show_info=legacy_show)
        )
        self.assertTrue(
            await resolve_ollama_reasoning("custom-model", "", "http://localhost:11434", warn, show_info=legacy_show)
        )

        # 6. Call with show_info=None fetches metadata via _show_model
        with patch("ollama_agent.core.models._show_model", AsyncMock(return_value=glm_show)):
            self.assertEqual(
                await resolve_ollama_reasoning("glm-5.3-flash:cloud", "max", "http://localhost:11434", warn),
                "max",
            )

    @patch("ollama_agent.core.models.model_supports_tools")
    async def test_ensure_model_supports_tools_raises(self, mock_supports: AsyncMock) -> None:
        mock_supports.return_value = False
        with self.assertRaises(ModelCapabilityError):
            await ensure_model_supports_tools("test-model", "http://localhost:11434")

    async def test_resolve_context_window_explicit_value(self) -> None:
        resolved = await resolve_context_window("test-model", 4096, "http://localhost:11434")
        self.assertEqual(resolved, 4096)

    async def test_resolve_context_window_string_int(self) -> None:
        resolved = await resolve_context_window("test-model", "16384", "http://localhost:11434")
        self.assertEqual(resolved, 16384)

    async def test_resolve_context_window_invalid_explicit_value_raises(self) -> None:
        with self.assertRaises(ModelContextWindowError):
            await resolve_context_window("test-model", 0, "http://localhost:11434")

        with self.assertRaises(ModelContextWindowError):
            await resolve_context_window("test-model", -10, "http://localhost:11434")

        with self.assertRaises(ModelContextWindowError):
            await resolve_context_window("test-model", "invalid", "http://localhost:11434")

    @patch("ollama_agent.core.models._show_model")
    async def test_resolve_context_window_max(self, mock_show: AsyncMock) -> None:
        mock_show.return_value = MagicMock(
            model_info={"llama.context_length": 131072},
            parameters="",
            modelfile="",
        )
        resolved = await resolve_context_window("llama3.3:70b", "max", "http://localhost:11434")
        self.assertEqual(resolved, 131072)

        # Case-insensitive
        resolved_upper = await resolve_context_window("llama3.3:70b", "MAX", "http://localhost:11434")
        self.assertEqual(resolved_upper, 131072)

        # Ollama SDK modelinfo attribute format (without underscore)
        mock_show.return_value = MagicMock(
            model_info=None,
            modelinfo={"gemma4.context_length": 262144},
            parameters="",
            modelfile="",
        )
        resolved_sdk = await resolve_context_window("gemma4:26b-a4b-it-qat", "max", "http://localhost:11434")
        self.assertEqual(resolved_sdk, 262144)

    @patch("ollama_agent.core.models._show_model")
    async def test_resolve_context_window_max_modelfile_fallback(self, mock_show: AsyncMock) -> None:
        mock_show.return_value = MagicMock(
            model_info={},
            parameters="PARAMETER num_ctx 32768",
            modelfile="",
        )
        resolved = await resolve_context_window("custom-model", "max", "http://localhost:11434")
        self.assertEqual(resolved, 32768)

    @patch("ollama_agent.core.models._show_model")
    async def test_resolve_context_window_max_unresolved_raises(self, mock_show: AsyncMock) -> None:
        mock_show.return_value = MagicMock(
            model_info={},
            parameters="",
            modelfile="",
        )
        with self.assertRaises(ModelContextWindowError):
            await resolve_context_window("unknown-model", "max", "http://localhost:11434")

    @patch("ollama_agent.core.models._show_model")
    @patch("ollama_agent.core.models.resolve_context_window", AsyncMock(return_value=8192))
    @patch("ollama_agent.core.models.resolve_ollama_reasoning", AsyncMock(return_value=True))
    async def test_create_ollama_chat_model(self, mock_show: AsyncMock) -> None:
        mock_show.return_value = MagicMock(parameters="", modelfile="")
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
            warn_callback=lambda _msg: None,
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
        text = "PARAMETER temperature \"0.65\"\nPARAMETER top_p '0.85'\nPARAMETER top_k 30\nrepeat_penalty 1.15"
        self.assertEqual(_parse_modelfile_param(text, "temperature"), "0.65")
        self.assertEqual(_parse_modelfile_param(text, "top_p"), "0.85")
        self.assertEqual(_parse_modelfile_param(text, "top_k"), "30")
        self.assertEqual(_parse_modelfile_param(text, "repeat_penalty"), "1.15")
        self.assertIsNone(_parse_modelfile_param(text, "min_p"))
        self.assertIsNone(_parse_modelfile_param("", "temperature"))

    def test_extended_show_response(self) -> None:
        resp = ExtendedShowResponse.model_validate({
            "thinking": {"values": ["low", "high", "max"], "default": "max"},
            "model_info": {},
            "capabilities": ["tools", "thinking"],
        })
        self.assertEqual(resp.thinking, {"values": ["low", "high", "max"], "default": "max"})
        self.assertEqual(resp.capabilities, ["tools", "thinking"])

    @patch("ollama_agent.core.models._show_model")
    async def test_resolve_model_parameters_precedence(self, mock_show: AsyncMock) -> None:
        mock_show.return_value = MagicMock(
            parameters="temperature 0.65\ntop_p 0.85\nrepetition_penalty 1.25",
            modelfile=None,
        )
        warnings_log: list[str] = []

        # 1. User overrides temperature; top_p and repeat_penalty resolve from metadata; unconfigured omitted
        resolved = await resolve_model_parameters(
            "test-model",
            "http://localhost:11434",
            temperature=0.3,
            top_p=None,
            top_k=None,
            min_p=None,
            presence_penalty=None,
            repeat_penalty=None,
            warn_callback=warnings_log.append,
        )

        # User value
        self.assertEqual(resolved["temperature"], (0.3, "user"))
        # Modelfile values
        self.assertEqual(resolved["top_p"], (0.85, "modelfile"))
        self.assertEqual(resolved["repeat_penalty"], (1.25, "modelfile"))
        # Unset values should not be present in resolved
        self.assertNotIn("top_k", resolved)
        self.assertNotIn("min_p", resolved)
        self.assertNotIn("presence_penalty", resolved)
        self.assertEqual(warnings_log, [])

    @patch("ollama_agent.core.models._show_model")
    async def test_resolve_model_parameters_invalid_value_warns(self, mock_show: AsyncMock) -> None:
        mock_show.return_value = MagicMock(
            parameters="temperature not-a-number",
            modelfile="",
        )
        warnings_log: list[str] = []

        resolved = await resolve_model_parameters(
            "test-model",
            "http://localhost:11434",
            warn_callback=warnings_log.append,
        )

        self.assertNotIn("temperature", resolved)
        self.assertEqual(len(warnings_log), 1)
        self.assertIn("not-a-number", warnings_log[0])
        self.assertIn("temperature", warnings_log[0])

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
            warn_callback=lambda _msg: None,
        )
        self.assertIsNone(model.temperature)
        self.assertIsNone(model.top_p)
        self.assertIsNone(model.top_k)
        self.assertIsNone(model.min_p)
        self.assertIsNone(model.presence_penalty)
        self.assertIsNone(model.repeat_penalty)
        self.assertEqual(model.effective_params, {})

    @patch("httpx.get")
    def test_get_ollama_version_success(self, mock_get: MagicMock) -> None:
        mock_response = MagicMock()
        mock_response.json.return_value = {"version": "0.34.3"}
        mock_get.return_value = mock_response

        version = get_ollama_version("http://localhost:11434")
        self.assertEqual(version, "0.34.3")
        mock_get.assert_called_once_with("http://localhost:11434/api/version", timeout=5.0)

    @patch("httpx.get")
    def test_get_ollama_version_connect_error(self, mock_get: MagicMock) -> None:
        mock_get.side_effect = httpx.ConnectError("Connection refused")
        with self.assertRaises(ModelCapabilityError) as cm:
            get_ollama_version("http://localhost:11434")
        self.assertIn("Could not connect to Ollama", str(cm.exception))

    @patch("httpx.get")
    def test_get_ollama_version_http_error(self, mock_get: MagicMock) -> None:
        mock_response = MagicMock()
        mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "500 Internal Server Error", request=MagicMock(), response=mock_response
        )
        mock_get.return_value = mock_response

        with self.assertRaises(ModelCapabilityError) as cm:
            get_ollama_version("http://localhost:11434")
        self.assertIn("Could not connect to Ollama", str(cm.exception))

    @patch("ollama_agent.core.models.get_ollama_version", return_value="0.34.3")
    def test_check_ollama_version_equal(self, mock_get_ver: MagicMock) -> None:
        version = check_ollama_version("http://localhost:11434")
        self.assertEqual(version, "0.34.3")
        mock_get_ver.assert_called_once_with("http://localhost:11434")

    @patch("ollama_agent.core.models.get_ollama_version", return_value="0.35.0")
    def test_check_ollama_version_higher(self, mock_get_ver: MagicMock) -> None:
        version = check_ollama_version("http://localhost:11434")
        self.assertEqual(version, "0.35.0")

    @patch("ollama_agent.core.models.get_ollama_version", return_value="0.34.2")
    def test_check_ollama_version_lower_raises(self, mock_get_ver: MagicMock) -> None:
        with self.assertRaises(OllamaVersionError) as cm:
            check_ollama_version("http://localhost:11434")
        self.assertIn("0.34.2", str(cm.exception))
        self.assertIn("0.34.3", str(cm.exception))

    @patch("ollama_agent.core.models.get_ollama_version", return_value="0.10.0")
    def test_check_ollama_version_much_lower_raises(self, mock_get_ver: MagicMock) -> None:
        with self.assertRaises(OllamaVersionError) as cm:
            check_ollama_version("http://localhost:11434")
        self.assertIn("0.10.0", str(cm.exception))
        self.assertIn("0.34.3", str(cm.exception))

    @patch("ollama_agent.core.models.get_ollama_version", return_value="0.40.0")
    def test_check_ollama_version_custom_min(self, mock_get_ver: MagicMock) -> None:
        version = check_ollama_version("http://localhost:11434", min_version="0.40.0")
        self.assertEqual(version, "0.40.0")

        with self.assertRaises(OllamaVersionError):
            check_ollama_version("http://localhost:11434", min_version="0.41.0")


if __name__ == "__main__":
    unittest.main()
