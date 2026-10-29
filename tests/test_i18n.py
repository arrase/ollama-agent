"""Tests for internationalization (i18n) system."""

import ast
import json
import os
import re
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pytest

from ollama_agent.i18n import (
    SUPPORTED_LOCALES,
    _,
    detect_system_language,
    get_locale,
    set_locale,
)
from ollama_agent.interfaces.cli import create_argument_parser
from ollama_agent.settings import Settings, reset_config


class TestI18nDetection(unittest.TestCase):
    """Test system language detection and fallbacks."""

    def test_detect_language_from_language_env(self) -> None:
        with patch.dict(os.environ, {"LANGUAGE": "es_ES:es"}, clear=True):
            assert detect_system_language() == "es"

    def test_detect_language_from_lc_all_env(self) -> None:
        with patch.dict(os.environ, {"LC_ALL": "fr_FR.UTF-8"}, clear=True):
            assert detect_system_language() == "fr"

    def test_detect_language_from_lc_messages_env(self) -> None:
        with patch.dict(os.environ, {"LC_MESSAGES": "de_DE@euro"}, clear=True):
            assert detect_system_language() == "de"

    def test_detect_language_from_lang_env(self) -> None:
        with patch.dict(os.environ, {"LANG": "it_IT.UTF-8"}, clear=True):
            assert detect_system_language() == "it"

    def test_detect_common_languages_with_country_codes(self) -> None:
        test_cases = [
            ("pt_BR.UTF-8", "pt"),
            ("zh_CN.UTF-8", "zh"),
            ("zh_TW.UTF-8", "zh"),
            ("zh_HK", "zh"),
            ("hi_IN.UTF-8", "hi"),
            ("ja_JP.eucJP", "ja"),
            ("ru_RU.CP1251", "ru"),
            ("es_MX.UTF-8", "es"),
            ("en_US.UTF-8", "en"),
            ("en_GB", "en"),
        ]
        for env_val, expected in test_cases:
            with patch.dict(os.environ, {"LANG": env_val}, clear=True):
                assert detect_system_language() == expected, f"Failed for {env_val}"

    def test_detect_unsupported_language_fallback_to_english(self) -> None:
        test_cases = ["sv_SE", "ca_ES", "da_DK", "fi_FI", "el_GR", "he_IL", "th_TH", "vi_VN", "unknown", "C", "POSIX"]
        for env_val in test_cases:
            with (
                patch.dict(os.environ, {"LANG": env_val}, clear=True),
                patch("locale.getlocale", return_value=(None, None)),
            ):
                assert detect_system_language() == "en", f"Failed for {env_val}"

    def test_detect_empty_environment_fallback_to_english(self) -> None:
        with patch.dict(os.environ, {}, clear=True), patch("locale.getlocale", return_value=(None, None)):
            assert detect_system_language() == "en"


class TestI18nLocaleManagement(unittest.TestCase):
    """Test setting the active locale and the supported locales registry."""

    def test_supported_locales_list(self) -> None:
        locales = SUPPORTED_LOCALES
        assert "en" in locales
        assert "es" in locales
        assert "fr" in locales
        assert "de" in locales
        assert "it" in locales
        assert "pt" in locales
        assert "zh" in locales
        assert "ja" in locales
        assert "ru" in locales
        assert "hi" in locales
        assert "ko" in locales
        assert "ar" in locales
        assert "tr" in locales
        assert "pl" in locales
        assert "nl" in locales
        assert "uk" in locales

    def test_set_valid_locale(self) -> None:
        for loc in SUPPORTED_LOCALES:
            assert set_locale(loc) == loc

    def test_set_locale_with_region_suffix(self) -> None:
        assert set_locale("es_ES.UTF-8") == "es"
        assert set_locale("fr-FR") == "fr"
        assert set_locale("hi_IN.UTF-8") == "hi"
        assert set_locale("zh_CN") == "zh"

    def test_set_unsupported_locale_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="Unsupported language: unsupported_lang"):
            set_locale("unsupported_lang")

    def test_set_locale_none_and_default(self) -> None:
        with patch("ollama_agent.i18n.detect_system_language", return_value="es"):
            assert set_locale(None) == "es"
            assert get_locale() == "es"
            assert set_locale() == "es"

    def test_get_locale(self) -> None:
        set_locale("en")
        assert get_locale() == "en"
        set_locale("es")
        assert get_locale() == "es"
        set_locale("fr")
        assert get_locale() == "fr"


class TestI18nTranslations(unittest.TestCase):
    """Test string translations across languages and parameter formatting."""

    def test_english_translation(self) -> None:
        set_locale("en")
        assert _("Manage saved tasks") == "Manage saved tasks"
        assert _("Exit the REPL") == "Exit the REPL"

    def test_spanish_translation(self) -> None:
        set_locale("es")
        assert _("Manage saved tasks") == "Gestionar tareas guardadas"
        assert _("Exit the REPL") == "Salir del REPL"
        assert _("No skills found.") == "No se encontraron habilidades."
        assert _("Tool:") == "Herramienta:"
        assert _("Arguments:") == "Argumentos:"
        assert _("Copy") == "Copiar"
        assert _("Paste") == "Pegar"
        assert _("Interrupt") == "Interrumpir"
        assert _("Show or clear the prompt queue") == "Mostrar o limpiar la cola de prompts"
        assert _("Approval cancelled.") == "Aprobación cancelada."

    def test_french_translation(self) -> None:
        set_locale("fr")
        assert _("Manage saved tasks") == "Gérer les tâches enregistrées"
        assert _("Exit the REPL") == "Quitter le REPL"
        assert _("Tool:") == "Outil :"
        assert _("Arguments:") == "Arguments :"
        assert _("Show or clear the prompt queue") == "Afficher ou vider la file d'attente des invites"
        assert _("Approval cancelled.") == "Approbation annulée."

    def test_german_translation(self) -> None:
        set_locale("de")
        assert _("Manage saved tasks") == "Gespeicherte Aufgaben verwalten"
        assert _("Exit the REPL") == "REPL beenden"
        assert _("Tool:") == "Werkzeug:"
        assert _("Arguments:") == "Argumente:"
        assert _("Show or clear the prompt queue") == "Prompt-Warteschlange anzeigen oder leeren"
        assert _("Approval cancelled.") == "Genehmigung abgebrochen."

    def test_italian_translation(self) -> None:
        set_locale("it")
        assert _("Manage saved tasks") == "Gestisci le attività salvate"
        assert _("Exit the REPL") == "Uscire dalla REPL"
        assert _("Tool:") == "Strumento:"
        assert _("Arguments:") == "Argomenti:"

    def test_portuguese_translation(self) -> None:
        set_locale("pt")
        assert _("Manage saved tasks") == "Gerenciar tarefas salvas"
        assert _("Exit the REPL") == "Sair do REPL"
        assert _("Tool:") == "Ferramenta:"
        assert _("Arguments:") == "Argumentos:"

    def test_chinese_translation(self) -> None:
        set_locale("zh")
        assert _("Manage saved tasks") == "管理保存的任务"
        assert _("Exit the REPL") == "退出 REPL"

    def test_japanese_translation(self) -> None:
        set_locale("ja")
        assert _("Manage saved tasks") == "保存済みタスクを管理"
        assert _("Exit the REPL") == "REPLを終了"

    def test_russian_translation(self) -> None:
        set_locale("ru")
        assert _("Manage saved tasks") == "Управление сохраненными задачами"
        assert _("Exit the REPL") == "Выйти из REPL"

    def test_hindi_translation(self) -> None:
        set_locale("hi")
        assert _("Manage saved tasks") == "सहेजे गए कार्य प्रबंधित करें"
        assert _("Exit the REPL") == "REPL से बाहर निकलें"
        assert _("No skills found.") == "कोई कौशल नहीं मिला।"

    def test_korean_translation(self) -> None:
        set_locale("ko")
        assert _("Manage saved tasks") == "저장된 작업 관리"
        assert _("Exit the REPL") == "REPL 종료"

    def test_arabic_translation(self) -> None:
        set_locale("ar")
        assert _("Manage saved tasks") == "إدارة المهام المحفوظة"
        assert _("Exit the REPL") == "اخرج من REPL"

    def test_turkish_translation(self) -> None:
        set_locale("tr")
        assert _("Manage saved tasks") == "Kaydedilen görevleri yönet"
        assert _("Exit the REPL") == "REPL'den çık"

    def test_polish_translation(self) -> None:
        set_locale("pl")
        assert _("Manage saved tasks") == "Zarządzaj zapisanymi zadaniami"
        assert _("Exit the REPL") == "Zakończ działanie REPL"

    def test_dutch_translation(self) -> None:
        set_locale("nl")
        assert _("Manage saved tasks") == "Opgeslagen taken beheren"
        assert _("Exit the REPL") == "Sluit de REPL af"

    def test_ukrainian_translation(self) -> None:
        set_locale("uk")
        assert _("Manage saved tasks") == "Керування збереженими завданнями"
        assert _("Exit the REPL") == "Вийти з REPL"

    def test_parameter_interpolation(self) -> None:
        set_locale("en")
        assert (
            _("Settings file must contain a YAML mapping: {path}", path="demo")
            == "Settings file must contain a YAML mapping: demo"
        )
        set_locale("es")
        assert (
            _("Settings file must contain a YAML mapping: {path}", path="demo")
            == "El archivo de ajustes debe contener un mapeo YAML: demo"
        )
        set_locale("fr")
        assert (
            _("Settings file must contain a YAML mapping: {path}", path="demo")
            == "Le fichier de paramètres doit contenir un mappage YAML : demo"
        )
        set_locale("hi")
        assert (
            _("Settings file must contain a YAML mapping: {path}", path="demo")
            == "सेटिंग्स फ़ाइल में YAML मैपिंग होनी चाहिए: demo"
        )
        set_locale("en")
        assert _("Queued prompts ({count}):", count=1) == "Queued prompts (1):"
        assert _("Prompt queue cleared ({count} removed).", count=3) == "Prompt queue cleared (3 removed)."
        assert _("Removed #{pos} from prompt queue: {text}", pos=2, text="foo") == "Removed #2 from prompt queue: foo"
        assert (
            _("Invalid queue position '{pos}'. Usage: /queue rm <position>", pos="xyz")
            == "Invalid queue position 'xyz'. Usage: /queue rm <position>"
        )
        assert (
            _("Queue position {pos} out of range (queue has {count} items).", pos=5, count=2)
            == "Queue position 5 out of range (queue has 2 items)."
        )
        assert _("Queued ({count})", count=4) == "Queued (4)"
        set_locale("es")
        assert _("Queued prompts ({count}):", count=1) == "Prompts en cola (1):"
        assert _("Prompt queue cleared ({count} removed).", count=3) == "Cola de prompts limpiada (3 eliminados)."
        assert (
            _("Removed #{pos} from prompt queue: {text}", pos=2, text="foo")
            == "Eliminado #2 de la cola de prompts: foo"
        )
        assert (
            _("Invalid queue position '{pos}'. Usage: /queue rm <position>", pos="xyz")
            == "Posición de cola no válida 'xyz'. Uso: /queue rm <posición>"
        )
        assert (
            _("Queue position {pos} out of range (queue has {count} items).", pos=5, count=2)
            == "Posición de cola 5 fuera de rango (la cola tiene 2 elementos)."
        )
        assert _("Queued ({count})", count=4) == "En cola (4)"

    def test_unknown_string_in_english_is_source_language(self) -> None:
        set_locale("en")
        unknown = "This is a non-registered string"
        assert _(unknown) == unknown

    def test_missing_translation_in_non_default_locale_returns_original(self) -> None:
        set_locale("es")
        unknown = "This is a non-registered string"
        assert _(unknown) == unknown


def _extract_file_strings(py_file: Path) -> set[str]:
    tree = ast.parse(py_file.read_text(encoding="utf-8"), filename=str(py_file))
    module_constants = {
        target.id: stmt.value.value
        for stmt in tree.body
        if isinstance(stmt, ast.Assign) and isinstance(stmt.value, ast.Constant) and isinstance(stmt.value.value, str)
        for target in stmt.targets
        if isinstance(target, ast.Name)
    }
    strings: set[str] = set()
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and node.args):
            continue
        func = node.func
        is_translate = (isinstance(func, ast.Name) and func.id == "_") or (
            isinstance(func, ast.Attribute) and func.attr == "_"
        )
        if not is_translate:
            continue
        first_arg = node.args[0]
        if isinstance(first_arg, ast.Constant) and isinstance(first_arg.value, str):
            strings.add(first_arg.value)
        elif isinstance(first_arg, ast.Name) and first_arg.id in module_constants:
            strings.add(module_constants[first_arg.id])
    return strings


class TestCatalogCompleteness(unittest.TestCase):
    """Verify all non-English locale JSON catalogs are complete and consistent."""

    def test_all_locale_catalogs_parity(self) -> None:
        locales_dir = Path(__file__).parent.parent / "ollama_agent" / "i18n" / "locales"
        non_en_locales = [loc for loc in SUPPORTED_LOCALES if loc != "en"]
        assert not (locales_dir / "en.json").exists(), (
            "en.json should not exist as English is the default in-code language"
        )

        ref_file = locales_dir / f"{non_en_locales[0]}.json"
        assert ref_file.exists(), f"Reference file {ref_file} must exist"

        ref_data: dict[str, str] = json.loads(ref_file.read_text(encoding="utf-8"))

        placeholder_pattern = re.compile(r"\{([a-zA-Z0-9_]+)\}")

        for loc in non_en_locales:
            loc_file = locales_dir / f"{loc}.json"
            assert loc_file.exists(), f"{loc}.json must exist"

            loc_data: dict[str, str] = json.loads(loc_file.read_text(encoding="utf-8"))

            # Check key parity with reference catalog (keys are original English strings)
            missing_keys = set(ref_data.keys()) - set(loc_data.keys())
            extra_keys = set(loc_data.keys()) - set(ref_data.keys())
            assert missing_keys == set(), f"{loc}.json is missing keys: {missing_keys}"
            assert extra_keys == set(), f"{loc}.json has unexpected extra keys: {extra_keys}"

            # Check placeholder consistency against original English key
            for key, loc_val in loc_data.items():
                key_placeholders = set(placeholder_pattern.findall(key))
                loc_placeholders = set(placeholder_pattern.findall(loc_val))
                assert key_placeholders == loc_placeholders, (
                    f"Placeholder mismatch in {loc}.json for key '{key}': {key_placeholders} vs {loc_placeholders}"
                )

    def test_all_codebase_strings_present_in_catalogs(self) -> None:
        pkg_dir = Path(__file__).parent.parent / "ollama_agent"
        locales_dir = pkg_dir / "i18n" / "locales"

        extracted_strings: set[str] = set()
        for py_file in pkg_dir.rglob("*.py"):
            extracted_strings.update(_extract_file_strings(py_file))

        for loc in [loc_code for loc_code in SUPPORTED_LOCALES if loc_code != "en"]:
            loc_file = locales_dir / f"{loc}.json"
            catalog = json.loads(loc_file.read_text(encoding="utf-8"))
            missing = [s for s in extracted_strings if s not in catalog]
            assert missing == [], f"{loc}.json is missing translation for in-code strings: {missing}"
            dead = [s for s in catalog if s not in extracted_strings]
            assert dead == [], f"{loc}.json contains dead/unused translation keys: {dead}"


class TestCLIAndSettingsI18n(unittest.TestCase):
    """Test CLI language argument and runtime settings integration."""

    def test_cli_parser_language_arg(self) -> None:
        parser = create_argument_parser()
        args = parser.parse_args(["--lang", "es"])
        assert args.language == "es"

        args2 = parser.parse_args(["-l", "hi"])
        assert args2.language == "hi"

        args3 = parser.parse_args(["-l", "zh"])
        assert args3.language == "zh"

    def test_settings_language_field(self) -> None:
        s = Settings()
        assert s.runtime.language == ""
        s.runtime.language = "hi"
        d = s.to_dict()
        assert d["runtime"]["language"] == "hi"

        s2 = Settings.from_dict({"runtime": {"language": "zh"}})
        assert s2.runtime.language == "zh"

    def test_reset_config_localization(self) -> None:
        set_locale("es")
        # tmp_path keeps the test from clobbering a real /tmp file.
        target = Path(str(self.enterContext(tempfile.TemporaryDirectory()))) / "dummy_settings.yaml"
        msgs = reset_config("config-file", settings_path=target)
        assert any("Reinicio:" in m or "restauró" in m for m in msgs), f"Unexpected messages: {msgs}"


if __name__ == "__main__":
    unittest.main()
