"""Tests for internationalization (i18n) system."""

import ast
import json
import os
import re
import unittest
from pathlib import Path
from unittest.mock import patch

from ollama_agent.i18n import (
    SUPPORTED_LOCALES,
    _,
    detect_system_language,
    set_locale,
)
from ollama_agent.interfaces.cli import create_argument_parser
from ollama_agent.settings import Settings, reset_config


class TestI18nDetection(unittest.TestCase):
    """Test system language detection and fallbacks."""

    def tearDown(self) -> None:
        set_locale("en")

    def test_detect_language_from_language_env(self) -> None:
        with patch.dict(os.environ, {"LANGUAGE": "es_ES:es"}, clear=True):
            self.assertEqual(detect_system_language(), "es")

    def test_detect_language_from_lc_all_env(self) -> None:
        with patch.dict(os.environ, {"LC_ALL": "fr_FR.UTF-8"}, clear=True):
            self.assertEqual(detect_system_language(), "fr")

    def test_detect_language_from_lc_messages_env(self) -> None:
        with patch.dict(os.environ, {"LC_MESSAGES": "de_DE@euro"}, clear=True):
            self.assertEqual(detect_system_language(), "de")

    def test_detect_language_from_lang_env(self) -> None:
        with patch.dict(os.environ, {"LANG": "it_IT.UTF-8"}, clear=True):
            self.assertEqual(detect_system_language(), "it")

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
                self.assertEqual(detect_system_language(), expected, f"Failed for {env_val}")

    def test_detect_unsupported_language_fallback_to_english(self) -> None:
        test_cases = ["sv_SE", "ca_ES", "da_DK", "fi_FI", "el_GR", "he_IL", "th_TH", "vi_VN", "unknown", "C", "POSIX"]
        for env_val in test_cases:
            with (
                patch.dict(os.environ, {"LANG": env_val}, clear=True),
                patch("locale.getlocale", return_value=(None, None)),
            ):
                self.assertEqual(detect_system_language(), "en", f"Failed for {env_val}")

    def test_detect_empty_environment_fallback_to_english(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            with patch("locale.getlocale", return_value=(None, None)):
                self.assertEqual(detect_system_language(), "en")


class TestI18nLocaleManagement(unittest.TestCase):
    """Test setting the active locale and the supported locales registry."""

    def tearDown(self) -> None:
        set_locale("en")

    def test_supported_locales_list(self) -> None:
        locales = SUPPORTED_LOCALES
        self.assertIn("en", locales)
        self.assertIn("es", locales)
        self.assertIn("fr", locales)
        self.assertIn("de", locales)
        self.assertIn("it", locales)
        self.assertIn("pt", locales)
        self.assertIn("zh", locales)
        self.assertIn("ja", locales)
        self.assertIn("ru", locales)
        self.assertIn("hi", locales)
        self.assertIn("ko", locales)
        self.assertIn("ar", locales)
        self.assertIn("tr", locales)
        self.assertIn("pl", locales)
        self.assertIn("nl", locales)
        self.assertIn("uk", locales)

    def test_set_valid_locale(self) -> None:
        for loc in SUPPORTED_LOCALES:
            self.assertEqual(set_locale(loc), loc)

    def test_set_locale_with_region_suffix(self) -> None:
        self.assertEqual(set_locale("es_ES.UTF-8"), "es")
        self.assertEqual(set_locale("fr-FR"), "fr")
        self.assertEqual(set_locale("hi_IN.UTF-8"), "hi")
        self.assertEqual(set_locale("zh_CN"), "zh")

    def test_set_unsupported_locale_raises_value_error(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unsupported language: unsupported_lang"):
            set_locale("unsupported_lang")


class TestI18nTranslations(unittest.TestCase):
    """Test string translations across languages and parameter formatting."""

    def tearDown(self) -> None:
        set_locale("en")

    def test_english_translation(self) -> None:
        set_locale("en")
        self.assertEqual(_("Manage saved tasks"), "Manage saved tasks")
        self.assertEqual(_("Exit the REPL"), "Exit the REPL")

    def test_spanish_translation(self) -> None:
        set_locale("es")
        self.assertEqual(_("Manage saved tasks"), "Gestionar tareas guardadas")
        self.assertEqual(_("Exit the REPL"), "Salir del REPL")
        self.assertEqual(_("No skills found."), "No se encontraron habilidades.")
        self.assertEqual(_("Tool:"), "Herramienta:")
        self.assertEqual(_("Arguments:"), "Argumentos:")
        self.assertEqual(_("Copy"), "Copiar")
        self.assertEqual(_("Paste"), "Pegar")
        self.assertEqual(_("Interrupt"), "Interrumpir")
        self.assertEqual(_("Show or clear the prompt queue"), "Mostrar o limpiar la cola de prompts")
        self.assertEqual(_("Approval cancelled."), "Aprobación cancelada.")

    def test_french_translation(self) -> None:
        set_locale("fr")
        self.assertEqual(_("Manage saved tasks"), "Gérer les tâches enregistrées")
        self.assertEqual(_("Exit the REPL"), "Quitter le REPL")
        self.assertEqual(_("Tool:"), "Outil :")
        self.assertEqual(_("Arguments:"), "Arguments :")
        self.assertEqual(_("Show or clear the prompt queue"), "Afficher ou vider la file d'attente des invites")
        self.assertEqual(_("Approval cancelled."), "Approbation annulée.")

    def test_german_translation(self) -> None:
        set_locale("de")
        self.assertEqual(_("Manage saved tasks"), "Gespeicherte Aufgaben verwalten")
        self.assertEqual(_("Exit the REPL"), "REPL beenden")
        self.assertEqual(_("Tool:"), "Werkzeug:")
        self.assertEqual(_("Arguments:"), "Argumente:")
        self.assertEqual(_("Show or clear the prompt queue"), "Prompt-Warteschlange anzeigen oder leeren")
        self.assertEqual(_("Approval cancelled."), "Genehmigung abgebrochen.")

    def test_italian_translation(self) -> None:
        set_locale("it")
        self.assertEqual(_("Manage saved tasks"), "Gestisci le attività salvate")
        self.assertEqual(_("Exit the REPL"), "Uscire dalla REPL")
        self.assertEqual(_("Tool:"), "Strumento:")
        self.assertEqual(_("Arguments:"), "Argomenti:")

    def test_portuguese_translation(self) -> None:
        set_locale("pt")
        self.assertEqual(_("Manage saved tasks"), "Gerenciar tarefas salvas")
        self.assertEqual(_("Exit the REPL"), "Sair do REPL")
        self.assertEqual(_("Tool:"), "Ferramenta:")
        self.assertEqual(_("Arguments:"), "Argumentos:")

    def test_chinese_translation(self) -> None:
        set_locale("zh")
        self.assertEqual(_("Manage saved tasks"), "管理保存的任务")
        self.assertEqual(_("Exit the REPL"), "退出 REPL")

    def test_japanese_translation(self) -> None:
        set_locale("ja")
        self.assertEqual(_("Manage saved tasks"), "保存済みタスクを管理")
        self.assertEqual(_("Exit the REPL"), "REPLを終了")

    def test_russian_translation(self) -> None:
        set_locale("ru")
        self.assertEqual(_("Manage saved tasks"), "Управление сохраненными задачами")
        self.assertEqual(_("Exit the REPL"), "Выйти из REPL")

    def test_hindi_translation(self) -> None:
        set_locale("hi")
        self.assertEqual(_("Manage saved tasks"), "सहेजे गए कार्य प्रबंधित करें")
        self.assertEqual(_("Exit the REPL"), "REPL से बाहर निकलें")
        self.assertEqual(_("No skills found."), "कोई कौशल नहीं मिला।")

    def test_korean_translation(self) -> None:
        set_locale("ko")
        self.assertEqual(_("Manage saved tasks"), "저장된 작업 관리")
        self.assertEqual(_("Exit the REPL"), "REPL 종료")

    def test_arabic_translation(self) -> None:
        set_locale("ar")
        self.assertEqual(_("Manage saved tasks"), "إدارة المهام المحفوظة")
        self.assertEqual(_("Exit the REPL"), "اخرج من REPL")

    def test_turkish_translation(self) -> None:
        set_locale("tr")
        self.assertEqual(_("Manage saved tasks"), "Kaydedilen görevleri yönet")
        self.assertEqual(_("Exit the REPL"), "REPL'den çık")

    def test_polish_translation(self) -> None:
        set_locale("pl")
        self.assertEqual(_("Manage saved tasks"), "Zarządzaj zapisanymi zadaniami")
        self.assertEqual(_("Exit the REPL"), "Zakończ działanie REPL")

    def test_dutch_translation(self) -> None:
        set_locale("nl")
        self.assertEqual(_("Manage saved tasks"), "Opgeslagen taken beheren")
        self.assertEqual(_("Exit the REPL"), "Sluit de REPL af")

    def test_ukrainian_translation(self) -> None:
        set_locale("uk")
        self.assertEqual(_("Manage saved tasks"), "Керування збереженими завданнями")
        self.assertEqual(_("Exit the REPL"), "Вийти з REPL")

    def test_parameter_interpolation(self) -> None:
        set_locale("en")
        self.assertEqual(
            _("Settings file must contain a YAML mapping: {path}", path="demo"),
            "Settings file must contain a YAML mapping: demo",
        )
        set_locale("es")
        self.assertEqual(
            _("Settings file must contain a YAML mapping: {path}", path="demo"),
            "El archivo de ajustes debe contener un mapeo YAML: demo",
        )
        set_locale("fr")
        self.assertEqual(
            _("Settings file must contain a YAML mapping: {path}", path="demo"),
            "Le fichier de paramètres doit contenir un mappage YAML : demo",
        )
        set_locale("hi")
        self.assertEqual(
            _("Settings file must contain a YAML mapping: {path}", path="demo"),
            "सेटिंग्स फ़ाइल में YAML मैपिंग होनी चाहिए: demo",
        )
        set_locale("en")
        self.assertEqual(
            _("Queued prompts ({count}):", count=1),
            "Queued prompts (1):",
        )
        self.assertEqual(
            _("Prompt queue cleared ({count} removed).", count=3),
            "Prompt queue cleared (3 removed).",
        )
        self.assertEqual(
            _("Removed #{pos} from prompt queue: {text}", pos=2, text="foo"),
            "Removed #2 from prompt queue: foo",
        )
        self.assertEqual(
            _("Invalid queue position '{pos}'. Usage: /queue rm <position>", pos="xyz"),
            "Invalid queue position 'xyz'. Usage: /queue rm <position>",
        )
        self.assertEqual(
            _("Queue position {pos} out of range (queue has {count} items).", pos=5, count=2),
            "Queue position 5 out of range (queue has 2 items).",
        )
        self.assertEqual(
            _("Queued ({count})", count=4),
            "Queued (4)",
        )
        set_locale("es")
        self.assertEqual(
            _("Queued prompts ({count}):", count=1),
            "Prompts en cola (1):",
        )
        self.assertEqual(
            _("Prompt queue cleared ({count} removed).", count=3),
            "Cola de prompts limpiada (3 eliminados).",
        )
        self.assertEqual(
            _("Removed #{pos} from prompt queue: {text}", pos=2, text="foo"),
            "Eliminado #2 de la cola de prompts: foo",
        )
        self.assertEqual(
            _("Invalid queue position '{pos}'. Usage: /queue rm <position>", pos="xyz"),
            "Posición de cola no válida 'xyz'. Uso: /queue rm <posición>",
        )
        self.assertEqual(
            _("Queue position {pos} out of range (queue has {count} items).", pos=5, count=2),
            "Posición de cola 5 fuera de rango (la cola tiene 2 elementos).",
        )
        self.assertEqual(
            _("Queued ({count})", count=4),
            "En cola (4)",
        )

    def test_unknown_string_in_english_is_source_language(self) -> None:
        set_locale("en")
        unknown = "This is a non-registered string"
        self.assertEqual(_(unknown), unknown)

    def test_missing_translation_in_non_default_locale_returns_original(self) -> None:
        set_locale("es")
        unknown = "This is a non-registered string"
        self.assertEqual(_(unknown), unknown)


class TestCatalogCompleteness(unittest.TestCase):
    """Verify all non-English locale JSON catalogs are complete and consistent."""

    def test_all_locale_catalogs_parity(self) -> None:
        locales_dir = Path(__file__).parent.parent / "ollama_agent" / "i18n" / "locales"
        non_en_locales = [loc for loc in SUPPORTED_LOCALES if loc != "en"]
        self.assertFalse(
            (locales_dir / "en.json").exists(), "en.json should not exist as English is the default in-code language"
        )

        ref_file = locales_dir / f"{non_en_locales[0]}.json"
        self.assertTrue(ref_file.exists(), f"Reference file {ref_file} must exist")

        with open(ref_file, "r", encoding="utf-8") as f:
            ref_data: dict[str, str] = json.load(f)

        placeholder_pattern = re.compile(r"\{([a-zA-Z0-9_]+)\}")

        for loc in non_en_locales:
            loc_file = locales_dir / f"{loc}.json"
            self.assertTrue(loc_file.exists(), f"{loc}.json must exist")

            with open(loc_file, "r", encoding="utf-8") as f:
                loc_data: dict[str, str] = json.load(f)

            # Check key parity with reference catalog (keys are original English strings)
            missing_keys = set(ref_data.keys()) - set(loc_data.keys())
            extra_keys = set(loc_data.keys()) - set(ref_data.keys())
            self.assertEqual(
                missing_keys,
                set(),
                f"{loc}.json is missing keys: {missing_keys}",
            )
            self.assertEqual(
                extra_keys,
                set(),
                f"{loc}.json has unexpected extra keys: {extra_keys}",
            )

            # Check placeholder consistency against original English key
            for key, loc_val in loc_data.items():
                key_placeholders = set(placeholder_pattern.findall(key))
                loc_placeholders = set(placeholder_pattern.findall(loc_val))
                self.assertEqual(
                    key_placeholders,
                    loc_placeholders,
                    f"Placeholder mismatch in {loc}.json for key '{key}': {key_placeholders} vs {loc_placeholders}",
                )

    def test_all_codebase_strings_present_in_catalogs(self) -> None:
        pkg_dir = Path(__file__).parent.parent / "ollama_agent"
        locales_dir = pkg_dir / "i18n" / "locales"

        extracted_strings: set[str] = set()
        for py_file in pkg_dir.rglob("*.py"):
            with open(py_file, "r", encoding="utf-8") as f:
                tree = ast.parse(f.read(), filename=str(py_file))
            for node in ast.walk(tree):
                if isinstance(node, ast.Call):
                    func = node.func
                    if (isinstance(func, ast.Name) and func.id == "_") or (
                        isinstance(func, ast.Attribute) and func.attr == "_"
                    ):
                        if node.args and isinstance(node.args[0], ast.Constant) and isinstance(node.args[0].value, str):
                            extracted_strings.add(node.args[0].value)

        for loc in [loc_code for loc_code in SUPPORTED_LOCALES if loc_code != "en"]:
            loc_file = locales_dir / f"{loc}.json"
            with open(loc_file, "r", encoding="utf-8") as f:
                catalog = json.load(f)
            missing = [s for s in extracted_strings if s not in catalog]
            self.assertEqual(missing, [], f"{loc}.json is missing translation for in-code strings: {missing}")
            dead = [s for s in catalog if s not in extracted_strings]
            self.assertEqual(dead, [], f"{loc}.json contains dead/unused translation keys: {dead}")


class TestCLIAndSettingsI18n(unittest.TestCase):
    """Test CLI language argument and runtime settings integration."""

    def tearDown(self) -> None:
        set_locale("en")

    def test_cli_parser_language_arg(self) -> None:
        parser = create_argument_parser()
        args = parser.parse_args(["--lang", "es"])
        self.assertEqual(args.language, "es")

        args2 = parser.parse_args(["-l", "hi"])
        self.assertEqual(args2.language, "hi")

        args3 = parser.parse_args(["-l", "zh"])
        self.assertEqual(args3.language, "zh")

    def test_settings_language_field(self) -> None:
        s = Settings()
        self.assertEqual(s.runtime.language, "")
        s.runtime.language = "hi"
        d = s.to_dict()
        self.assertEqual(d["runtime"]["language"], "hi")

        s2 = Settings.from_dict({"runtime": {"language": "zh"}})
        self.assertEqual(s2.runtime.language, "zh")

    def test_reset_config_localization(self) -> None:
        set_locale("es")
        msgs = reset_config("config-file", settings_path=Path("/tmp/dummy_settings.yaml"))
        self.assertTrue(any("Reinicio:" in m or "restauró" in m for m in msgs), f"Unexpected messages: {msgs}")


if __name__ == "__main__":
    unittest.main()
