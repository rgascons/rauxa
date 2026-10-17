import os
import tempfile
import unittest
from contextlib import redirect_stderr
from io import StringIO
from pathlib import Path
from unittest.mock import patch

from rauxa.cli import build_parser
from rauxa.pipeline.config import CopilotConfig, load_config, load_settings, resolve_copilot


BASE = '''
[run]
name = "example"
target_size = 10
[[sources]]
type = "procedural"
task = "castellanismes_verbals"
'''


class TestCopilotConfig(unittest.TestCase):
    def load(self, extra=""):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.toml"
            path.write_text(extra + BASE, encoding="utf-8")
            run, copilot = load_settings(path)
            self.assertEqual(load_config(path), run)
            return run, copilot

    def test_fingerprint_excludes_copilot(self):
        run, missing = self.load()
        self.assertFalse(resolve_copilot(missing).enabled)
        for extra in (
            "[runners]",
            "[runners.copilot]",
            "[runners.copilot]\nworkers=2\nmax_calls=10",
            "[runners.copilot]\nenabled=true\nworkers=2\nmax_calls=10",
            "[runners.copilot]\nenabled=false\nworkers=8\nmax_calls=200",
        ):
            modified, _ = self.load(extra)
            self.assertEqual(run.fingerprint(), modified.fingerprint())
        modified.target_size += 1
        self.assertNotEqual(run.fingerprint(), modified.fingerprint())

    def test_section_presence_and_explicit_enabled(self):
        for extra, present, enabled, origin in (
            ("", False, False, "default"),
            ("[runners]", False, False, "default"),
            ("[runners.copilot]", True, False, "default"),
            ("[runners.copilot]\nmax_calls=5", True, False, "default"),
            ("[runners.copilot]\nworkers=2\nmax_calls=10", True, False, "default"),
            ("[runners.copilot]\nenabled=true\nmax_calls=5", True, True, "toml:runners.copilot.enabled"),
            ("[runners.copilot]\nenabled=false", True, False, "toml:runners.copilot.enabled"),
        ):
            with self.subTest(extra=extra):
                _, config = self.load(extra)
                self.assertEqual(config.enabled, enabled)
                self.assertEqual(config.section_present, present)
                settings = resolve_copilot(config, environ={})
                self.assertEqual(settings.enabled, enabled)
                self.assertEqual(settings.origins["enabled"], origin)

    def test_only_enabled_runner_requires_explicit_cap(self):
        for extra in ("[runners.copilot]", "[runners.copilot]\nworkers=2"):
            with self.subTest(extra=extra):
                _, config = self.load(extra)
                self.assertFalse(resolve_copilot(config).enabled)
                _, config = self.load(extra + "\nenabled=true")
                self.assertTrue(config.enabled)
                with self.assertRaisesRegex(ValueError, r"requires runners\.copilot\.max_calls"):
                    resolve_copilot(config)
                _, capped = self.load(extra + "\nenabled=true\nmax_calls=5")
                self.assertTrue(resolve_copilot(capped).enabled)
        _, disabled = self.load("[runners.copilot]\nenabled=false")
        self.assertFalse(resolve_copilot(disabled).enabled)

    def test_toml_settings_and_default_origins(self):
        _, config = self.load("[runners.copilot]\nenabled=true\nworkers=2\nmax_calls=10")
        settings = resolve_copilot(config, environ={})
        self.assertEqual((settings.enabled, settings.workers, settings.max_calls), (True, 2, 10))
        self.assertEqual(settings.origins["workers"], "toml:runners.copilot.workers")
        self.assertEqual(settings.origins["max_calls"], "toml:runners.copilot.max_calls")
        self.assertEqual(settings.origins["enabled"], "toml:runners.copilot.enabled")
        _, config = self.load("[runners.copilot]\nmax_calls=5")
        default = resolve_copilot(config, environ={})
        self.assertEqual(default.workers, 4)
        self.assertEqual(default.origins["workers"], "default")

    def test_executable_precedence_and_no_fallback(self):
        config = CopilotConfig()
        env = {"RAUXA_COPILOT_BIN": "environment-cli"}
        settings = resolve_copilot(config, environ=env)
        self.assertEqual(settings.command, "environment-cli")
        self.assertEqual(settings.origins["command"], "env:RAUXA_COPILOT_BIN")
        with patch("shutil.which", return_value=None):
            with self.assertRaisesRegex(ValueError, "environment-cli"):
                settings.executable()
        settings = resolve_copilot(config, environ={})
        self.assertEqual((settings.command, settings.origins["command"]), ("copilot", "path"))
        with patch("shutil.which", return_value="relative copilot"):
            self.assertEqual(settings.executable(), os.path.abspath("relative copilot"))
        with self.assertRaisesRegex(ValueError, "empty Copilot"):
            resolve_copilot(config, environ={"RAUXA_COPILOT_BIN": ""}).executable()

    def test_copilot_cli_overrides_are_rejected(self):
        parser = build_parser()
        for flags in (["--copilot"], ["--no-copilot"], ["--copilot-workers", "2"],
                      ["--copilot-max-calls", "5"], ["--copilot-command", "copilot"]):
            with self.subTest(flags=flags), redirect_stderr(StringIO()), self.assertRaises(SystemExit) as error:
                parser.parse_args(["run", "example.toml", *flags])
            self.assertEqual(error.exception.code, 2)
        self.assertTrue(parser.parse_args(["run", "example.toml", "--resume"]).resume)

    def test_validation(self):
        for extra in ('workers=0', 'workers=true', 'max_calls=-1', 'enabled="yes"', 'executable="copilot"'):
            with self.subTest(extra=extra), self.assertRaisesRegex(ValueError, r"runners\.copilot"):
                self.load("[runners.copilot]\n" + extra)
        with self.assertRaisesRegex(ValueError, "requires"):
            resolve_copilot(CopilotConfig(enabled=True))
        self.assertFalse(resolve_copilot(CopilotConfig(enabled=False)).enabled)

    def test_runner_structure_and_unknown_names_are_rejected(self):
        for extra, error in (
            ('runners = "copilot"', r"\[runners\] must be a TOML table"),
            ("[[runners]]", r"\[runners\] must be a TOML table"),
            ("[runners]\ncopilot = true", r"\[runners\.copilot\] must be a TOML table"),
            ("[[runners.copilot]]", r"\[runners\.copilot\] must be a TOML table"),
            ("[runners.copliot]", r"unknown \[runners\] keys"),
            ("[runners]\nenabled = true", r"unknown \[runners\] keys"),
        ):
            with self.subTest(extra=extra), self.assertRaisesRegex(ValueError, error):
                self.load(extra)

    def test_legacy_section_requires_explicit_migration(self):
        for extra in (
            "[copilot]",
            "[copilot]\nworkers=2\nmax_calls=10",
            "[copilot]\nenabled=true\nmax_calls=10",
            "[copilot]\nenabled=false",
            "[copilot]\n[runners.copilot]\nenabled=true\nmax_calls=10",
        ):
            with self.subTest(extra=extra), self.assertRaisesRegex(
                ValueError, r"\[copilot\] has moved to \[runners\.copilot\]; set enabled = true explicitly",
            ):
                self.load(extra)


if __name__ == "__main__":
    unittest.main()
