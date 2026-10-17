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
            path.write_text(BASE + extra, encoding="utf-8")
            run, copilot = load_settings(path)
            self.assertEqual(load_config(path), run)
            return run, copilot

    def test_fingerprint_excludes_copilot(self):
        run, missing = self.load()
        self.assertFalse(resolve_copilot(missing).enabled)
        for extra in (
            "[copilot]",
            "[copilot]\nworkers=2\nmax_calls=10",
            "[copilot]\nenabled=true\nworkers=2\nmax_calls=10",
            "[copilot]\nenabled=false\nworkers=8\nmax_calls=200",
        ):
            modified, _ = self.load(extra)
            self.assertEqual(run.fingerprint(), modified.fingerprint())
        modified.target_size += 1
        self.assertNotEqual(run.fingerprint(), modified.fingerprint())

    def test_section_presence_and_explicit_enabled(self):
        for extra, enabled, origin in (
            ("", False, "default"),
            ("[copilot]\nmax_calls=5", True, "toml:copilot"),
            ("[copilot]\nworkers=2\nmax_calls=10", True, "toml:copilot"),
            ("[copilot]\nenabled=true\nmax_calls=5", True, "toml:copilot.enabled"),
            ("[copilot]\nenabled=false", False, "toml:copilot.enabled"),
        ):
            with self.subTest(extra=extra):
                _, config = self.load(extra)
                self.assertEqual(config.enabled, enabled)
                self.assertEqual(config.section_present, bool(extra))
                settings = resolve_copilot(config, environ={})
                self.assertEqual(settings.enabled, enabled)
                self.assertEqual(settings.origins["enabled"], origin)

    def test_present_section_still_requires_explicit_cap(self):
        for extra in ("[copilot]", "[copilot]\nworkers=2"):
            with self.subTest(extra=extra):
                _, config = self.load(extra)
                self.assertTrue(config.enabled)
                with self.assertRaisesRegex(ValueError, "requires"):
                    resolve_copilot(config)
                _, capped = self.load(extra + "\nmax_calls=5")
                self.assertTrue(resolve_copilot(capped).enabled)
        _, disabled = self.load("[copilot]\nenabled=false")
        self.assertFalse(resolve_copilot(disabled).enabled)

    def test_toml_settings_and_default_origins(self):
        _, config = self.load("[copilot]\nenabled=true\nworkers=2\nmax_calls=10")
        settings = resolve_copilot(config, environ={})
        self.assertEqual((settings.enabled, settings.workers, settings.max_calls), (True, 2, 10))
        self.assertEqual(settings.origins["workers"], "toml:copilot.workers")
        self.assertEqual(settings.origins["max_calls"], "toml:copilot.max_calls")
        self.assertEqual(settings.origins["enabled"], "toml:copilot.enabled")
        _, config = self.load("[copilot]\nmax_calls=5")
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
            with self.subTest(extra=extra), self.assertRaises(ValueError):
                self.load("[copilot]\n" + extra)
        with self.assertRaisesRegex(ValueError, "requires"):
            resolve_copilot(CopilotConfig(enabled=True))
        self.assertFalse(resolve_copilot(CopilotConfig(enabled=False)).enabled)


if __name__ == "__main__":
    unittest.main()
