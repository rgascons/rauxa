import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

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
            "[copilot]\nenabled=true\nworkers=2\nmax_calls=10",
            "[copilot]\nenabled=false\nworkers=8\nmax_calls=200",
        ):
            modified, _ = self.load(extra)
            self.assertEqual(run.fingerprint(), modified.fingerprint())
        modified.target_size += 1
        self.assertNotEqual(run.fingerprint(), modified.fingerprint())

    def test_precedence_and_origins(self):
        _, config = self.load("[copilot]\nenabled=true\nworkers=2\nmax_calls=10")
        settings = resolve_copilot(config, environ={})
        self.assertEqual((settings.enabled, settings.workers, settings.max_calls), (True, 2, 10))
        self.assertEqual(settings.origins["workers"], "toml:copilot.workers")
        override = resolve_copilot(config, workers=8, max_calls=20, enabled=False, environ={})
        self.assertEqual((override.enabled, override.workers, override.max_calls), (False, 8, 20))
        self.assertEqual(override.origins["enabled"], "cli:--no-copilot")
        default = resolve_copilot(CopilotConfig(), enabled=True, max_calls=5, environ={})
        self.assertEqual(default.workers, 4)
        self.assertEqual(default.origins["workers"], "default")

    def test_executable_precedence_and_no_fallback(self):
        config = CopilotConfig()
        env = {"RAUXA_COPILOT_BIN": "environment-cli"}
        self.assertEqual(resolve_copilot(config, command="chosen-cli", environ=env).command, "chosen-cli")
        settings = resolve_copilot(config, environ=env)
        self.assertEqual(settings.origins["command"], "env:RAUXA_COPILOT_BIN")
        with patch("shutil.which", return_value=None):
            with self.assertRaisesRegex(ValueError, "environment-cli"):
                settings.executable()
        settings = resolve_copilot(config, environ={})
        self.assertEqual((settings.command, settings.origins["command"]), ("copilot", "path"))
        with patch("shutil.which", return_value="relative copilot"):
            self.assertEqual(settings.executable(), os.path.abspath("relative copilot"))
        with self.assertRaisesRegex(ValueError, "empty Copilot"):
            resolve_copilot(config, command="", environ=env).executable()

    def test_validation(self):
        for extra in ('workers=0', 'workers=true', 'max_calls=-1', 'enabled="yes"', 'executable="copilot"'):
            with self.subTest(extra=extra), self.assertRaises(ValueError):
                self.load("[copilot]\n" + extra)
        with self.assertRaisesRegex(ValueError, "requires"):
            resolve_copilot(CopilotConfig(enabled=True))
        self.assertFalse(resolve_copilot(CopilotConfig(enabled=True), enabled=False).enabled)


if __name__ == "__main__":
    unittest.main()
