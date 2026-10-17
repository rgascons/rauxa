import unittest
from pathlib import Path

from rauxa.pipeline import load_config
from rauxa.pipeline.config import load_settings, resolve_copilot

CONFIGS = Path(__file__).resolve().parent.parent / "configs"


class TestConfigs(unittest.TestCase):
    def test_every_config_loads(self):
        paths = sorted(CONFIGS.glob("*.toml"))
        self.assertTrue(paths)
        for path in paths:
            with self.subTest(config=path.name):
                load_config(path)

    def test_copilot_examples_explicitly_enable_runner(self):
        for name in ("castellanismes_runners.toml", "castellanismes_runners_10k.toml"):
            with self.subTest(config=name):
                _, copilot = load_settings(CONFIGS / name)
                settings = resolve_copilot(copilot, environ={})
                self.assertTrue(settings.enabled)
                self.assertEqual(settings.origins["enabled"], "toml:runners.copilot.enabled")


if __name__ == "__main__":
    unittest.main()
