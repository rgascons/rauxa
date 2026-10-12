import unittest
from pathlib import Path

from rauxa.pipeline import load_config

CONFIGS = Path(__file__).resolve().parent.parent / "configs"


class TestConfigs(unittest.TestCase):
    def test_every_config_loads(self):
        paths = sorted(CONFIGS.glob("*.toml"))
        self.assertTrue(paths)
        for path in paths:
            with self.subTest(config=path.name):
                load_config(path)


if __name__ == "__main__":
    unittest.main()
