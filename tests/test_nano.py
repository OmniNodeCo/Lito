"""Lito-Nano custom micro-LLM tests."""

from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


class IntentNetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        from lito.nano.intent import IntentNet
        from lito.nano.runtime import default_weights_dir

        path = default_weights_dir() / "lito-intent.bin"
        if not path.exists():
            raise unittest.SkipTest("intent weights not trained")
        cls.net = IntentNet.load(path)

    def test_routes(self) -> None:
        cases = {
            "hello": "chat",
            "help": "help",
            "calculate 3+4": "calc",
            "what is mqtt": "search",
            "remember wifi is x": "remember",
            "recall wifi": "recall",
            "time": "time",
        }
        for text, label in cases.items():
            pred, probs = self.net.predict(text)
            self.assertEqual(pred, label, msg=f"{text} -> {pred}")
            self.assertGreater(max(probs), 0.5)


class NanoAgentTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        os.environ["LITO_DATA"] = self.tmp.name
        os.environ["LITO_SHOW_THOUGHTS"] = "1"
        os.environ.pop("LITO_FORCE_LOCAL", None)
        from lito.agent import Agent

        self.agent = Agent()

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_nano_loaded(self) -> None:
        self.assertIsNotNone(self.agent.nano_reasoner)

    def test_calc(self) -> None:
        r = self.agent.handle("calculate 6*7")
        self.assertIn("42", r.text)
        self.assertIn("nano", r.text.lower())

    def test_memory(self) -> None:
        self.agent.handle("remember project is lito")
        r = self.agent.handle("recall project")
        self.assertIn("lito", r.text.lower())

    def test_help_brands_nano(self) -> None:
        r = self.agent.handle("help")
        self.assertIn("Lito-Nano", r.text)


if __name__ == "__main__":
    unittest.main()
