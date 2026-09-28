"""Lito-Nano generative stack tests."""

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


class GenMemTests(unittest.TestCase):
    def test_generate_coherent(self) -> None:
        from lito.nano.markov import MarkovGen
        from lito.nano.runtime import default_weights_dir

        path = default_weights_dir() / "lito-markov.json"
        if not path.exists():
            raise unittest.SkipTest("markov missing")
        m = MarkovGen.load(path)
        hello = m.generate("hello")
        self.assertTrue(len(hello) > 8)
        self.assertNotIn("equals", hello.lower())
        mqtt = m.generate("what is mqtt")
        self.assertIn("mqtt", mqtt.lower())
        ocean = m.generate("tell me about the ocean")
        self.assertIn("ocean", ocean.lower())


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

    def test_memory(self) -> None:
        self.agent.handle("remember project is lito")
        r = self.agent.handle("recall project")
        self.assertIn("lito", r.text.lower())

    def test_generates_not_offline(self) -> None:
        r = self.agent.handle("help")
        self.assertNotIn("offline", r.text.lower())
        self.assertTrue(len(r.text) > 20)


if __name__ == "__main__":
    unittest.main()
