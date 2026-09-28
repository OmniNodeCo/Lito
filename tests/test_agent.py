"""Core agent tests — generative nano path."""

from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


class CalcTests(unittest.TestCase):
    def test_basic(self) -> None:
        from lito.tools import tool_calc

        self.assertEqual(tool_calc("2+3*4"), "14")
        self.assertEqual(tool_calc("2^10"), "1024")

    def test_rejects_names(self) -> None:
        from lito.tools import tool_calc

        self.assertIn("error", tool_calc("__import__('os')").lower())


class MemoryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        os.environ["LITO_DATA"] = self.tmp.name

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_kv(self) -> None:
        from lito import memory as mem

        mem.remember("wifi", "orchard")
        self.assertEqual(mem.recall("wifi"), "orchard")


class AgentGenTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        os.environ["LITO_DATA"] = self.tmp.name
        os.environ["LITO_SHOW_THOUGHTS"] = "1"
        os.environ.pop("LITO_FORCE_LOCAL", None)
        from lito.agent import Agent

        self.agent = Agent()

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_generates_hello(self) -> None:
        r = self.agent.handle("hello")
        self.assertTrue(len(r.text) > 5)
        # should not be the old offline stub
        self.assertNotIn("generative core is offline", r.text.lower())

    def test_calc(self) -> None:
        r = self.agent.handle("calculate 6*7")
        self.assertIn("42", r.text)

    def test_remember_recall(self) -> None:
        self.agent.handle("remember project is lito")
        r = self.agent.handle("recall project")
        self.assertIn("lito", r.text.lower())

    def test_varies_hello(self) -> None:
        # generation / mutation should not crash on repeats
        texts = {self.agent.handle("hello").text for _ in range(3)}
        self.assertTrue(all("offline" not in t.lower() for t in texts))

    def test_thought_optional(self) -> None:
        os.environ["LITO_SHOW_THOUGHTS"] = "1"
        from lito.agent import Agent

        r = Agent().handle("time")
        self.assertTrue(r.ok)


class ShellSafetyTests(unittest.TestCase):
    def test_blocks_rm_rf(self) -> None:
        from lito.tools import tool_shell

        out = tool_shell("rm -rf /")
        self.assertIn("blocked", out.lower())


if __name__ == "__main__":
    unittest.main()
