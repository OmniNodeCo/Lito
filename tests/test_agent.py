"""Core agent / reasoner / tools tests — offline, no network required for most."""

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


class ReasonerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        os.environ["LITO_DATA"] = self.tmp.name
        os.environ["LITO_SHOW_THOUGHTS"] = "1"
        os.environ["LITO_FORCE_LOCAL"] = "1"
        os.environ.pop("LITO_LLM_URL", None)
        from lito.agent import Agent

        self.agent = Agent()

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_help(self) -> None:
        r = self.agent.handle("help")
        self.assertIn("Tools", r.text)
        self.assertIn("thinking", r.text.lower())

    def test_calc_path(self) -> None:
        r = self.agent.handle("calculate 2^8")
        self.assertIn("256", r.text)

    def test_remember_recall(self) -> None:
        self.agent.handle("remember project is lito")
        r = self.agent.handle("recall project")
        self.assertIn("lito", r.text.lower())

    def test_greet(self) -> None:
        r = self.agent.handle("hello")
        self.assertIn("Lito", r.text)

    def test_web_search_mocked(self) -> None:
        with mock.patch(
            "lito.tools.tool_web_search",
            return_value="A walrus is a large marine mammal.\n\nsource: https://example.com",
        ):
            # patch via call_tool path — reasoner imports call_tool
            with mock.patch("lito.reasoner.call_tool", side_effect=self._fake_call):
                r = self.agent.handle("what is a walrus")
        self.assertIn("walrus", r.text.lower())
        self.assertIn("thinking", r.text.lower())

    def _fake_call(self, tools, name, args=None):
        if name == "search":
            return "A walrus is a large flippered marine mammal.\n\nsource: https://ex.com/w"
        from lito.tools import call_tool as real

        return real(tools, name, args)

    def test_thought_trace_present(self) -> None:
        r = self.agent.handle("time")
        self.assertIn("thinking", r.text.lower())


class ShellSafetyTests(unittest.TestCase):
    def test_blocks_rm_rf(self) -> None:
        from lito.tools import tool_shell

        out = tool_shell("rm -rf /")
        self.assertIn("blocked", out.lower())


if __name__ == "__main__":
    unittest.main()
