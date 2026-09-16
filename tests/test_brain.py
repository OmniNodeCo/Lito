"""Fast unit tests - no network, no GUI."""

from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path

# Ensure package importable when run as script
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


class BrainTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._tmpdir = tempfile.TemporaryDirectory()
        os.environ["LITO_DATA"] = cls._tmpdir.name
        # Import after env so paths point at temp
        from lito.brain import Brain

        cls.Brain = Brain

    @classmethod
    def tearDownClass(cls) -> None:
        cls._tmpdir.cleanup()

    def setUp(self) -> None:
        self.brain = self.Brain()

    def test_help(self) -> None:
        r = self.brain.handle("help")
        self.assertTrue(r.ok)
        self.assertIn("Lito", r.text)

    def test_calc(self) -> None:
        r = self.brain.handle("calc 2 + 3 * 4")
        self.assertTrue(r.ok)
        self.assertIn("14", r.text)

    def test_calc_bare(self) -> None:
        r = self.brain.handle("10 / 2")
        self.assertTrue(r.ok)
        self.assertIn("5", r.text)

    def test_note_and_show(self) -> None:
        r = self.brain.handle("note buy oat milk")
        self.assertTrue(r.ok)
        self.assertIn("oat milk", r.text)
        r2 = self.brain.handle("show notes")
        self.assertIn("oat milk", r2.text)

    def test_remember_recall(self) -> None:
        r = self.brain.handle("remember wifi is blueorchid")
        self.assertTrue(r.ok)
        r2 = self.brain.handle("what is wifi")
        self.assertIn("blueorchid", r2.text)

    def test_time(self) -> None:
        r = self.brain.handle("time")
        self.assertTrue(r.ok)
        # Avoid curly/smart apostrophe mismatches across locales
        self.assertTrue("It" in r.text and ":" in r.text, r.text)

    def test_ram(self) -> None:
        r = self.brain.handle("how much ram")
        self.assertTrue(r.ok)
        self.assertIn("RAM", r.text)

    def test_sysinfo(self) -> None:
        r = self.brain.handle("system info")
        self.assertTrue(r.ok)
        self.assertIn("OS", r.text)

    def test_greet(self) -> None:
        r = self.brain.handle("hello")
        self.assertTrue(r.ok)
        self.assertIn("Lito", r.text)

    def test_shell_echo(self) -> None:
        import platform

        from lito.actions import run_shell

        # Keep the command trivial so cmd.exe / sh quoting never bites CI
        if platform.system() == "Windows":
            cmd = "cmd /c echo hello-lito"
            via_brain = "run cmd /c echo hello-lito"
        else:
            cmd = "echo hello-lito"
            via_brain = "run echo hello-lito"

        ok, msg = run_shell(cmd)
        self.assertTrue(ok, msg)
        self.assertIn("hello-lito", msg.replace("\r", ""))

        r = self.brain.handle(via_brain)
        self.assertTrue(r.ok, r.text)
        self.assertIn("hello-lito", r.text.replace("\r", ""))

    def test_dangerous_shell_blocked(self) -> None:
        r = self.brain.handle("run rm -rf /")
        self.assertFalse(r.ok)
        self.assertIn("Blocked", r.text)

    def test_status(self) -> None:
        r = self.brain.handle("status")
        self.assertTrue(r.ok)
        self.assertIn("Lito", r.text)

    def test_list_apps(self) -> None:
        r = self.brain.handle("list apps")
        self.assertTrue(r.ok)

    def test_unknown(self) -> None:
        r = self.brain.handle("blorptastic quantum waffle")
        self.assertFalse(r.ok)
        self.assertIn("help", r.text.lower())


class CalcSafetyTests(unittest.TestCase):
    def test_rejects_names(self) -> None:
        from lito.actions import safe_calc

        ok, msg = safe_calc("__import__('os').system('id')")
        self.assertFalse(ok)

    def test_pow(self) -> None:
        from lito.actions import safe_calc

        ok, msg = safe_calc("2 ** 10")
        self.assertTrue(ok)
        self.assertIn("1024", msg)


class MemoryTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()
        os.environ["LITO_DATA"] = self._tmpdir.name
        # reload paths by re-importing is heavy; functions read env each time via config paths
        import importlib
        import lito.config as cfg
        import lito.memory as mem

        importlib.reload(cfg)
        importlib.reload(mem)
        self.mem = mem

    def tearDown(self) -> None:
        self._tmpdir.cleanup()

    def test_kv(self) -> None:
        self.mem.remember("city", "Lisbon")
        self.assertEqual(self.mem.recall("city"), "Lisbon")
        self.assertTrue(self.mem.forget("city"))
        self.assertIsNone(self.mem.recall("city"))


if __name__ == "__main__":
    unittest.main()
