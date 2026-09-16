"""Tests for last-used tracking, uninstall planning, and smart search."""

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


class UsageTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        os.environ["LITO_DATA"] = self.tmp.name
        os.environ["LITO_NO_DESKTOP_SCAN"] = "1"
        from lito import usage

        self.usage = usage

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_record_and_format(self) -> None:
        self.usage.record_launch("Firefox", "firefox")
        self.usage.record_launch("Firefox", "firefox")
        u = self.usage.get_usage("firefox")
        self.assertIsNotNone(u)
        assert u is not None
        self.assertEqual(u["launches"], 2)
        self.assertTrue(u["last_used"] > 0)
        ago = self.usage.format_ago(u["last_used"])
        self.assertTrue("ago" in ago or ago == "just now")

    def test_enrich(self) -> None:
        self.usage.record_launch("Code", "code")
        info = self.usage.enrich_last_used("Code", "code")
        self.assertEqual(info["launches"], 1)
        self.assertNotEqual(info["last_used_ago"], "never")


class UninstallTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        os.environ["LITO_DATA"] = self.tmp.name
        os.environ["LITO_NO_DESKTOP_SCAN"] = "1"
        from lito.apps import AppEntry, AppRegistry
        from lito import uninstall

        self.uninstall = uninstall
        self.reg = AppRegistry()
        self.reg._apps = [
            AppEntry("DemoFlat", "flatpak run org.demo.App", description="org.demo.App", source="flatpak"),
            AppEntry("DemoSnap", "snap run demosnap", source="snap"),
            AppEntry("Mystery", "mystery-bin", source="desktop"),
        ]
        self.reg._loaded = True

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_plan_flatpak(self) -> None:
        with mock.patch.object(self.uninstall, "registry", self.reg):
            with mock.patch("lito.uninstall.shutil.which", side_effect=lambda x: "/usr/bin/flatpak" if x == "flatpak" else None):
                plan, err = self.uninstall.plan_uninstall("DemoFlat")
        self.assertEqual(err, "")
        assert plan is not None
        self.assertEqual(plan.method, "flatpak")
        self.assertIn("org.demo.App", plan.command)

    def test_requires_confirm(self) -> None:
        with mock.patch.object(self.uninstall, "registry", self.reg):
            with mock.patch("lito.uninstall.shutil.which", side_effect=lambda x: "/usr/bin/flatpak" if x == "flatpak" else None):
                ok, msg = self.uninstall.uninstall_app("DemoFlat", confirm=False)
        self.assertTrue(ok)
        self.assertIn("confirm uninstall", msg.lower())


class WebSearchTests(unittest.TestCase):
    def test_smart_search_offline_graceful(self) -> None:
        from lito import websearch

        with mock.patch.object(websearch, "duckduckgo_instant", return_value=None):
            with mock.patch.object(websearch, "wikipedia_summary", return_value=None):
                with mock.patch.object(websearch, "duckduckgo_html_results", return_value=[]):
                    ok, msg = websearch.smart_search("zzzz-nonexistent-xyz")
        self.assertIn("Web search", msg)

    def test_smart_search_with_instant(self) -> None:
        from lito import websearch
        from lito.websearch import SearchHit

        instant = {
            "AbstractText": "A walrus is a large flippered marine mammal.",
            "Heading": "Walrus",
            "AbstractURL": "https://example.com/walrus",
            "Answer": "",
            "RelatedTopics": [],
        }
        hits = [SearchHit("Walrus - Wiki", "https://example.com/w", "big mammal")]
        with mock.patch.object(websearch, "duckduckgo_instant", return_value=instant):
            with mock.patch.object(websearch, "duckduckgo_html_results", return_value=hits):
                ok, msg = websearch.smart_search("walrus")
        self.assertTrue(ok)
        self.assertIn("Walrus", msg)
        self.assertIn("flippered", msg)
        self.assertIn("Top results", msg)


class BrainFeatureTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        os.environ["LITO_DATA"] = self.tmp.name
        os.environ["LITO_NO_DESKTOP_SCAN"] = "1"
        from lito.brain import Brain

        self.brain = Brain()

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_list_by_recent_intent(self) -> None:
        r = self.brain.handle("list apps by recent")
        self.assertTrue(r.ok)
        self.assertIn("app", r.text.lower())

    def test_uninstall_intent_plans(self) -> None:
        with mock.patch("lito.actions.uninstall_app_text", return_value=(True, "Ready to uninstall demo")):
            r = self.brain.handle("uninstall demoapp")
        self.assertIn("Ready to uninstall", r.text)

    def test_web_search_intent(self) -> None:
        with mock.patch("lito.actions.smart_web_search", return_value=(True, "Web search: cats\nmeow")):
            r = self.brain.handle("search web cats")
        self.assertTrue(r.ok)
        self.assertIn("cats", r.text.lower())

    def test_last_used_intent(self) -> None:
        from lito import usage

        usage.record_launch("DemoApp", "demo")
        with mock.patch("lito.actions.app_last_used_text", return_value="**DemoApp**\n- last used: just now"):
            r = self.brain.handle("when was DemoApp last used")
        self.assertIn("DemoApp", r.text)


if __name__ == "__main__":
    unittest.main()
