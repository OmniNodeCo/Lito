"""Update checker unit tests - no network required."""

from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


class VersionCompareTests(unittest.TestCase):
    def test_is_newer(self) -> None:
        from lito.update import is_newer

        self.assertTrue(is_newer("0.2.0", "0.1.0"))
        self.assertTrue(is_newer("v1.0.0", "0.9.9"))
        self.assertFalse(is_newer("0.1.0", "0.1.0"))
        self.assertFalse(is_newer("0.1.0", "0.2.0"))

    def test_platform_tag(self) -> None:
        from lito.update import current_platform_tag

        tag = current_platform_tag()
        self.assertTrue(
            tag.startswith(("linux-", "macos-", "windows-")),
            tag,
        )


class CheckUpdateTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        os.environ["LITO_DATA"] = self.tmp.name
        import importlib
        import lito.config as cfg
        import lito.update as upd

        importlib.reload(cfg)
        importlib.reload(upd)
        self.upd = upd

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_from_latest_json_newer(self) -> None:
        payload = {
            "version": "9.9.9",
            "tag_name": "v9.9.9",
            "html_url": "https://example.com/r",
            "notes": "big release",
            "assets": {
                self.upd.current_platform_tag(): {
                    "name": "lito-bin",
                    "url": "https://example.com/lito-bin",
                    "sha256": "abc",
                }
            },
        }

        with mock.patch.object(self.upd, "_http_json", return_value=payload):
            info = self.upd.check_for_update(current="0.1.0")
        self.assertIsNotNone(info)
        assert info is not None
        self.assertEqual(info.version, "9.9.9")
        self.assertIn("lito-bin", info.asset_name or "")

    def test_up_to_date(self) -> None:
        payload = {"version": "0.0.1", "tag_name": "v0.0.1", "assets": {}}
        with mock.patch.object(self.upd, "_http_json", return_value=payload):
            info = self.upd.check_for_update(current="0.1.0")
        self.assertIsNone(info)

    def test_format_status(self) -> None:
        text = self.upd.format_update_status(None)
        self.assertIn("up to date", text.lower())

    def test_pick_asset(self) -> None:
        assets = [
            {"name": "latest.json", "browser_download_url": "x"},
            {
                "name": f"lito-1.0.0-{self.upd.current_platform_tag()}.zip",
                "browser_download_url": "https://example.com/a.zip",
            },
            {
                "name": "lito-1.0.0-other-arch",
                "browser_download_url": "https://example.com/b",
            },
        ]
        picked = self.upd._pick_asset(assets)
        self.assertIsNotNone(picked)
        assert picked is not None
        self.assertIn("lito-1.0.0", picked["name"])


class BrainUpdateIntentTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        os.environ["LITO_DATA"] = self.tmp.name
        os.environ["LITO_NO_DESKTOP_SCAN"] = "1"
        os.environ["LITO_AUTO_UPDATE"] = "0"
        from lito.brain import Brain

        self.brain = Brain()

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_check_update_intent(self) -> None:
        with mock.patch("lito.actions.check_update", return_value=(True, "up to date mock")):
            r = self.brain.handle("check update")
        self.assertTrue(r.ok)
        self.assertIn("up to date mock", r.text)

    def test_help_mentions_update(self) -> None:
        r = self.brain.handle("help")
        self.assertIn("check update", r.text.lower())


if __name__ == "__main__":
    unittest.main()
