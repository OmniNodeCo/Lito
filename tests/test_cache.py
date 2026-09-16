"""Cache scanner / cleaner tests - uses a fake home tree, never touches real caches."""

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


class CacheCleanerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.home = Path(self.tmp.name)
        os.environ["LITO_DATA"] = str(self.home / "lito-data")
        # Build a mini cache tree under fake HOME
        self.ff = self.home / ".cache" / "mozilla" / "firefox"
        self.ff.mkdir(parents=True)
        (self.ff / "big.bin").write_bytes(b"x" * 4096)
        self.pip = self.home / ".cache" / "pip"
        self.pip.mkdir(parents=True)
        (self.pip / "wheel.dat").write_bytes(b"y" * 2048)
        self.chrome = self.home / ".cache" / "google-chrome"
        self.chrome.mkdir(parents=True)
        (self.chrome / "f").write_bytes(b"z" * 1024)
        # Active-looking path for a fake running app
        self.code_cache = self.home / ".config" / "Code" / "Cache"
        self.code_cache.mkdir(parents=True)
        (self.code_cache / "c").write_bytes(b"c" * 512)

        # Patch Path.home used by cache module
        import lito.cache as cache

        self.cache = cache
        self._home_patch = mock.patch.object(cache, "_home", return_value=self.home)
        self._home_patch.start()

    def tearDown(self) -> None:
        self._home_patch.stop()
        self.tmp.cleanup()

    def test_scan_finds_owners(self) -> None:
        with mock.patch.object(self.cache, "_running_processes", return_value=set()):
            with mock.patch.object(self.cache, "_owner_installed", return_value=True):
                entries = self.cache.scan_caches(include_system=False, max_secs=5.0)
        paths = {str(e.path) for e in entries}
        self.assertTrue(any("mozilla" in p for p in paths), paths)
        self.assertTrue(any("pip" in p for p in paths), paths)
        owners = {e.owner_id for e in entries}
        self.assertIn("firefox", owners)
        self.assertIn("pip", owners)

    def test_running_app_marked_active(self) -> None:
        with mock.patch.object(
            self.cache, "_running_processes", return_value={"code", "Xorg"}
        ):
            with mock.patch.object(self.cache, "_owner_installed", return_value=True):
                entries = self.cache.scan_caches(max_secs=5.0)
        code_entries = [e for e in entries if e.owner_id == "code"]
        self.assertTrue(code_entries)
        self.assertTrue(all(e.status == "active" for e in code_entries))

    def test_clear_unused_skips_active(self) -> None:
        with mock.patch.object(
            self.cache, "_running_processes", return_value={"code"}
        ):
            with mock.patch.object(self.cache, "_owner_installed", return_value=True):
                result = self.cache.clear_caches(unused_only=True, dry_run=False)
        # code cache should remain
        self.assertTrue(self.code_cache.exists())
        # unused ones should be gone
        self.assertFalse(self.ff.exists())
        self.assertFalse(self.pip.exists())
        self.assertGreater(result.cleared, 0)
        self.assertGreater(result.freed_bytes, 0)

    def test_dry_run_deletes_nothing(self) -> None:
        with mock.patch.object(self.cache, "_running_processes", return_value=set()):
            with mock.patch.object(self.cache, "_owner_installed", return_value=True):
                result = self.cache.clear_caches(unused_only=True, dry_run=True)
        self.assertTrue(self.ff.exists())
        self.assertTrue(self.pip.exists())
        self.assertGreater(result.cleared, 0)  # would-clear count
        self.assertGreater(result.freed_bytes, 0)

    def test_clear_owner_filter(self) -> None:
        with mock.patch.object(self.cache, "_running_processes", return_value=set()):
            with mock.patch.object(self.cache, "_owner_installed", return_value=True):
                result = self.cache.clear_caches(owner="pip", unused_only=True)
        self.assertFalse(self.pip.exists())
        self.assertTrue(self.ff.exists())  # other apps untouched
        self.assertGreaterEqual(result.cleared, 1)

    def test_protected_home_not_deleted(self) -> None:
        ok, msg, _ = self.cache._safe_delete(self.home)
        self.assertFalse(ok)

    def test_orphaned_when_not_installed(self) -> None:
        with mock.patch.object(self.cache, "_running_processes", return_value=set()):
            with mock.patch.object(self.cache, "_owner_installed", return_value=False):
                entries = self.cache.scan_caches(max_secs=5.0)
        # entries with process tokens should be orphaned
        ff = [e for e in entries if e.owner_id == "firefox"]
        self.assertTrue(ff)
        self.assertTrue(all(e.status == "orphaned" for e in ff))

    def test_format_scan(self) -> None:
        with mock.patch.object(self.cache, "_running_processes", return_value=set()):
            with mock.patch.object(self.cache, "_owner_installed", return_value=True):
                entries = self.cache.scan_caches(max_secs=5.0)
        text = self.cache.format_scan(entries)
        self.assertIn("Cache scan", text)
        self.assertIn("clear unused caches", text.lower())


class BrainCacheIntentTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        os.environ["LITO_DATA"] = self.tmp.name
        os.environ["LITO_NO_DESKTOP_SCAN"] = "1"
        from lito.brain import Brain

        self.brain = Brain()

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_scan_intent(self) -> None:
        r = self.brain.handle("scan caches")
        self.assertTrue(r.ok)
        self.assertIn("Cache scan", r.text)

    def test_voice_free_up(self) -> None:
        r = self.brain.handle("free up cache space dry run")
        self.assertIn("Dry run", r.text)
        self.assertIn("Scanned", r.text)

    def test_clear_for_owner_dry(self) -> None:
        r = self.brain.handle("clear cache for pip dry run")
        self.assertTrue(r.ok)
        self.assertIn("Dry run", r.text)

    def test_help_mentions_cache(self) -> None:
        r = self.brain.handle("help")
        self.assertIn("scan caches", r.text.lower())




class LastUsedCacheTests(unittest.TestCase):
    """Caches for recently-used apps stay; idle ones clear."""

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.home = Path(self.tmp.name)
        os.environ["LITO_DATA"] = str(self.home / "lito-data")
        self.ff = self.home / ".cache" / "mozilla" / "firefox"
        self.ff.mkdir(parents=True)
        (self.ff / "big.bin").write_bytes(b"x" * 4096)
        self.pip = self.home / ".cache" / "pip"
        self.pip.mkdir(parents=True)
        (self.pip / "wheel.dat").write_bytes(b"y" * 2048)
        self.chrome = self.home / ".cache" / "google-chrome"
        self.chrome.mkdir(parents=True)
        (self.chrome / "f").write_bytes(b"z" * 1024)

        import lito.cache as cache
        import lito.usage as usage

        self.cache = cache
        self.usage = usage
        self._home_patch = mock.patch.object(cache, "_home", return_value=self.home)
        self._home_patch.start()

    def tearDown(self) -> None:
        self._home_patch.stop()
        self.tmp.cleanup()

    def test_recent_marked_when_used_today(self) -> None:
        self.usage.record_launch("Firefox", "firefox")
        with mock.patch.object(self.cache, "_running_processes", return_value=set()):
            with mock.patch.object(self.cache, "_owner_installed", return_value=True):
                entries = self.cache.scan_caches(idle_days=7, max_secs=5.0)
        ff = [e for e in entries if e.owner_id == "firefox"]
        self.assertTrue(ff)
        self.assertTrue(all(e.status == "recent" for e in ff), [e.status for e in ff])
        self.assertTrue(ff[0].last_used)

    def test_recent_usage_skips_clear(self) -> None:
        self.usage.record_launch("Firefox", "firefox")
        with mock.patch.object(self.cache, "_running_processes", return_value=set()):
            with mock.patch.object(self.cache, "_owner_installed", return_value=True):
                result = self.cache.clear_caches(unused_only=True, dry_run=False, idle_days=7)
        # firefox kept (recent), pip/chrome cleared (no last-used -> unused)
        self.assertTrue(self.ff.exists(), "recent firefox cache should remain")
        self.assertFalse(self.pip.exists())
        self.assertGreater(result.cleared, 0)
        skip_txt = "\n".join(result.lines).lower()
        self.assertTrue(
            "within 7d" in skip_txt or "[~]" in skip_txt or "used just now" in skip_txt,
            skip_txt,
        )

    def test_idle_old_usage_clears(self) -> None:
        # Record a launch then backdate it to 40 days ago
        self.usage.record_launch("Firefox", "firefox")
        data = self.usage._load()
        import time as _t
        data["firefox"]["last_used"] = _t.time() - 40 * 86400
        self.usage._save(data)
        with mock.patch.object(self.cache, "_running_processes", return_value=set()):
            with mock.patch.object(self.cache, "_owner_installed", return_value=True):
                result = self.cache.clear_caches(unused_only=True, dry_run=False, idle_days=7)
        self.assertFalse(self.ff.exists())
        self.assertGreater(result.cleared, 0)

    def test_older_than_window(self) -> None:
        self.usage.record_launch("Firefox", "firefox")
        # Used today: with 1-day window still recent; with 0-day window eligible
        with mock.patch.object(self.cache, "_running_processes", return_value=set()):
            with mock.patch.object(self.cache, "_owner_installed", return_value=True):
                r0 = self.cache.clear_caches(unused_only=True, dry_run=True, idle_days=0)
                entries = self.cache.scan_caches(idle_days=0, max_secs=5.0)
        ff = [e for e in entries if e.owner_id == "firefox"]
        # idle_days=0 means anything with last_used has idle_val >= 0 and
        # idle_val < 0 is false, so if idle_val is 0.0, 0 < 0 is false -> unused
        self.assertTrue(ff)
        self.assertTrue(all(e.status in {"unused", "recent"} for e in ff))

    def test_parse_idle_days(self) -> None:
        self.assertEqual(self.cache._parse_idle_days("14d"), 14.0)
        self.assertEqual(self.cache._parse_idle_days("48h"), 2.0)
        self.assertEqual(self.cache._parse_idle_days("2w"), 14.0)
        self.assertEqual(self.cache._parse_idle_days(30), 30.0)

    def test_brain_older_than_intent(self) -> None:
        from lito.brain import Brain

        b = Brain()
        with mock.patch("lito.actions.cache_clear", return_value=(True, "cleared idle")) as m:
            r = b.handle("clear caches older than 30 days dry run")
        self.assertTrue(r.ok)
        self.assertIn("cleared", r.text.lower())
        kwargs = m.call_args.kwargs
        self.assertEqual(kwargs.get("idle_days"), 30.0)
        self.assertTrue(kwargs.get("dry_run"))

    def test_format_shows_last_used(self) -> None:
        self.usage.record_launch("pip", "pip")
        with mock.patch.object(self.cache, "_running_processes", return_value=set()):
            with mock.patch.object(self.cache, "_owner_installed", return_value=True):
                entries = self.cache.scan_caches(max_secs=5.0)
        text = self.cache.format_scan(entries)
        self.assertIn("last used", text.lower())
        self.assertIn("keep", text.lower())


if __name__ == "__main__":
    unittest.main()
