"""App discovery tests."""

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


class DesktopParseTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def _write_desktop(self, name: str, body: str) -> Path:
        p = self.root / f"{name}.desktop"
        p.write_text(body, encoding="utf-8")
        return p

    def test_parse_basic(self) -> None:
        from lito.apps import _parse_desktop_file

        p = self._write_desktop(
            "foo",
            "[Desktop Entry]\n"
            "Type=Application\n"
            "Name=Foo Editor\n"
            "Exec=foo %F\n"
            "Comment=A foo app\n",
        )
        with mock.patch("lito.apps.shutil.which", return_value="/usr/bin/foo"):
            app = _parse_desktop_file(p)
        self.assertIsNotNone(app)
        assert app is not None
        self.assertEqual(app.name, "Foo Editor")
        self.assertEqual(app.command, "foo")
        self.assertIn("foo editor", app.aliases)

    def test_skip_nodisplay(self) -> None:
        from lito.apps import _parse_desktop_file

        p = self._write_desktop(
            "hidden",
            "[Desktop Entry]\nType=Application\nName=Hidden\nExec=true\nNoDisplay=true\n",
        )
        self.assertIsNone(_parse_desktop_file(p))

    def test_discover_from_dir(self) -> None:
        from lito import apps as apps_mod

        self._write_desktop(
            "alpha",
            "[Desktop Entry]\nType=Application\nName=Alpha App\nExec=true\n",
        )
        self._write_desktop(
            "beta",
            "[Desktop Entry]\nType=Application\nName=Beta App\nExec=true\n",
        )
        with mock.patch.object(apps_mod, "_desktop_search_dirs", return_value=[self.root]):
            with mock.patch.object(apps_mod.shutil, "which", return_value="/bin/true"):
                found = apps_mod.discover_desktop_apps()
        names = {a.name for a in found}
        self.assertIn("Alpha App", names)
        self.assertIn("Beta App", names)

    def test_list_apps_shows_all(self) -> None:
        os.environ["LITO_NO_DESKTOP_SCAN"] = "1"
        os.environ["LITO_DATA"] = self.tmp.name
        from lito.apps import AppEntry, AppRegistry
        from lito import actions

        reg = AppRegistry()
        reg._apps = [
            AppEntry("Zebra", "zebra", source="desktop"),
            AppEntry("Apple", "apple", source="desktop"),
            AppEntry("Mango", "mango", source="flatpak"),
        ]
        reg._loaded = True
        with mock.patch.object(actions, "registry", reg):
            text = actions.list_apps_text("")
        self.assertIn("Zebra", text)
        self.assertIn("Apple", text)
        self.assertIn("Mango", text)
        self.assertIn("All installed apps", text)
        # No silent 40-cap: all three present
        self.assertIn("3", text)

    def test_list_apps_filter(self) -> None:
        os.environ["LITO_NO_DESKTOP_SCAN"] = "1"
        from lito.apps import AppEntry, AppRegistry
        from lito import actions

        reg = AppRegistry()
        reg._apps = [
            AppEntry("Firefox Web", "firefox", aliases=("firefox",), source="desktop"),
            AppEntry("Chrome", "chrome", source="desktop"),
        ]
        reg._loaded = True
        with mock.patch.object(actions, "registry", reg):
            text = actions.list_apps_text("fire")
        self.assertIn("Firefox", text)
        self.assertNotIn("Chrome", text)


if __name__ == "__main__":
    unittest.main()
