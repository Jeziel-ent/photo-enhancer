"""Unit tests for backend/_frozen.py and the frozen-aware workspace root in
backend/workspace.py -- the two path-resolution changes made for
installed-path compatibility (see packaging/README.md)."""

import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from backend import _frozen, workspace


class AppRootTestCase(unittest.TestCase):
    def test_dev_mode_resolves_to_repo_root(self):
        # Not frozen in the test process -- matches source-tree behavior.
        self.assertFalse(getattr(sys, "frozen", False))
        root = _frozen.app_root()
        self.assertTrue((root / "backend" / "_frozen.py").is_file())
        self.assertTrue((root / "image_enhancer" / "src" / "enhance.py").is_file())

    def test_frozen_mode_resolves_to_meipass(self):
        with mock.patch.object(sys, "frozen", True, create=True), \
                mock.patch.object(sys, "_MEIPASS", r"C:\fake\bundle\root", create=True):
            self.assertEqual(_frozen.app_root(), Path(r"C:\fake\bundle\root"))


class WorkspaceRootTestCase(unittest.TestCase):
    def test_dev_default_is_unchanged(self):
        with mock.patch.dict("os.environ", {}, clear=False):
            import os
            os.environ.pop("ADINN_INSTALLED", None)
            root = workspace._default_workspace_root()
        self.assertEqual(root, Path(workspace.__file__).resolve().parent / ".workspace")

    def test_installed_redirects_to_localappdata(self):
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.dict(
                "os.environ", {"ADINN_INSTALLED": "1", "LOCALAPPDATA": tmp}, clear=False,
            ):
                root = workspace._default_workspace_root()
            self.assertEqual(root, Path(tmp) / "Adinn4KImageEnhancer" / "workspace")

    def test_installed_without_localappdata_falls_back(self):
        import os
        with mock.patch.dict("os.environ", {"ADINN_INSTALLED": "1"}, clear=False):
            os.environ.pop("LOCALAPPDATA", None)
            root = workspace._default_workspace_root()
        self.assertEqual(root, Path(workspace.__file__).resolve().parent / ".workspace")


if __name__ == "__main__":
    unittest.main()
