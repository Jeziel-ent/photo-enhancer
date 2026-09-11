"""Tests for the optional static-file serving added to server.py for the
pywebview shell. Confirms it's purely additive: /api/* and /health are
unaffected, and static serving only activates when static_dir is given.
"""

import json
import tempfile
import threading
import unittest
import urllib.error
import urllib.request
from pathlib import Path

from backend.jobs import JobManager
from backend.server import make_server


def _get(base_url, path):
    req = urllib.request.Request(base_url + path, method="GET")
    try:
        with urllib.request.urlopen(req) as resp:
            return resp.status, resp.read(), dict(resp.headers)
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read(), dict(exc.headers)


class StaticServingTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        static_dir = Path(cls._tmp.name)
        (static_dir / "index.html").write_text("<html>root</html>", encoding="utf-8")
        (static_dir / "app.js").write_text("console.log('hi');", encoding="utf-8")
        assets = static_dir / "assets"
        assets.mkdir()
        (assets / "style.css").write_text("body{}", encoding="utf-8")

        cls.manager = JobManager(process_fn=lambda i, o: None)
        cls.httpd = make_server(port=0, job_manager=cls.manager, static_dir=static_dir)
        cls.base_url = f"http://127.0.0.1:{cls.httpd.server_address[1]}"
        cls.thread = threading.Thread(target=cls.httpd.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.httpd.shutdown()
        cls.httpd.server_close()
        cls.thread.join(timeout=5)
        cls._tmp.cleanup()

    def test_health_still_works_alongside_static_dir(self):
        status, body, _ = _get(self.base_url, "/health")
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(body), {"ok": True})

    def test_root_serves_index_html(self):
        status, body, headers = _get(self.base_url, "/")
        self.assertEqual(status, 200)
        self.assertEqual(body, b"<html>root</html>")
        self.assertEqual(headers["Content-Type"], "text/html")
        self.assertNotIn("Content-Disposition", headers)

    def test_nested_asset_served_with_correct_mime(self):
        status, body, headers = _get(self.base_url, "/assets/style.css")
        self.assertEqual(status, 200)
        self.assertEqual(body, b"body{}")
        self.assertEqual(headers["Content-Type"], "text/css")

    def test_js_file_served(self):
        status, body, headers = _get(self.base_url, "/app.js")
        self.assertEqual(status, 200)
        self.assertEqual(body, b"console.log('hi');")

    def test_unknown_client_route_falls_back_to_index_html(self):
        # e.g. a client-side router path like /some/react/route
        status, body, _ = _get(self.base_url, "/some/react/route")
        self.assertEqual(status, 200)
        self.assertEqual(body, b"<html>root</html>")

    def test_api_routes_never_fall_back_to_static(self):
        status, body, _ = _get(self.base_url, "/api/does-not-exist")
        self.assertEqual(status, 404)
        self.assertFalse(json.loads(body)["ok"])

    def test_path_traversal_is_rejected(self):
        status, _body, _headers = _get(self.base_url, "/../../../etc/passwd")
        # urllib normalizes ".." itself for the request line in some cases;
        # assert it never returns a 200 with unexpected content either way.
        self.assertIn(status, (200, 404))
        if status == 200:
            _status2, body2, _h = _get(self.base_url, "/../../../etc/passwd")
            self.assertEqual(body2, b"<html>root</html>")  # fell back to index, not escaped


class NoStaticDirTestCase(unittest.TestCase):
    """Without static_dir, behavior must be byte-identical to before this
    feature existed: no static fallback, plain 404 on unknown paths."""

    @classmethod
    def setUpClass(cls):
        cls.manager = JobManager(process_fn=lambda i, o: None)
        cls.httpd = make_server(port=0, job_manager=cls.manager)  # no static_dir
        cls.base_url = f"http://127.0.0.1:{cls.httpd.server_address[1]}"
        cls.thread = threading.Thread(target=cls.httpd.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.httpd.shutdown()
        cls.httpd.server_close()
        cls.thread.join(timeout=5)

    def test_root_is_404_without_static_dir(self):
        status, body, _ = _get(self.base_url, "/")
        self.assertEqual(status, 404)
        self.assertFalse(json.loads(body)["ok"])

    def test_health_unaffected(self):
        status, body, _ = _get(self.base_url, "/health")
        self.assertEqual(status, 200)


if __name__ == "__main__":
    unittest.main()
