"""Local-only HTTP Application/API layer for the Adinn 4K Image Enhancer.

Binds to 127.0.0.1 only. This is meant to run inside the desktop app's own
process (a pywebview window points at it) — never exposed on the network,
never talking to an external AI API.

Endpoints (all JSON except the result-download route):

    POST /api/jobs                 create a job from one or more image files
    GET  /api/jobs/<job_id>         job status: status/progress/current_file/
                                    completed_count/total_count/errors
    GET  /api/jobs/<job_id>/result  the enhanced image (1 input), or a ZIP
                                    of every successfully enhanced image
                                    (>1 input)
    GET  /health                   liveness probe

Optionally also serves a built static frontend (e.g. frontend/dist) from
this same origin when `make_server(static_dir=...)` is given one — see
`_serve_static`. That's what lets the desktop shell (backend/shell.py) load
the page and the API from one server with no CORS involved; it changes
nothing about the routes above, and callers that don't pass `static_dir`
(every existing test, and `main()` by default) get byte-identical behavior
to before this was added.

No PPT/campaign/site/billboard concepts here — see engine_adapter.py for how
this layer keeps the engine's own billboard-regions config from ever firing
on a generic uploaded photo.
"""

from __future__ import annotations

import json
import mimetypes
import os
import re
import sys
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

from .jobs import STATUS_COMPLETED, JobManager
from .multipart import MultipartError, parse_multipart_files

PORT = int(os.environ.get("ADINN_BACKEND_PORT", "8787"))
MAX_BODY_BYTES = int(os.environ.get("ADINN_MAX_BODY_BYTES", str(200 * 1024 * 1024)))
MAX_FILE_BYTES = int(os.environ.get("ADINN_MAX_FILE_BYTES", str(40 * 1024 * 1024)))
MAX_FILES_PER_JOB = int(os.environ.get("ADINN_MAX_FILES_PER_JOB", "50"))

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png"}
ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png"}


class ApiError(Exception):
    """A 4xx error whose message is safe to return to the client."""

    def __init__(self, message: str, status: int = 400):
        super().__init__(message)
        self.status = status


def _validate_upload(filename: str, content_type: str, data: bytes) -> None:
    label = filename or "file"
    if not data:
        raise ApiError(f"{label}: empty file")
    if len(data) > MAX_FILE_BYTES:
        raise ApiError(f"{label}: file exceeds the {MAX_FILE_BYTES} byte limit")
    ext = Path(filename or "").suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise ApiError(
            f"{label}: unsupported file type {ext!r} (allowed: .jpg, .jpeg, .png)")
    declared = (content_type or "").split(";")[0].strip().lower()
    if declared and declared not in ALLOWED_CONTENT_TYPES:
        # Client-declared Content-Type isn't trusted on its own — the real
        # content check happens when the engine decodes the image — but an
        # outright mismatch (e.g. text/plain) is rejected before a job is
        # even created.
        raise ApiError(f"{label}: unsupported content type {content_type!r}")


class ApiHandler(BaseHTTPRequestHandler):
    server_version = "AdinnImageEnhancer/0.1"

    # ------------------------------------------------------------------ utils
    def _cors(self) -> None:
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Accept")

    def _send_json(self, status: int, obj: dict) -> None:
        body = json.dumps(obj).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self._cors()
        self.end_headers()
        self.wfile.write(body)

    def _send_error_json(self, status: int, message: str) -> None:
        self._send_json(status, {"ok": False, "error": message})

    def _write_file_response(self, path: Path, mime: str, disposition: Optional[str] = None) -> None:
        size = path.stat().st_size
        self.send_response(200)
        self.send_header("Content-Type", mime)
        self.send_header("Content-Length", str(size))
        if disposition:
            self.send_header("Content-Disposition", disposition)
        self._cors()
        self.end_headers()
        with path.open("rb") as fh:
            self.wfile.write(fh.read())

    def _send_file(self, path: Path, mime: str, download_name: str) -> None:
        if not path.is_file():
            self._send_error_json(404, "result file not found on the server")
            return
        self._write_file_response(path, mime, disposition=f'attachment; filename="{download_name}"')

    def _job_manager(self) -> JobManager:
        return self.server.job_manager  # type: ignore[attr-defined]

    def _serve_static(self, path: str) -> bool:
        """Serves a file from this server's static_dir (if one was configured
        via make_server), falling back to index.html for any path that isn't
        a real file so client-side routing works. Returns False (serving
        nothing) when no static_dir is configured or the path resolves
        outside of it — callers fall through to the normal 404 in that case.
        """
        static_root = getattr(self.server, "static_root", None)
        if not static_root:
            return False
        static_root = Path(static_root).resolve()

        rel = path.lstrip("/") or "index.html"
        candidate = (static_root / rel).resolve()
        try:
            candidate.relative_to(static_root)
        except ValueError:
            return False  # e.g. "/../../secret" — outside the static root

        if not candidate.is_file():
            candidate = static_root / "index.html"
            if not candidate.is_file():
                return False

        mime = mimetypes.guess_type(str(candidate))[0] or "application/octet-stream"
        self._write_file_response(candidate, mime)
        return True

    # ------------------------------------------------------------------ routes
    def do_OPTIONS(self) -> None:
        self.send_response(204)
        self._cors()
        self.end_headers()

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path == "/health":
            self._send_json(200, {"ok": True})
            return

        m = re.fullmatch(r"/api/jobs/([^/]+)/result", path)
        if m:
            self._handle_get_result(m.group(1))
            return

        m = re.fullmatch(r"/api/jobs/([^/]+)", path)
        if m:
            self._handle_get_status(m.group(1))
            return

        if not path.startswith("/api/") and self._serve_static(path):
            return

        self._send_error_json(404, "not found")

    def _handle_get_status(self, job_id: str) -> None:
        job = self._job_manager().get_job(job_id)
        if job is None:
            self._send_error_json(404, "job not found")
            return
        self._send_json(200, {"ok": True, **job.to_status_dict()})

    def _handle_get_result(self, job_id: str) -> None:
        job = self._job_manager().get_job(job_id)
        if job is None:
            self._send_error_json(404, "job not found")
            return
        status_dict = job.to_status_dict()
        if status_dict["status"] != STATUS_COMPLETED:
            self._send_error_json(
                409, f"job is not completed (status: {status_dict['status']})")
            return
        if job.result_path is None or not Path(job.result_path).is_file():
            self._send_error_json(500, "job completed but no result file was found")
            return
        result_path = Path(job.result_path)
        mime = "application/zip" if result_path.suffix == ".zip" else "image/png"
        self._send_file(result_path, mime, job.result_filename or result_path.name)

    def do_POST(self) -> None:
        if urlparse(self.path).path != "/api/jobs":
            self._send_error_json(404, "not found")
            return
        try:
            job = self._handle_create_job()
        except ApiError as exc:
            self._send_error_json(exc.status, str(exc))
        except Exception as exc:  # noqa: BLE001
            traceback.print_exc(file=sys.stderr)
            self._send_error_json(500, f"unexpected server error: {exc}")
        else:
            self._send_json(200, {
                "ok": True,
                "job_id": job.id,
                "status": job.status,
                "total_count": job.total_count,
            })

    def _handle_create_job(self):
        content_type = self.headers.get("Content-Type", "")
        if "multipart/form-data" not in content_type:
            raise ApiError("expected multipart/form-data with one or more image files")

        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            raise ApiError("invalid Content-Length") from None
        if length <= 0:
            raise ApiError("empty request body")
        if length > MAX_BODY_BYTES:
            # Drain the socket so the connection stays usable for the error response.
            try:
                self.rfile.read(length)
            except Exception:  # noqa: BLE001
                pass
            raise ApiError(f"request body exceeds the {MAX_BODY_BYTES} byte limit")

        body = self.rfile.read(length)
        try:
            uploaded = parse_multipart_files(body, content_type)
        except MultipartError as exc:
            raise ApiError(str(exc)) from exc

        if not uploaded:
            raise ApiError("at least one image file is required")
        if len(uploaded) > MAX_FILES_PER_JOB:
            raise ApiError(f"too many files (max {MAX_FILES_PER_JOB} per job)")
        for f in uploaded:
            _validate_upload(f.filename, f.content_type, f.data)

        return self._job_manager().create_job([(f.filename, f.data) for f in uploaded])

    def log_message(self, fmt: str, *args) -> None:  # noqa: A002
        sys.stderr.write("[adinn-backend] %s - %s\n" % (self.address_string(), fmt % args))


def make_server(
    port: Optional[int] = None,
    job_manager: Optional[JobManager] = None,
    static_dir: Optional[Path] = None,
) -> ThreadingHTTPServer:
    # port=0 must reach the socket layer as 0 (OS-assigned ephemeral port,
    # used by tests) rather than being folded into the PORT default by a
    # truthiness check.
    bind_port = PORT if port is None else port
    httpd = ThreadingHTTPServer(("127.0.0.1", bind_port), ApiHandler)
    httpd.job_manager = job_manager or JobManager()  # type: ignore[attr-defined]
    httpd.static_root = static_dir  # type: ignore[attr-defined]  # None => unchanged prior behavior
    return httpd


def main(argv=None) -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Adinn 4K Image Enhancer — local API")
    parser.add_argument("--port", type=int, default=None)
    args = parser.parse_args(argv)
    httpd = make_server(args.port)
    bound_port = httpd.server_address[1]
    print(f"Adinn Image Enhancer backend listening on http://127.0.0.1:{bound_port}")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nshutting down")
    finally:
        httpd.server_close()


if __name__ == "__main__":
    main()
