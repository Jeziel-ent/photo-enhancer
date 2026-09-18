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
                                    (>1 input). Two optional query params
                                    (single-image results only; the ZIP path
                                    always stays as-is):
                                      ?billboard=<url-encoded JSON array>
                                        composites confirmed editor rects
                                        onto a copy of the PNG result
                                      ?adjust=<url-encoded JSON object>
                                        applies manual post-processing
                                        adjustments (brightness/contrast/
                                        highlights/shadows/saturation/detail)
                                        BEFORE any billboard rects are drawn
                                        (see adjustment_overlay.py — the same
                                        endpoint powers both the live preview
                                        fetch and the final download, so
                                        there is exactly one code path)
    GET  /api/jobs/<job_id>/results/<result_id>
                                    one image's own clean enhanced PNG from a
                                    multi-file (gallery) job — never the ZIP,
                                    never another file's image. Powers the
                                    gallery's main preview and thumbnails.
                                    Also accepts an optional ?adjust= query,
                                    same semantics as above.
    POST /api/jobs/<job_id>/export  batch export: one ZIP built fresh from a
                                    JSON body {format, include_outlines,
                                    adjust, images: [{result_id, rects}]} —
                                    one COMMON format (and, when given, one
                                    common `adjust`) for every image, each
                                    image getting only its own rects burned
                                    in (see output_manager.build_batch_export)
    GET  /api/settings              current processing-device preference +
                                    what's actually detected/active
    PUT  /api/settings              persist a new processing-device
                                    preference ("auto"/"gpu"/"cpu")
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
from urllib.parse import parse_qs, urlparse

from . import engine_adapter, settings
from .device import VALID_PREFERENCES
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
        self.send_header("Access-Control-Allow-Methods", "GET, POST, PUT, OPTIONS")
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

    def _send_bytes_response(
        self,
        status: int,
        data: bytes,
        mime: str,
        disposition: Optional[str] = None,
    ) -> None:
        self.send_response(status)
        self.send_header("Content-Type", mime)
        self.send_header("Content-Length", str(len(data)))
        if disposition:
            self.send_header("Content-Disposition", disposition)
        self._cors()
        self.end_headers()
        self.wfile.write(data)

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

        if path == "/api/settings":
            self._handle_get_settings()
            return

        m = re.fullmatch(r"/api/jobs/([^/]+)/result", path)
        if m:
            self._handle_get_result(m.group(1), parse_qs(urlparse(self.path).query))
            return

        m = re.fullmatch(r"/api/jobs/([^/]+)/results/([^/]+)", path)
        if m:
            self._handle_get_individual_result(
                m.group(1), m.group(2), parse_qs(urlparse(self.path).query))
            return

        m = re.fullmatch(r"/api/jobs/([^/]+)", path)
        if m:
            self._handle_get_status(m.group(1))
            return

        if not path.startswith("/api/") and self._serve_static(path):
            return

        self._send_error_json(404, "not found")

    def _handle_get_settings(self) -> None:
        preference = settings.get_processing_device_preference()
        state = engine_adapter.get_device_state()
        self._send_json(200, {
            "ok": True,
            "processing_device": preference,
            # What's actually running in this process right now -- may
            # briefly be None just after launch, before the JobManager
            # worker thread's one-time device resolution has run (see
            # jobs.py's JobManager._run) -- and may legitimately differ
            # from `processing_device` if the user changed the setting
            # after this process already resolved a device (see
            # `restart_required` on PUT below).
            "effective_device": state["effective_device"],
            "detected_gpu": state["detected_gpu"],
            "warning": state["warning"],
            "restart_required": (
                state["effective_device"] is not None
                and state["preference"] != preference
            ),
        })

    def _handle_get_status(self, job_id: str) -> None:
        job = self._job_manager().get_job(job_id)
        if job is None:
            self._send_error_json(404, "job not found")
            return
        self._send_json(200, {"ok": True, **job.to_status_dict()})

    def _handle_get_result(self, job_id: str, query: Optional[dict] = None) -> None:
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
        if result_path.suffix == ".zip":
            self._send_file(result_path, "application/zip", job.result_filename or result_path.name)
            return

        billboard_rects = self._parse_billboard_query(query or {})
        adjust = self._parse_adjust_query(query or {})
        if billboard_rects or adjust:
            # Browser-download / preview path: composite adjustments + rects
            # backend-side so the image never crosses the frontend twice.
            # Only a PNG is ever produced here (single-image results are
            # PNG); the ZIP route returned above. This is the SAME function
            # the live preview fetch and the final download both call — see
            # adjustment_overlay.py's own module docstring.
            try:
                from .adjustment_overlay import compose_bytes

                png_data = compose_bytes(result_path, billboard_rects or [], adjust, "png")
            except Exception as exc:  # noqa: BLE001 -- any decode/encode failure surfaces as a 500
                self._send_error_json(500, f"could not render the adjusted image: {exc}")
                return
            self._send_bytes_response(
                200, png_data, "image/png",
                disposition=f'attachment; filename="{job.result_filename or result_path.name}"')
            return

        self._send_file(result_path, "image/png", job.result_filename or result_path.name)

    def _handle_get_individual_result(
        self, job_id: str, result_id: str, query: Optional[dict] = None
    ) -> None:
        job = self._job_manager().get_job(job_id)
        if job is None:
            self._send_error_json(404, "job not found")
            return
        status_dict = job.to_status_dict()
        if status_dict["status"] != STATUS_COMPLETED:
            self._send_error_json(
                409, f"job is not completed (status: {status_dict['status']})")
            return
        file_result = job.get_file_result(result_id)
        if file_result is None or not Path(file_result.output_path).is_file():
            self._send_error_json(404, "result not found for that image")
            return

        adjust = self._parse_adjust_query(query or {})
        if adjust:
            try:
                from .adjustment_overlay import compose_bytes

                png_data = compose_bytes(Path(file_result.output_path), [], adjust, "png")
            except Exception as exc:  # noqa: BLE001
                self._send_error_json(500, f"could not render the adjusted image: {exc}")
                return
            self._send_bytes_response(200, png_data, "image/png")
            return

        self._write_file_response(Path(file_result.output_path), "image/png")

    def _parse_billboard_query(self, query: dict) -> Optional[list]:
        """Extracts an optional ``?billboard=<url-encoded JSON array>`` query
        param. Returns None (no overlay) for a missing/malformed payload and
        a raw list for well-formed ones; actual clamping happens downstream in
        billboard_overlay.clamp_billboard_rects."""
        raw = (query or {}).get("billboard")
        if not raw or not raw[0]:
            return None
        try:
            value = json.loads(raw[0])
        except (ValueError, json.JSONDecodeError):
            return None
        return value if isinstance(value, list) else None

    def _parse_adjust_query(self, query: dict) -> Optional[dict]:
        """Extracts an optional ``?adjust=<url-encoded JSON object>`` query
        param (brightness/contrast/highlights/shadows/saturation/detail).
        Returns None for a missing/malformed/default-valued payload so the
        caller's fast, no-op file-serving path is unaffected; actual
        clamping happens downstream in adjustments.normalize_adjustments."""
        raw = (query or {}).get("adjust")
        if not raw or not raw[0]:
            return None
        try:
            value = json.loads(raw[0])
        except (ValueError, json.JSONDecodeError):
            return None
        if not isinstance(value, dict):
            return None
        from .adjustment_overlay import _resolve_image_enhancer_src

        _resolve_image_enhancer_src()
        import adjustments as adj  # image_enhancer/src/adjustments.py

        return None if adj.is_default(value) else value

    def do_POST(self) -> None:
        path = urlparse(self.path).path
        if path == "/api/jobs":
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
            return

        m = re.fullmatch(r"/api/jobs/([^/]+)/export", path)
        if m:
            try:
                self._handle_export(m.group(1))
            except ApiError as exc:
                self._send_error_json(exc.status, str(exc))
            except Exception as exc:  # noqa: BLE001
                traceback.print_exc(file=sys.stderr)
                self._send_error_json(500, f"unexpected server error: {exc}")
            return

        self._send_error_json(404, "not found")

    def _read_json_body(self) -> dict:
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            raise ApiError("invalid Content-Length") from None
        body = self.rfile.read(length) if length > 0 else b"{}"
        try:
            payload = json.loads(body or b"{}")
        except (ValueError, json.JSONDecodeError):
            raise ApiError("invalid JSON body") from None
        if not isinstance(payload, dict):
            raise ApiError("invalid JSON body")
        return payload

    def _handle_export(self, job_id: str) -> None:
        job = self._job_manager().get_job(job_id)
        if job is None:
            raise ApiError("job not found", 404)
        status_dict = job.to_status_dict()
        if status_dict["status"] != STATUS_COMPLETED:
            raise ApiError(
                f"job is not completed (status: {status_dict['status']})", 409)

        payload = self._read_json_body()
        image_format = payload.get("format")
        if image_format not in ("png", "jpg", "jpeg"):
            raise ApiError("format must be one of 'png', 'jpg', 'jpeg'")
        include_outlines = bool(payload.get("include_outlines"))
        adjust = payload.get("adjust")
        adjust = adjust if isinstance(adjust, dict) else None
        images = payload.get("images")
        if not isinstance(images, list) or not images:
            raise ApiError("images must be a non-empty list")

        items = []
        for entry in images:
            if not isinstance(entry, dict):
                raise ApiError("each image entry must be an object")
            result_id = entry.get("result_id")
            if not isinstance(result_id, str):
                raise ApiError("each image entry needs a result_id")
            file_result = job.get_file_result(result_id)
            if file_result is None:
                raise ApiError(f"unknown result_id: {result_id!r}", 404)
            rects = entry.get("rects") or []
            items.append((file_result.original_filename, file_result.output_path, rects))

        from . import workspace
        from .output_manager import build_batch_export

        try:
            zip_path = build_batch_export(
                items, workspace.export_zip_path(job.id), image_format, include_outlines, adjust)
        except Exception as exc:  # noqa: BLE001
            self._send_error_json(500, f"could not build the export: {exc}")
            return

        self._send_file(zip_path, "application/zip", f"enhanced_images_{job.id[:8]}.zip")

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

    def do_PUT(self) -> None:
        if urlparse(self.path).path != "/api/settings":
            self._send_error_json(404, "not found")
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            body = self.rfile.read(length) if length > 0 else b"{}"
            payload = json.loads(body or b"{}")
        except (ValueError, json.JSONDecodeError):
            self._send_error_json(400, "invalid JSON body")
            return
        requested = payload.get("processing_device")
        if requested not in VALID_PREFERENCES:
            self._send_error_json(
                400, f"processing_device must be one of {list(VALID_PREFERENCES)}")
            return

        stored = settings.set_processing_device_preference(requested)
        state = engine_adapter.get_device_state()
        self._send_json(200, {
            "ok": True,
            "processing_device": stored,
            "effective_device": state["effective_device"],
            "detected_gpu": state["detected_gpu"],
            "warning": state["warning"],
            # This process already resolved a device (or hasn't yet); either
            # way, changing the on-disk preference now does NOT retroactively
            # change what's running -- see
            # engine_adapter.apply_device_preference's own docstring for why
            # (CUDA visibility is fixed for a process's lifetime once torch
            # has looked at it). True whenever a device has already been
            # resolved in this process and the new choice differs from it.
            "restart_required": (
                state["effective_device"] is not None
                and state["preference"] != stored
            ),
        })

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
