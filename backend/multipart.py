"""Minimal, dependency-free multipart/form-data parser.

Only what ``POST /api/jobs`` needs: pull every *file* part (one with a
``filename`` on its Content-Disposition) out of the body. Plain form fields
are ignored. This is not a general-purpose MIME parser — request bodies here
are whole image uploads already fully read into memory by the HTTP layer, so
a simple split-on-boundary parser is enough and keeps the API layer
dependency-light.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


class MultipartError(ValueError):
    """The request body is not well-formed multipart/form-data."""


@dataclass
class UploadedFile:
    field_name: str
    filename: str
    content_type: str
    data: bytes


_BOUNDARY_RE = re.compile(r'boundary="?([^";]+)"?')


def _parse_boundary(content_type: str) -> bytes:
    match = _BOUNDARY_RE.search(content_type or "")
    if not match:
        raise MultipartError("multipart/form-data request is missing a boundary")
    return b"--" + match.group(1).encode("utf-8")


def _parse_part_headers(raw_headers: bytes) -> dict[str, str]:
    headers: dict[str, str] = {}
    for line in raw_headers.split(b"\r\n"):
        if not line or b":" not in line:
            continue
        key, _, value = line.partition(b":")
        headers[key.strip().lower().decode("latin-1")] = value.strip().decode("latin-1")
    return headers


def _parse_content_disposition(value: str) -> dict[str, str]:
    params: dict[str, str] = {}
    for piece in value.split(";"):
        piece = piece.strip()
        if "=" not in piece:
            continue
        key, _, val = piece.partition("=")
        params[key.strip().lower()] = val.strip().strip('"')
    return params


def parse_multipart_files(body: bytes, content_type: str) -> list[UploadedFile]:
    """Extracts every file part from a multipart/form-data body.

    Raises MultipartError if the boundary is missing or the body is
    malformed enough that no parts can be located at all (an empty result
    for a well-formed body with no file parts is not an error — the caller
    decides whether "no files" is acceptable).
    """
    boundary = _parse_boundary(content_type)
    if boundary not in body:
        raise MultipartError("multipart/form-data body does not contain its boundary")

    files: list[UploadedFile] = []
    segments = body.split(boundary)
    # segments[0] is the preamble before the first boundary; the last
    # segment is the closing "--\r\n" marker. Both are dropped.
    for segment in segments[1:-1]:
        if segment.startswith(b"\r\n"):
            segment = segment[2:]
        if segment.endswith(b"\r\n"):
            segment = segment[:-2]
        if b"\r\n\r\n" not in segment:
            continue
        raw_headers, data = segment.split(b"\r\n\r\n", 1)
        headers = _parse_part_headers(raw_headers)
        params = _parse_content_disposition(headers.get("content-disposition", ""))
        filename = params.get("filename")
        if not filename:
            continue  # a plain form field, not a file part
        files.append(UploadedFile(
            field_name=params.get("name", ""),
            filename=filename,
            content_type=headers.get("content-type", "application/octet-stream"),
            data=data,
        ))
    return files
