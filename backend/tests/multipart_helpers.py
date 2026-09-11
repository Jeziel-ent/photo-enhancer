"""Shared helper to build real multipart/form-data request bodies for tests,
so both the parser unit tests and the HTTP-level server tests exercise the
exact wire format a browser or `curl -F` would send."""

from __future__ import annotations

import uuid


def build_multipart_body(
    files: list[tuple[str, str, str, bytes]],
    extra_fields: dict[str, str] | None = None,
) -> tuple[str, bytes]:
    """``files`` entries are (field_name, filename, content_type, data).

    Returns (content_type_header_value, body_bytes).
    """
    boundary = f"----adinntest{uuid.uuid4().hex}"
    lines: list[bytes] = []
    for field_name, filename, content_type, data in files:
        lines.append(f"--{boundary}".encode())
        lines.append(
            f'Content-Disposition: form-data; name="{field_name}"; filename="{filename}"'.encode()
        )
        lines.append(f"Content-Type: {content_type}".encode())
        lines.append(b"")
        lines.append(data)
    for name, value in (extra_fields or {}).items():
        lines.append(f"--{boundary}".encode())
        lines.append(f'Content-Disposition: form-data; name="{name}"'.encode())
        lines.append(b"")
        lines.append(value.encode())
    lines.append(f"--{boundary}--".encode())
    lines.append(b"")
    body = b"\r\n".join(lines)
    return f"multipart/form-data; boundary={boundary}", body
