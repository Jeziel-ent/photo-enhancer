import unittest

from backend.multipart import MultipartError, parse_multipart_files
from backend.tests.multipart_helpers import build_multipart_body


class TestParseMultipartFiles(unittest.TestCase):
    def test_single_file(self):
        content_type, body = build_multipart_body([
            ("files", "photo.jpg", "image/jpeg", b"\xff\xd8fake-jpeg-bytes"),
        ])
        files = parse_multipart_files(body, content_type)
        self.assertEqual(len(files), 1)
        self.assertEqual(files[0].filename, "photo.jpg")
        self.assertEqual(files[0].content_type, "image/jpeg")
        self.assertEqual(files[0].data, b"\xff\xd8fake-jpeg-bytes")
        self.assertEqual(files[0].field_name, "files")

    def test_multiple_files_preserve_order(self):
        content_type, body = build_multipart_body([
            ("files", "a.png", "image/png", b"AAA"),
            ("files", "b.png", "image/png", b"BBB"),
            ("files", "c.png", "image/png", b"CCC"),
        ])
        files = parse_multipart_files(body, content_type)
        self.assertEqual([f.filename for f in files], ["a.png", "b.png", "c.png"])
        self.assertEqual([f.data for f in files], [b"AAA", b"BBB", b"CCC"])

    def test_plain_form_fields_are_ignored(self):
        content_type, body = build_multipart_body(
            [("files", "photo.jpg", "image/jpeg", b"DATA")],
            extra_fields={"note": "not a file"},
        )
        files = parse_multipart_files(body, content_type)
        self.assertEqual(len(files), 1)
        self.assertEqual(files[0].filename, "photo.jpg")

    def test_binary_data_with_boundary_like_bytes_survives(self):
        # Real image bytes are arbitrary binary; make sure a chunk containing
        # \r\n sequences round-trips intact.
        data = bytes(range(256)) * 4
        content_type, body = build_multipart_body([
            ("files", "photo.png", "image/png", data),
        ])
        files = parse_multipart_files(body, content_type)
        self.assertEqual(files[0].data, data)

    def test_missing_boundary_raises(self):
        with self.assertRaises(MultipartError):
            parse_multipart_files(b"whatever", "multipart/form-data")

    def test_boundary_not_in_body_raises(self):
        with self.assertRaises(MultipartError):
            parse_multipart_files(b"no boundary here", "multipart/form-data; boundary=xyz")

    def test_no_file_parts_returns_empty_list(self):
        content_type, body = build_multipart_body(
            [], extra_fields={"note": "just a field"})
        files = parse_multipart_files(body, content_type)
        self.assertEqual(files, [])


if __name__ == "__main__":
    unittest.main()
