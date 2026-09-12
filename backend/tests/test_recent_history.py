"""Unit tests for backend/recent_history.py's JSON-file persistence."""

import tempfile
import unittest
from pathlib import Path
from unittest import mock

from backend import recent_history


class RecentHistoryTestCase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self._patch = mock.patch.object(
            recent_history.workspace, "WORKSPACE_ROOT", Path(self._tmp.name))
        self._patch.start()
        self.addCleanup(self._patch.stop)

    def test_load_with_no_file_yet_is_empty(self):
        self.assertEqual(recent_history.load(), [])

    def test_add_persists_newest_first(self):
        recent_history.add({"id": "a", "filename": "one.png"})
        recent_history.add({"id": "b", "filename": "two.png"})
        entries = recent_history.load()
        self.assertEqual([e["id"] for e in entries], ["b", "a"])

    def test_add_survives_a_fresh_load_call(self):
        recent_history.add({"id": "a", "filename": "one.png"})
        # A brand-new call to load() re-reads from disk rather than any
        # in-memory state, proving this actually persists.
        self.assertEqual(recent_history.load(), [{"id": "a", "filename": "one.png"}])

    def test_entries_are_capped_at_max(self):
        for i in range(recent_history.MAX_ENTRIES + 10):
            recent_history.add({"id": str(i)})
        entries = recent_history.load()
        self.assertEqual(len(entries), recent_history.MAX_ENTRIES)
        self.assertEqual(entries[0]["id"], str(recent_history.MAX_ENTRIES + 9))

    def test_corrupt_file_is_treated_as_empty_not_a_crash(self):
        path = Path(self._tmp.name) / "recent_history.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{not valid json", encoding="utf-8")
        self.assertEqual(recent_history.load(), [])
        # add() must still work afterwards (overwrites the corrupt file).
        recent_history.add({"id": "a"})
        self.assertEqual(recent_history.load(), [{"id": "a"}])


if __name__ == "__main__":
    unittest.main()
