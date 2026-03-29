"""Tests for submission pending/archive path helpers."""

import unittest
from pathlib import Path

from tfbot.submissions import _storage_root_from_meta_path


class SubmissionStorageRootTests(unittest.TestCase):
    def test_legacy_flat_layout(self) -> None:
        root = Path("/fake/pending")
        meta = root / "abc123token" / "meta.json"
        self.assertEqual(_storage_root_from_meta_path(meta, root), root)

    def test_per_channel_layout(self) -> None:
        root = Path("/fake/pending")
        meta = root / "ch_999" / "abc123token" / "meta.json"
        self.assertEqual(_storage_root_from_meta_path(meta, root), root / "ch_999")

    def test_rejects_deep_or_odd_paths(self) -> None:
        root = Path("/fake/pending")
        self.assertIsNone(_storage_root_from_meta_path(root / "a" / "b" / "c" / "meta.json", root))


if __name__ == "__main__":
    unittest.main()
