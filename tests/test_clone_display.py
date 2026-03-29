"""Tests for frozen clone-of labels (tfbot.clone_display)."""

import unittest

from tfbot.clone_display import clone_copied_source_display_name


class CloneCopiedSourceDisplayTests(unittest.TestCase):
    def test_empty_record(self) -> None:
        self.assertIsNone(clone_copied_source_display_name({}))

    def test_prefers_character_name_in_snapshot(self) -> None:
        rec = {"cloned_visual_snapshot": {"character_name": "Connie Smith", "identity_display_name": "ignored"}}
        self.assertEqual(clone_copied_source_display_name(rec), "Connie Smith")

    def test_falls_back_to_identity_in_snapshot(self) -> None:
        rec = {"cloned_visual_snapshot": {"character_name": "", "identity_display_name": "Connie"}}
        self.assertEqual(clone_copied_source_display_name(rec), "Connie")

    def test_strips_whitespace(self) -> None:
        rec = {"cloned_visual_snapshot": {"character_name": "  Maria  "}}
        self.assertEqual(clone_copied_source_display_name(rec), "Maria")


if __name__ == "__main__":
    unittest.main()
