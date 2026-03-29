"""Tests for tfbot.prefix_tail token counting."""

import unittest

from tfbot.prefix_tail import count_prefix_tail_tokens


class PrefixTailTokenTests(unittest.TestCase):
    def test_empty(self) -> None:
        self.assertEqual(count_prefix_tail_tokens(""), 0)
        self.assertEqual(count_prefix_tail_tokens("   "), 0)

    def test_single_and_multi(self) -> None:
        self.assertEqual(count_prefix_tail_tokens("kat"), 1)
        self.assertEqual(count_prefix_tail_tokens("how do i"), 3)

    def test_quoted_counts_as_one(self) -> None:
        self.assertEqual(count_prefix_tail_tokens('"John Smith"'), 1)
        self.assertEqual(count_prefix_tail_tokens("'Jane Doe' extra"), 2)


if __name__ == "__main__":
    unittest.main()
