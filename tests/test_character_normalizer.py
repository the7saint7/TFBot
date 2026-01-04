import tempfile
import unittest
from pathlib import Path

from tfbot.character_lookup import CharacterNameNormalizer, CharacterNormalizationError


class CharacterNameNormalizerTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.repo_root = Path(self._tmp.name)
        characters_dir = self.repo_root / "characters"
        characters_dir.mkdir(parents=True, exist_ok=True)
        (characters_dir / "johnGB").mkdir()
        (characters_dir / "alice").mkdir()

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_case_insensitive_lookup_returns_canonical_form(self) -> None:
        normalizer = CharacterNameNormalizer(self.repo_root)
        self.assertEqual(normalizer.normalize("johnGB"), "johnGB")
        self.assertEqual(normalizer.normalize("JOHNgb"), "johnGB")
        self.assertEqual(normalizer.normalize("alice"), "alice")

    def test_unknown_character_raises_error(self) -> None:
        normalizer = CharacterNameNormalizer(self.repo_root)
        with self.assertRaises(CharacterNormalizationError):
            normalizer.normalize("unknown-character")

    def test_conflicting_names_are_rejected(self) -> None:
        normalizer = CharacterNameNormalizer(
            self.repo_root,
            extra_candidates=["johnGB", "JOHNgb", "JoHnGb"],
        )
        with self.assertRaises(CharacterNormalizationError) as ctx:
            normalizer.normalize("johnGB")
        self.assertIn("Multiple canonical characters", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
