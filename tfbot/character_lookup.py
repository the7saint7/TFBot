"""Utilities for canonical character folder normalization."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set

logger = logging.getLogger("tfbot.character_lookup")


class CharacterNormalizationError(Exception):
    """Raised when a submitted character name cannot be normalized."""


class CharacterNameNormalizer:
    """Normalize submitted character folder names to their canonical casing."""

    def __init__(self, repo_root: Path, *, extra_candidates: Optional[Sequence[str]] = None):
        self.repo_root = repo_root
        self._extra_candidates = tuple(
            str(candidate).strip() for candidate in (extra_candidates or []) if str(candidate).strip()
        )
        self._lookup: Dict[str, str] = {}
        self._conflicts: Dict[str, List[str]] = {}
        self.refresh()

    def refresh(self) -> None:
        """Rebuild the lookup table from the characters repo."""
        lookup: Dict[str, str] = {}
        conflicts: Dict[str, List[str]] = {}
        for canonical in self._candidate_names():
            lowered = canonical.lower()
            if lowered in conflicts:
                if canonical not in conflicts[lowered]:
                    conflicts[lowered].append(canonical)
                continue

            existing = lookup.get(lowered)
            if existing is None:
                lookup[lowered] = canonical
                continue

            # Collision detected. Track all conflicting canonical names and drop from lookup.
            names: Set[str] = set(conflicts.get(lowered, []))
            names.add(existing)
            names.add(canonical)
            formatted = sorted(names, key=str.lower)
            conflicts[lowered] = formatted
            lookup.pop(lowered, None)
            logger.error(
                "Duplicate character folder names detected for '%s': %s",
                lowered,
                ", ".join(formatted),
            )

        self._lookup = lookup
        self._conflicts = conflicts

    def normalize(self, submitted: str) -> str:
        """Return the canonical folder name for the submitted value."""
        if not submitted:
            raise CharacterNormalizationError("Provide a valid character folder name.")

        lowered = submitted.strip().lower()
        conflict = self._conflicts.get(lowered)
        if conflict:
            conflict_names = ", ".join(conflict)
            raise CharacterNormalizationError(
                f"Multiple canonical characters share the name '{submitted.lower()}': {conflict_names}. "
                "Contact staff to resolve the duplicate before continuing."
            )

        canonical = self._lookup.get(lowered)
        if canonical:
            return canonical

        raise CharacterNormalizationError(f"Unknown character '{submitted}'. Check the spelling and try again.")

    def _candidate_names(self) -> Iterable[str]:
        names: Set[str] = set()
        characters_dir = self.repo_root / "characters"
        if characters_dir.exists():
            for entry in characters_dir.iterdir():
                if entry.is_dir():
                    candidate = entry.name.strip()
                    if candidate:
                        names.add(candidate)

        for candidate in self._extra_candidates:
            names.add(candidate)

        return sorted(names, key=str.lower)


__all__ = ["CharacterNameNormalizer", "CharacterNormalizationError"]
