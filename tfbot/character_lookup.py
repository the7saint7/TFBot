"""Utilities for canonical character folder normalization."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Set

logger = logging.getLogger("tfbot.character_lookup")

PACKAGE_DIR = Path(__file__).resolve().parent
DEFAULT_BOT_BASE = PACKAGE_DIR.parent


class CharacterNormalizationError(Exception):
    """Raised when a submitted character name cannot be normalized."""


def resolve_characters_git_root(base_dir: Optional[Path] = None) -> Optional[Path]:
    """Directory containing `.git` for the configured characters repository."""
    root_base = (base_dir or DEFAULT_BOT_BASE).resolve()
    repo_setting = os.getenv("TFBOT_CHARACTERS_REPO_DIR", "characters_repo").strip() or "characters_repo"
    repo_dir = Path(repo_setting)
    if not repo_dir.is_absolute():
        repo_dir = (root_base / repo_dir).resolve()
    if not repo_dir.exists() or not (repo_dir / ".git").exists():
        return None
    return repo_dir


def resolve_characters_content_root(base_dir: Optional[Path] = None) -> Optional[Path]:
    """Directory whose immediate subfolders are character names (matches `TFBOT_VN_ASSET_ROOT` / startup)."""
    git_root = resolve_characters_git_root(base_dir)
    if git_root is None:
        return None
    subdir = os.getenv("TFBOT_CHARACTERS_REPO_SUBDIR", "characters").strip()
    content = (git_root / subdir) if subdir else git_root
    if not content.is_dir():
        return None
    return content.resolve()


def _normalize_path_slashes(value: str) -> str:
    return value.replace("\\", "/").strip().strip("/")


def _folder_basename(canonical: str) -> str:
    norm = _normalize_path_slashes(canonical)
    if not norm:
        return ""
    return norm.split("/")[-1]


class CharacterNameNormalizer:
    """Normalize submitted character folder names to their canonical casing.

    ``content_root`` is the directory that directly contains character folders (not the git root unless
    ``TFBOT_CHARACTERS_REPO_SUBDIR`` is empty).
    """

    def __init__(
        self,
        content_root: Path,
        *,
        extra_candidates: Optional[Sequence[str]] = None,
        name_aliases: Optional[Mapping[str, str]] = None,
    ):
        self.content_root = content_root
        self._extra_candidates = tuple(
            _normalize_path_slashes(str(candidate))
            for candidate in (extra_candidates or [])
            if str(candidate).strip()
        )
        self._name_aliases: Dict[str, str] = {
            k.strip().lower(): _normalize_path_slashes(v)
            for k, v in (name_aliases or {}).items()
            if str(k).strip() and str(v).strip()
        }
        self._lookup: Dict[str, str] = {}
        self._conflicts: Dict[str, List[str]] = {}
        self._canonical_names: tuple[str, ...] = ()
        self.refresh()

    def refresh(self) -> None:
        """Rebuild the lookup table from the characters repo."""
        raw_candidates = list(self._collect_raw_candidates())
        self._canonical_names = tuple(sorted(set(raw_candidates), key=str.lower))

        lookup: Dict[str, str] = {}
        conflicts: Dict[str, List[str]] = {}
        for canonical in self._canonical_names:
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

        stripped = submitted.strip()
        lowered_key = _normalize_path_slashes(stripped).lower()
        conflict = self._conflicts.get(lowered_key)
        if conflict:
            conflict_names = ", ".join(conflict)
            raise CharacterNormalizationError(
                f"Multiple canonical characters share the name '{stripped.lower()}': {conflict_names}. "
                "Contact staff to resolve the duplicate before continuing."
            )

        canonical = self._lookup.get(lowered_key)
        if canonical:
            return canonical

        alias_folder = self._name_aliases.get(lowered_key)
        if alias_folder:
            af_l = alias_folder.lower()
            hit = self._lookup.get(af_l)
            if hit:
                return hit
            fuzzy_alias = self._fuzzy_resolve(af_l)
            if len(fuzzy_alias) == 1:
                return fuzzy_alias[0]

        fuzzy = self._fuzzy_resolve(lowered_key)
        if len(fuzzy) == 1:
            return fuzzy[0]
        if len(fuzzy) > 1:
            preview = ", ".join(sorted(fuzzy, key=str.lower))
            raise CharacterNormalizationError(
                f"Ambiguous character '{submitted}': could mean {preview}. Use a fuller or more specific name."
            )

        raise CharacterNormalizationError(f"Unknown character '{submitted}'. Check the spelling and try again.")

    def _fuzzy_resolve(self, lowered: str) -> List[str]:
        """Match basename or first word of basename; return unique canonical paths."""
        if not lowered:
            return []
        hits: list[str] = []
        seen: Set[str] = set()
        for canonical in self._canonical_names:
            norm = _normalize_path_slashes(canonical)
            if not norm:
                continue
            full_l = norm.lower()
            if full_l == lowered:
                if canonical not in seen:
                    seen.add(canonical)
                    hits.append(canonical)
                continue
            base = _folder_basename(canonical)
            base_l = base.lower()
            if base_l == lowered:
                if canonical not in seen:
                    seen.add(canonical)
                    hits.append(canonical)
                continue
            parts = base_l.split()
            if parts and parts[0] == lowered:
                if canonical not in seen:
                    seen.add(canonical)
                    hits.append(canonical)
        return hits

    def _collect_raw_candidates(self) -> Iterable[str]:
        names: Set[str] = set()
        if self.content_root.exists():
            try:
                for entry in self.content_root.iterdir():
                    if not entry.is_dir():
                        continue
                    candidate = entry.name.strip()
                    if not candidate:
                        continue
                    names.add(candidate)
                    try:
                        for sub in entry.iterdir():
                            if sub.is_dir():
                                sub_name = sub.name.strip()
                                if sub_name:
                                    names.add(f"{candidate}/{sub_name}")
                    except OSError as exc:
                        logger.debug("Skipping subdirs under %s: %s", entry, exc)
            except OSError as exc:
                logger.warning("Failed to list character directories under %s: %s", self.content_root, exc)

        for candidate in self._extra_candidates:
            if candidate:
                names.add(candidate)

        return sorted(names, key=str.lower)


__all__ = [
    "CharacterNameNormalizer",
    "CharacterNormalizationError",
    "resolve_characters_git_root",
    "resolve_characters_content_root",
]
