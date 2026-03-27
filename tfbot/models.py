"""Dataclasses and shared type definitions for TFBot."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, Optional, Sequence, Tuple


TransformKey = Tuple[int, int]


@dataclass(frozen=True)
class TFCharacter:
    name: str
    avatar_path: str
    message: str
    speech_color: Optional[str] = None
    display_name: Optional[str] = None
    folder: Optional[str] = None
    # Variables and links for reroll filtering and swap commands (VN only)
    gender: Optional[str] = None
    age: Optional[str] = None
    type: Optional[str] = None  # species, e.g. human
    _pack_name: Optional[str] = None  # pack file, e.g. characters_ST
    genderswap: Optional[str] = None
    ageswap: Optional[str] = None
    gender_age_swap: Optional[str] = None


@dataclass(frozen=True)
class OutfitAsset:
    name: str
    base_path: Path
    accessory_layers: Sequence[Path]


@dataclass
class TransformationState:
    user_id: int
    guild_id: int
    character_name: str
    character_avatar_path: str
    character_message: str
    original_nick: Optional[str]
    started_at: datetime
    expires_at: datetime
    duration_label: str
    character_folder: Optional[str] = None
    avatar_applied: bool = False
    original_display_name: str = ""
    is_inanimate: bool = False
    inanimate_responses: Tuple[str, ...] = field(default_factory=tuple)
    form_owner_user_id: Optional[int] = None
    identity_display_name: Optional[str] = None
    is_pillow: bool = False


@dataclass
class ReplyContext:
    author: str
    text: str


__all__ = [
    "TFCharacter",
    "OutfitAsset",
    "TransformationState",
    "ReplyContext",
    "TransformKey",
]
