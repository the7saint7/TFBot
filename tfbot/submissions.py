"""Submission workflow for custom VN sprite approvals."""

from __future__ import annotations

import asyncio
import io
import json
import logging
import os
import secrets
import shutil
import subprocess
import time
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, FrozenSet, Optional, Sequence

import discord
from discord.ext import commands

from tfbot.character_lookup import (
    CharacterNameNormalizer,
    CharacterNormalizationError,
    resolve_characters_content_root,
    resolve_characters_git_root,
)

# Match CharacterNameNormalizer folder normalization for alias cache equality.
def _submission_norm_folder(value: str) -> str:
    return value.replace("\\", "/").strip().strip("/")
from tfbot.models import TransformationState
from tfbot.panel_executor import run_panel_render_vn
from tfbot.panels import VN_CACHE_DIR, compose_game_avatar, parse_discord_formatting, render_vn_panel
from tfbot.utils import _parse_quoted_comma_list, get_setting, is_admin, utc_now

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None  # type: ignore[misc, assignment]

try:
    import yaml  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    yaml = None

logger = logging.getLogger("tfbot.submissions")

PACKAGE_DIR = Path(__file__).resolve().parent
BASE_DIR = PACKAGE_DIR.parent


def _character_content_directory(content_root: Path, character: str) -> Path:
    """Join content_root with a canonical character path that may contain `/` (pack bucket + name)."""
    rel = character.replace("\\", "/").strip().strip("/")
    path = content_root
    for part in rel.split("/"):
        p = part.strip()
        if p:
            path = path / p
    return path


def _tf_character_folder_candidates() -> tuple[str, ...]:
    try:
        from tf_characters import TF_CHARACTERS
    except ImportError:
        return ()
    seen: set[str] = set()
    out: list[str] = []
    for entry in TF_CHARACTERS:
        if not isinstance(entry, dict):
            continue
        raw = _submission_norm_folder((entry.get("folder") or "").strip().replace("\\", "/"))
        if raw and raw not in seen:
            seen.add(raw)
            out.append(raw)
    return tuple(out)


def _tf_character_name_folder_aliases() -> dict[str, str]:
    """Map pool `name` (and unique first words) to canonical `folder` for !submit normalization."""
    try:
        from tf_characters import TF_CHARACTERS
    except ImportError:
        return {}
    full_name_aliases: dict[str, str] = {}
    first_word_hits: dict[str, list[str]] = {}
    for entry in TF_CHARACTERS:
        if not isinstance(entry, dict):
            continue
        name = (entry.get("name") or "").strip()
        folder = _submission_norm_folder((entry.get("folder") or "").strip().replace("\\", "/"))
        if not name or not folder:
            continue
        key_full = name.lower()
        if key_full not in full_name_aliases:
            full_name_aliases[key_full] = folder
        parts = name.split()
        if parts:
            first_word_hits.setdefault(parts[0].lower(), []).append(folder)
    out = dict(full_name_aliases)
    for fw, folders in first_word_hits.items():
        uniq = sorted(set(folders))
        if len(uniq) != 1:
            continue
        if fw in out and out[fw] != uniq[0]:
            continue
        if fw not in out:
            out[fw] = uniq[0]
    return out


def _validate_character_folder_token(character: str) -> Optional[str]:
    if not character or "\\" in character:
        return "Provide a valid character folder name."
    if ".." in character or character.startswith("/"):
        return "Provide a valid character folder name."
    parts = [p for p in character.split("/") if p]
    if not parts:
        return "Provide a valid character folder name."
    for part in parts:
        segment = part.strip()
        if not segment:
            return "Provide a valid character folder name."
        for ch in segment:
            if not (ch.isalnum() or ch in (" ", "_", "-")):
                return "Character path may only use letters, numbers, spaces, underscores, hyphens, and `/` between folders."
    return None

SUBMISSIONS_DIR = BASE_DIR / "submissions"
PENDING_DIR = SUBMISSIONS_DIR / "pending"
ARCHIVE_DIR = SUBMISSIONS_DIR / "archive"

ACCEPT_EMOJI = "✅"
DECLINE_EMOJI = "❌"

_ALLOWED_SUBMISSION_CHANNEL_IDS: FrozenSet[int] = frozenset()


def set_submission_channel_allowlist(*channel_ids: int) -> None:
    """Register Discord channel IDs where !submit / !mirror / !synch are allowed (from bot.py get_channel_id)."""
    global _ALLOWED_SUBMISSION_CHANNEL_IDS
    _ALLOWED_SUBMISSION_CHANNEL_IDS = frozenset(i for i in channel_ids if i > 0)


def _submission_channel_bucket_name(channel_id: int) -> str:
    return f"ch_{int(channel_id)}"


def _pending_root_for_channel(channel_id: int) -> Path:
    return PENDING_DIR / _submission_channel_bucket_name(channel_id)


def _archive_bucket_for_channel(channel_id: int) -> Path:
    return ARCHIVE_DIR / _submission_channel_bucket_name(channel_id)


def _storage_root_from_meta_path(meta_path: Path, root: Path) -> Optional[Path]:
    """Legacy: root/<token>/meta.json -> storage_root root. New: root/ch_<id>/<token>/meta.json -> root/ch_<id>."""
    try:
        rel = meta_path.relative_to(root)
    except ValueError:
        return None
    parts = rel.parts
    if len(parts) == 2 and parts[1] == "meta.json":
        return root
    if len(parts) == 3 and parts[0].startswith("ch_") and parts[2] == "meta.json":
        return root / parts[0]
    return None


def _reject_if_submission_channel_forbidden(channel_id: int) -> bool:
    """True if this channel may not run submission commands."""
    if not _ALLOWED_SUBMISSION_CHANNEL_IDS:
        return False
    return channel_id not in _ALLOWED_SUBMISSION_CHANNEL_IDS


def _ensure_directories() -> None:
    for path in (SUBMISSIONS_DIR, PENDING_DIR, ARCHIVE_DIR):
        path.mkdir(parents=True, exist_ok=True)


@dataclass
class SubmissionRecord:
    token: str
    guild_id: int
    channel_id: int
    submitter_id: int
    submitter_name: str
    character_name: str
    pose_name: str
    outfit_name: str
    attachment_url: str
    target_relative_path: str
    image_filename: str
    message_id: Optional[int] = None
    created_at: float = field(default_factory=lambda: utc_now().timestamp())
    status: str = "pending"
    approver_id: Optional[int] = None
    approver_name: Optional[str] = None
    storage_root: Path = field(default=PENDING_DIR, repr=False)

    @property
    def directory(self) -> Path:
        return self.storage_root / self.token

    @property
    def meta_path(self) -> Path:
        return self.directory / "meta.json"

    @property
    def image_path(self) -> Path:
        return self.directory / self.image_filename

    @property
    def jump_url(self) -> Optional[str]:
        if not self.message_id:
            return None
        return f"https://discord.com/channels/{self.guild_id}/{self.channel_id}/{self.message_id}"

    def to_dict(self) -> dict:
        return {
            "token": self.token,
            "guild_id": self.guild_id,
            "channel_id": self.channel_id,
            "submitter_id": self.submitter_id,
            "submitter_name": self.submitter_name,
            "character_name": self.character_name,
            "pose_name": self.pose_name,
            "outfit_name": self.outfit_name,
            "attachment_url": self.attachment_url,
            "target_relative_path": self.target_relative_path,
            "image_filename": self.image_filename,
            "message_id": self.message_id,
            "created_at": self.created_at,
            "status": self.status,
            "approver_id": self.approver_id,
            "approver_name": self.approver_name,
        }

    @classmethod
    def from_meta(cls, data: dict, storage_root: Path) -> Optional["SubmissionRecord"]:
        token = data.get("token")
        if not token:
            return None
        try:
            return cls(
                token=str(token),
                guild_id=int(data.get("guild_id") or 0),
                channel_id=int(data.get("channel_id") or 0),
                submitter_id=int(data.get("submitter_id") or 0),
                submitter_name=str(data.get("submitter_name") or ""),
                character_name=str(data.get("character_name") or ""),
                pose_name=str(data.get("pose_name") or ""),
                outfit_name=str(data.get("outfit_name") or ""),
                attachment_url=str(data.get("attachment_url") or ""),
                target_relative_path=str(data.get("target_relative_path") or ""),
                image_filename=str(data.get("image_filename") or "submission.png"),
                message_id=int(data["message_id"]) if data.get("message_id") else None,
                created_at=float(data.get("created_at") or utc_now().timestamp()),
                status=str(data.get("status") or "pending"),
                approver_id=int(data["approver_id"]) if data.get("approver_id") else None,
                approver_name=str(data.get("approver_name") or "") or None,
                storage_root=storage_root,
            )
        except (TypeError, ValueError):
            return None

    def save(self) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        self.meta_path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    def move_to_archive(self) -> None:
        try:
            self.directory.resolve().relative_to(ARCHIVE_DIR.resolve())
            return
        except ValueError:
            pass
        bucket = _archive_bucket_for_channel(self.channel_id)
        target_dir = bucket / self.token
        bucket.mkdir(parents=True, exist_ok=True)
        if target_dir.exists():
            shutil.rmtree(target_dir)
        shutil.move(str(self.directory), str(target_dir))
        self.storage_root = bucket


class SubmissionManager:
    def __init__(self, bot: commands.Bot):
        self.bot = bot
        self.records_by_token: Dict[str, SubmissionRecord] = {}
        self.pending_by_message: Dict[int, SubmissionRecord] = {}
        self._locks: Dict[str, asyncio.Lock] = {}
        self._git_root: Optional[Path] = None
        self._content_root: Optional[Path] = None
        self._character_normalizer: Optional[CharacterNameNormalizer] = None
        _ensure_directories()
        self._load_existing()

    def _load_existing(self) -> None:
        for root in (PENDING_DIR, ARCHIVE_DIR):
            if not root.exists():
                continue
            for meta_path in root.rglob("meta.json"):
                storage_root = _storage_root_from_meta_path(meta_path, root)
                if storage_root is None:
                    continue
                try:
                    data = json.loads(meta_path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError) as exc:
                    logger.warning("Failed to read submission metadata %s: %s", meta_path, exc)
                    continue
                record = SubmissionRecord.from_meta(data, storage_root=storage_root)
                if record is None:
                    logger.warning("Submission metadata malformed: %s", meta_path)
                    continue
                token = record.token
                self.records_by_token[token] = record
                if record.status == "pending" and record.message_id:
                    self.pending_by_message[record.message_id] = record
                self._locks.setdefault(token, asyncio.Lock())

    async def submit_command(self, ctx: commands.Context, character: str, pose: str, outfit: str) -> None:
        author = ctx.author
        guild = ctx.guild
        if not isinstance(author, discord.Member) or guild is None:
            await ctx.reply("Run this command inside a server.", mention_author=False)
            return
        if _reject_if_submission_channel_forbidden(ctx.channel.id):
            await ctx.reply("This command can only be used in the designated submission channel.", mention_author=False)
            return
        attachment = self._select_attachment(ctx.message.attachments)
        if attachment is None:
            await ctx.reply("Attach an image to your submission.", mention_author=False)
            return
        validation = self._validate_inputs(character, pose, outfit)
        if validation:
            await ctx.reply(validation, mention_author=False)
            return
        git_root = self._resolve_git_root()
        content_root = self._resolve_content_root()
        if git_root is None or content_root is None:
            await ctx.reply("characters_repo is not configured; submissions are unavailable.", mention_author=False)
            return
        character, normalization_error = self._normalize_character_name(character, content_root=content_root)
        if normalization_error:
            await ctx.reply(normalization_error, mention_author=False)
            return
        char_root = _character_content_directory(content_root, character)
        pose_dir = char_root / pose
        target_dir = pose_dir / "outfits"
        try:
            pose_dir.mkdir(parents=True, exist_ok=True)
            target_dir.mkdir(exist_ok=True)
        except OSError as exc:
            logger.warning("Failed to create character/pose directory %s: %s", target_dir, exc)
            await ctx.reply("Unable to create character or pose folders in characters_repo.", mention_author=False)
            return
        relative_path = self._build_relative_outfit_path(
            character, pose, outfit, git_root=git_root, content_root=content_root
        )
        dest_file = git_root / relative_path
        if dest_file.exists():
            await ctx.reply("An outfit with that name already exists. Choose a different outfit name.", mention_author=False)
            return
        try:
            image = await self._load_attachment_image(attachment)
        except ValueError as exc:
            await ctx.reply(str(exc), mention_author=False)
            return
        record = self._create_record(
            guild_id=guild.id,
            channel_id=ctx.channel.id,
            author=author,
            character=character,
            pose=pose,
            outfit=outfit,
            attachment_url=attachment.url,
            relative_path=relative_path.as_posix(),
        )
        try:
            self._store_image(record, image)
        except OSError as exc:
            logger.warning("Failed to store submission image: %s", exc)
            await ctx.reply("Failed to save your submission image. Try again later.", mention_author=False)
            self._discard_record(record)
            return
        finally:
            image.close()
        preview_text = f"pose: {pose}\noutfit: {outfit}"
        panel_file = await run_panel_render_vn(self._render_preview, record, preview_text)
        if panel_file is None:
            await ctx.reply("Unable to render the VN panel preview. Please contact staff.", mention_author=False)
            self._discard_record(record)
            return

        content = (
            f"Submission from {author.mention} for `{character}` pose `{pose}` outfit `{outfit}`.\n"
            f"{ACCEPT_EMOJI} to approve / {DECLINE_EMOJI} to decline."
        )
        try:
            sent_message = await ctx.channel.send(content, file=panel_file)
        except discord.HTTPException as exc:
            logger.warning("Failed to send submission preview: %s", exc)
            await ctx.reply("Could not post the submission preview.", mention_author=False)
            self._discard_record(record)
            return

        record.message_id = sent_message.id
        record.save()
        self.records_by_token[record.token] = record
        self.pending_by_message[sent_message.id] = record
        self._locks.setdefault(record.token, asyncio.Lock())

        for emoji in (ACCEPT_EMOJI, DECLINE_EMOJI):
            try:
                await sent_message.add_reaction(emoji)
            except discord.HTTPException:
                logger.debug("Failed to add reaction %s to submission message %s", emoji, sent_message.id)
        await ctx.reply("Submission posted for approval.", mention_author=False)

    async def mirror_command(self, ctx: commands.Context, character: str, pose: str, outfit: str) -> None:
        author = ctx.author
        guild = ctx.guild
        if not isinstance(author, discord.Member) or guild is None:
            await ctx.reply("Run this command inside a server.", mention_author=False)
            return
        if _reject_if_submission_channel_forbidden(ctx.channel.id):
            await ctx.reply("This command can only be used in the designated submission channel.", mention_author=False)
            return
        if not self._has_approval_power(author):
            await ctx.reply("You lack permission to mirror outfits.", mention_author=False)
            return
        validation = self._validate_inputs(character, pose, outfit)
        if validation:
            await ctx.reply(validation, mention_author=False)
            return
        git_root = self._resolve_git_root()
        content_root = self._resolve_content_root()
        if git_root is None or content_root is None:
            await ctx.reply("characters_repo is not configured; mirroring is unavailable.", mention_author=False)
            return
        character, normalization_error = self._normalize_character_name(character, content_root=content_root)
        if normalization_error:
            await ctx.reply(normalization_error, mention_author=False)
            return
        char_root = _character_content_directory(content_root, character)
        target_dir = char_root / pose / "outfits"
        outfit_path = target_dir / f"{outfit}.png"
        if not outfit_path.exists():
            await ctx.reply("That outfit image does not exist yet.", mention_author=False)
            return

        success, detail = await asyncio.to_thread(
            self._mirror_outfit_image,
            git_root,
            outfit_path,
            author.display_name,
            character,
            pose,
        )
        if success:
            message = (
                f"Mirrored `{character}` pose `{pose}` outfit `{outfit}` and pushed the update."
            )
            await ctx.reply(message, mention_author=False)
        else:
            await ctx.reply(f"Mirrored locally but git push failed: {detail}", mention_author=False)

    async def sync_command(self, ctx: commands.Context) -> None:
        author = ctx.author
        guild = ctx.guild
        if not isinstance(author, discord.Member) or guild is None:
            await ctx.reply("Run this command inside a server.", mention_author=False)
            return
        if _reject_if_submission_channel_forbidden(ctx.channel.id):
            await ctx.reply("This command can only be used in the designated submission channel.", mention_author=False)
            return
        if not self._has_approval_power(author):
            await ctx.reply("You lack permission to run repository sync.", mention_author=False)
            return
        git_root = self._resolve_git_root()
        if git_root is None:
            await ctx.reply("characters_repo is not configured.", mention_author=False)
            return

        success, detail = await asyncio.to_thread(self._sync_characters_repo, git_root)
        if success:
            await ctx.reply("characters_repo synced successfully.", mention_author=False)
        else:
            await ctx.reply(f"Sync failed: {detail}", mention_author=False)

    def _select_attachment(self, attachments: Sequence[discord.Attachment]) -> Optional[discord.Attachment]:
        for attachment in attachments:
            if attachment.content_type and attachment.content_type.startswith("image/"):
                return attachment
        return attachments[0] if attachments else None

    def _validate_inputs(self, character: str, pose: str, outfit: str) -> Optional[str]:
        char_err = _validate_character_folder_token(character)
        if char_err:
            return char_err
        if not pose or "/" in pose or "\\" in pose:
            return "Provide a valid pose folder name."
        if not outfit or " " in outfit:
            return "Outfit names must not contain spaces."
        if not all(ch.isalnum() or ch == "_" for ch in outfit):
            return "Outfit names may only contain letters, numbers, or underscores."
        return None

    def _build_relative_outfit_path(
        self,
        character: str,
        pose: str,
        outfit: str,
        *,
        git_root: Path,
        content_root: Path,
    ) -> Path:
        char_root = _character_content_directory(content_root, character)
        dest = (char_root / pose / "outfits" / f"{outfit}.png").resolve()
        return dest.relative_to(git_root.resolve())

    def _normalize_character_name(
        self,
        character: str,
        *,
        content_root: Optional[Path] = None,
    ) -> tuple[Optional[str], Optional[str]]:
        root = content_root or self._resolve_content_root()
        if root is None:
            return None, "characters_repo is not configured."
        extras = _tf_character_folder_candidates()
        name_aliases = _tf_character_name_folder_aliases()
        normalizer = self._character_normalizer
        cached_extras = getattr(normalizer, "_extra_candidates", None)
        cached_aliases = getattr(normalizer, "_name_aliases", None)
        if (
            normalizer is None
            or normalizer.content_root != root
            or cached_extras != extras
            or cached_aliases != name_aliases
        ):
            normalizer = CharacterNameNormalizer(
                root, extra_candidates=extras, name_aliases=name_aliases
            )
            self._character_normalizer = normalizer
        else:
            normalizer.refresh()
        try:
            canonical = normalizer.normalize(character)
        except CharacterNormalizationError as exc:
            logger.warning("Character normalization failed for '%s': %s", character, exc)
            return None, str(exc)
        return canonical, None

    def _ensure_canonical_record(self, record: SubmissionRecord, git_root: Path, content_root: Path) -> Optional[str]:
        canonical, error = self._normalize_character_name(record.character_name, content_root=content_root)
        if error:
            return error
        if canonical and canonical != record.character_name:
            logger.info(
                "Updating submission %s character folder %s -> %s",
                record.token,
                record.character_name,
                canonical,
            )
            record.character_name = canonical
        relative_path = self._build_relative_outfit_path(
            record.character_name,
            record.pose_name,
            record.outfit_name,
            git_root=git_root,
            content_root=content_root,
        )
        new_relative = relative_path.as_posix()
        if record.target_relative_path != new_relative:
            record.target_relative_path = new_relative
        return None

    async def _load_attachment_image(self, attachment: discord.Attachment) -> "Image.Image":
        try:
            from PIL import Image
        except ImportError as exc:  # pragma: no cover - dependency missing guard
            raise ValueError("Image support is not available.") from exc
        data = await attachment.read()
        if not data:
            raise ValueError("Attachment is empty.")
        try:
            image = Image.open(io.BytesIO(data))
        except OSError as exc:
            raise ValueError("Attachment is not a valid image.") from exc
        if getattr(image, "is_animated", False):
            image.seek(0)
        return image.convert("RGBA")

    def _create_record(
        self,
        *,
        guild_id: int,
        channel_id: int,
        author: discord.Member,
        character: str,
        pose: str,
        outfit: str,
        attachment_url: str,
        relative_path: str,
    ) -> SubmissionRecord:
        token = secrets.token_hex(8)
        record = SubmissionRecord(
            token=token,
            guild_id=guild_id,
            channel_id=channel_id,
            submitter_id=author.id,
            submitter_name=author.display_name,
            character_name=character,
            pose_name=pose,
            outfit_name=outfit,
            attachment_url=attachment_url,
            target_relative_path=relative_path,
            image_filename="submission.png",
            storage_root=_pending_root_for_channel(channel_id),
        )
        record.save()
        return record

    def _store_image(self, record: SubmissionRecord, image: "Image.Image") -> None:
        image.save(record.image_path, format="PNG")

    def _mirror_outfit_image(
        self,
        repo_root: Path,
        outfit_path: Path,
        actor_name: str,
        character: str,
        pose: str,
    ) -> tuple[bool, str]:
        try:
            from PIL import Image, ImageOps
        except ImportError:
            return False, "image processing is unavailable."
        if not outfit_path.exists():
            return False, "outfit image missing."
        try:
            with Image.open(outfit_path) as img:
                mirrored = ImageOps.mirror(img.convert("RGBA"))
                mirrored.save(outfit_path)
        except OSError as exc:
            logger.warning("Failed to mirror outfit %s: %s", outfit_path, exc)
            return False, "failed to process outfit image."

        self._clear_vn_cache_entry(character, pose)
        compose_game_avatar.cache_clear()
        relative_path = outfit_path.relative_to(repo_root)
        commit_message = f"mirrored by {actor_name}"
        return self._commit_submission(repo_root, relative_path, commit_message)

    def _sync_characters_repo(self, repo_root: Path) -> tuple[bool, str]:
        git_executable = shutil.which("git")
        if not git_executable:
            return False, "git executable not found."
        pull_rc, pull_detail = self._run_git_command(git_executable, repo_root, ["pull"])
        if pull_rc == 0:
            return True, "Repository pulled successfully."
        logger.warning("characters_repo pull failed: %s", pull_detail)

        push_rc, push_detail = self._run_git_command(git_executable, repo_root, ["push"])
        if push_rc == 0:
            retry_rc, _ = self._run_git_command(git_executable, repo_root, ["pull"])
            if retry_rc == 0:
                return True, "Local changes pushed and repository pulled."
        else:
            logger.warning("characters_repo push failed: %s", push_detail)

        reset_success = self._hard_reset_repo(git_executable, repo_root)
        if reset_success:
            return True, "Repository hard-reset to remote."
        return False, "Unable to synchronize characters_repo automatically."

    def _run_git_command(self, git_executable: str, repo_root: Path, args: list[str]) -> tuple[int, str]:
        cmd = [git_executable, "-C", str(repo_root), *args]
        result = subprocess.run(cmd, capture_output=True, text=True)
        detail = result.stderr.strip() or result.stdout.strip() or "no output"
        return result.returncode, detail

    def _detect_upstream_ref(self, git_executable: str, repo_root: Path) -> Optional[str]:
        rc, detail = self._run_git_command(
            git_executable,
            repo_root,
            ["rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{u}"],
        )
        if rc == 0 and detail:
            return detail.splitlines()[0].strip()
        logger.warning("Failed to detect upstream ref for %s: %s", repo_root, detail)
        return None

    def _hard_reset_repo(self, git_executable: str, repo_root: Path) -> bool:
        upstream = self._detect_upstream_ref(git_executable, repo_root) or "origin/main"
        fetch_rc, fetch_detail = self._run_git_command(
            git_executable,
            repo_root,
            ["fetch", "--all", "--tags", "--prune"],
        )
        if fetch_rc != 0:
            logger.warning("characters_repo fetch failed: %s", fetch_detail)
            return False
        reset_rc, reset_detail = self._run_git_command(
            git_executable,
            repo_root,
            ["reset", "--hard", upstream],
        )
        if reset_rc != 0:
            logger.warning("characters_repo reset --hard failed: %s", reset_detail)
            return False
        clean_rc, clean_detail = self._run_git_command(
            git_executable,
            repo_root,
            ["clean", "-fd"],
        )
        if clean_rc != 0:
            logger.warning("characters_repo clean failed: %s", clean_detail)
            return False
        return True

    def _render_preview(self, record: SubmissionRecord, preview_text: str) -> Optional[discord.File]:
        try:
            from PIL import Image, ImageOps
        except ImportError:
            return None
        if not record.image_path.exists():
            return None
        try:
            with Image.open(record.image_path) as img:
                avatar_image = img.convert("RGBA")
                if self._should_flip_avatar(record):
                    avatar_image = ImageOps.mirror(avatar_image)
                state = self._build_preview_state(record)
                segments = list(parse_discord_formatting(preview_text))
                return render_vn_panel(
                    state=state,
                    message_content=preview_text,
                    character_display_name=record.character_name,
                    original_name=record.submitter_name,
                    attachment_id=record.token,
                    formatted_segments=segments,
                    custom_emoji_images={},
                    reply_context=None,
                    selection_scope=None,
                    gacha_star_count=None,
                    gacha_rudy=None,
                    gacha_frog=None,
                    gacha_border=None,
                    override_avatar_image=avatar_image,
                )
        except OSError as exc:
            logger.warning("Failed to render submission preview for %s: %s", record.token, exc)
            return None

    def _should_flip_avatar(self, record: SubmissionRecord) -> bool:
        if yaml is None:
            return False
        content_root = self._resolve_content_root()
        if content_root is None:
            return False
        character_dir = _character_content_directory(content_root, record.character_name)
        config_path = character_dir / "character.yml"
        if not config_path.exists():
            return False
        try:
            config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        except Exception as exc:  # pragma: no cover - defensive
            logger.debug("Unable to parse character config %s: %s", config_path, exc)
            return False
        if not isinstance(config, dict):
            return False
        poses = config.get("poses")
        if not isinstance(poses, dict):
            return False
        pose_entry = poses.get(record.pose_name)
        if pose_entry is None:
            # fallback to case-insensitive lookup
            for key, value in poses.items():
                if isinstance(key, str) and key.lower() == record.pose_name.lower():
                    pose_entry = value
                    break
        if not isinstance(pose_entry, dict):
            return False
        facing = pose_entry.get("facing")
        if not isinstance(facing, str):
            return False
        return facing.strip().lower() == "left"

    def _clear_vn_cache_entry(self, character: str, pose: str) -> None:
        if VN_CACHE_DIR is None:
            return
        slash = character.replace("\\", "/")
        flat = slash.replace("/", "_")
        candidates = {character, character.lower(), slash, slash.lower(), flat, flat.lower()}
        pose_candidates = {pose, pose.lower()}
        for char_token in candidates:
            char_dir = VN_CACHE_DIR / char_token
            for pose_token in pose_candidates:
                cache_dir = char_dir / pose_token
                if cache_dir.exists():
                    shutil.rmtree(cache_dir, ignore_errors=True)
                    logger.debug("Cleared VN cache directory %s", cache_dir)
            if char_dir.exists():
                try:
                    next(char_dir.iterdir())
                except StopIteration:
                    shutil.rmtree(char_dir, ignore_errors=True)

    def _build_preview_state(self, record: SubmissionRecord) -> TransformationState:
        now = utc_now()
        return TransformationState(
            user_id=record.submitter_id,
            guild_id=record.guild_id,
            character_name=record.character_name,
            character_avatar_path=str(record.image_path),
            character_message=record.outfit_name,
            original_nick=None,
            started_at=now,
            expires_at=now,
            duration_label="",
            character_folder=record.pose_name,
            original_display_name=record.submitter_name,
        )

    def _discard_record(self, record: SubmissionRecord) -> None:
        try:
            if record.directory.exists():
                shutil.rmtree(record.directory, ignore_errors=True)
        finally:
            self.records_by_token.pop(record.token, None)
            if record.message_id:
                self.pending_by_message.pop(record.message_id, None)

    def _resolve_git_root(self) -> Optional[Path]:
        if self._git_root and self._git_root.exists() and (self._git_root / ".git").exists():
            return self._git_root
        repo = resolve_characters_git_root(BASE_DIR)
        if repo and repo.exists():
            self._git_root = repo
        return self._git_root

    def _resolve_content_root(self) -> Optional[Path]:
        if self._content_root and self._content_root.exists():
            return self._content_root
        root = resolve_characters_content_root(BASE_DIR)
        if root and root.exists():
            self._content_root = root
        return self._content_root

    async def on_raw_reaction_add(self, payload: discord.RawReactionActionEvent) -> None:
        if payload.user_id == self.bot.user.id:
            return
        record = self.pending_by_message.get(payload.message_id)
        if record is None:
            return
        emoji_name = str(payload.emoji)
        if emoji_name not in (ACCEPT_EMOJI, DECLINE_EMOJI):
            return
        guild = self.bot.get_guild(record.guild_id)
        member = payload.member
        if member is None and guild:
            member = guild.get_member(payload.user_id)
        if member is None or guild is None:
            return
        if not self._has_approval_power(member):
            await self._remove_reaction(payload)
            return
        lock = self._locks.setdefault(record.token, asyncio.Lock())
        async with lock:
            if record.status != "pending":
                return
            if emoji_name == ACCEPT_EMOJI:
                await self._handle_approval(record, member)
            elif emoji_name == DECLINE_EMOJI:
                await self._handle_decline(record, member)

    def _has_approval_power(self, member: discord.Member) -> bool:
        if is_admin(member):
            return True
        if member.guild.owner_id == member.id:
            return True
        if load_dotenv:
            load_dotenv()
        raw = os.getenv("TFBOT_TEST", "").strip().upper()
        test_mode = raw in ("YES", "TRUE", "1", "ON")
        names_str = get_setting("TFBOT_SUBMISSION_APPROVER_ROLE_NAMES", "", test_mode).strip()
        if not names_str:
            return False
        allowed_names = {s.lower() for s in _parse_quoted_comma_list(names_str)}
        return any(role.name.lower() in allowed_names for role in getattr(member, "roles", []))

    async def _remove_reaction(self, payload: discord.RawReactionActionEvent) -> None:
        channel = self.bot.get_channel(payload.channel_id)
        if channel is None:
            try:
                channel = await self.bot.fetch_channel(payload.channel_id)
            except discord.HTTPException:
                return
        if not isinstance(channel, (discord.TextChannel, discord.Thread)):
            return
        try:
            message = await channel.fetch_message(payload.message_id)
        except discord.HTTPException:
            return
        emoji = payload.emoji
        try:
            await message.remove_reaction(emoji, payload.member or discord.Object(id=payload.user_id))
        except discord.HTTPException:
            logger.debug("Failed to remove unauthorized reaction on message %s", payload.message_id)

    async def _handle_approval(self, record: SubmissionRecord, member: discord.Member) -> None:
        git_root = self._resolve_git_root()
        content_root = self._resolve_content_root()
        if git_root is None or content_root is None:
            await self._notify_channel(record, "characters_repo is unavailable. Cannot approve submissions right now.")
            return
        normalization_error = self._ensure_canonical_record(record, git_root, content_root)
        if normalization_error:
            await self._notify_channel(record, f"Cannot approve submission: {normalization_error}")
            return
        result = await asyncio.to_thread(self._apply_submission_changes, record, git_root, member.display_name)
        success, detail = result
        if success:
            record.status = "approved"
            record.approver_id = member.id
            record.approver_name = member.display_name
            record.move_to_archive()
            record.save()
            self.pending_by_message.pop(record.message_id or 0, None)
            message = (
                f"{member.mention} approved the submission from <@{record.submitter_id}> for "
                f"`{record.character_name}` `{record.pose_name}` outfit `{record.outfit_name}`.\n"
                f"Saved to `{record.target_relative_path}`."
            )
            await self._notify_channel(record, message)
        else:
            await self._notify_channel(
                record,
                f"Failed to approve submission for `{record.outfit_name}`: {detail}",
            )

    async def _handle_decline(self, record: SubmissionRecord, member: discord.Member) -> None:
        record.status = "declined"
        record.approver_id = member.id
        record.approver_name = member.display_name
        record.move_to_archive()
        record.save()
        self.pending_by_message.pop(record.message_id or 0, None)
        message = (
            f"{member.mention} declined the submission from <@{record.submitter_id}> for "
            f"`{record.character_name}` `{record.pose_name}` outfit `{record.outfit_name}`."
        )
        await self._notify_channel(record, message)

    async def _notify_channel(self, record: SubmissionRecord, content: str) -> None:
        channel = self.bot.get_channel(record.channel_id)
        if channel is None:
            try:
                channel = await self.bot.fetch_channel(record.channel_id)
            except discord.HTTPException:
                logger.warning("Unable to locate channel %s for submission updates", record.channel_id)
                return
        if isinstance(channel, (discord.TextChannel, discord.Thread)):
            jump = record.jump_url
            suffix = f"\n<{jump}>" if jump else ""
            try:
                await channel.send(f"{content}{suffix}")
            except discord.HTTPException as exc:
                logger.warning("Failed to send submission status message: %s", exc)

    def _apply_submission_changes(
        self,
        record: SubmissionRecord,
        git_root: Path,
        approver_name: str,
    ) -> tuple[bool, str]:
        content_root = self._resolve_content_root()
        if content_root is None:
            return False, "characters_repo content root is not available."
        try:
            relative_path = self._build_relative_outfit_path(
                record.character_name,
                record.pose_name,
                record.outfit_name,
                git_root=git_root,
                content_root=content_root,
            )
            record.target_relative_path = relative_path.as_posix()
            destination = (git_root / relative_path).resolve()
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(record.image_path, destination)
        except (OSError, ValueError) as exc:
            logger.warning("Failed to copy submission image into repo: %s", exc)
            return False, "failed to copy image into repository."
        commit_message = f"aprouved by {approver_name}"
        return self._commit_submission(git_root, relative_path, commit_message)

    def _commit_submission(self, repo_root: Path, relative_path: Path, commit_message: str) -> tuple[bool, str]:
        git_executable = shutil.which("git")
        if not git_executable:
            return False, "git executable not found."

        add_cmd = [git_executable, "-C", str(repo_root), "add", str(relative_path)]
        add_result = subprocess.run(add_cmd, capture_output=True, text=True)
        if add_result.returncode != 0:
            detail = add_result.stderr.strip() or add_result.stdout.strip() or "git add failed"
            return False, detail

        staged_check = [
            git_executable,
            "-C",
            str(repo_root),
            "diff",
            "--cached",
            "--name-only",
            "--",
            str(relative_path),
        ]
        staged_result = subprocess.run(staged_check, capture_output=True, text=True)
        staged_names = (staged_result.stdout or "").strip()
        if not staged_names:
            retry_add = [git_executable, "-C", str(repo_root), "add", "-A", "--", str(relative_path)]
            retry_result = subprocess.run(retry_add, capture_output=True, text=True)
            if retry_result.returncode != 0:
                detail = retry_result.stderr.strip() or retry_result.stdout.strip() or "git add -A failed"
                return False, detail
            staged_result = subprocess.run(staged_check, capture_output=True, text=True)
            staged_names = (staged_result.stdout or "").strip()
            if not staged_names:
                status_cmd = [git_executable, "-C", str(repo_root), "status", "--porcelain"]
                status_result = subprocess.run(status_cmd, capture_output=True, text=True)
                detail = status_result.stdout.strip() or status_result.stderr.strip() or "git status failed"
                return False, detail

        commit_cmd = [git_executable, "-C", str(repo_root), "commit", "-m", commit_message]
        commit_result = subprocess.run(commit_cmd, capture_output=True, text=True)
        if commit_result.returncode != 0:
            detail = commit_result.stderr.strip() or commit_result.stdout.strip() or "git commit failed"
            return False, detail

        push_cmd = [git_executable, "-C", str(repo_root), "push"]
        pull_cmd = [git_executable, "-C", str(repo_root), "pull", "--rebase", "--autostash"]

        last_detail = ""
        for _ in range(3):
            push_result = subprocess.run(push_cmd, capture_output=True, text=True)
            if push_result.returncode == 0:
                return True, "pushed successfully"
            last_detail = (push_result.stderr or "").strip() or (push_result.stdout or "").strip() or "git push failed"

            if "cannot lock ref 'refs/heads/main'" in last_detail or "cannot lock ref \"refs/heads/main\"" in last_detail:
                time.sleep(0.6 + (random.random() * 1.4))

            pull_result = subprocess.run(pull_cmd, capture_output=True, text=True)
            if pull_result.returncode != 0:
                detail = pull_result.stderr.strip() or pull_result.stdout.strip() or "git pull failed"
                return False, detail

        return False, last_detail or "git push failed"


def setup_submission_features(bot: commands.Bot) -> SubmissionManager:
    manager = SubmissionManager(bot)

    @bot.command(name="submit")
    async def submit_command(ctx: commands.Context, character: str, pose: str, outfit: str):
        await manager.submit_command(ctx, character, pose, outfit)

    @bot.command(name="mirror")
    async def mirror_command(ctx: commands.Context, character: str, pose: str, outfit: str):
        await manager.mirror_command(ctx, character, pose, outfit)

    @bot.command(name="synch")
    async def sync_command(ctx: commands.Context):
        await manager.sync_command(ctx)

    bot.add_listener(manager.on_raw_reaction_add)
    return manager


__all__ = ["set_submission_channel_allowlist", "setup_submission_features", "SubmissionManager"]
