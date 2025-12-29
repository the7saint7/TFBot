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
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Sequence

import discord
from discord.ext import commands

from tfbot.models import TransformationState
from tfbot.panels import VN_CACHE_DIR, compose_game_avatar, parse_discord_formatting, render_vn_panel
from tfbot.utils import is_admin, utc_now

try:
    import yaml  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    yaml = None

logger = logging.getLogger("tfbot.submissions")

PACKAGE_DIR = Path(__file__).resolve().parent
BASE_DIR = PACKAGE_DIR.parent

SUBMISSIONS_DIR = BASE_DIR / "submissions"
PENDING_DIR = SUBMISSIONS_DIR / "pending"
ARCHIVE_DIR = SUBMISSIONS_DIR / "archive"

ACCEPT_EMOJI = "✅"
DECLINE_EMOJI = "❌"
APROUVER_ROLE = "aprouver"
SUBMISSION_CHANNEL_ID = int(os.getenv("TFBOT_SUBMISSION_CHANNEL_ID", "0") or 0)


def _resolve_characters_repo_root() -> Optional[Path]:
    repo_setting = os.getenv("TFBOT_CHARACTERS_REPO_DIR", "characters_repo").strip() or "characters_repo"
    repo_dir = Path(repo_setting)
    if not repo_dir.is_absolute():
        repo_dir = (BASE_DIR / repo_dir).resolve()
    repo_git = repo_dir / ".git"
    if not repo_dir.exists() or not repo_git.exists():
        return None
    return repo_dir


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
        if self.storage_root == ARCHIVE_DIR:
            return
        target_dir = ARCHIVE_DIR / self.token
        target_dir.parent.mkdir(parents=True, exist_ok=True)
        if target_dir.exists():
            shutil.rmtree(target_dir)
        shutil.move(str(self.directory), str(target_dir))
        self.storage_root = ARCHIVE_DIR


class SubmissionManager:
    def __init__(self, bot: commands.Bot):
        self.bot = bot
        self.records_by_token: Dict[str, SubmissionRecord] = {}
        self.pending_by_message: Dict[int, SubmissionRecord] = {}
        self._locks: Dict[str, asyncio.Lock] = {}
        self._repo_root: Optional[Path] = None
        _ensure_directories()
        self._load_existing()

    def _load_existing(self) -> None:
        for root in (PENDING_DIR, ARCHIVE_DIR):
            if not root.exists():
                continue
            for meta_path in root.glob("*/meta.json"):
                try:
                    data = json.loads(meta_path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError) as exc:
                    logger.warning("Failed to read submission metadata %s: %s", meta_path, exc)
                    continue
                record = SubmissionRecord.from_meta(data, storage_root=root)
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
        if SUBMISSION_CHANNEL_ID and ctx.channel.id != SUBMISSION_CHANNEL_ID:
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
        repo_root = self._resolve_repo()
        if repo_root is None:
            await ctx.reply("characters_repo is not configured; submissions are unavailable.", mention_author=False)
            return
        pose_dir = repo_root / "characters" / character / pose
        target_dir = pose_dir / "outfits"
        try:
            pose_dir.mkdir(parents=True, exist_ok=True)
            target_dir.mkdir(exist_ok=True)
        except OSError as exc:
            logger.warning("Failed to create character/pose directory %s: %s", target_dir, exc)
            await ctx.reply("Unable to create character or pose folders in characters_repo.", mention_author=False)
            return
        dest_file = target_dir / f"{outfit}.png"
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
            relative_path=str(dest_file.relative_to(repo_root)),
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
        panel_file = self._render_preview(record, preview_text)
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
        if SUBMISSION_CHANNEL_ID and ctx.channel.id != SUBMISSION_CHANNEL_ID:
            await ctx.reply("This command can only be used in the designated submission channel.", mention_author=False)
            return
        if not self._has_approval_power(author):
            await ctx.reply("You lack permission to mirror outfits.", mention_author=False)
            return
        validation = self._validate_inputs(character, pose, outfit)
        if validation:
            await ctx.reply(validation, mention_author=False)
            return
        repo_root = self._resolve_repo()
        if repo_root is None:
            await ctx.reply("characters_repo is not configured; mirroring is unavailable.", mention_author=False)
            return
        target_dir = repo_root / "characters" / character / pose / "outfits"
        outfit_path = target_dir / f"{outfit}.png"
        if not outfit_path.exists():
            await ctx.reply("That outfit image does not exist yet.", mention_author=False)
            return

        success, detail = await asyncio.to_thread(
            self._mirror_outfit_image,
            repo_root,
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
        if SUBMISSION_CHANNEL_ID and ctx.channel.id != SUBMISSION_CHANNEL_ID:
            await ctx.reply("This command can only be used in the designated submission channel.", mention_author=False)
            return
        if not self._has_approval_power(author):
            await ctx.reply("You lack permission to run repository sync.", mention_author=False)
            return
        repo_root = self._resolve_repo()
        if repo_root is None:
            await ctx.reply("characters_repo is not configured.", mention_author=False)
            return

        success, detail = await asyncio.to_thread(self._sync_characters_repo, repo_root)
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
        if not character or "/" in character or "\\" in character:
            return "Provide a valid character folder name."
        if not pose or "/" in pose or "\\" in pose:
            return "Provide a valid pose folder name."
        if not outfit or " " in outfit:
            return "Outfit names must not contain spaces."
        if not all(ch.isalnum() or ch == "_" for ch in outfit):
            return "Outfit names may only contain letters, numbers, or underscores."
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
            storage_root=PENDING_DIR,
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
        repo_root = self._resolve_repo()
        if repo_root is None:
            return False
        character_dir = repo_root / "characters" / record.character_name
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
        candidates = {character, character.lower()}
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

    def _resolve_repo(self) -> Optional[Path]:
        if self._repo_root and self._repo_root.exists():
            return self._repo_root
        repo = _resolve_characters_repo_root()
        if repo and repo.exists():
            self._repo_root = repo
        return self._repo_root

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
        normalized = APROUVER_ROLE.lower()
        return any(role.name.lower() == normalized for role in getattr(member, "roles", []))

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
        repo_root = self._resolve_repo()
        if repo_root is None:
            await self._notify_channel(record, "characters_repo is unavailable. Cannot approve submissions right now.")
            return
        result = await asyncio.to_thread(self._apply_submission_changes, record, repo_root, member.display_name)
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
        repo_root: Path,
        approver_name: str,
    ) -> tuple[bool, str]:
        try:
            relative_path = Path(record.target_relative_path)
            destination = (repo_root / relative_path).resolve()
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(record.image_path, destination)
        except OSError as exc:
            logger.warning("Failed to copy submission image into repo: %s", exc)
            return False, "failed to copy image into repository."
        commit_message = f"aprouved by {approver_name}"
        return self._commit_submission(repo_root, relative_path, commit_message)

    def _commit_submission(self, repo_root: Path, relative_path: Path, commit_message: str) -> tuple[bool, str]:
        git_executable = shutil.which("git")
        if not git_executable:
            return False, "git executable not found."

        commands = [
            [git_executable, "-C", str(repo_root), "add", str(relative_path)],
            [git_executable, "-C", str(repo_root), "commit", "-m", commit_message],
        ]
        for cmd in commands:
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                detail = result.stderr.strip() or result.stdout.strip() or "unknown git error"
                return False, detail

        push_cmd = [git_executable, "-C", str(repo_root), "push"]
        push_result = subprocess.run(push_cmd, capture_output=True, text=True)
        if push_result.returncode == 0:
            return True, "pushed successfully"

        pull_cmd = [git_executable, "-C", str(repo_root), "pull", "--rebase", "--autostash"]
        pull_result = subprocess.run(pull_cmd, capture_output=True, text=True)
        if pull_result.returncode != 0:
            detail = pull_result.stderr.strip() or pull_result.stdout.strip() or "git pull failed"
            return False, detail

        retry_result = subprocess.run(push_cmd, capture_output=True, text=True)
        if retry_result.returncode != 0:
            detail = retry_result.stderr.strip() or retry_result.stdout.strip() or "git push failed"
            return False, detail
        return True, "pushed after pulling"


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


__all__ = ["setup_submission_features", "SubmissionManager"]
