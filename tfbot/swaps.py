"""Helpers for managing swap state and unwind logic."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple, Set

from .models import TransformationState
from .state import active_transformations, persist_states


@dataclass(frozen=True)
class SwapTransition:
    """Represents a user's before/after forms when a swap chain unwinds."""

    user_id: int
    before_form: str
    after_form: str


def ensure_form_owner(state: TransformationState) -> None:
    """Guarantee that the state tracks who owns the current form."""
    if state.form_owner_user_id is None:
        state.form_owner_user_id = state.user_id


def _collect_swap_maps(
    guild_id: int,
) -> Tuple[Dict[int, int], Dict[int, int], Dict[int, TransformationState]]:
    owner_to_holder: Dict[int, int] = {}
    holder_to_owner: Dict[int, int] = {}
    user_states: Dict[int, TransformationState] = {}
    for (g_id, _), state in active_transformations.items():
        if g_id != guild_id:
            continue
        user_states[state.user_id] = state
        owner_id = state.form_owner_user_id or state.user_id
        owner_to_holder[owner_id] = state.user_id
        holder_to_owner[state.user_id] = owner_id
    return owner_to_holder, holder_to_owner, user_states


def _discover_chain(
    trigger_user_id: int,
    owner_to_holder: Dict[int, int],
    holder_to_owner: Dict[int, int],
) -> Set[int]:
    stack: List[int] = [trigger_user_id]
    visited: Set[int] = set()
    while stack:
        current = stack.pop()
        if current in visited:
            continue
        visited.add(current)
        holder = owner_to_holder.get(current)
        if holder is not None and holder not in visited:
            stack.append(holder)
        owner = holder_to_owner.get(current)
        if owner is not None and owner not in visited:
            stack.append(owner)
    return visited


def unswap_chain(guild_id: int, trigger_user_id: int) -> Optional[List[SwapTransition]]:
    """Unwind every swap connected to trigger_user_id in the guild.

    Returns the list of swap transitions (for announcements) when a change
    occurred, otherwise None.
    """

    owner_to_holder, holder_to_owner, user_states = _collect_swap_maps(guild_id)
    if trigger_user_id not in user_states and trigger_user_id not in owner_to_holder:
        return None

    related = _discover_chain(trigger_user_id, owner_to_holder, holder_to_owner)
    participants: List[int] = []
    for user_id in related:
        holder = owner_to_holder.get(user_id)
        owner = holder_to_owner.get(user_id)
        if holder is not None and holder != user_id:
            participants.append(user_id)
            continue
        if owner is not None and owner != user_id:
            participants.append(user_id)
    if not participants:
        return None

    # Snapshot every owner's form before overwriting anything.
    form_payloads: Dict[int, Tuple[str, Optional[str], str, str, bool, Tuple[str, ...]]] = {}
    for owner_id in participants:
        holder_id = owner_to_holder.get(owner_id)
        if holder_id is None:
            continue
        holder_state = user_states.get(holder_id)
        if holder_state is None:
            continue
        form_payloads[owner_id] = (
            holder_state.character_name,
            holder_state.character_folder,
            holder_state.character_avatar_path,
            holder_state.character_message,
            holder_state.is_inanimate,
            holder_state.inanimate_responses,
        )

    transitions: List[SwapTransition] = []
    changed = False
    for owner_id in participants:
        state = user_states.get(owner_id)
        payload = form_payloads.get(owner_id)
        if state is None or payload is None:
            continue
        before_form = state.character_name
        (
            state.character_name,
            state.character_folder,
            state.character_avatar_path,
            state.character_message,
            state.is_inanimate,
            state.inanimate_responses,
        ) = payload
        state.form_owner_user_id = owner_id
        state.identity_display_name = None
        transitions.append(
            SwapTransition(
                user_id=owner_id,
                before_form=before_form,
                after_form=state.character_name,
            )
        )
        changed = True

    if changed:
        persist_states()
        return transitions
    return None


__all__ = ["SwapTransition", "ensure_form_owner", "unswap_chain"]
