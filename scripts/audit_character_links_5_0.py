"""
Audit pack link correctness for BunniBotRP 5.0.

This script is intentionally "pure AST" (no bot imports) so it can run in dev
without Discord/bot runtime dependencies.

Checks:
1) Orphan tails: every `characters_STVariants/<Tail>` referenced in ST + Shorts
   must exist as a variant row `name` inside characters_STVariants.py.
2) Teen quad: genderswap pairing + ageswap to expected adult + plan-F cross-swap
   for the five `characters_ST/`-genderswap teen pairs (male ageswap vs female
   gender_age_swap, and vice versa).
3) Swap-semantics edges: for each state (ST, STVariants, Shorts), verify that
   genderswap/ageswap/gender_age_swap resolve to the expected opposite
   gender/age corner (based on current row gender/age).
4) Plan G: block membership — on ST + Shorts rows, every
   `characters_STVariants/<Tail>` must belong to that row's base block's variant
   `name` set (from `default_character` in characters_STVariants.py).
5) Plan H: no bogus mixed family — a `...GBAP` / `...GBAR` STV tail requires the
   base block to declare at least one variant name with that suffix family.
6) Plan I: characters_STVariants.py must not author internal
   `characters_STVariants/...` cross-links (only `variants_*` + `characters_ST/`
   style).

CI: `python audit_character_links_5_0.py --ci` exits with code 1 if any check
reports failures.

Output:
- prints a tail-mismatch summary
- prints a detailed list of incorrect/missing swap edges with "suggested"
  corrected link strings that match the pack's authoring style.
"""

from __future__ import annotations

import argparse
import ast
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


TFBOT_DIR = Path(__file__).resolve().parent
ROOT_DIR = TFBOT_DIR.parent
PACKS_DIR = ROOT_DIR / "characters_repo" / "packs"

ST_FILE = PACKS_DIR / "characters_ST.py"
STV_FILE = PACKS_DIR / "characters_STVariants.py"
SHORTS_FILE = PACKS_DIR / "characters_Shorts.py"


LINK_STV_PREFIX = "characters_STVariants/"
LINK_ST_PREFIX = "characters_ST/"
LINK_SHORTS_PREFIX = "characters_Shorts/"


def _literal_eval_assignment(path: Path, var_name: str) -> Any:
    """
    Return the literal value for `var_name = ...` where the RHS is literal-evaluable.
    """
    tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            t = node.targets[0]
            if isinstance(t, ast.Name) and t.id == var_name:
                return ast.literal_eval(node.value)
    raise KeyError(f"{var_name} not found in {path}")


def _norm_folder_token(folder: str | None) -> str:
    if not folder:
        return ""
    return folder.strip().replace("\\", "/").strip("/").lower()


def parse_variant_base_from_name(name: str | None) -> str | None:
    if not name:
        return None
    n = name.strip()
    # Strip suffixes in the same order as the naming patterns we use.
    for suf in ("GBAP", "GBAR", "GB", "AP", "AR"):
        if n.endswith(suf):
            return n[: -len(suf)].lower()
    return None


@dataclass(frozen=True)
class State:
    pack: str  # characters_ST | characters_STVariants | characters_Shorts
    name: str
    gender: str
    age: str  # teen | adult
    folder: str
    genderswap: str | None
    ageswap: str | None
    gender_age_swap: str | None
    base: str


def flatten_st_rows() -> list[State]:
    rows: list[dict[str, Any]] = _literal_eval_assignment(ST_FILE, "TF_CHARACTERS")
    out: list[State] = []
    for r in rows:
        base = (r.get("folder") or "").strip().lower()
        # ST teen folders often encode female teen as <base>GB (e.g. bradGB).
        if base.endswith("gb") and len(base) > 2:
            base = base[: -2]

        out.append(
            State(
                pack="characters_ST",
                name=r["name"],
                gender=r.get("gender"),
                age=r.get("age"),
                folder=r.get("folder"),
                genderswap=r.get("genderswap"),
                ageswap=r.get("ageswap"),
                gender_age_swap=r.get("gender_age_swap"),
                base=base,
            )
        )
    return out


def flatten_shorts_rows() -> list[State]:
    rows: list[dict[str, Any]] = _literal_eval_assignment(SHORTS_FILE, "TF_CHARACTERS")
    out: list[State] = []
    for r in rows:
        ageswap = r.get("ageswap")
        base = None
        if isinstance(ageswap, str) and ageswap.startswith(LINK_STV_PREFIX):
            tail = ageswap.split("/")[-1].strip()
            base = parse_variant_base_from_name(tail)
        base = base or (r.get("folder") or "").strip().lower()

        out.append(
            State(
                pack="characters_Shorts",
                name=r["name"],
                gender=r.get("gender"),
                age=r.get("age"),
                folder=r.get("folder"),
                genderswap=r.get("genderswap"),
                ageswap=r.get("ageswap"),
                gender_age_swap=r.get("gender_age_swap"),
                base=base,
            )
        )
    return out


def flatten_stvariants_rows() -> list[State]:
    blocks: list[dict[str, Any]] = _literal_eval_assignment(STV_FILE, "CHARACTER_BLOCKS")
    out: list[State] = []
    for blk in blocks:
        variants: list[dict[str, Any]] = blk.get("variants", []) or []
        for v in variants:
            variant_name = v["name"]
            base = parse_variant_base_from_name(variant_name)
            out.append(
                State(
                    pack="characters_STVariants",
                    name=variant_name,
                    gender=v.get("gender"),
                    age=v.get("age"),
                    folder=v.get("folder"),
                    genderswap=v.get("genderswap"),
                    ageswap=v.get("ageswap"),
                    gender_age_swap=v.get("gender_age_swap"),
                    base=base or "",
                )
            )
    return out


def resolve_authoring_link(link_value: str, src_pack: str, *, indices: dict[str, Any]) -> State | None:
    """
    Resolve an authoring link string to a State.

    This matches the *authoring* strings inside the pack files:
    - `characters_STVariants/<VariantName>` -> STVariants state by name
    - `characters_ST/<folder>` -> ST state by folder
    - `characters_Shorts/<folder>` -> Shorts state by folder
    - unqualified `variants_*` (no '/') -> STVariants state by folder
    """
    link_value = link_value.strip()
    if not link_value:
        return None

    if link_value.startswith(LINK_STV_PREFIX):
        tail = link_value.split("/")[-1].strip()
        return indices["stv_by_name"].get(tail)

    if link_value.startswith(LINK_ST_PREFIX):
        tail = link_value.split("/")[-1].strip()
        return indices["st_by_folder"].get(_norm_folder_token(tail))

    if link_value.startswith(LINK_SHORTS_PREFIX):
        tail = link_value.split("/")[-1].strip()
        return indices["shorts_by_folder"].get(_norm_folder_token(tail))

    # Folder-only inside STVariants uses `variants_*` folder tokens.
    if "/" not in link_value:
        return indices["stv_by_folder"].get(_norm_folder_token(link_value))

    # Anything else isn't part of the simple authoring format.
    return None


def build_indices(states: Iterable[State]) -> dict[str, Any]:
    st_by_folder: dict[str, State] = {}
    stv_by_name: dict[str, State] = {}
    stv_by_folder: dict[str, State] = {}
    shorts_by_folder: dict[str, State] = {}

    for s in states:
        if s.pack == "characters_ST":
            st_by_folder[_norm_folder_token(s.folder)] = s
        elif s.pack == "characters_STVariants":
            stv_by_name[s.name] = s
            stv_by_folder[_norm_folder_token(s.folder)] = s
        elif s.pack == "characters_Shorts":
            shorts_by_folder[_norm_folder_token(s.folder)] = s
    return {
        "st_by_folder": st_by_folder,
        "stv_by_name": stv_by_name,
        "stv_by_folder": stv_by_folder,
        "shorts_by_folder": shorts_by_folder,
    }


def expected_corner_states(states_by_key: dict[tuple[str, str, str], list[State]], base: str, gender: str, age: str) -> State | None:
    """
    Pick the expected state instance using the same heuristic as earlier audits:
    - prefer characters_ST for teen corners
    - prefer characters_STVariants for adult corners
    - if the preferred pack isn't present, pick the first state in insertion order
    """
    lst = states_by_key.get((base, gender, age), [])
    if not lst:
        return None
    prefer = "characters_ST" if age == "teen" else "characters_STVariants"
    for x in lst:
        if x.pack == prefer:
            return x
    return lst[0]


def iter_states_in_pack_order(st: list[State], shorts: list[State], stv: list[State]) -> list[State]:
    # Insertion order matters for the "fallback to first" behavior in expected_corner_states.
    return [*st, *shorts, *stv]


def audit_orphan_stvariants_tails(st: list[State], shorts: list[State], stv: list[State]) -> list[str]:
    stv_names = {s.name for s in stv if s.pack == "characters_STVariants"}

    tails: set[str] = set()
    for src in [*st, *shorts]:
        for link in (src.genderswap, src.ageswap, src.gender_age_swap):
            if isinstance(link, str) and link.startswith(LINK_STV_PREFIX):
                tails.add(link.split("/")[-1].strip())

    missing = sorted(t for t in tails if t not in stv_names)
    return missing


def load_character_blocks() -> list[dict[str, Any]]:
    return _literal_eval_assignment(STV_FILE, "CHARACTER_BLOCKS")


def build_block_name_sets(blocks: list[dict[str, Any]]) -> dict[str, set[str]]:
    """
    Map ST `default_character` slug (e.g. characters_ST/brad -> brad) to the
    set of canonical variant `name` values in that CHARACTER_BLOCKS entry.
    """
    out: dict[str, set[str]] = {}
    for blk in blocks:
        dc = blk.get("default_character") or ""
        if not isinstance(dc, str) or not dc.startswith(LINK_ST_PREFIX):
            continue
        slug = dc.split("/")[-1].strip().lower()
        names = {
            v["name"]
            for v in (blk.get("variants") or [])
            if isinstance(v, dict) and v.get("name")
        }
        out.setdefault(slug, set()).update(names)
    return out


def iter_all_string_values(obj: Any) -> Iterable[str]:
    if isinstance(obj, str):
        yield obj
    elif isinstance(obj, dict):
        for v in obj.values():
            yield from iter_all_string_values(v)
    elif isinstance(obj, list):
        for item in obj:
            yield from iter_all_string_values(item)


def audit_block_membership_g(
    st_rows: list[State], shorts_rows: list[State], block_sets: dict[str, set[str]]
) -> list[str]:
    """Plan G: ST + Shorts rows — every STV tail must be in that base's block name set."""
    faults: list[str] = []
    for src in [*st_rows, *shorts_rows]:
        b = src.base
        if not b:
            continue
        names = block_sets.get(b)
        for field in ("genderswap", "ageswap", "gender_age_swap"):
            link = getattr(src, field)
            if not isinstance(link, str) or not link.startswith(LINK_STV_PREFIX):
                continue
            tail = link.split("/")[-1].strip()
            if names is None:
                faults.append(
                    f"G: {src.pack} {src.name!r} base={b!r} {field}={link!r} "
                    f"(no CHARACTER_BLOCKS default_character for this base)"
                )
            elif tail not in names:
                faults.append(
                    f"G: {src.pack} {src.name!r} base={b!r} {field} tail {tail!r} "
                    f"not in block name set"
                )
    return faults


def audit_bogus_mixed_family_h(
    st_rows: list[State], shorts_rows: list[State], block_sets: dict[str, set[str]]
) -> list[str]:
    """
    Plan H: if a link uses a `...GBAP` / `...GBAR` tail, the base block must
    actually define that suffix family (guards AP vs AR mixed authoring).
    """
    faults: list[str] = []
    for src in [*st_rows, *shorts_rows]:
        b = src.base
        if not b:
            continue
        names = block_sets.get(b) or set()
        for field in ("genderswap", "ageswap", "gender_age_swap"):
            link = getattr(src, field)
            if not isinstance(link, str) or not link.startswith(LINK_STV_PREFIX):
                continue
            tail = link.split("/")[-1].strip()
            if tail.endswith("GBAP") and names and not any(n.endswith("GBAP") for n in names):
                faults.append(
                    f"H: {src.pack} {src.name!r} base={b!r} {field}={link!r} "
                    f"(block has no *GBAP variant)"
                )
            if tail.endswith("GBAR") and names and not any(n.endswith("GBAR") for n in names):
                faults.append(
                    f"H: {src.pack} {src.name!r} base={b!r} {field}={link!r} "
                    f"(block has no *GBAR variant)"
                )
    return faults


def audit_no_stv_cross_refs_i(blocks: list[dict[str, Any]]) -> list[str]:
    """Plan I: STVariants pack must not embed characters_STVariants/... strings."""
    faults: list[str] = []
    for blk in blocks:
        for s in iter_all_string_values(blk):
            if LINK_STV_PREFIX in s:
                faults.append(f"I: forbidden nested STV path in pack literal: {s!r}")
    return faults


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit ST / STVariants / Shorts character links (5.0 packs).")
    parser.add_argument(
        "--ci",
        action="store_true",
        help="Exit with code 1 if any audit reports failures (for CI pipelines).",
    )
    args = parser.parse_args()
    st_rows = flatten_st_rows()
    shorts_rows = flatten_shorts_rows()
    stv_rows = flatten_stvariants_rows()
    character_blocks = load_character_blocks()
    block_name_sets = build_block_name_sets(character_blocks)

    all_states = iter_states_in_pack_order(st_rows, shorts_rows, stv_rows)
    indices = build_indices(all_states)

    # Track which corners exist so we can treat "missing expected corners"
    # as graph-incomplete only when a base clearly represents a quad.
    teen_male_exists: dict[str, bool] = {}
    teen_female_exists: dict[str, bool] = {}
    for s in all_states:
        if s.age == "teen":
            if s.gender == "male":
                teen_male_exists[s.base] = True
            if s.gender == "female":
                teen_female_exists[s.base] = True

    # --- 1) Orphan tail audit ---
    missing_tails = audit_orphan_stvariants_tails(st_rows, shorts_rows, stv_rows)
    print(f"STVariants tail audit: missing_tails={len(missing_tails)}")
    for t in missing_tails:
        print(f"  - {t}")

    g_faults = audit_block_membership_g(st_rows, shorts_rows, block_name_sets)
    print(f"Block membership (G): faults={len(g_faults)}")
    for line in g_faults[:80]:
        print(f"  - {line}")
    if len(g_faults) > 80:
        print(f"  ... ({len(g_faults) - 80} more)")

    h_faults = audit_bogus_mixed_family_h(st_rows, shorts_rows, block_name_sets)
    print(f"Mixed family suffix (H): faults={len(h_faults)}")
    for line in h_faults[:80]:
        print(f"  - {line}")
    if len(h_faults) > 80:
        print(f"  ... ({len(h_faults) - 80} more)")

    i_faults = audit_no_stv_cross_refs_i(character_blocks)
    print(f"STVariants nested STV paths (I): faults={len(i_faults)}")
    for line in i_faults:
        print(f"  - {line}")

    # --- 2) Swap semantics + teen quad ---
    by_key: dict[tuple[str, str, str], list[State]] = {}
    for s in all_states:
        by_key.setdefault((s.base, s.gender, s.age), []).append(s)

    # --- Optional teen-quad semantic audit (genderswap + ageswap only) ---
    teen_quad_faults: list[str] = []
    st_by_base_gender_age: dict[tuple[str, str], State] = {}
    for s in st_rows:
        if s.age == "teen" and s.gender in ("male", "female"):
            st_by_base_gender_age[(s.base, s.gender)] = s

    for (base, _) in list(st_by_base_gender_age.keys()):
        male = st_by_base_gender_age.get((base, "male"))
        female = st_by_base_gender_age.get((base, "female"))
        if not male or not female:
            continue

        male_to_female = male.genderswap
        if male_to_female:
            resolved = resolve_authoring_link(male_to_female, male.pack, indices=indices)
            if resolved is not female:
                teen_quad_faults.append(f"{base}: ST male teen genderswap not paired (actual={male_to_female!r})")

        female_to_male = female.genderswap
        if female_to_male:
            resolved = resolve_authoring_link(female_to_male, female.pack, indices=indices)
            if resolved is not male:
                teen_quad_faults.append(f"{base}: ST female teen genderswap not paired (actual={female_to_male!r})")

        # ageswap: teen -> adult, same gender
        expected_adult_male = expected_corner_states(by_key, base, "male", "adult")
        if expected_adult_male and male.ageswap:
            resolved = resolve_authoring_link(male.ageswap, male.pack, indices=indices)
            if resolved is not expected_adult_male:
                teen_quad_faults.append(f"{base}: ST male teen ageswap not reaching expected adult male")

        expected_adult_female = expected_corner_states(by_key, base, "female", "adult")
        if expected_adult_female and female.ageswap:
            resolved = resolve_authoring_link(female.ageswap, female.pack, indices=indices)
            if resolved is not expected_adult_female:
                teen_quad_faults.append(f"{base}: ST female teen ageswap not reaching expected adult female")

        # Plan F: `characters_ST/`-genderswap pairs — ageswap vs partner's gender_age_swap cross-swapped.
        mgs = male.genderswap
        fgs = female.genderswap
        if (
            isinstance(mgs, str)
            and mgs.startswith(LINK_ST_PREFIX)
            and isinstance(fgs, str)
            and fgs.startswith(LINK_ST_PREFIX)
        ):
            r_m_to_f = resolve_authoring_link(mgs, male.pack, indices=indices)
            r_f_to_m = resolve_authoring_link(fgs, female.pack, indices=indices)
            if r_m_to_f is female and r_f_to_m is male:
                ma, mfga = male.ageswap, male.gender_age_swap
                fa, ffga = female.ageswap, female.gender_age_swap
                r_ma = (
                    resolve_authoring_link(ma, male.pack, indices=indices)
                    if isinstance(ma, str) and ma.strip()
                    else None
                )
                r_mfga = (
                    resolve_authoring_link(mfga, male.pack, indices=indices)
                    if isinstance(mfga, str) and mfga.strip()
                    else None
                )
                r_fa = (
                    resolve_authoring_link(fa, female.pack, indices=indices)
                    if isinstance(fa, str) and fa.strip()
                    else None
                )
                r_ffga = (
                    resolve_authoring_link(ffga, female.pack, indices=indices)
                    if isinstance(ffga, str) and ffga.strip()
                    else None
                )
                if r_ma and r_ffga and r_ma is not r_ffga:
                    teen_quad_faults.append(
                        f"{base}: teen ST-pair cross-swap: male.ageswap resolves to {r_ma.name!r} "
                        f"but female.gender_age_swap resolves to {r_ffga.name!r}"
                    )
                if r_mfga and r_fa and r_mfga is not r_fa:
                    teen_quad_faults.append(
                        f"{base}: teen ST-pair cross-swap: male.gender_age_swap resolves to {r_mfga.name!r} "
                        f"but female.ageswap resolves to {r_fa.name!r}"
                    )

    print(f"Teen quad check (genderswap+ageswap+cross-swap): faults={len(teen_quad_faults)}")
    for line in teen_quad_faults[:40]:
        print(f"  - {line}")

    faults: list[tuple[str, str, str, str, str | None, str | None, str]] = []
    # columns: base, src_node, field, kind, actual_link, suggested_link, expected_node_name

    def suggested_link_for(src: State, expected: State) -> str:
        if src.pack in ("characters_ST", "characters_Shorts"):
            if expected.pack == "characters_STVariants":
                return f"{LINK_STV_PREFIX}{expected.name}"
            if expected.pack == "characters_ST":
                return f"{LINK_ST_PREFIX}{expected.folder}"
            if expected.pack == "characters_Shorts":
                return f"{LINK_SHORTS_PREFIX}{expected.folder}"
        if src.pack == "characters_STVariants":
            if expected.pack == "characters_STVariants":
                # STVariants internal edges are folder-only `variants_*`
                return expected.folder
            if expected.pack == "characters_ST":
                return f"{LINK_ST_PREFIX}{expected.folder}"
            if expected.pack == "characters_Shorts":
                return f"{LINK_SHORTS_PREFIX}{expected.folder}"
        return ""

    def expected_for_field(src: State, field: str) -> tuple[str, str] | None:
        opp_gender = "female" if src.gender == "male" else "male" if src.gender == "female" else None
        opp_age = "adult" if src.age == "teen" else "teen" if src.age == "adult" else None
        if field == "genderswap":
            return (opp_gender, src.age)
        if field == "ageswap":
            return (src.gender, opp_age)
        if field == "gender_age_swap":
            return (opp_gender, opp_age)
        return None

    def synthesize_missing_stvariants_variant_name(base: str, expected_gender: str) -> str | None:
        """
        Synthesize the missing STVariants adult identity name for incomplete quads.

        This is primarily to cover the CorneliaGBAP gap where STVariants
        contains only `CorneliaAP` (adult female) and does not contain the
        adult-male corner identity.
        """
        adult_variants = [v for v in stv_rows if v.base == base and v.age == "adult"]
        if not adult_variants:
            return None

        # Only try the AP/GBAP complement logic for now.
        present_suffixes: set[str] = set()
        prefix: str | None = None
        for v in adult_variants:
            nm = v.name
            if nm.endswith("GBAP"):
                present_suffixes.add("GBAP")
                prefix = nm[: -len("GBAP")]
            elif nm.endswith("AP"):
                present_suffixes.add("AP")
                prefix = nm[: -len("AP")]

        if not prefix:
            return None

        # If AP exists but GBAP is missing, the missing gender is the complement.
        if "AP" in present_suffixes and "GBAP" not in present_suffixes:
            return prefix + "GBAP"
        if "GBAP" in present_suffixes and "AP" not in present_suffixes:
            return prefix + "AP"

        return None

    for src in all_states:
        if src.gender not in ("male", "female") or src.age not in ("teen", "adult"):
            continue
        base = src.base
        for field in ("genderswap", "ageswap", "gender_age_swap"):
            actual_link = getattr(src, field)
            if actual_link is not None and isinstance(actual_link, str) and not actual_link.strip():
                actual_link = None

            exp_gender_age = expected_for_field(src, field)
            if exp_gender_age is None:
                continue
            exp_gender, exp_age = exp_gender_age
            if exp_gender is None or exp_age is None:
                continue

            expected_state = expected_corner_states(by_key, base, exp_gender, exp_age)
            if expected_state is None:
                # Treat missing expected adult corners as graph-incomplete
                # only for bases that have both teen corners (quad-shaped
                # graphs), e.g. Cornelia missing CorneliaGBAP.
                should_flag_incomplete = (
                    exp_age == "adult"
                    and teen_male_exists.get(base, False)
                    and teen_female_exists.get(base, False)
                )
                if not should_flag_incomplete:
                    continue

                missing_variant_name = synthesize_missing_stvariants_variant_name(base, exp_gender)
                if not missing_variant_name:
                    continue

                # Use an explicit fully-qualified link form in the report.
                suggested = f"{LINK_STV_PREFIX}{missing_variant_name}"
                faults.append((base, src.pack, src.name, field, actual_link, suggested, missing_variant_name))
                continue

            actual_resolved = None
            if isinstance(actual_link, str) and actual_link.strip():
                actual_resolved = resolve_authoring_link(actual_link, src.pack, indices=indices)

            # Determine whether current edge matches the expected node.
            ok = actual_resolved is expected_state
            if not ok:
                suggested = ""
                suggested = suggested_link_for(src, expected_state)
                faults.append((base, src.pack, src.name, field, actual_link, suggested, expected_state.name))

    print(f"Swap semantics audit faults={len(faults)}")
    for row in faults:
        base, src_pack, src_name, field, actual_link, suggested_link, expected_name = row
        print(
            f"{base} | {src_pack}/{src_name} | {field} | actual={actual_link!r} | "
            f"expected_node={expected_name} | suggested={suggested_link!r}"
        )

    if args.ci:
        ci_failed = bool(
            missing_tails
            or g_faults
            or h_faults
            or i_faults
            or teen_quad_faults
            or faults
        )
        if ci_failed:
            print("\n--ci: one or more audits failed (exit 1).")
            sys.exit(1)
        print("\n--ci: all audits passed (exit 0).")


if __name__ == "__main__":
    main()

