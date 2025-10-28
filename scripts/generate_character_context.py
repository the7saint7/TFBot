"""
Utility script to extract character descriptions from the Student Transfer ``char_db.py``
file and emit a minimal JSON mapping of ``full name -> description``.

Usage (from repository root):

    python scripts/generate_character_context.py \
        --char-db "C:/WorkStuff/Downloads/RenPy/renpy-8.2.1-sdk/student-transfer/game/char_db.py" \
        --output data/character_context.json

The script executes the Ren'Py character database in an isolated namespace with a stub
``persistent`` object so the default description paths are used. Only the character
names and their default descriptions are retained.
"""

from __future__ import annotations

import argparse
import json
import types
from pathlib import Path
from typing import Dict


class _PersistentStub:
    """Return ``False`` for any attribute lookup."""

    def __getattr__(self, name: str) -> bool:
        return False

    def __setattr__(self, name: str, value) -> None:  # pragma: no cover - unused but explicit
        object.__setattr__(self, name, value)


def _load_character_descriptions(char_db_path: Path) -> Dict[str, str]:
    persistent = _PersistentStub()
    renpy_stub = types.SimpleNamespace(store=types.SimpleNamespace(persistent=persistent))

    namespace: Dict[str, object] = {
        "__file__": str(char_db_path),
        "__name__": "char_db_extractor",
        "persistent": persistent,
        "renpy": renpy_stub,
    }

    code = compile(char_db_path.read_text(encoding="utf-8"), str(char_db_path), "exec")
    exec(code, namespace)  # pylint: disable=exec-used

    profile_cls = namespace.get("Profile")
    if not isinstance(profile_cls, type):
        raise RuntimeError("char_db.py did not define a Profile base class.")

    result: Dict[str, str] = {}
    for obj in namespace.values():
        if not isinstance(obj, type):
            continue
        if obj is profile_cls or not issubclass(obj, profile_cls):
            continue
        try:
            instance = obj()
        except Exception:  # pylint: disable=broad-except
            continue

        full_name = getattr(instance, "name", None)
        if not isinstance(full_name, str) or not full_name.strip():
            continue

        desc = getattr(instance, "description", None)
        if callable(desc):  # guard against stray callables
            try:
                desc = desc()
            except Exception:
                continue
        if not isinstance(desc, str):
            continue

        result[full_name.strip()] = " ".join(desc.split())
    return dict(sorted(result.items()))


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract Student Transfer character descriptions.")
    parser.add_argument(
        "--char-db",
        type=Path,
        required=True,
        help="Path to student-transfer/game/char_db.py from the Ren'Py project.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Destination JSON file for the Name -> Description mapping.",
    )
    args = parser.parse_args()

    char_db_path: Path = args.char_db.expanduser().resolve()
    if not char_db_path.exists():
        raise SystemExit(f"char_db file not found: {char_db_path}")

    descriptions = _load_character_descriptions(char_db_path)
    if not descriptions:
        raise SystemExit("No character descriptions were extracted.")

    output_path: Path = args.output.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(descriptions, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {len(descriptions)} character descriptions to {output_path}")


if __name__ == "__main__":
    main()
