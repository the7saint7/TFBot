import json
from pathlib import Path

RATING_MAP = {
    "1": "common",
    "2": "rare",
    "3": "epic",
}


def prompt_choice(display_name: str, current: str) -> str:
    while True:
        choice = input(f"{display_name} [{current}] -> ").strip()
        if choice == "":
            return current
        rarity = RATING_MAP.get(choice)
        if rarity:
            return rarity
        print("Please enter 1 (common), 2 (rare), 3 (epic), or press Enter to keep current.")


def main() -> None:
    config_path = Path("gacha_config.json")
    if not config_path.exists():
        raise SystemExit("gacha_config.json not found. Run script from repo root.")

    data = json.loads(config_path.read_text(encoding="utf-8"))
    characters = data.get("characters")
    if not isinstance(characters, dict):
        raise SystemExit("Invalid config: 'characters' must be an object.")

    print("Assign rarity for each character: 1=common, 2=rare, 3=epic (Enter to keep current).")

    for slug, entry in sorted(characters.items()):
        if not isinstance(entry, dict):
            continue
        display = entry.get("display_name") or slug.title()
        current = entry.get("rarity") or "common"
        entry["rarity"] = prompt_choice(display, current)

    config_path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("Updated", config_path)


if __name__ == "__main__":
    main()
