#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Tuple


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Flip pitch sign for spatial.pitch_deg and bfov.pitch_deg in all metadata.json files."
    )
    p.add_argument(
        "--root",
        default="/workspace/wcp/pano_data_generation/base_data_generation/results_syn/metadatabfov",
        help="Root directory to search recursively for metadata.json",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Only report what would change, do not write files.",
    )
    p.add_argument(
        "--backup-ext",
        default="",
        help="Optional backup suffix, e.g. .bak . If set, writes a backup before modifying each file.",
    )
    p.add_argument(
        "--indent",
        type=int,
        default=2,
        help="JSON indent for rewritten files.",
    )
    return p.parse_args()


def maybe_flip_pitch(d: Dict[str, Any], key_path: str) -> Tuple[bool, Any, Any]:
    """
    key_path is one of:
      - spatial.pitch_deg
      - bfov.pitch_deg
    """
    head, tail = key_path.split(".")
    sub = d.get(head)
    if not isinstance(sub, dict):
        return False, None, None
    if tail not in sub:
        return False, None, None

    old_val = sub[tail]
    if not isinstance(old_val, (int, float)):
        return False, old_val, old_val

    new_val = -float(old_val)
    # keep int if original was int-like and exact
    if isinstance(old_val, int) and float(new_val).is_integer():
        new_val = int(new_val)

    sub[tail] = new_val
    return True, old_val, new_val


def process_file(path: Path, dry_run: bool = False, backup_ext: str = "", indent: int = 2) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    entities = data.get("entities")
    if not isinstance(entities, list):
        return {
            "file": str(path),
            "changed": False,
            "reason": "missing_or_invalid_entities",
            "entity_changes": 0,
        }

    entity_changes = 0
    details = []

    for idx, ent in enumerate(entities):
        if not isinstance(ent, dict):
            continue

        ent_changed = False
        entry = {
            "entity_index": idx,
            "entity_id": ent.get("entity_id"),
        }

        changed, old_v, new_v = maybe_flip_pitch(ent, "spatial.pitch_deg")
        if changed:
            ent_changed = True
            entry["spatial.pitch_deg"] = {"old": old_v, "new": new_v}

        changed, old_v, new_v = maybe_flip_pitch(ent, "bfov.pitch_deg")
        if changed:
            ent_changed = True
            entry["bfov.pitch_deg"] = {"old": old_v, "new": new_v}

        if ent_changed:
            entity_changes += 1
            details.append(entry)

    if entity_changes > 0 and not dry_run:
        if backup_ext:
            backup_path = path.with_name(path.name + backup_ext)
            backup_path.write_text(path.read_text(encoding="utf-8"), encoding="utf-8")
        with path.open("w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=indent)
            f.write("\n")

    return {
        "file": str(path),
        "changed": entity_changes > 0,
        "entity_changes": entity_changes,
        "details": details,
    }


def main() -> int:
    args = parse_args()
    root = Path(args.root)

    if not root.exists():
        raise SystemExit(f"Root does not exist: {root}")

    metadata_files = sorted(root.rglob("metadata.json"))
    if not metadata_files:
        print(json.dumps({
            "root": str(root),
            "metadata_files_found": 0,
            "changed_files": 0,
            "changed_entities": 0,
            "dry_run": args.dry_run,
        }, ensure_ascii=False, indent=2))
        return 0

    changed_files = 0
    changed_entities = 0
    examples = []

    for path in metadata_files:
        result = process_file(
            path,
            dry_run=args.dry_run,
            backup_ext=args.backup_ext,
            indent=args.indent,
        )
        if result["changed"]:
            changed_files += 1
            changed_entities += int(result["entity_changes"])
            if len(examples) < 10:
                examples.append(result)

    print(json.dumps({
        "root": str(root),
        "metadata_files_found": len(metadata_files),
        "changed_files": changed_files,
        "changed_entities": changed_entities,
        "dry_run": args.dry_run,
        "backup_ext": args.backup_ext,
        "examples": examples,
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
