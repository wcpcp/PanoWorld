#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

from _common import load_cfg

from erp_meta.io_utils import dump_json, ensure_dir, load_json
from erp_meta.relations import build_entity_contexts, build_relations


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", required=True)
    ap.add_argument("--entities_json", required=True)
    ap.add_argument("--out_json", required=True)
    ap.add_argument("--near_thr", type=float, default=0.35)
    args = ap.parse_args()

    _ = load_cfg(args.cfg)
    obj = load_json(args.entities_json)
    rels = build_relations(obj["entities"], near_thr_rad=args.near_thr)
    contexts = build_entity_contexts(obj["entities"])
    dump_json(
        args.out_json,
        {
            "scene_id": obj["scene_id"],
            "viewpoint_id": obj["viewpoint_id"],
            "entity_contexts": contexts,
            "relations": rels,
        },
    )
    print(f"relations={len(rels)} -> {args.out_json}")


if __name__ == "__main__":
    main()
