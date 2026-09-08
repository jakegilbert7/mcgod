#!/usr/bin/env python3
"""Maintain the human-verifiable view of canonical world structures.

    python3 render_structures.py --clean
    python3 render_structures.py --id s_ab12cd34ef56
    python3 render_structures.py --list

``--clean`` archives every top-level PNG that is not backed by a current structure before
rendering.  It never deletes the old evidence.  Each current object then gets exactly one
four-view, entity-free sheet fetched from the live world and named with its current label.
"""

from __future__ import annotations

import argparse
import datetime
from pathlib import Path

from classify import render_of
from render import REVIEW_DIR, save_for_review
from store import Store
from structures import audit, current, facts_of


def archive_stale(valid_ids: set[str], folder: Path = REVIEW_DIR) -> list[Path]:
    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    archive = folder / "archive" / stamp
    moved = []
    for path in sorted(folder.glob("*.png")):
        sid = path.name.split("__", 1)[0]
        if sid in valid_ids:
            continue
        archive.mkdir(parents=True, exist_ok=True)
        target = archive / path.name
        # Preserve both files if a prior cleanup used the same second.
        if target.exists():
            target = archive / f"{path.stem}-{len(moved)}{path.suffix}"
        path.replace(target)
        moved.append(target)
    return moved


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--clean", action="store_true",
                        help="archive renders not backed by current structure rows")
    parser.add_argument("--list", action="store_true", help="list state without fetching")
    parser.add_argument("--check", action="store_true",
                        help="verify database and one-render-per-object invariants")
    parser.add_argument("--id", action="append", default=[], help="render only this id")
    args = parser.parse_args(argv)

    store = Store()
    rows = current(store)
    valid = {row["id"] for row in rows}
    if args.check:
        from masses import audit as audit_masses
        issues = audit(store) + audit_masses(store)
        rendered = {}
        for path in REVIEW_DIR.glob("*.png"):
            rendered.setdefault(path.name.split("__", 1)[0], []).append(path)
        for sid in sorted(valid - set(rendered)):
            issues.append(f"{sid}: no top-level review render")
        for sid in sorted(set(rendered) - valid):
            issues.append(f"{sid}: render has no current structure")
        for sid, paths in rendered.items():
            if len(paths) != 1:
                issues.append(f"{sid}: {len(paths)} current renders, expected one")
        if issues:
            for issue in issues:
                print(f"ERROR  {issue}")
            store.close()
            return 1
        from masses import current as current_masses
        print(f"ok: {len(rows)} canonical structures, one render each, over "
              f"{len(current_masses(store))} built masses")
        store.close()
        return 0
    if args.clean:
        moved = archive_stale(valid)
        print(f"archived {len(moved)} stale render(s)")

    if args.id:
        wanted = set(args.id)
        unknown = wanted - valid
        if unknown:
            print("unknown current structure(s): " + ", ".join(sorted(unknown)))
        rows = [row for row in rows if row["id"] in wanted]

    rows.sort(key=lambda row: (-facts_of(row).get("blocks", 0), row["id"]))
    if args.list:
        for row in rows:
            name = store.structure_name(row["id"])
            facts = facts_of(row)
            print(f"{row['id']}  {facts.get('blocks', '?'):>5} blocks  "
                  f"{name['object'] if name else 'unnamed'}")
        store.close()
        return 0

    written, failed = 0, 0
    for row in rows:
        name = store.structure_name(row["id"])
        label = name["object"] if name else "unnamed"
        confidence = float(name["confidence"]) if name else None
        picture = render_of(row)
        if not picture:
            failed += 1
            print(f"FAILED  {row['id']} ({label}): live voxel read unavailable")
            continue
        path = save_for_review(picture, row["id"], label, confidence)
        if path:
            written += 1
            print(f"wrote   {path.relative_to(REVIEW_DIR.parent)}")
        else:
            failed += 1
    store.close()
    print(f"{written} current render(s) written, {failed} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
