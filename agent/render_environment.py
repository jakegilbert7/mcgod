#!/usr/bin/env python3
"""Render the live terrain around a player for human verification.

    .venv/bin/python render_environment.py
    .venv/bin/python render_environment.py --pos 173 65 311 --span 40

The default point is the most recently observed player position. Output lives below
``renders/environments`` so the one-render-per-canonical-structure audit remains exact.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from god import terrain_view
from environment import summarize_voxels
from store import Store


def latest_player(store: Store):
    rows = sorted(store.all("players"), key=lambda row: row["last_seen_tick"] or 0,
                  reverse=True)
    for row in rows:
        try:
            pos = json.loads(row["profile"] or "{}").get("last_pos")
        except (TypeError, ValueError):
            continue
        if pos:
            return row["dim"], pos
    return None, None


async def run(args) -> int:
    store = Store()
    dim, pos = latest_player(store)
    store.close()
    dim = args.dim or dim or "overworld"
    pos = args.pos or pos
    if not pos:
        print("no player position is recorded; pass --pos X Y Z")
        return 1
    picture, detail, voxels = await terrain_view(dim, pos, args.span)
    if not picture:
        print(f"environment render failed: {detail}")
        return 1
    folder = Path(__file__).parent / "renders" / "environments"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"current-{dim}.png"
    path.write_bytes(picture)
    print(f"wrote {path.relative_to(Path(__file__).parent)} ({detail})")
    print(summarize_voxels(voxels, pos))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pos", nargs=3, type=int, metavar=("X", "Y", "Z"))
    parser.add_argument("--dim")
    parser.add_argument("--span", type=int, default=40)
    return asyncio.run(run(parser.parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
