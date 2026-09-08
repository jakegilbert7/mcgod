"""Run the segmentation pipeline over a whole area, with no player and no event history.

At runtime the god never sweeps the world, and that is deliberate: Minecraft does not tick
unloaded chunks, so re-reading a region nobody has visited can only return what was already
known. Scanning is for correction, not discovery.

Testing is the exception. A downloaded map or a generated village is exactly the case the
event-driven path cannot reach — nobody built any of it while we were watching, so no
disturbance ever marks it, and the pipeline has nothing to look at. This walks a grid over
an area instead and re-segments every patch that has anything built in it, which is the
only way to find out how segmentation behaves on buildings we did not watch go up.

It is a test harness. It must not become how the god works.

    python3 survey.py --centre 170 66 305 --radius 120
    python3 survey.py --centre 0 70 0 --radius 300 --spacing 40 --dry-run
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import time

from scan import request_voxels, to_voxels
from segment import built_blocks
from store import Store

URL = "ws://127.0.0.1:8765"


async def probe(dim: str, centre, span: int) -> int:
    """How much built material is in one patch. Cheap enough to run over a grid.

    Chunks load on demand for a scan, so this works with nobody in the world — but it does
    mean a wide survey pulls a lot of chunks through the server, which is why the caller
    paces itself.
    """
    x, y, z = centre
    try:
        result = await request_voxels(URL, dim, [x - span, max(-64, y - 16), z - span],
                                      [x + span, y + 30, z + span])
    except Exception:
        return 0
    return len(built_blocks(to_voxels(result))) if result.get("ok") else 0


async def survey(god, dim: str, centre, radius: int, spacing: int, span: int,
                 floor: int, dry_run: bool, pause: float,
                 origin: str = "world_generated") -> dict:
    """Walk a grid, re-segmenting wherever there is enough built material to be worth it."""
    cx, cy, cz = centre
    cells = [(x, cy, z)
             for x in range(cx - radius, cx + radius + 1, spacing)
             for z in range(cz - radius, cz + radius + 1, spacing)]
    report = {"cells": len(cells), "with_builds": 0, "segmented": 0, "structures": 0,
              "seconds": 0.0, "hits": []}
    started = time.time()

    for i, cell in enumerate(cells, 1):
        found = await probe(dim, cell, span)
        marker = f"[{i}/{len(cells)}] {cell[0]:>6},{cell[2]:<6}"
        if found < floor:
            print(f"{marker}  {found:5d} built  —")
            continue
        report["with_builds"] += 1
        report["hits"].append({"cell": list(cell), "built": found})
        if dry_run:
            print(f"{marker}  {found:5d} built  would segment")
            continue
        print(f"{marker}  {found:5d} built  segmenting...", flush=True)
        try:
            wrote = await god.resegment(dim, list(cell), span=span,
                                        origin=origin)
        except Exception as e:
            print(f"    failed: {type(e).__name__}: {e}")
            continue
        report["segmented"] += 1
        report["structures"] += wrote
        await asyncio.sleep(pause)

    report["seconds"] = round(time.time() - started, 1)
    return report


def summarise(store: Store, dim: str, origin: str = "world_generated") -> None:
    """Everything on record, largest first — the thing you actually read afterwards."""
    from structures import current, current_generated
    rows = (current_generated(store, dim) if origin == "world_generated"
            else current(store, dim))
    facts = []
    for row in rows:
        value = json.loads(row["value"] or "{}")
        if (value.get("blocks") or 0) < 25:
            continue
        named = store.structure_name(row["id"])
        facts.append((value["blocks"], value.get("dims", ""),
                      named["object"] if named else "-",
                      (row["min_x"], row["min_y"], row["min_z"])))
    facts.sort(reverse=True)
    noun = "generated features" if origin == "world_generated" else "player structures"
    print(f"\n{len(facts)} {noun} on record\n")
    for blocks, dims, name, at in facts:
        print(f"  {blocks:5d}b {dims:>10}  ({at[0]},{at[1]},{at[2]})  {name}")


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--centre", nargs=3, type=int, required=True,
                        metavar=("X", "Y", "Z"))
    parser.add_argument("--radius", type=int, default=120,
                        help="how far out from the centre to walk, in blocks")
    parser.add_argument("--spacing", type=int, default=36,
                        help="distance between patch centres; below the span so patches "
                             "overlap and a building on a seam is still seen whole")
    parser.add_argument("--span", type=int, default=26, help="half-width of each patch")
    parser.add_argument("--floor", type=int, default=40,
                        help="built blocks a patch needs before it is worth segmenting")
    parser.add_argument("--dim", default="overworld")
    parser.add_argument("--pause", type=float, default=1.0,
                        help="seconds between patches, to leave the server room")
    parser.add_argument("--dry-run", action="store_true",
                        help="probe only: report where the builds are, segment nothing")
    parser.add_argument("--plan-view", action="store_true",
                        help="include the slower orthographic view for an explicit eval")
    parser.add_argument("--origin", choices=("world_generated", "player_built"),
                        default="world_generated",
                        help="which separate world-state layer receives segmented objects")
    args = parser.parse_args()
    os.environ["MCGOD_PLAN_VIEW"] = "1" if args.plan_view else "0"

    store = Store()
    god = None
    if not args.dry_run:
        from god import God
        god = God(store, use_model=True)
        god.tick = 999999

    report = await survey(god, args.dim, args.centre, args.radius, args.spacing,
                          args.span, args.floor, args.dry_run, args.pause, args.origin)
    print(f"\n{report['cells']} patches probed, {report['with_builds']} had builds, "
          f"{report['segmented']} segmented, {report['structures']} structures written "
          f"in {report['seconds']}s")
    if not args.dry_run:
        summarise(store, args.dim, args.origin)
    store.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
