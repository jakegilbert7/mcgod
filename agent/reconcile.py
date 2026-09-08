#!/usr/bin/env python3
"""L4: keeping beliefs honest about places someone has been.

    python3 reconcile.py sessions/corpus/*.jsonl            # build the work queue
    python3 reconcile.py --queue                            # what most needs a look
    python3 reconcile.py --verify placement_173_63_293      # confirm one structure exists

There is no world sweep here, deliberately. Minecraft does not tick unloaded chunks: nothing
grows, spawns, burns, flows or smelts where no player is, so re-reading an empty region can
only return what was already known. Scanning the world is not merely expensive, it is
mostly meaningless.

Scanning is not for discovery. It is for correction, and it is needed even inside the tiny
scope of "where players are", because the event tap cannot see everything that happens in a
loaded chunk with a player standing in it:

  - explosions (now record what they destroyed, so this one is closed without scanning)
  - fire spread, lava and water flow, falling sand and gravel
  - endermen moving blocks, zombies breaking doors
  - events dropped when the capture queue is full
  - our own accounting drift: a bed is one event and two blocks

So: three triggers, all scoped to where someone has been.

  verify-on-reference   before asserting a structure exists, read its bounding box
  dirty regions         something happened we could not fully account for; look there
  staleness             a cell with activity since it was last read, oldest first
"""

from __future__ import annotations

import argparse
import asyncio
import collections
import json
import sys
import time
from pathlib import Path

from consumer import BOLD, DIM, GREEN, RED, RESET, YELLOW
from detectors import analyse
from store import NON_PLAYER_ACTORS, Belief, Provenance, Store

REGION = 64

#: Events that mean "our model of this cell may now be wrong". An explosion is not a world
#: event so much as an invalidation signal.
DIRTYING = {"explosion"}


def cell_id(dim: str, x: int, z: int) -> tuple[str, int, int]:
    cx, cz = x // REGION, z // REGION
    return f"{dim}:{cx}:{cz}", cx, cz


def index_regions(store: Store, events: list[dict]) -> int:
    """Records every cell a player occupied, and which of them are suspect."""
    cells: dict[str, dict] = {}
    for e in events:
        pos = e.get("pos")
        if not pos:
            continue
        # The god's own commands carry no place. Indexing them would put a region cell at
        # the world origin and send the sweep to read ground nobody has ever stood on.
        if e.get("actor") == "god":
            continue
        observed_ms = int(e.get("t") or time.time() * 1000)
        rid, cx, cz = cell_id(e.get("dim", "?"), pos[0], pos[2])
        row = cells.setdefault(rid, {
            "id": rid, "dim": e.get("dim", "?"), "cx": cx, "cz": cz,
            "last_activity_t": 0, "last_scanned_t": None,
            "last_activity_ms": 0, "last_scanned_ms": None,
            "dirty": 0, "dirty_reason": None, "actors": set(),
        })
        row["last_activity_t"] = max(row["last_activity_t"], observed_ms)
        row["last_activity_ms"] = max(row["last_activity_ms"], observed_ms)
        if e.get("actor") and e["actor"] not in NON_PLAYER_ACTORS:
            row["actors"].add(e["actor"])
        if e["type"] in DIRTYING:
            row["dirty"] = 1
            lost = e.get("destroyed") or {}
            row["dirty_reason"] = (
                f"{e['type']}: {e.get('blocks', '?')} blocks"
                + (f" ({', '.join(k.replace('minecraft:', '') for k in list(lost)[:3])})"
                   if lost else " of unknown composition"))

    for rid, row in cells.items():
        existing = store.get("regions", rid)
        if existing:
            row["last_scanned_t"] = existing["last_scanned_t"]
            row["last_scanned_ms"] = existing["last_scanned_ms"]
            row["last_activity_t"] = max(row["last_activity_t"],
                                         existing["last_activity_t"] or 0)
            row["last_activity_ms"] = max(row["last_activity_ms"],
                                          existing["last_activity_ms"] or 0)
            row["dirty"] = max(row["dirty"], existing["dirty"] or 0)
            row["dirty_reason"] = row["dirty_reason"] or existing["dirty_reason"]
            try:
                row["actors"].update(json.loads(existing["actors"] or "[]"))
            except (TypeError, ValueError):
                pass
        row["actors"] = json.dumps(sorted(row["actors"]))
        store.put("regions", row, Belief(provenance=Provenance.OBSERVED,
                                          verified_at_ms=row["last_activity_ms"] or None))
    return len(cells)


async def confirm_before_speaking(store: Store, structure_id: str, url: str,
                                  floor: float = 0.5) -> dict | None:
    """Verify a structure if what we believe about it has gone stale.

    The gate before the god names something. A belief derived from events can be wrong for
    reasons the event stream will never mention — someone blew it up, it burned, the
    capture queue dropped the breaks — and the longer since anyone looked, the likelier
    that is. Cheap: one read of a box whose corners are already known, and only when the
    belief has decayed past the floor.
    """
    row = store.get("structures", structure_id)
    if row is None:
        return None
    if not store.stale("structures", structure_id, floor):
        return None
    try:
        return await verify(store, structure_id, url)
    except (OSError, TimeoutError) as e:
        # Not being able to look is not evidence of anything.
        return {"ok": False, "error": f"could not reach the world: {type(e).__name__}"}


def work_queue(store: Store) -> list:
    """What most needs reading, worst first.

    Dirty cells outrank stale ones: a cell we know is wrong is worth more than a cell we
    merely have not checked lately. Within each group, oldest reading first.
    """
    rows = store.all("regions")

    def rank(r):
        activity = r["last_activity_ms"] or r["last_activity_t"] or 0
        scanned = r["last_scanned_ms"] or 0
        stale = activity - scanned
        return (-(r["dirty"] or 0), -max(0, stale))

    return sorted(rows, key=rank)


async def verify(store: Store, structure_id: str, url: str) -> dict:
    """Read a structure's bounding box and compare it to what we believe.

    The gate before the god speaks about a named structure. A belief derived from events can
    be stale for reasons the event stream will never mention — someone blew it up, it burned,
    the capture queue dropped the breaks. Confirming costs one bounded read of a box we
    already know the corners of.
    """
    from scan import request_scan

    row = store.get("structures", structure_id)
    if row is None:
        return {"ok": False, "error": f"no such structure: {structure_id}"}

    believed = json.loads(row["materials"] or "{}")
    result = await request_scan(
        url, row["dim"],
        [row["min_x"], row["min_y"], row["min_z"]],
        [row["max_x"], row["max_y"], row["max_z"]])
    if not result.get("ok"):
        return result

    # A scan that finds nothing at all where we believed a solid structure stood is far
    # more likely a failed read than a vanished building — chunks still loading, a world
    # not yet initialised, a timeout. Treating it as evidence once destroyed a correct
    # belief permanently: it was written at SCANNED, which outranks DERIVED, so
    # re-deriving from events could never repair it. Absence of evidence is refused here.
    if believed and result.get("solid", 0) == 0:
        return {"ok": False, "id": structure_id,
                "error": "scan returned an empty region where a structure was believed to "
                         "stand; treating as a failed read, not a demolition"}

    # Both sides must be keyed the same way. The census strips the namespace and the scan
    # keeps it, so a raw dict lookup silently missed every material and reported a standing
    # house as 0% survived — a wrong answer that looked entirely plausible, because the
    # display strips the namespace from both and they printed identically.
    def bare(d: dict) -> dict:
        return {k.replace("minecraft:", ""): v for k, v in d.items()}

    found = bare(result["materials"])
    believed = bare(believed)
    verdict = {"ok": True, "id": structure_id, "believed": believed, "found": {},
               "missing": {}, "extra": {}}
    for material, count in believed.items():
        actual = found.get(material, 0)
        verdict["found"][material] = actual
        if actual < count:
            verdict["missing"][material] = count - actual
    survived = sum(min(v, found.get(k, 0)) for k, v in believed.items())
    total = sum(believed.values()) or 1
    verdict["survival"] = round(survived / total, 3)

    # A live census of what is in there. Transient by nature — animals wander and despawn —
    # so it is recorded with the scan that saw it and never merged into the build record.
    observed_ms = int(time.time() * 1000)
    if result.get("entities"):
        store.put("relationships", {
            "id": f"holds:{structure_id}",
            "subject": structure_id, "predicate": "holds",
            "object": json.dumps(result["entities"]),
        }, Belief(provenance=Provenance.SCANNED, verified_at_tick=result["tick"],
                  verified_at_ms=observed_ms,
                  value=json.dumps({"when": observed_ms})))

    # Two different facts, kept in two different rows. What was BUILT is history, derived
    # from events, and cannot change — the blocks were placed whatever happened afterwards.
    # What STANDS now is current state from a scan. Writing the scan over the build record
    # conflated them and let one bad read erase what a player actually did.
    store.put("relationships", {
        "id": f"survives:{structure_id}",
        "subject": structure_id,
        "predicate": "survives",
        "object": f"{verdict['survival']:.3f}",
    }, Belief(provenance=Provenance.SCANNED, verified_at_tick=result["tick"],
              verified_at_ms=observed_ms,
              value=json.dumps({"found": found, "missing": verdict["missing"],
                                "when": observed_ms})))
    rid, _, _ = cell_id(row["dim"], row["min_x"], row["min_z"])
    region = store.get("regions", rid)
    if region:
        store.put("regions", {
            "id": rid, "dim": region["dim"], "cx": region["cx"], "cz": region["cz"],
            "last_activity_t": region["last_activity_t"],
            "last_scanned_t": region["last_scanned_t"],
            "last_activity_ms": region["last_activity_ms"] or region["last_activity_t"],
            "last_scanned_ms": observed_ms, "dirty": 0, "dirty_reason": None,
            "actors": region["actors"],
        }, Belief(provenance=Provenance.OBSERVED, verified_at_ms=observed_ms))
    return verdict


def main(argv: list[str]) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("sessions", nargs="*", type=Path)
    p.add_argument("--queue", action="store_true", help="show the scan work queue")
    p.add_argument("--verify", metavar="STRUCTURE_ID")
    p.add_argument("--url", default="ws://127.0.0.1:8765")
    args = p.parse_args(argv)

    store = Store()
    if args.sessions:
        total = 0
        for path in sorted(args.sessions):
            _, _, events = analyse([path])
            total = index_regions(store, events)
        print(f"indexed {total} region cells a player has occupied")

    if args.verify:
        verdict = asyncio.run(verify(store, args.verify, args.url))
        if not verdict.get("ok"):
            print(f"{RED}{verdict.get('error')}{RESET}")
            return 1
        colour = GREEN if verdict["survival"] > 0.95 else (
            YELLOW if verdict["survival"] > 0.5 else RED)
        print(f"{BOLD}{verdict['id']}{RESET}: "
              f"{colour}{verdict['survival'] * 100:.0f}% of it is still there{RESET}")
        for material, believed in verdict["believed"].items():
            found = verdict["found"].get(material, 0)
            flag = "" if found >= believed else f"  {RED}-{believed - found}{RESET}"
            print(f"  {material.replace('minecraft:', ''):24} believed {believed:4d}  "
                  f"found {found:4d}{flag}")
        return 0

    if args.queue or args.sessions:
        rows = work_queue(store)
        dirty = [r for r in rows if r["dirty"]]
        print(f"\n{BOLD}scan queue{RESET}  {len(rows)} cells occupied, "
              f"{len(dirty)} dirty, {sum(1 for r in rows if not r['last_scanned_ms'])} "
              f"never read")
        for r in rows[:12]:
            mark = f"{RED}DIRTY{RESET}" if r["dirty"] else f"{DIM}stale{RESET}"
            print(f"  {mark}  {r['id']:22} "
                  + (f"{r['dirty_reason']}" if r["dirty_reason"] else
                     f"{DIM}never scanned{RESET}" if not r["last_scanned_ms"] else ""))
    store.close()
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
