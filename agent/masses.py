#!/usr/bin/env python3
"""The complete ledger of built things: every connected mass of non-natural blocks.

    python3 masses.py --list                # what is on record, biggest first
    python3 masses.py --audit               # invariants over masses and their landmarks
    python3 masses.py --rebuild [--wipe]    # re-read every cell a player has occupied
    python3 masses.py --roles               # ask what the unnamed ones are for
    python3 masses.py --relink              # attach history to the masses it happened in

Before this module, a player-built thing existed in the present tense only if the vision
model chose to draw a box around it, and the segmentation prompt told it outright to leave
out paths, scattered dirt and pillars. That made the model the gate on *existence*, not
merely on meaning: something it declined to name had no current record at all, and nearly
half of all block events could not be attached to any place. A god asked "who built that
pillar" had nothing to point at.

So this layer is deterministic and complete. Every connected group of non-natural blocks in
a patch that has been read becomes one row, whatever its size and whoever put it there —
a house, a scaffold pillar, a torch, a village wall the world generated. Code measures it,
code attributes it from the exact block history, code matches its identity across reads.
No model decides whether a mass exists. What a mass IS remains a model's judgement: a
landmark in ``structures`` is a named grouping of masses, and an unnamed mass carries a
separate INFERRED ``role`` belief saying what it seems to be for.

Three rules that keep it honest:

- **A read has no information beyond its own edge.** A mass pressed against the boundary of
  the patch is recorded as ``partial`` and never overwrites a whole observation of the same
  thing. The neighbouring ground is queued for a look instead.
- **Only measured absence retires a row.** A mass fully inside a fresh read with none of its
  blocks left is gone; one whose blocks now sit in a differently-shaped mass is absorbed,
  with lineage recorded. Being unmentioned by anything retires nothing.
- **Authorship is replayed, never assumed.** The latest recorded mutation at each block is
  compared to what stands there now. A mass nobody was watching being built has an unknown
  builder, and says so, rather than inheriting the nearest village or the nearest player.
"""

from __future__ import annotations

import argparse
import asyncio
import collections
import hashlib
import json
import sys
import time
import uuid
from typing import Literal

from pydantic import BaseModel, Field

from render import is_natural
from segment import MIN_STRUCTURE_BLOCKS, REACH, built_blocks
from store import Belief, Provenance, Store
from structures import (PLAYER_ORIGIN_SHARE, assign_identities, box_of, current as
                        current_structures, current_generated, facts_of, generated_near,
                        match_score, volume)

SCHEMA_VERSION = 1
KIND = "built_mass"

#: Two runs of block work inside one mass separated by at least this long are two edits.
EDIT_GAP_MS = 10 * 60_000

#: Below this a mass is recorded but not asked about: the same universal evidence floor
#: segmentation uses. A single torch is a fact worth keeping and not worth a model call.
ROLE_FLOOR = MIN_STRUCTURE_BLOCKS

ROLES = ("landmark", "utility", "decoration", "debris", "unknown")

FACES = (("west", "east"), ("below", "above"), ("north", "south"))


def bare(material: str) -> str:
    return str(material or "").split("[")[0].replace("minecraft:", "")


def new_id(existing_ids=()) -> str:
    used = set(existing_ids)
    while True:
        mid = f"m_{uuid.uuid4().hex[:12]}"
        if mid not in used:
            return mid


def is_mass(row) -> bool:
    return facts_of(row).get("kind") == KIND


def current(store, dim: str | None = None) -> list:
    rows = store.all("masses", "dim = ?", (dim,)) if dim else store.all("masses")
    return [row for row in rows if row["min_x"] is not None and is_mass(row)]


def within(store, dim: str, lo, hi) -> list:
    """Masses whose boxes intersect a window."""
    return [row for row in store.all(
        "masses",
        "dim = ? AND max_x >= ? AND min_x <= ? AND max_y >= ? AND min_y <= ? "
        "AND max_z >= ? AND min_z <= ?",
        (dim, lo[0], hi[0], lo[1], hi[1], lo[2], hi[2])) if is_mass(row)]


def inside(row, lo, hi) -> bool:
    a, b = box_of(row)
    return all(lo[i] <= a[i] and b[i] <= hi[i] for i in range(3))


# --------------------------------------------------------------------------- measuring

def measure(group: list, voxels: dict, built: dict, lo, hi) -> dict:
    """Every fact code can state about one connected mass, with no interpretation.

    The geometry facts here are deliberately generic — where it rests, what it touches, how
    it is distributed by height — so that a model can tell a bridge from a pillar from a
    wall without any shape-specific rule living in code.
    """
    xs, ys, zs = zip(*group)
    glo = [min(xs), min(ys), min(zs)]
    ghi = [max(xs), max(ys), max(zs)]
    dims = [ghi[i] - glo[i] + 1 for i in range(3)]
    members = set(group)
    materials = collections.Counter(bare(built[p]) for p in group)
    levels = collections.Counter(p[1] for p in group)

    footing = {"on_ground": 0, "over_air": 0, "over_water": 0, "on_built": 0}
    natural_contact = 0
    for p in group:
        below = (p[0], p[1] - 1, p[2])
        if below not in members and below[1] >= lo[1]:
            under = voxels.get(below)
            if under is None:
                footing["over_air"] += 1
            elif is_natural(under):
                if bare(under) in ("water", "lava"):
                    footing["over_water"] += 1
                else:
                    footing["on_ground"] += 1
            else:
                footing["on_built"] += 1
        for dx, dz in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            side = voxels.get((p[0] + dx, p[1], p[2] + dz))
            if side is not None and is_natural(side) and bare(side) not in (
                    "water", "lava", "short_grass", "tall_grass", "fern", "seagrass",
                    "kelp", "kelp_plant", "dead_bush"):
                natural_contact += 1
                break

    # Connectivity reaches REACH blocks, so a mass can continue past the edge without a
    # block sitting on it. Anything within reach of the edge may be cut.
    partial = []
    for axis, (low_name, high_name) in enumerate(FACES):
        if glo[axis] <= lo[axis] + REACH - 1:
            partial.append(low_name)
        if ghi[axis] >= hi[axis] - REACH + 1:
            partial.append(high_name)

    return {
        "lo": glo, "hi": ghi, "blocks": len(group),
        "dims": f"{dims[0]}x{dims[1]}x{dims[2]}",
        "fill": round(len(group) / max(1, dims[0] * dims[1] * dims[2]), 3),
        "vertical": dims[1] >= 3,
        "materials": dict(materials.most_common()),
        "levels": [[y, n] for y, n in sorted(levels.items())],
        "footing": footing,
        "natural_contact": natural_contact,
        "partial": partial,
        "_blocks": members,
        "_material_at": {p: bare(built[p]) for p in group},
    }


_OFFSETS = [(dx, dy, dz) for dx in range(-REACH, REACH + 1) for dy in range(-REACH, REACH + 1)
            for dz in range(-REACH, REACH + 1) if (dx, dy, dz) != (0, 0, 0)]


def components(built: dict, reach: int = REACH) -> list[list]:
    """Connected groups of built blocks, in a deterministic order.

    Same connectivity as ``segment.clusters`` but linear in the number of blocks: that
    scans every remaining block per step, which is quadratic and took tens of seconds on
    a village-sized read. Neighbour lookups against a set do not.
    """
    offsets = _OFFSETS if reach == REACH else [
        (dx, dy, dz) for dx in range(-reach, reach + 1) for dy in range(-reach, reach + 1)
        for dz in range(-reach, reach + 1) if (dx, dy, dz) != (0, 0, 0)]
    remaining = set(built)
    groups = []
    for seed in sorted(built):
        if seed not in remaining:
            continue
        remaining.discard(seed)
        group, frontier = [seed], [seed]
        while frontier:
            x, y, z = frontier.pop()
            for dx, dy, dz in offsets:
                q = (x + dx, y + dy, z + dz)
                if q in remaining:
                    remaining.discard(q)
                    group.append(q)
                    frontier.append(q)
        groups.append(group)
    return sorted(groups, key=lambda g: (-len(g), min(g)))


LOG_LIKE = ("_log", "_wood", "_stem", "_hyphae")
LOG_EXACT = {"mushroom_stem", "bamboo_block"}
LEAF_LIKE = ("_leaves",)
LEAF_EXACT = {"nether_wart_block", "warped_wart_block", "shroomlight",
              "red_mushroom_block", "brown_mushroom_block"}


def _log_like(state: str) -> bool:
    name = bare(state)
    return name in LOG_EXACT or name.endswith(LOG_LIKE)


def _leaf_like(state) -> bool:
    if state is None:
        return False
    name = bare(state)
    return name in LEAF_EXACT or name.endswith(LEAF_LIKE)


def clip(voxels: dict, lo, hi) -> dict:
    """Only what lies inside the window. A read is already clipped by the plugin; a caller
    holding a larger dictionary must not leak blocks beyond the edge into a window's facts,
    or ``partial`` and retirement both lie."""
    return {p: m for p, m in voxels.items()
            if lo[0] <= p[0] <= hi[0] and lo[1] <= p[1] <= hi[1] and lo[2] <= p[2] <= hi[2]}


def ledger_built(store, dim: str, voxels: dict, lo, hi, credit: dict | None = None) -> dict:
    """What the ledger counts as built.

    Three adjustments to the renderer's terrain filter, each resolved by evidence rather
    than by a shape rule:

    - ``is_natural`` drops dirt, which is what most pillars and ramps are made of. Any
      block the exact history credits a player with placing counts as built, whatever
      its material. The world stays the authority on what stands; the ledger only says
      which of it was put there on purpose.
    - Logs are deliberately not natural, because a log is as often a wall as a tree. A
      connected mass made only of logs, credited to nobody, and touching leaves is a
      tree, and is left to the forest. A log cabin touches no leaves; a felled and
      re-stacked trunk is credited.
    """
    voxels = clip(voxels, lo, hi)
    if credit is None:
        credit = credited(block_events(store, dim, lo, hi), voxels)
    built = built_blocks(voxels)
    for pos in credit:
        if pos in voxels:
            built[pos] = voxels[pos]
    for group in components(built):
        if any(p in credit for p in group) or not all(_log_like(built[p]) for p in group):
            continue
        touches_leaves = any(_leaf_like(voxels.get((p[0] + dx, p[1] + dy, p[2] + dz)))
                             for p in group
                             for dx, dy, dz in ((1, 0, 0), (-1, 0, 0), (0, 1, 0),
                                                (0, -1, 0), (0, 0, 1), (0, 0, -1)))
        # A log column running out of the read is undecidable: its canopy, if it has one,
        # is exactly what was not read. Recording the fragment as a wall and deleting it
        # when the slab above shows the leaves churned the ledger on every rebuild.
        cut = any(p[i] <= lo[i] or p[i] >= hi[i] for p in group for i in range(3))
        if touches_leaves or cut:
            for p in group:
                del built[p]
    return built


def landmark_boxes(store, dim: str) -> list:
    """Every measured landmark box, smallest first: structures and generated buildings."""
    boxes = []
    for row in current_structures(store, dim) + current_generated(store, dim):
        lo, hi = box_of(row)
        boxes.append((volume(lo, hi), row["id"], lo, hi))
    boxes.sort()
    return [(rid, lo, hi) for _, rid, lo, hi in boxes]


def partitioned_components(built: dict, boxes: list, credit: dict) -> list[list]:
    """Connected groups that never cross a landmark boundary or an authorship seam.

    Plain connectivity fused a whole settlement — hall, pen, tower and rails, joined by
    fences and paths — into one 632-block mass that no landmark could claim. The boxes the
    model drew ARE the judgement of where one thing ends, so the ledger honours them: a
    block inside a landmark's box clusters only with blocks in the same box. And a block
    the history credits to a player never clusters with one it does not, so a dirt pillar
    leaned against a village wall stays the player's pillar rather than vanishing into
    the wall.
    """
    def key(p):
        for rid, lo, hi in boxes:
            if all(lo[i] <= p[i] <= hi[i] for i in range(3)):
                return (rid, p in credit)
        return (None, p in credit)

    groups: dict = collections.defaultdict(dict)
    for p, state in built.items():
        groups[key(p)][p] = state
    out = []
    for part in groups.values():
        out.extend(components(part))
    return sorted(out, key=lambda g: (-len(g), min(g)))


def extract(voxels: dict, built: dict, lo, hi, boxes: list = (),
            credit: dict | None = None) -> list[dict]:
    """Every connected built mass in a read, largest first. No floor: a torch is a mass."""
    groups = (partitioned_components(built, list(boxes), credit or {})
              if boxes or credit else components(built))
    return [measure(group, voxels, built, lo, hi) for group in groups]


def block_events(store, dim: str, lo, hi, pad: int = 1) -> list:
    """Every recorded block mutation in a window, oldest first. Fetched once per read."""
    return store.all(
        "event_history",
        "dim = ? AND kind IN ('block_place', 'block_break') "
        "AND x BETWEEN ? AND ? AND y BETWEEN ? AND ? AND z BETWEEN ? AND ? "
        "ORDER BY event_t, id",
        (dim, lo[0] - pad, hi[0] + pad, lo[1] - pad, hi[1] + pad, lo[2] - pad, hi[2] + pad))


def credited(rows: list, voxels: dict) -> dict:
    """Blocks whose latest recorded mutation is a player placing exactly what stands there.

    That is the whole authorship test: not "someone once placed something here", but "the
    last thing that happened here was a player putting this very material down, and it is
    still there". A block placed and later broken credits nobody; a block whose placement
    was never captured credits nobody. Maps position to actor.
    """
    latest: dict = {}
    for row in rows:
        latest[(row["x"], row["y"], row["z"])] = row
    out = {}
    for pos, row in latest.items():
        if row["kind"] != "block_place" or not row["actor"] or row["actor"] == "world":
            continue
        state = voxels.get(pos)
        if state is None:
            continue
        try:
            after = json.loads(row["payload"] or "{}").get("after")
        except (TypeError, ValueError):
            continue
        if bare(after) == bare(state):
            out[pos] = row["actor"]
    return out


def history(rows: list, facts: dict, credit: dict) -> dict:
    """What the event ledger says about one mass: who, when, and how often.

    ``builders`` is each actor's share of the mass's standing blocks by exact credit, so
    ``observed_share`` says how much of the mass the ledger can vouch for and ``builder``
    is the actor who supplied at least half. Edits are runs of block events inside the box
    (padded by one, so taking a wall down counts) separated by ``EDIT_GAP_MS``.
    """
    lo, hi = facts["lo"], facts["hi"]
    members = facts["_blocks"]
    times: list[int] = []
    actors: set = set()
    placed = broken = 0
    for row in rows:
        if not (lo[0] - 1 <= row["x"] <= hi[0] + 1 and lo[1] - 1 <= row["y"] <= hi[1] + 1
                and lo[2] - 1 <= row["z"] <= hi[2] + 1):
            continue
        if row["event_t"] is not None:
            times.append(int(row["event_t"]))
        if row["actor"] and row["actor"] != "world":
            actors.add(row["actor"])
        if row["kind"] == "block_place":
            placed += 1
        else:
            broken += 1

    counted: collections.Counter = collections.Counter(
        credit[p] for p in members if p in credit)
    first_built = None
    if counted:
        by_pos = {(row["x"], row["y"], row["z"]): row for row in rows}
        stamps = [int(by_pos[p]["event_t"]) for p in members
                  if p in credit and p in by_pos and by_pos[p]["event_t"] is not None]
        first_built = min(stamps) if stamps else None

    blocks = max(1, facts["blocks"])
    shares = {actor: round(n / blocks, 3) for actor, n in counted.most_common()}
    builder = None
    if counted:
        actor, n = counted.most_common(1)[0]
        if n / blocks >= PLAYER_ORIGIN_SHARE:
            builder = actor

    times.sort()
    edits = 0
    last = None
    for t in times:
        if last is None or t - last > EDIT_GAP_MS:
            edits += 1
        last = t

    return {
        "builders": shares,
        "builder": builder,
        "observed_share": round(sum(counted.values()) / blocks, 3),
        "first_built_ms": first_built,
        "first_touched_ms": times[0] if times else None,
        "last_edited_ms": times[-1] if times else None,
        "edits": edits,
        "placed": placed,
        "broken": broken,
        "actors": sorted(actors),
    }


def origin_of(store, dim: str, facts: dict, hist: dict) -> tuple[str, str, str | None]:
    """(origin, basis, generated site id). Absence of history is ``unknown``, never a guess.

    The site is the broad generated place, never one of its component buildings: a
    component with a box claims a mass only by containing it (see ``_containing_place``),
    and falling back to the nearest component anchored sixty blocks of village paths to
    one house.
    """
    from structures import generated_root

    if hist.get("builder"):
        return "player_built", "placement history", None
    near = generated_near(store, dim, facts)
    if near is not None:
        root = generated_root(store, near) or near
        return "world_generated", "inside a known generated site", root["id"]
    return "unknown", "no placement history and no known generated site", None


# --------------------------------------------------------------------------- identity

def _containing_place(blocks: set, lo, hi, structures: list, generated: list,
                      origin: str, generated_id: str | None) -> str | None:
    """The landmark or generated building that holds at least half of a mass.

    Exact block membership when the caller has it, box intersection otherwise. The
    smallest container wins so a pen beside a hall is the pen's, not the hall's.
    """
    best = None
    for kind, rows in (("structure", structures), ("generated", generated)):
        for row in rows:
            a, b = box_of(row)
            if blocks:
                share = sum(1 for p in blocks
                            if all(a[i] <= p[i] <= b[i] for i in range(3))) / len(blocks)
            else:
                size = [min(hi[i], b[i]) - max(lo[i], a[i]) + 1 for i in range(3)]
                shared = size[0] * size[1] * size[2] if all(n > 0 for n in size) else 0
                share = shared / max(1, volume(lo, hi))
            if share < 0.5:
                continue
            if best is None or volume(a, b) < best[0]:
                best = (volume(a, b), row["id"])
        if best:
            return best[1]
    if origin == "world_generated" and generated_id:
        return generated_id
    return None


def apply(store, dim: str, voxels: dict, built: dict, lo, hi, tick: int,
          now_ms: int | None = None) -> dict:
    """Replace the ledger's view of one read window with what the read actually shows.

    One transaction. Identity is matched one-to-one on geometry and materials; a partial
    observation never overwrites a whole one; rows fully inside the window that nothing
    continues are retired only because their blocks are measurably gone or measurably
    part of something else now.
    """
    now_ms = int(time.time() * 1000) if now_ms is None else now_ms
    voxels, built = clip(voxels, lo, hi), clip(built, lo, hi)
    rows = block_events(store, dim, lo, hi)
    credit = credited(rows, voxels)
    structures = current_structures(store, dim)
    generated = [row for row in current_generated(store, dim)]
    observations = extract(voxels, built, lo, hi, landmark_boxes(store, dim), credit)
    for o in observations:
        o["dim"] = dim
        o["history"] = history(rows, o, credit)
        o["origin"], o["origin_basis"], o["generated_id"] = origin_of(
            store, dim, o, o["history"])

    existing = within(store, dim, lo, hi)
    by_id = {row["id"]: row for row in existing}
    assigned = assign_identities(existing, observations)
    used = {row["id"] for row in store.all("masses")}

    def touching(row, blocks: set) -> bool:
        a, b = box_of(row)
        return any(all(a[i] <= p[i] <= b[i] for i in range(3)) for p in blocks)

    written, kept, retired = [], [], []
    with store.transaction():
        claimed: set = set()
        new_by_id: dict = {}
        for index, o in enumerate(observations):
            mid = assigned.get(index)
            prior = by_id.get(mid) if mid else None
            if o["partial"] and prior is None:
                # A fragment through the edge that no row won, yet lying against a row
                # known to extend beyond this window: one-to-one assignment gave that row
                # to a bigger fragment of the same thing. This is a view of it, not a new
                # mass.
                beyond = [r for r in by_id.values()
                          if not inside(r, lo, hi) and touching(r, o["_blocks"])]
                if beyond:
                    prior = max(beyond, key=lambda r: match_score(r, o))
                    mid = prior["id"]
                    claimed.update(r["id"] for r in beyond)
            if o["partial"] and prior is not None and not inside(prior, lo, hi):
                # A cut-off view of a thing known to continue past this window. A partial
                # view may only GROW a record — the union of what was known and what is
                # seen — and only a read that holds the whole record may shrink it.
                # Overwriting with the fragment lost the far end; leaving the record
                # untouched lost every extension a player built past the read edge.
                merged = _union_view(prior, o, lo, hi)
                _write(store, mid, merged, dim, tick, now_ms, prior, prior["place_id"])
                # Two fragments of one record in one read must accumulate, not race.
                by_id[mid] = store.get("masses", mid)
                kept.append(mid)
                claimed.add(mid)
                claimed.update(r["id"] for r in by_id.values() if touching(r, o["_blocks"]))
                continue
            lineage: dict = {}
            if prior is None:
                mid = new_id(used)
                used.add(mid)
                parents = sorted(r["id"] for r in existing if touching(r, o["_blocks"]))
                if parents:
                    lineage["split_from"] = parents
            else:
                lineage = dict(facts_of(prior).get("lineage") or {})
            o["_lineage"] = lineage
            place = _containing_place(o["_blocks"], o["lo"], o["hi"], structures,
                                      generated, o["origin"], o["generated_id"])
            _write(store, mid, o, dim, tick, now_ms, prior, place)
            written.append(mid)
            claimed.add(mid)
            new_by_id[mid] = o

        for row in existing:
            if row["id"] in claimed or not inside(row, lo, hi):
                continue
            a, b = box_of(row)
            still = sum(1 for p in built if all(a[i] <= p[i] <= b[i] for i in range(3)))
            if still == 0:
                reason = "not there on a fresh look"
            else:
                reason = "its blocks now belong to a differently shaped mass"
                for mid, o in new_by_id.items():
                    if touching(row, o["_blocks"]):
                        store.put("relationships", {
                            "id": f"absorbed:{mid}:{row['id']}", "subject": mid,
                            "predicate": "absorbed", "object": row["id"],
                        }, Belief(provenance=Provenance.SCANNED, verified_at_tick=tick,
                                  verified_at_ms=now_ms,
                                  value=json.dumps({"when": now_ms, "blocks_left": still})))
            _delete(store, row["id"])
            retired.append((row["id"], reason))

    return {"written": written, "kept": kept, "retired": retired,
            "partial": [o for o in observations if o["partial"]],
            "observations": observations}


def _union_view(prior, o: dict, window_lo, window_hi) -> dict:
    """What is known about a mass after a partial view of it: the union.

    Geometry is the union of boxes; the material census gains exactly the fragment's blocks
    that lie outside the old box, so an extension is counted and nothing is counted twice. A face stays or becomes cut only
    where nothing is known past the window edge: a fragment cut on the east of a window
    the record already crosses eastward has told us nothing new about that side. History
    keeps the better-observed authorship and takes the later edit.
    """
    pf = facts_of(prior)
    plo, phi = box_of(prior)
    lo = [min(plo[i], o["lo"][i]) for i in range(3)]
    hi = [max(phi[i], o["hi"][i]) for i in range(3)]
    cut = set(pf.get("partial") or [])
    for axis, (low_name, high_name) in enumerate(FACES):
        if low_name in o["partial"] and not plo[axis] < window_lo[axis]:
            cut.add(low_name)
        if high_name in o["partial"] and not phi[axis] > window_hi[axis]:
            cut.add(high_name)
    dims = [hi[i] - lo[i] + 1 for i in range(3)]
    # Blocks of the fragment outside the old box are certainly new; add exactly those.
    # Changes inside the old box cannot be told apart from what was counted before.
    materials = collections.Counter(json.loads(prior["materials"] or "{}"))
    materials.update(material for p, material in o.get("_material_at", {}).items()
                     if not all(plo[i] <= p[i] <= phi[i] for i in range(3)))
    blocks = sum(materials.values())
    levels: dict = {int(y): n for y, n in (pf.get("levels") or [])}
    for y, n in o["levels"]:
        levels[y] = max(levels.get(y, 0), n)
    footing = dict(pf.get("footing") or {})
    for key, n in o["footing"].items():
        footing[key] = max(footing.get(key, 0), n)
    old_hist = {key: pf.get(key) for key in (
        "builders", "builder", "observed_share", "first_built_ms", "first_touched_ms",
        "last_edited_ms", "edits", "placed", "broken", "actors")}
    new_hist = o["history"]
    better = (new_hist if (new_hist["observed_share"] or 0) > (old_hist["observed_share"] or 0)
              else old_hist)
    hist = dict(better)
    hist["last_edited_ms"] = max(v for v in (old_hist["last_edited_ms"],
                                             new_hist["last_edited_ms"]) if v) if (
        old_hist["last_edited_ms"] or new_hist["last_edited_ms"]) else None
    hist["edits"] = max(old_hist["edits"] or 0, new_hist["edits"] or 0)
    hist["placed"] = max(old_hist["placed"] or 0, new_hist["placed"] or 0)
    hist["broken"] = max(old_hist["broken"] or 0, new_hist["broken"] or 0)
    hist["actors"] = sorted(set(old_hist["actors"] or []) | set(new_hist["actors"] or []))
    origin = ("player_built" if hist.get("builder")
              else pf.get("origin") if pf.get("origin") != "player_built" else "unknown")
    return {
        "lo": lo, "hi": hi, "blocks": blocks,
        "dims": f"{dims[0]}x{dims[1]}x{dims[2]}",
        "fill": round(blocks / max(1, dims[0] * dims[1] * dims[2]), 3),
        "vertical": dims[1] >= 3,
        "materials": dict(materials.most_common()),
        "levels": [[y, n] for y, n in sorted(levels.items())],
        "footing": footing,
        "natural_contact": max(pf.get("natural_contact") or 0, o["natural_contact"]),
        "partial": sorted(cut),
        "_blocks": o["_blocks"],
        "history": hist,
        "origin": origin,
        "origin_basis": (pf.get("origin_basis") if origin == pf.get("origin")
                         else o["origin_basis"]),
        "generated_id": pf.get("generated_id") or o["generated_id"],
        "_lineage": dict(pf.get("lineage") or {}),
        "union_of_views": True,
    }


def _write(store, mid: str, o: dict, dim: str, tick: int, now_ms: int, prior,
           place_id: str | None) -> None:
    hist = o["history"]
    prior_facts = facts_of(prior) if prior is not None else {}
    value = {
        "schema_version": SCHEMA_VERSION, "kind": KIND,
        "observed_at_ms": now_ms,
        "first_observed_ms": prior_facts.get("first_observed_ms") or now_ms,
        "blocks": o["blocks"], "dims": o["dims"], "fill": o["fill"],
        "vertical": o["vertical"], "levels": o["levels"], "footing": o["footing"],
        "natural_contact": o["natural_contact"], "partial": o["partial"],
        "origin": o["origin"], "origin_basis": o["origin_basis"],
        "generated_id": o["generated_id"],
        "builders": hist["builders"], "builder": hist["builder"],
        "observed_share": hist["observed_share"],
        "first_built_ms": hist["first_built_ms"],
        "first_touched_ms": hist["first_touched_ms"],
        "last_edited_ms": hist["last_edited_ms"],
        "edits": hist["edits"], "placed": hist["placed"], "broken": hist["broken"],
        "actors": hist["actors"],
        "lineage": o.get("_lineage") or {},
        "union_of_views": bool(o.get("union_of_views")),
        # When a vision pass last drew boxes over this mass. Carried across re-measurement
        # so a thing the model has already declined to name is not asked about again
        # until it changes enough to be worth it.
        "looked_at_ms": prior_facts.get("looked_at_ms"),
    }
    store.put("masses", {
        "id": mid, "dim": dim, "actor": hist["builder"], "place_id": place_id,
        "min_x": o["lo"][0], "min_y": o["lo"][1], "min_z": o["lo"][2],
        "max_x": o["hi"][0], "max_y": o["hi"][1], "max_z": o["hi"][2],
        "materials": json.dumps(o["materials"]),
    }, Belief(provenance=Provenance.SCANNED, confidence=1.0,
              first_seen_tick=(prior["first_seen_tick"] if prior is not None else tick),
              verified_at_tick=tick, verified_at_ms=now_ms,
              value=json.dumps(value)))


def _delete(store, mid: str) -> None:
    store.db.execute("DELETE FROM masses WHERE id = ?", (mid,))
    store.db.execute("DELETE FROM relationships WHERE subject = ?", (mid,))
    store.db.execute("UPDATE masses SET place_id = NULL WHERE place_id = ?", (mid,))
    store.commit()


# --------------------------------------------------------------------------- landmarks

def anchor(store, dim: str | None = None) -> int:
    """Attach every mass to the landmark or generated building holding it, and record on
    each structure which masses it comprises.

    Run after any pass that rewrites structures. Whole-dimension on purpose: a structure
    near the edge of a read may comprise masses outside it.
    """
    changed = 0
    dims = [dim] if dim else sorted({row["dim"] for row in current(store)})
    for d in dims:
        structures = current_structures(store, d)
        generated = current_generated(store, d)
        comprises: dict = collections.defaultdict(list)
        structure_ids = {row["id"] for row in structures}
        for row in current(store, d):
            facts = facts_of(row)
            place = _containing_place(set(), *box_of(row), structures, generated,
                                      facts.get("origin", "unknown"),
                                      facts.get("generated_id"))
            if place != row["place_id"]:
                store.db.execute("UPDATE masses SET place_id = ? WHERE id = ?",
                                 (place, row["id"]))
                changed += 1
            if place in structure_ids:
                comprises[place].append(row["id"])
        for row in structures:
            value = facts_of(row)
            now = sorted(comprises.get(row["id"], []))
            if value.get("comprises") != now:
                value["comprises"] = now
                store.db.execute("UPDATE structures SET value = ? WHERE id = ?",
                                 (json.dumps(value), row["id"]))
                changed += 1
    store.commit()
    return changed


def mark_looked(store, dim: str, lo, hi, now_ms: int) -> int:
    """Stamp every mass in a window as having been shown to the boxing model."""
    changed = 0
    for row in within(store, dim, lo, hi):
        value = facts_of(row)
        value["looked_at_ms"] = now_ms
        store.db.execute("UPDATE masses SET value = ? WHERE id = ?",
                         (json.dumps(value), row["id"]))
        changed += 1
    store.commit()
    return changed


def unnamed_whole(store, dim: str, lo, hi, floor: int = ROLE_FLOOR) -> list:
    """Player-built masses in a window, seen whole, that no landmark holds and no vision
    pass has ever been shown. The one case that earns a look regardless of any threshold:
    a thing exists that nobody has had the chance to name."""
    out = []
    for row in within(store, dim, lo, hi):
        facts = facts_of(row)
        if (facts.get("origin") == "player_built" and not facts.get("partial")
                and not _has_landmark(store, row) and not facts.get("looked_at_ms")
                and (facts.get("blocks") or 0) >= floor):
            out.append(row)
    return out


def refresh_landmarks(store, dim: str, now_ms: int, tick: int | None = None) -> int:
    """Bring every anchored structure's census up to what the ledger measured.

    A structure's box is the model's and waits for the next boxing pass; what stands inside
    that box is code's, re-measured on every read. Two gold blocks set into a cobblestone
    elephant were reported as cobblestone for as long as the look was deferred, because
    nothing carried the ledger's fresh census onto the row that answers questions. Now the
    row's materials and block count are the sum of the masses it comprises, and a structure
    marked out of date by an edit is reaffirmed: its contents are known again, only its
    name may lag.
    """
    changed = 0
    for row in current_structures(store, dim):
        value = facts_of(row)
        ids = value.get("comprises") or []
        if not ids:
            continue
        materials: collections.Counter = collections.Counter()
        missing = False
        for mid in ids:
            mass = store.get("masses", mid)
            if mass is None:
                missing = True
                break
            materials.update(json.loads(mass["materials"] or "{}"))
        if missing or not materials:
            continue
        census = dict(materials.most_common())
        blocks = sum(materials.values())
        if census != json.loads(row["materials"] or "{}") or value.get("blocks") != blocks:
            value["blocks"] = blocks
            store.db.execute("UPDATE structures SET materials = ?, value = ? WHERE id = ?",
                             (json.dumps(census), json.dumps(value), row["id"]))
            changed += 1
        if store.reaffirm("structures", row["id"], now_ms, tick):
            changed += 1
    store.commit()
    return changed


def landmark_blocks(store, row) -> int | None:
    """A structure's current size from the masses it comprises, which are re-measured on
    every read, rather than from its own row, which waits for the next boxing pass."""
    ids = facts_of(row).get("comprises") or []
    if not ids:
        return None
    total = 0
    for mid in ids:
        mass = store.get("masses", mid)
        if mass is None:
            return None
        total += facts_of(mass).get("blocks") or 0
    return total


def mass_at(store, dim: str, pos, pad: int = 1):
    """The smallest mass whose box contains a point, or None."""
    if not pos:
        return None
    rows = store.all(
        "masses",
        "dim = ? AND min_x - ? <= ? AND max_x + ? >= ? AND min_y - ? <= ? "
        "AND max_y + ? >= ? AND min_z - ? <= ? AND max_z + ? >= ?",
        (dim, pad, pos[0], pad, pos[0], pad, pos[1], pad, pos[1], pad, pos[2], pad, pos[2]))
    rows = [row for row in rows if is_mass(row)]
    if not rows:
        return None
    return min(rows, key=lambda row: (volume(*box_of(row)), row["id"]))


def place_of(store, dim: str, pos):
    """The row an event at ``pos`` belongs to: its landmark if the mass has one, else the
    mass itself. Structures and generated components are checked by the caller first."""
    mass = mass_at(store, dim, pos)
    if mass is None:
        return None
    if mass["place_id"]:
        owner = (store.get("structures", mass["place_id"])
                 or store.get("generated_features", mass["place_id"]))
        if owner is not None:
            return owner
    return mass


def relink(store, dim: str | None = None, lo=None, hi=None) -> int:
    """Attach history rows that have no place, or a mass as their place, to current masses.

    Only rows with a null place or a mass-valued place are touched: a row already inside a
    structure or a generated building keeps that, because those were resolved from more
    specific evidence (a measured box, a workstation, a loot table).
    """
    changed = 0
    dims = [dim] if dim else sorted({row["dim"] for row in current(store)})
    for d in dims:
        rows = current(store, d)
        boxes = [(volume(*box_of(row)), box_of(row), row) for row in rows]
        boxes.sort(key=lambda item: item[0])
        landmark_of = {}
        for _, _, row in boxes:
            owner = row["place_id"]
            if owner and not (store.get("structures", owner)
                              or store.get("generated_features", owner)):
                owner = None
            landmark_of[row["id"]] = owner or row["id"]
        for table in ("event_history", "activities"):
            clause = (f"dim = ? AND x IS NOT NULL AND (place_id IS NULL OR place_id LIKE 'm\\_%' "
                      f"ESCAPE '\\')")
            params: tuple = (d,)
            if lo is not None:
                clause += " AND x BETWEEN ? AND ? AND y BETWEEN ? AND ? AND z BETWEEN ? AND ?"
                params += (lo[0], hi[0], lo[1], hi[1], lo[2], hi[2])
            for row in store.db.execute(
                    f"SELECT id, place_id, x, y, z FROM {table} WHERE {clause}",
                    params).fetchall():
                place = None
                for _, (a, b), mass in boxes:
                    if (a[0] - 1 <= row["x"] <= b[0] + 1 and a[1] - 1 <= row["y"] <= b[1] + 1
                            and a[2] - 1 <= row["z"] <= b[2] + 1):
                        place = landmark_of[mass["id"]]
                        break
                if place != row["place_id"]:
                    store.db.execute(f"UPDATE {table} SET place_id = ? WHERE id = ?",
                                     (place, row["id"]))
                    changed += 1
    store.commit()
    return changed


# --------------------------------------------------------------------------- roles

ROLE_SYSTEM = """\
You are given measured facts about one connected mass of blocks standing in a Minecraft
world. Nobody has named it. Say what it most likely is FOR, from the facts alone.

You cannot see it. Everything you have was measured by code from a fresh read of the world
and from the recorded history of who placed what. Read the facts as a builder would:
- `dims` is width x height x depth; `levels` is how many blocks sit at each height.
- `footing` says what the lowest blocks rest on: natural ground, open air, water, or other
  built blocks. A mass mostly over air or water is spanning something.
- `natural_contact` counts blocks pressed against natural terrain — a thing dug into or
  leaning on a hillside, cliff or cave wall.
- `elapsed_ms` is how long its construction took, `edits` how many separate visits it took.

Roles:
- landmark: a thing a person would point at and name — a house, tower, pen, statue, bridge
  someone would call a bridge, a rail line, a garden.
- utility: made to get somewhere or do a job and then left — a pillar climbed to escape a
  hole or reach a cliff, a scaffold, a line of torches, a dirt ramp, a temporary wall.
- decoration: small deliberate ornament — a flower ring, a marker post, a sign.
- debris: leftover or accidental — a scattering of blocks, a dropped stack, spillage.
- unknown: the facts genuinely do not say.

Reply as JSON only:
{"role": "landmark|utility|decoration|debris|unknown",
 "noun": "<two or three plain words, e.g. 'dirt pillar', 'plank footbridge'>",
 "confidence": <0-1>,
 "why": "<one short sentence citing the facts>"}"""


class RoleOutput(BaseModel):
    role: Literal["landmark", "utility", "decoration", "debris", "unknown"] = "unknown"
    noun: str = ""
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    why: str = ""


def role_prompt(row) -> str:
    """Facts only. Coordinates and ids are withheld: they are not evidence of purpose."""
    facts = facts_of(row)
    elapsed = None
    if facts.get("first_built_ms") and facts.get("last_edited_ms"):
        elapsed = max(0, int(facts["last_edited_ms"]) - int(facts["first_built_ms"]))
    described = {
        "blocks": facts.get("blocks"), "dims": facts.get("dims"), "fill": facts.get("fill"),
        "levels": facts.get("levels"), "footing": facts.get("footing"),
        "natural_contact": facts.get("natural_contact"),
        "materials": dict(list(json.loads(row["materials"] or "{}").items())[:8]),
        "origin": facts.get("origin"),
        "builders_observed": len(facts.get("builders") or {}),
        "observed_share": facts.get("observed_share"),
        "edits": facts.get("edits"), "elapsed_ms": elapsed,
        "placed_events": facts.get("placed"), "broken_events": facts.get("broken"),
        "cut_by_read_edge": facts.get("partial") or [],
        "inside_generated_site": bool(facts.get("generated_id")),
    }
    return json.dumps(described, indent=1, sort_keys=True)


def role_key(prompt: str, model: str) -> str:
    return "role:" + hashlib.sha256((model + prompt).encode()).hexdigest()[:16]


def _has_landmark(store, row) -> bool:
    place = row["place_id"]
    if not place:
        return False
    if store.get("structures", place):
        return True
    generated = store.get("generated_features", place)
    return bool(generated and generated["min_x"] is not None)


#: How much a mass must change before its role is worth asking about again.
#:
#: The role prompt is content-hashed, so a single block joining a path made a new hash and
#: a new call. One mass was asked four times in three minutes and came back "landmark" then
#: "utility" for the same dirt path: the answers were not more accurate, only more
#: expensive, and they flapped. What a thing is FOR does not change when it grows by a
#: block; it changes when it grows into something else.
ROLE_RECHECK_CHANGE = 0.25


def _role_is_current(existing, row, key: str) -> bool:
    """Whether a stored role still describes this mass well enough to keep."""
    try:
        keys = json.loads(existing["source_event_ids"] or "[]")
    except (TypeError, ValueError):
        keys = []
    if key in keys:
        return True                     # identical facts; asking again is pure waste
    try:
        asked = json.loads(existing["value"] or "{}") or {}
    except (TypeError, ValueError):
        return False
    was = asked.get("blocks")
    now = facts_of(row).get("blocks")
    if not was or not now:
        return False
    return abs(now - was) / max(1, was) < ROLE_RECHECK_CHANGE


def pending_roles(store, model: str, dim: str | None = None, lo=None, hi=None) -> list:
    """Masses worth asking about: no role yet, or one whose subject has materially changed."""
    rows = within(store, dim, lo, hi) if (dim and lo is not None) else current(store, dim)
    out = []
    for row in rows:
        if (facts_of(row).get("blocks") or 0) < ROLE_FLOOR or _has_landmark(store, row):
            continue
        existing = store.get("relationships", f"role:{row['id']}")
        key = role_key(role_prompt(row), model)
        if existing is not None and _role_is_current(existing, row, key):
            continue
        out.append(row)
    return out


async def classify_roles(store, rows: list, model: str | None = None,
                         concurrency: int = 3, log=None) -> int:
    """One cheap text call per mass, cached by the facts it was asked about."""
    from config import CLASSIFY_MODEL, thinking_for
    from model_api import model_client

    model = model or CLASSIFY_MODEL
    client = model_client(asynchronous=True)
    gate = asyncio.Semaphore(concurrency)
    done = 0

    async def one(row) -> bool:
        prompt = role_prompt(row)
        async with gate:
            try:
                reply = await client.messages.parse(
                    model=model, max_tokens=400, thinking=thinking_for(model),
                    system=ROLE_SYSTEM, output_format=RoleOutput,
                    messages=[{"role": "user", "content": [{"type": "text", "text": prompt}]}])
            except Exception as e:  # noqa: BLE001 - a failed call is logged, never stored
                if log:
                    log(f"role call failed for {row['id']}: {type(e).__name__}: {e}")
                return False
        parsed = reply.parsed_output
        if parsed is None:
            return False
        store.put("relationships", {
            "id": f"role:{row['id']}", "subject": row["id"], "predicate": "role",
            "object": parsed.role,
        }, Belief(provenance=Provenance.INFERRED, confidence=float(parsed.confidence),
                  verified_at_tick=row["verified_at_tick"],
                  verified_at_ms=int(time.time() * 1000),
                  source_event_ids=(role_key(prompt, model),),
                  value=json.dumps({"noun": parsed.noun, "why": parsed.why,
                                    "model": model,
                                    # What it was when asked, so growing by a block does
                                    # not buy another opinion.
                                    "blocks": facts_of(row).get("blocks")})))
        if log:
            log(f"  {row['id']}: {parsed.role} — {parsed.noun} ({parsed.confidence:.2f})")
        return True

    for ok in await asyncio.gather(*(one(row) for row in rows)):
        done += int(bool(ok))
    return done


# --------------------------------------------------------------------------- views

def nearby_summary(store, dim: str, focus, radius: int = 96) -> str:
    """One line on the unnamed built things near a point, for the god's context."""
    if not focus:
        return ""
    roles = {r["subject"]: r for r in store.all("relationships", "predicate = 'role'")}
    grouped: dict = collections.defaultdict(list)
    small: collections.Counter = collections.Counter()
    small_count = 0
    for row in current(store, dim):
        if _has_landmark(store, row):
            continue
        cx = (row["min_x"] + row["max_x"]) / 2
        cz = (row["min_z"] + row["max_z"]) / 2
        if max(abs(cx - focus[0]), abs(cz - focus[2])) > radius:
            continue
        facts = facts_of(row)
        if (facts.get("blocks") or 0) < ROLE_FLOOR:
            small_count += 1
            top = next(iter(json.loads(row["materials"] or "{}")), "?")
            small[top] += 1
            continue
        role = roles.get(row["id"])
        if role is None:
            grouped["unclassified"].append(facts.get("dims", "?"))
            continue
        try:
            noun = json.loads(role["value"] or "{}").get("noun") or facts.get("dims")
        except (TypeError, ValueError):
            noun = facts.get("dims")
        grouped[role["object"]].append(noun)
    if not grouped and not small_count:
        return ""
    parts = []
    for role, nouns in sorted(grouped.items()):
        counted = collections.Counter(nouns)
        parts.append(f"{len(nouns)} {role} ("
                     + ", ".join(f"{n}x {noun}" if n > 1 else noun
                                 for noun, n in counted.most_common(4)) + ")")
    if small_count:
        parts.append(f"{small_count} small placements under {ROLE_FLOOR} blocks ("
                     + ", ".join(f"{k}" for k, _ in small.most_common(3)) + ")")
    return "unnamed built things standing nearby: " + "; ".join(parts)


def audit(store) -> list[str]:
    """Mechanical invariants over the ledger and its link to landmarks."""
    issues = []
    rows = store.all("masses")
    for row in rows:
        if not is_mass(row):
            issues.append(f"{row['id']}: row in masses without the ledger schema marker")
            continue
        lo, hi = box_of(row)
        if any(lo[i] > hi[i] for i in range(3)):
            issues.append(f"{row['id']}: inside-out bounding box")
        if row["provenance"] != "SCANNED":
            issues.append(f"{row['id']}: a mass must be SCANNED, is {row['provenance']}")
        if not row["verified_at_ms"]:
            issues.append(f"{row['id']}: no wall-clock observation time")
        place = row["place_id"]
        if place and not (store.get("structures", place)
                          or store.get("generated_features", place)):
            issues.append(f"{row['id']}: place {place} does not exist")
        facts = facts_of(row)
        if facts.get("origin") not in ("player_built", "world_generated", "unknown"):
            issues.append(f"{row['id']}: origin {facts.get('origin')!r} is not a known kind")
        if facts.get("origin") == "player_built" and not row["actor"]:
            issues.append(f"{row['id']}: player_built with no builder")
    live = [row for row in rows if is_mass(row)]
    for i, left in enumerate(live):
        for right in live[i + 1:]:
            if left["dim"] != right["dim"]:
                continue
            rlo, rhi = box_of(right)
            if match_score(left, {"dim": right["dim"], "lo": rlo, "hi": rhi,
                                  "materials": right["materials"]}) >= 0.85:
                issues.append(f"{left['id']} and {right['id']}: likely duplicate mass")
    by_place: dict = collections.defaultdict(set)
    for row in live:
        if row["place_id"]:
            by_place[row["place_id"]].add(row["id"])
    for row in current_structures(store):
        comprises = set(facts_of(row).get("comprises") or [])
        if not comprises:
            issues.append(f"{row['id']}: structure anchored to no mass")
        elif comprises != by_place.get(row["id"], set()):
            issues.append(f"{row['id']}: comprises does not match masses' place_id")
        for mid in comprises:
            if not store.get("masses", mid):
                issues.append(f"{row['id']}: comprises missing mass {mid}")
    for name in ("event_history", "activities"):
        stray = store.db.execute(
            f"SELECT COUNT(*) FROM {name} WHERE place_id LIKE 'm\\_%' ESCAPE '\\' "
            f"AND place_id NOT IN (SELECT id FROM masses)").fetchone()[0]
        if stray:
            issues.append(f"{name}: {stray} rows point at masses that no longer exist")
    return issues


# --------------------------------------------------------------------------- rebuild

REGION = 64
MARGIN = 8
MAX_HEIGHT = 48          # the plugin's cap on one voxel read
OVERLAP = 8


def windows_for_cell(store, dim: str, cx: int, cz: int) -> list[tuple[list, list]]:
    """Read windows covering one occupied region cell, sized to the plugin's caps.

    Height comes from where events actually happened in the cell, not the whole column:
    nothing is built where nobody has been. Tall cells are read in overlapping slabs; a
    mass cut by a slab edge is recorded partial and a later live read makes it whole.
    """
    x0, z0 = cx * REGION, cz * REGION
    row = store.db.execute(
        "SELECT MIN(y), MAX(y) FROM event_history WHERE dim = ? AND y IS NOT NULL "
        "AND x BETWEEN ? AND ? AND z BETWEEN ? AND ?",
        (dim, x0, x0 + REGION - 1, z0, z0 + REGION - 1)).fetchone()
    if row is None or row[0] is None:
        return []
    y_lo, y_hi = max(-64, int(row[0]) - 6), min(319, int(row[1]) + 12)
    out = []
    y = y_lo
    while True:
        top = min(y_hi, y + MAX_HEIGHT - 1)
        out.append(([x0 - MARGIN, y, z0 - MARGIN],
                    [x0 + REGION - 1 + MARGIN, top, z0 + REGION - 1 + MARGIN]))
        if top >= y_hi:
            break
        y = top - OVERLAP + 1
    return out


async def rebuild(store, url: str, wipe: bool = False, log=print) -> dict:
    from scan import request_voxels, to_voxels

    if wipe:
        store.db.execute("DELETE FROM relationships WHERE subject LIKE 'm\\_%' ESCAPE '\\'")
        store.db.execute("DELETE FROM masses")
        store.commit()
        log("wiped the ledger")
    cells = store.all("regions")
    reads = failed = written = retired = 0
    for cell in cells:
        for lo, hi in windows_for_cell(store, cell["dim"], cell["cx"], cell["cz"]):
            reads += 1
            try:
                result = await request_voxels(url, cell["dim"], lo, hi)
            except Exception as e:  # noqa: BLE001
                failed += 1
                log(f"  {cell['id']} {lo}..{hi}: {type(e).__name__}")
                continue
            if not result.get("ok"):
                failed += 1
                log(f"  {cell['id']} {lo}..{hi}: {result.get('error')}")
                continue
            voxels = to_voxels(result)
            if not voxels:
                failed += 1
                log(f"  {cell['id']} {lo}..{hi}: empty read, skipped")
                continue
            report = apply(store, cell["dim"], voxels,
                           ledger_built(store, cell["dim"], voxels, lo, hi), lo, hi,
                           int(result.get("tick") or 0))
            written += len(report["written"])
            retired += len(report["retired"])
            log(f"  {cell['id']} y{lo[1]}..{hi[1]}: {len(report['written'])} masses"
                + (f", {len(report['partial'])} partial" if report["partial"] else "")
                + (f", {len(report['retired'])} retired" if report["retired"] else ""))
    anchored = anchor(store)
    linked = relink(store)
    return {"cells": len(cells), "reads": reads, "failed": failed, "written": written,
            "retired": retired, "anchored": anchored, "relinked": linked,
            "total": len(current(store))}


def main(argv: list[str]) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--list", action="store_true")
    p.add_argument("--audit", action="store_true")
    p.add_argument("--rebuild", action="store_true")
    p.add_argument("--wipe", action="store_true", help="with --rebuild: start from nothing")
    p.add_argument("--roles", action="store_true", help="classify unnamed masses")
    p.add_argument("--relink", action="store_true")
    p.add_argument("--url", default="ws://127.0.0.1:8765")
    p.add_argument("--dim", default=None)
    args = p.parse_args(argv)

    store = Store()
    try:
        if args.rebuild:
            report = asyncio.run(rebuild(store, args.url, wipe=args.wipe))
            print(json.dumps(report))
        if args.relink:
            print(f"relinked {relink(store, args.dim)} history rows; "
                  f"anchored {anchor(store, args.dim)} changes")
        if args.roles:
            from config import CLASSIFY_MODEL, have_key
            if not have_key():
                print("no model key configured")
                return 1
            rows = pending_roles(store, CLASSIFY_MODEL, args.dim)
            print(f"{len(rows)} mass(es) to classify with {CLASSIFY_MODEL}")
            done = asyncio.run(classify_roles(store, rows, CLASSIFY_MODEL, log=print))
            print(f"classified {done}")
        if args.list:
            roles = {r["subject"]: r for r in store.all("relationships", "predicate = 'role'")}
            names = store.all_names()
            rows = sorted(current(store, args.dim),
                          key=lambda r: -(facts_of(r).get("blocks") or 0))
            for row in rows:
                facts = facts_of(row)
                place = row["place_id"] or "-"
                label = ""
                if row["place_id"] in names:
                    label = f"part of {names[row['place_id']]['object']!r}"
                elif row["id"] in roles:
                    role = roles[row["id"]]
                    label = f"{role['object']}: " + (
                        json.loads(role["value"] or "{}").get("noun") or "")
                print(f"{row['id']}  {facts.get('blocks', 0):5d} {facts.get('dims', ''):>10}"
                      f"  {facts.get('origin', '?'):15} {row['actor'] or '-':38}"
                      f"  {place:24} {label}"
                      + ("  PARTIAL" if facts.get("partial") else ""))
            print(f"{len(rows)} masses")
        if args.audit:
            issues = audit(store)
            for issue in issues:
                print(f"ERROR  {issue}")
            print("ok" if not issues else f"{len(issues)} issue(s)")
            return 1 if issues else 0
    finally:
        store.close()
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
