#!/usr/bin/env python3
"""Deciding what counts as a structure, from the world rather than from the event log.

    python3 segment.py --at 232 68 295

The census that came before this clustered *events* inside one episode. That made a
structure a temporal object: what you happened to place in one sitting. Coming back a week
later to add a floor produced a second structure, two touching buildings merged because
their blocks touched, and a build nobody watched being made did not exist at all.

Segmentation here is spatial and re-run on demand. The world is the source of truth; events
only say where to look. Every time a place is disturbed it is read again and re-segmented
from scratch, so nothing is permanent and a structure is free to grow, shrink, split or
merge as the world does.

Two decisions are deliberately separated. The model proposes semantic boundaries from
rendered and coordinate-bearing views. Code clips, measures and rejects partial or empty
boxes, then assigns stable identities one-to-one from geometry and material similarity.
Names never determine identity and missing model output never proves demolition.
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
from typing import Literal

from pydantic import BaseModel, Field

from render import is_natural

#: How far apart two placed blocks can be and still be the same thing.
REACH = 2

# A semantic proposal still has to contain enough measured world geometry to be more than
# an accidental click.  Eight admits compact pixel art and signs; recognition remains the
# model's job, so this is one universal evidence floor rather than a catalogue of shapes.
MIN_STRUCTURE_BLOCKS = 8

#: Families of material. Two touching groups made of different families are a likely seam —
#: a fence line against a cobblestone wall, a dirt path meeting a plank floor — so they are
#: proposed as separate and the model is asked. Within a family, touching means the same
#: thing.
FAMILIES = {
    "fence": ("fence", "fence_gate", "wall"),
    "wood": ("planks", "log", "wood", "stairs_oak", "door", "trapdoor"),
    "stone": ("cobblestone", "stone", "deepslate", "andesite", "brick", "tuff"),
    "path": ("path", "farmland", "gravel"),
    "rail": ("rail",),
    "glass": ("glass",),
}


def family_of(material: str) -> str:
    name = material.split("[")[0].replace("minecraft:", "")
    for family, marks in FAMILIES.items():
        if any(mark in name for mark in marks):
            return family
    return "other"


def built_blocks(voxels: dict) -> dict:
    """Just what someone put there — not the ground, the ore, the trees or the flowers."""
    return {p: m for p, m in voxels.items() if not is_natural(m)}


def clusters(built: dict, reach: int = REACH, split_families: bool = False) -> list:
    """Connected groups of placed blocks.

    Plain connectivity by default. Cutting at every material seam was tried first and is
    far too eager: a cottage of stone footings, plank walls and a fence around it came out
    as three separate structures, which is not what anybody means by a house. Material
    composition is reported instead, and the model is asked where the seams really are —
    a question about meaning, which is not connectivity's to answer.
    """
    remaining = set(built)
    groups = []
    while remaining:
        seed = remaining.pop()
        group, frontier = [seed], [seed]
        seed_family = family_of(built[seed])
        while frontier:
            cx, cy, cz = frontier.pop()
            near = [p for p in remaining
                    if abs(p[0] - cx) <= reach and abs(p[1] - cy) <= reach
                    and abs(p[2] - cz) <= reach
                    and (not split_families or family_of(built[p]) == seed_family)]
            for p in near:
                remaining.discard(p)
                group.append(p)
                frontier.append(p)
        groups.append(group)
    return sorted(groups, key=len, reverse=True)


def describe_cluster(group: list, built: dict) -> dict:
    """The facts about one proposed structure."""
    xs, ys, zs = zip(*group)
    dims = [max(xs) - min(xs) + 1, max(ys) - min(ys) + 1, max(zs) - min(zs) + 1]
    materials = collections.Counter(
        built[p].split("[")[0].replace("minecraft:", "") for p in group)
    return {
        "lo": [min(xs), min(ys), min(zs)], "hi": [max(xs), max(ys), max(zs)],
        "blocks": len(group),
        "dims": f"{dims[0]}x{dims[1]}x{dims[2]}",
        "fill": round(len(group) / max(1, dims[0] * dims[1] * dims[2]), 3),
        "vertical": dims[1] >= 3,
        "materials": dict(materials.most_common(8)),
        "family": family_of(built[group[0]]),
    }


#: What the model is asked, when it is asked to draw the boundaries itself.
#:
#: Block-level rules proposed the segments before this, and they could not be made to work.
#: Cutting at material seams split one cottage into three; plain connectivity fused a house,
#: a pen and two paths into a single lump; and every fix for one case broke another, because
#: "where does this building end" is not a fact about block adjacency. It is a judgement
#: about what someone meant, and there is no rule that survives contact with a real world.
#:
#: So the model draws the boxes. It is given the picture, a set of horizontal slices with
#: real coordinates so it can be precise, and whatever is already known to stand there.
SEGMENT = """\
You are looking at one patch of a Minecraft world. Say what separate structures stand in it.

The main image is always one canonical 2x2 sheet: perspective at upper left, orthographic
top at upper right, front elevation at lower left, and side elevation at lower right. Empty
space between blocks in those views is real air, not missing rendering. You are also given:
sometimes a separate plan view with a world-coordinate grid; horizontal slices as character
grids; the geometry of structures already on record there; and player-work footprints from
history. Prior names are deliberately withheld: classify fresh evidence independently.

You are also given `built_masses`: every connected mass of built blocks in the patch, as
code measured it. That geometry is exact. Anything you leave out of your answer stays on
record as an unnamed built mass, so omitting a pillar, a path or a scaffold loses nothing —
you are choosing what deserves a name, not what exists. A structure you name may group
several masses (a house and the fence touching it) or be one mass alone.

A player-work footprint is an attention cue, not a structure and not its current geometry.
It says that a player deliberately placed blocks in that area, possibly followed by removals.
Inspect the current image and slices there. If the surviving arrangement reads as one named
thing — including compact decoration, pixel art, a sign, or a mosaic — include it. If it is
merely scaffolding, mobility blocks, random edits, or no longer stands, leave it out. Apply
that same semantic test to every footprint; do not infer a particular shape from its history.

When a separate gridded plan is present, use it for horizontal boundaries: its grid is exact.
Use the perspective and elevations for height and for what things ARE. Without a gridded plan,
use the coordinate-bearing slices to make the boundary exact.

A structure is a thing a person would point at and name — a house, a tower, a pen, a rail
loop. Judge it as a person would:
- A house of stone footings, plank walls and a glass window is ONE structure, not three.
- A house with a pen leaning against it is TWO, even though they touch.
- A path, a scattering of dirt, a pillar someone climbed out of a hole on — these are not
  structures. Leave them out.
- Terrain is not a structure. Neither is a tree, an ore vein, or a patch of flowers.

Name the visual gestalt, not merely its material or construction method. For symbolic and
decorative work, prefer the familiar figure, mark, or object the arrangement depicts when
the views support one; use a generic phrase such as "pixel pattern" only when no more
specific reading is visually defensible.

For each real structure give a bounding box in world coordinates that contains it. The box
may include some ground and some of a neighbour — boxes are how a region is fetched, not
what a structure is — but it must be tight enough that the structure is plainly the subject.

Existing boxes prevent accidental duplication and show where edits may have occurred. Do not
decide their database identity; code matches observations to prior objects from measured
geometry and materials. Name every visible object from the current views, not prior belief.

Reply as JSON only:
{"structures": [{"label": "<short noun phrase>",
                 "category": "building|farm|path|excavation|other",
                 "min": [x, y, z], "max": [x, y, z],
                 "confidence": <0-1>,
                 "description": "<two or three sentences: what it is, what is in it,
                                  what it is for>"}]}"""


class Boundary(BaseModel):
    label: str
    category: Literal["building", "farm", "path", "excavation", "other"] = "other"
    min: list[int] = Field(min_length=3, max_length=3)
    max: list[int] = Field(min_length=3, max_length=3)
    confidence: float = Field(ge=0.0, le=1.0)
    description: str = ""


class SegmentationOutput(BaseModel):
    structures: list[Boundary]


#: The second look. The first pass draws boxes from a picture of a whole patch, and boxes
#: drawn that way cut things in half and swallow neighbours. So each box is rendered on its
#: own and handed back: now that you can see only this, is it one whole thing?
REFINE = """\
Each numbered object below is ONE box you drew, rendered with everything outside it removed.
Its most informative labelled view is supplied separately: top for horizontal planar work,
front or side for upright planar work, and all four views for volumetric work. Empty gaps are
real air. Together the supplied evidence is exactly what you claimed — nothing more.

Inspect every labelled view before naming the object. Thin planar work is often unreadable in
perspective: use the top view for work lying on the ground and the front or side elevation for
upright work. A horizontal symbol has no privileged viewing direction; horizontal-slice data
therefore includes a tightly cropped shape at all four quarter-turn rotations. Check those
before declaring it generic or ruined. They are for meaning only—the north-up `rows` remain
the source for coordinates. Read disconnected blocks together when they form a familiar
visual gestalt.
If the boundary is right but the label is not, return `keep` with the corrected label and
description.

Two ways a box goes wrong, and both are visible here:

1. **It caught two things.** If the picture shows a tower AND a rail loop, or a house AND
   the pen beside it, that is two structures. Split it: keep one in min/max and put the
   other in `also`. A house made of stone footings and plank walls is still ONE thing —
   split what a person would name separately, not what is made of different blocks.

2. **It cut a structure in half.** You are told, per face, how many built blocks sit
   immediately outside that face of the box. A wall of blocks pressed against one face
   means the structure carries on past it and the box is too small — grow it. A handful is
   a neighbour or a path and can be left alone.

Most boxes will be right. Only return the ones whose boundary or semantic label needs a
correction.

Reply as JSON only:
{"fixes": [{"index": <which picture>,
            "action": "grow|shrink|split|drop|keep",
            "min": [x,y,z], "max": [x,y,z],
            "also": [{"label": "...", "min": [x,y,z], "max": [x,y,z],
                      "description": "..."}],
            "label": "<corrected name, or the same one>",
            "description": "<corrected description>",
            "why": "<a few words>"}]}

Use `also` only when splitting: the box you keep goes in min/max, the other piece in `also`.
Omit an index entirely if it is fine."""


class SplitPiece(BaseModel):
    label: str
    min: list[int] = Field(min_length=3, max_length=3)
    max: list[int] = Field(min_length=3, max_length=3)
    description: str = ""
    confidence: float = Field(default=0.7, ge=0.0, le=1.0)


class BoundaryFix(BaseModel):
    index: int
    action: Literal["grow", "shrink", "split", "drop", "keep"] = "keep"
    min: list[int] | None = Field(default=None, min_length=3, max_length=3)
    max: list[int] | None = Field(default=None, min_length=3, max_length=3)
    also: list[SplitPiece] = Field(default_factory=list)
    label: str = ""
    description: str = ""
    why: str = ""
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)


class RefinementOutput(BaseModel):
    fixes: list[BoundaryFix]


def spilling(built: dict, lo, hi, reach: int = 3) -> dict:
    """How much built material presses against each face of a box, just outside it.

    A picture of a box with everything else removed cannot show that the building carries on
    past the edge — the part that would show it is exactly what was cropped away. This is the
    measurement that makes "you cut this in half" answerable: a wall of blocks against one
    face means the structure continues, where a handful means a neighbour or a path.
    """
    out = {}
    for axis, low_name, high_name in ((0, "west", "east"), (1, "below", "above"),
                                      (2, "north", "south")):
        for name, side in ((low_name, "lo"), (high_name, "hi")):
            count = 0
            for p in built:
                if any(lo[i] - reach <= p[i] <= hi[i] + reach for i in range(3)) is False:
                    continue
                inside_others = all(lo[i] <= p[i] <= hi[i]
                                    for i in range(3) if i != axis)
                if not inside_others:
                    continue
                if side == "lo" and lo[axis] - reach <= p[axis] < lo[axis]:
                    count += 1
                elif side == "hi" and hi[axis] < p[axis] <= hi[axis] + reach:
                    count += 1
            if count:
                out[name] = count
    return out


ADJUDICATE = """\
You are shown a picture of one connected mass of blocks a player has built, and a list of
the parts it breaks into by material. Decide how many separate things are really there.

Connectivity cannot answer this. A cottage of stone footings, plank walls and a glass window
is one building even though it is three materials. A house with a fenced animal pen leaning
against it is two things even though they touch. Only meaning separates them, which is why
you are being asked.

Group the parts. Every part must go in exactly one group. A group is one structure — the
thing a person would point at and name.

Reply as JSON only:
{"structures": [{"parts": [<part numbers>], "label": "<short noun phrase>",
                 "category": "building|farm|path|excavation|harvest|clearing|other",
                 "confidence": <0-1>,
                 "description": "<two or three sentences: what it is, what is in it, what
                                  it is for>"}]}"""


def parts_of(group: list, built: dict, floor: int = 8) -> list:
    """Break one connected mass into the pieces a seam might fall between.

    These are proposals, not conclusions. The split is by material family, which is where
    seams usually are, and small remnants are folded into the nearest larger part so the
    model is not handed forty fragments to reason about.
    """
    inner = {p: built[p] for p in group}
    pieces = [g for g in clusters(inner, split_families=True) if len(g) >= floor]
    placed = {p for piece in pieces for p in piece}
    leftovers = [p for p in group if p not in placed]
    for p in leftovers:
        if not pieces:
            break
        nearest = min(pieces, key=lambda piece: min(
            abs(p[0] - q[0]) + abs(p[1] - q[1]) + abs(p[2] - q[2]) for q in piece))
        nearest.append(p)
    return pieces


def overlap(a: dict, b_row) -> int:
    """How many blocks of a proposed cluster fall inside a structure already on record."""
    if b_row["min_x"] is None:
        return 0
    inside = 1
    for i, (lo_key, hi_key) in enumerate((("min_x", "max_x"), ("min_y", "max_y"),
                                          ("min_z", "max_z"))):
        lo = max(a["lo"][i], b_row[lo_key])
        hi = min(a["hi"][i], b_row[hi_key])
        if hi < lo:
            return 0
        inside *= hi - lo + 1
    return inside


def match(proposals: list, existing: list) -> list:
    """Pair each proposed structure with the one already on record it continues.

    Identity is spatial, so returning to a build a week later is an edit of that build
    rather than a new one beside it.

    Assignment is exclusive and settled by best overlap first. Letting several proposals
    claim the same record put a farmhouse and the pen beside it under one id, and the
    second silently overwrote the first — the house vanished and the pen inherited its
    name. A record continues exactly one thing.
    """
    claims = []
    for index, proposal in enumerate(proposals):
        for row in existing:
            shared = overlap(proposal, row)
            if shared > 0:
                claims.append((shared, index, row["id"]))
    claims.sort(reverse=True)

    taken_by: dict = {}
    given: dict = {}
    absorbed: dict = {}
    for shared, index, sid in claims:
        if sid in taken_by:
            # Already continued by a better-overlapping proposal. The loser notes it as
            # absorbed only if it is genuinely swallowing it, not merely adjacent.
            continue
        if index in given:
            absorbed.setdefault(index, []).append(sid)
            taken_by[sid] = index
            continue
        given[index] = sid
        taken_by[sid] = index

    return [{"facts": proposal,
             "continues": given.get(i),
             "absorbs": absorbed.get(i, [])}
            for i, proposal in enumerate(proposals)]


def slices(voxels: dict, lo, hi, levels: int = 4) -> list:
    """Horizontal cuts through a region as character grids with real coordinates.

    The picture says what things are; it cannot say where they are to the block. These can.
    Together they let the model give a boundary in world coordinates without anyone having
    to guess at a scale from an image.
    """
    from render import is_natural

    out = []
    occupied_y = sorted({p[1] for p, state in voxels.items()
                         if lo[0] <= p[0] <= hi[0]
                         and lo[1] <= p[1] <= hi[1]
                         and lo[2] <= p[2] <= hi[2]
                         and not is_natural(state)})
    if not occupied_y:
        return out
    if len(occupied_y) <= levels:
        selected_y = occupied_y
    elif levels <= 1:
        selected_y = [occupied_y[len(occupied_y) // 2]]
    else:
        # Sample actual occupied levels, never arbitrary heights in the scan volume. The
        # old fixed interval routinely returned four empty slices while a one-block-high
        # mosaic sat between them, depriving the model of the exact evidence promised here.
        indices = {round(i * (len(occupied_y) - 1) / (levels - 1))
                   for i in range(levels)}
        selected_y = [occupied_y[i] for i in sorted(indices)]
    for y in selected_y:
        legend: dict = {}
        rows = []
        for z in range(lo[2], hi[2] + 1):
            line = []
            for x in range(lo[0], hi[0] + 1):
                state = voxels.get((x, y, z))
                if state is None:
                    line.append(".")
                elif is_natural(state):
                    # Slices answer where player-made geometry is. Encoding every tuft of
                    # grass as a second glyph broke sparse symbols into visual noise and
                    # invited the model to read them as ruins. Terrain remains visible in
                    # the render; here it is intentionally the same empty background as air.
                    line.append(".")
                else:
                    name = state.split("[")[0].replace("minecraft:", "")
                    if name not in legend:
                        legend[name] = "#*o+x=%&$@ABCDEFGH"[len(legend) % 17]
                    line.append(legend[name])
            rows.append("".join(line))
        occupied = [(z, x) for z, row in enumerate(rows)
                    for x, char in enumerate(row) if char != "."]
        min_z = min(z for z, _ in occupied)
        max_z = max(z for z, _ in occupied)
        min_x = min(x for _, x in occupied)
        max_x = max(x for _, x in occupied)
        shape = [row[min_x:max_x + 1] for row in rows[min_z:max_z + 1]]

        def turn(grid):
            return ["".join(grid[len(grid) - 1 - row][column]
                            for row in range(len(grid)))
                    for column in range(len(grid[0]))]

        rotated = []
        for degrees in (0, 90, 180, 270):
            rotated.append({"degrees": degrees, "rows": shape})
            shape = turn(shape)
        out.append({"y": y, "x_from": lo[0], "z_from": lo[2],
                    "legend": {v: k for k, v in legend.items()},
                    "rows": rows,
                    # Coordinate-free rotations are semantic aids only. `rows` above stays
                    # north-up and is the sole source for exact boundaries.
                    "cropped_shape_rotations": rotated})
    return out


def main(argv: list[str]) -> int:
    import asyncio

    from scan import request_voxels, to_voxels
    from store import Store

    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--at", nargs=3, type=int, required=True, metavar=("X", "Y", "Z"))
    p.add_argument("--span", type=int, default=24)
    p.add_argument("--split", action="store_true",
                   help="also cut at material seams, to see where they fall")
    args = p.parse_args(argv)

    x, y, z = args.at
    s = args.span

    async def read():
        return await request_voxels("ws://127.0.0.1:8765", "overworld",
                                    [x - s, max(-64, y - 12), z - s],
                                    [x + s, y + 28, z + s])

    result = asyncio.run(read())
    if not result.get("ok"):
        print(result.get("error"))
        return 1
    built = built_blocks(to_voxels(result))
    groups = [g for g in clusters(built, split_families=args.split) if len(g) >= 12]
    store = Store()
    from structures import current
    existing = current(store)
    matched = match([describe_cluster(g, built) for g in groups], existing)
    print(f"{len(built)} placed blocks -> {len(groups)} proposed structures")
    for m in matched:
        f = m["facts"]
        print(f"  {f['blocks']:5d} {f['dims']:>10} {f['family']:8} at {f['lo']}"
              f"  {'continues ' + m['continues'] if m['continues'] else 'NEW'}"
              + (f"  absorbs {m['absorbs']}" if m["absorbs"] else ""))
        print(f"        {list(f['materials'].items())[:4]}")
    store.close()
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
