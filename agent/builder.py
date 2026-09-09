#!/usr/bin/env python3
"""The builder: a model whose whole job is turning a description into blocks.

Building is not conversation. Holding a shape in mind, deciding where its shoulders and its
roofline go, and writing the several hundred commands that put them there is a different
skill from answering a question well, and the statues showed it: asked for a humanoid figure,
a dialogue model produced something nobody would recognise as one.

So it is a separate model behind a separate tool, configured by ``MCGOD_BUILD_MODEL`` (or
``MCGOD_OPENROUTER_BUILD_MODEL``). The dialogue model decides what to build and where; this
decides what blocks that is. Neither has to be good at the other's job, and either can be
swapped without touching the other.
"""

from __future__ import annotations

import json
import os

from config import BUILD_MODEL, thinking_for
from evidence import complete_message
from model_api import model_client

#: A build is written once and then looked at, so it may be large. This is generous rather
#: than tuned: the cost of a truncated build is a half-finished statue in the world.
MAX_BUILD_TOKENS = 16000

#: How long the builder may take. Far longer than a chat reply, because it is a different
#: job: a figure worth looking at is worth waiting for, and the alternative to waiting is
#: not a faster build but no build.
BUILD_TIMEOUT_SECONDS = float(os.environ.get("MCGOD_BUILD_TIMEOUT", "300"))

#: How far around the anchor the site is surveyed. Wide enough to show what the thing has
#: to stand on and sit beside, small enough that the grids stay readable.
SITE_RADIUS = 24

SYSTEM = """\
You build things in Minecraft, block by block, and you are given the job because shape is
what you are good at.

You will be told what to build, where its anchor point is, sometimes which way it faces and
what it is made of, and what the ground there is actually like. Reply with the commands that
build it and nothing else.

Reading the site. You are given a survey of the ground rather than a picture of it:

- HEIGHT is a grid of the surface level of every column, written as an offset from the
  anchor's own Y. `0` is level with the anchor, `+3` is three blocks higher, `-2` is two
  lower, and `~` is a column with nothing solid in it. Read it as a contour map: that is
  where the ground is, so that is what your footing has to meet.
- SURFACE is the same grid again, one letter per material, with a legend. It tells you what
  you are building on: water and lava need a foundation, sand and gravel fall, leaves and
  logs are a tree you should probably clear or build around.
- STANDING lists anything already built inside the site and where its box is. Do not
  overwrite somebody's house. Build beside it, or on it if that is what was asked.

Both grids run west to east across a row (increasing X) and north to south down the rows
(increasing Z), and the corner coordinate is given so you can turn any cell into real
coordinates.

Curves, which are most of what separates a good build from a bad one. Minecraft has no
curves, so a round thing is a stack of circles and you have to work out each one:

- For a sphere or a balloon of radius R centred at height Yc, the layer at height y has
  radius r = sqrt(R^2 - (y - Yc)^2). Compute that for EVERY layer before you write any
  commands, and write the layers out widest-first so you can see the profile you are making.
  A balloon of R=9 goes 0, 4.0, 5.6, 6.7, 7.5, 8.1, 8.5, 8.8, 9.0 and back down. Guessing
  instead gives you a barrel with a lid, which is the single most common way a build fails.
- Draw each circle properly: a block at (x, z) is in the layer when x^2 + z^2 <= r^2, using
  the layer's own r. Do not reuse one circle at several heights.
- A dome is the top half of that. A cone or a spire shrinks linearly instead. A teardrop or
  a balloon envelope is a sphere that tapers to a neck over its bottom third.
- Never approximate a curve with a single `fill`. Fill the straight RUNS inside each circle,
  one per row of the circle, which is both round and cheap.

Scale. Build it as big as it was asked for. A "large" figure or vehicle is 15-25 blocks in
its longest dimension, a monumental one 30-50. Small builds read as models of the thing
rather than the thing, and detail you cannot fit in is worse than no detail.

How to build well:
- Work in absolute coordinates from the anchor you were given. The anchor is the point the
  thing stands on, at its centre unless told otherwise. Something described as floating or
  in the air hangs from the anchor rather than resting on it.
- Sit it on the ground the height grid describes. A figure floating two blocks up, or buried
  to the knee in a slope, is the most common way for a good shape to look wrong, and it is
  the one thing the survey exists to prevent. On a slope, either step the footing down to
  meet the ground or level a platform first.
- Prefer `setblock` for anything whose shape matters and `fill` for slabs, walls and solid
  runs. A figure made of fills looks like boxes, because it is boxes.
- Build in the order a person would: footing, then mass, then the details that make it
  recognisable. Put the recognisable parts in — a face, hands, a roofline, a doorway — since
  those are what tell someone what they are looking at.
- Use the block palette to shade. Different stones, wools, terracottas and concretes read as
  light and shadow, and a single material reads as a lump.
- Anything humanoid or animal needs proportion above all: head roughly an eighth of the
  height, shoulders wider than the head, limbs that reach where limbs reach. Get the
  silhouette right before any detail.
- Clear the space first if the ground would swallow it.

Take the room you need. There is no penalty for a long answer and a large one is expected:
several hundred commands for a figure is normal, and stopping early leaves a half-built
thing standing in somebody's world.

Work out the shape before you write the commands. State the overall dimensions and, for
anything rounded, the radius of each layer. Then write the commands, and check the count
against what you planned: a large detailed build is several hundred commands, and a hundred
means you left most of it out.

Reply as JSON only:
{"commands": ["setblock 100 64 100 minecraft:stone", ...],
 "describes": "<one short sentence naming what you built>"}"""


#: Things you cannot stand a building on. Its own list rather than the renderer's natural
#: one, because those answer different questions: ice, obsidian and moss are natural and are
#: perfectly good ground, while a fern is neither. Getting a plant wrong here costs one block
#: of reported height; getting grass_block wrong reports the whole site as empty air.
NOT_FOOTING_SUFFIX = (
    "_leaves", "_sapling", "_mushroom", "_flower", "_tulip", "_orchid", "_bush", "_fungus",
    "_roots", "_coral_fan", "_sprouts", "_vine", "_vines", "_lichen", "_carpet", "_sign",
    "_banner", "_torch", "_button", "_plant", "_petals",
)
NOT_FOOTING = {
    "air", "cave_air", "void_air", "short_grass", "tall_grass", "fern", "large_fern",
    "dead_bush", "vine", "glow_lichen", "cobweb", "sugar_cane", "cactus", "bamboo",
    "bamboo_sapling", "kelp", "kelp_plant", "seagrass", "tall_seagrass", "lily_pad",
    "sunflower", "lilac", "peony", "rose_bush", "torchflower", "pitcher_plant",
    "wildflowers", "leaf_litter", "short_dry_grass", "tall_dry_grass", "snow",
    "sea_pickle", "spore_blossom", "big_dripleaf", "big_dripleaf_stem", "small_dripleaf",
    "cave_vines", "cave_vines_plant", "weeping_vines", "weeping_vines_plant",
    "twisting_vines", "twisting_vines_plant", "chorus_plant", "chorus_flower",
    "hanging_roots", "sculk_vein", "wheat", "carrots", "potatoes", "beetroots",
    "nether_wart", "sweet_berry_bush", "azalea", "flowering_azalea", "bush",
    "moss_carpet", "pale_moss_carpet", "pale_hanging_moss", "fire", "soul_fire",
    "light", "structure_void", "bubble_column",
}


def _is_footing(state: str) -> bool:
    """Could a foundation rest on this block?"""
    name = state.split("[")[0].replace("minecraft:", "")
    return name not in NOT_FOOTING and not name.endswith(NOT_FOOTING_SUFFIX)


def survey(voxels: dict, anchor, radius: int = SITE_RADIUS) -> dict:
    """The ground around the anchor, as a builder needs to see it.

    Two grids and a list, never a block array. A render says what a place looks like; it
    cannot say that this column is three blocks lower than that one, and a builder that
    cannot read the slope sets a statue's feet in the air on one side and buries them on the
    other. The grids carry their own corner coordinate so every cell is locatable exactly.

    Height is measured to the topmost SOLID, non-vegetation block, because the top of a tree
    is not the ground.
    """
    ax, ay, az = int(anchor[0]), int(anchor[1]), int(anchor[2])
    lo_x, hi_x = ax - radius, ax + radius
    lo_z, hi_z = az - radius, az + radius

    tops: dict = {}
    for (x, y, z), state in voxels.items():
        if not (lo_x <= x <= hi_x and lo_z <= z <= hi_z):
            continue
        if not _is_footing(state):
            continue
        best = tops.get((x, z))
        if best is None or y > best[0]:
            tops[(x, z)] = (y, state)

    legend: dict = {}
    height_rows, surface_rows = [], []
    for z in range(lo_z, hi_z + 1):
        height, surface = [], []
        for x in range(lo_x, hi_x + 1):
            found = tops.get((x, z))
            if found is None:
                height.append("  ~")
                surface.append(".")
                continue
            delta = found[0] - ay
            height.append(f"{delta:+3d}" if delta else "  0")
            name = found[1].split("[")[0].replace("minecraft:", "")
            if name not in legend:
                legend[name] = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"[
                    len(legend) % 52]
            surface.append(legend[name])
        height_rows.append("".join(height))
        surface_rows.append("".join(surface))

    return {
        "corner_north_west": [lo_x, lo_z],
        "rows_run": "west to east across a row (X increasing), north to south down the "
                    "rows (Z increasing)",
        "anchor_y": ay,
        "height_offsets_from_anchor_y": height_rows,
        "surface": surface_rows,
        "surface_legend": {glyph: name for name, glyph in legend.items()},
    }


def _plain(reply) -> str:
    return "".join(block.text for block in reply.content if block.type == "text").strip()


async def design(what: str, anchor, facing: str = "", materials: str = "",
                 model: str | None = None, site: dict | None = None,
                 standing: list | None = None) -> dict:
    """Ask the builder for the commands that make one thing.

    Returns ``{"commands": [...], "describes": str}``, or ``{"error": ...}``. Nothing here
    validates the commands: they go through the same gate as anything else the god runs, and
    a build that names a forbidden command is refused there like any other.
    """
    model = model or BUILD_MODEL
    client = model_client(asynchronous=True)
    request = {
        "build": what,
        "anchor": [int(anchor[0]), int(anchor[1]), int(anchor[2])],
        "facing": facing or "unspecified",
        "materials": materials or "your choice",
    }
    if site:
        request["HEIGHT and SURFACE, the ground you are building on"] = site
    if standing:
        request["STANDING here already, do not overwrite it"] = standing
    try:
        reply = await complete_message(
            client, model=model, thinking=thinking_for(model), system=SYSTEM,
            max_tokens=MAX_BUILD_TOKENS, timeout=BUILD_TIMEOUT_SECONDS,
            messages=[{"role": "user", "content": json.dumps(request, indent=1)}])
    except Exception as error:  # noqa: BLE001 - a builder failure is not a world failure
        # With the class alone this read as a provider problem for four retries, while the
        # real cause was a value in our own request that would not serialise.
        return {"error": f"the builder did not answer ({type(error).__name__}: {error})"}
    text = _plain(reply)
    try:
        body = json.loads(text[text.find("{"):text.rfind("}") + 1])
    except (ValueError, json.JSONDecodeError):
        return {"error": "the builder did not return commands"}
    commands = [str(c) for c in (body.get("commands") or []) if str(c).strip()]
    if not commands:
        return {"error": "the builder returned no commands"}
    return {"commands": commands, "describes": str(body.get("describes") or what),
            "model": model, "count": len(commands)}
