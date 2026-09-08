#!/usr/bin/env python3
"""P8: the first real model call. Classification only.

    python3 classify.py                 # name every structure the census found
    python3 classify.py --dry-run       # build the prompts, call nothing
    python3 classify.py --check-tokens  # compare the local estimate to count_tokens

This is the "naming is the model's job, done lazily and cached" rule made real. A fresh
world read produced a canonical object's dimensions, fill and materials. Here a model reads
those facts and proposes a name. Historical placement/removal evidence lives separately in
``work_events`` and is not rendered as though it still stands.

Three rules hold the line:

Nothing the model says is evidence. A classification is stored as INFERRED with the model's
own confidence, and the store's threshold decides whether the god may ever state it without
hedging. It can never become SCANNED, and a later scan will overwrite it.

The model sees facts, never raw blocks. It gets the same summary a human would need, which
is also why it cannot quietly recount the world to suit its guess.

Results are cached by content. Re-running costs nothing for structures that have not
changed, because a classification is expensive and a structure is usually still itself.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import sys
from pathlib import Path

from typing import Literal

from pydantic import BaseModel, Field

from config import CLASSIFY_MODEL, MODEL_PROVIDER, VISION_MODEL, have_key
from consumer import BOLD, DIM, GREEN, RED, RESET, YELLOW
from model_api import (CONNECTION_ERRORS, NOT_FOUND_ERRORS, RATE_LIMIT_ERRORS,
                       STATUS_ERRORS, model_client)
from store import Belief, Provenance, Store

SYSTEM = """\
You name Minecraft structures. Where you are shown a picture it was rendered from the world
itself, from three angles; where you are not, you have measurements and nothing else.

You are given a present-tense structure's bounding box, how much of that box is filled, and
a material census. Nothing else exists.

Rules:
- Name what the measurements support and no more. "stone building" beats "wizard tower".
- fill near 1.0 is a solid mass. fill under ~0.2 in a tall box is hollow — walls, a shaft,
  or scattered work. A flat box one or two blocks tall is a floor, a field, or a path.
- Set confidence honestly. If the measurements fit several readings, say so and score low.
- `label` is free text and is what gets said aloud. `category` is a fixed bucket used to
  group things; pick the closest one even when the label is more specific. Cutting down
  trees is `harvest` whatever you call it.
- If you are told what people have DONE in a place, weigh it above the materials. Fences
  around farmland are a wheat field or an animal pen depending entirely on whether anyone
  has bred animals there, and the materials cannot tell you which.
- If you are shown a PICTURE, believe your eyes over the numbers. A shape with a doorway,
  a roof and windows is a building whatever its block count says; a neat cube of one
  material is a cube however many blocks went into it. The measurements tell you how big
  and what of; the picture tells you what it IS.
- If you are told what is ALIVE inside it, weigh that highest of all. Twelve chickens
  standing in a pen settles what the pen is; nothing else needs to be inferred. Villagers
  in a walled space make it a village or a trading hall, not a house.
- You are not being asked whether it is impressive, only what it is.

You may also be shown neighbouring current structures. Do not merge their identities merely
because their boxes overlap."""


#: A small closed set for grouping. The free-text label is what the god says; this is what
#: code groups by. Without it, "tree felling", "tree felling site" and "log removal" are
#: three separate things to every downstream consumer, and the same six trees get recalled
#: six times because no two of them agree on a name.
CATEGORIES = ("building", "excavation", "harvest", "clearing",
              "path", "farm", "demolition", "other")


class Naming(BaseModel):
    """What the model is allowed to say. Anything outside this shape is rejected."""

    label: str = Field(description="Short noun phrase, 1-4 words, lowercase")
    category: Literal[CATEGORIES] = Field(
        description="Which of the fixed categories this belongs to")
    confidence: float = Field(ge=0.0, le=1.0, description="Honest confidence, 0 to 1")
    rationale: str = Field(description="One sentence citing the measurements used")
    alternative: str = Field(description="Next most plausible reading, or empty string")


def _box(row) -> tuple:
    return (row["min_x"], row["min_y"], row["min_z"],
            row["max_x"], row["max_y"], row["max_z"])


def _overlaps(a, b, pad: int = 3) -> bool:
    ax0, ay0, az0, ax1, ay1, az1 = _box(a)
    bx0, by0, bz0, bx1, by1, bz1 = _box(b)
    return (ax0 - pad <= bx1 and bx0 - pad <= ax1
            and ay0 - pad <= by1 and by0 - pad <= ay1
            and az0 - pad <= bz1 and bz0 - pad <= az1)


def _kind(row) -> str:
    return "current world structure"


#: Events that say what a place is FOR. Materials say what something is made of and geometry
#: says what shape it is; neither can tell a chicken coop from a wheat field, because both
#: are fences around farmland. What separates them is that someone bred chickens in one.
TELLING = ("breed", "tame", "sleep", "smelt", "eat", "container_open", "container_put",
           "container_take", "mob_kill", "enchant", "villager_trade", "fish")


def instrumented_window(paths) -> tuple:
    """When the tap was actually recording the events that say what a place is for.

    The schema grew over time: the earliest sessions captured block work and movement only.
    For a structure built then, "nobody bred animals here" does not mean nobody did — it
    means nothing was watching. Reporting that silence as evidence is the same
    absence-of-evidence trap as reading an empty scan as a demolition, and it would let the
    classifier confidently call a busy place unused.
    """
    earliest = None
    for path in paths:
        for line in Path(path).open(encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if event.get("type") in TELLING:
                t = event.get("t")
                if t and (earliest is None or t < earliest):
                    earliest = t
                break
    return earliest


def activity_index(paths) -> dict:
    """Buckets telling events by region cell, once, for all structures."""
    index: dict = {}
    for path in paths:
        for line in Path(path).open(encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if event.get("type") not in TELLING:
                continue
            pos = event.get("pos")
            if not pos:
                continue
            index.setdefault((pos[0] // 16, pos[2] // 16), []).append(event)
    return index


def activity_near(row, index: dict, pad: int = 6) -> collections.Counter:
    """What people actually did inside and just around a structure."""
    counts = collections.Counter()
    lo = (row["min_x"] - pad, row["min_y"] - pad, row["min_z"] - pad)
    hi = (row["max_x"] + pad, row["max_y"] + pad, row["max_z"] + pad)
    for cx in range(lo[0] // 16, hi[0] // 16 + 1):
        for cz in range(lo[2] // 16, hi[2] // 16 + 1):
            for event in index.get((cx, cz), ()):
                p = event["pos"]
                if not all(lo[i] <= p[i] <= hi[i] for i in range(3)):
                    continue
                what = event.get("entity") or event.get("item") or ""
                label = event["type"] + (f" {what.replace('minecraft:', '')}" if what else "")
                counts[label] += 1
    return counts


def describe(row, others=(), activity=None, instrumented=True, living=None) -> str:
    """The facts, and only the facts, as the model will see them.

    Neighbouring current objects are included so overlap is not mistaken for one object.
    """
    facts = json.loads(row["value"] or "{}")
    materials = json.loads(row["materials"] or "{}")
    lines = [
        f"operation: {_kind(row)}",
        f"count: {facts.get('blocks')}",
        f"bounding box: {facts.get('dims')} at ({row['min_x']},{row['min_y']},{row['min_z']})",
        f"fill fraction: {facts.get('fill')}",
        f"tallest dimension >= 3 blocks: {facts.get('vertical')}",
        f"distinct materials: {facts.get('distinct_materials')}",
        "materials: " + ", ".join(f"{v} {k}" for k, v in materials.items()),
    ]
    if living:
        lines.append("")
        lines.append("alive inside it right now, counted directly:")
        for entity, n in sorted(living.items(), key=lambda kv: -kv[1])[:6]:
            lines.append(f"  - {n} {entity.replace('minecraft:', '')}")

    if activity:
        lines.append("")
        lines.append("what people have done here:")
        for label, n in activity.most_common(6):
            lines.append(f"  - {label} x{n}")
    elif instrumented is False:
        lines.append("")
        lines.append("what people have done here: NOT RECORDED — this was built before "
                     "such things were being watched. Do not read the silence as disuse.")

    near = [o for o in others if o["id"] != row["id"] and _overlaps(row, o)]
    if near:
        lines.append("")
        lines.append("neighbouring current structures:")
        for o in near[:4]:
            of = json.loads(o["value"] or "{}")
            om = json.loads(o["materials"] or "{}")
            same = ("SAME footprint exactly" if _box(o) == _box(row)
                    else f"{of.get('dims')} overlapping")
            lines.append(
                f"  - {same}: {of.get('blocks')} blocks, "
                + ", ".join(f"{v} {k}" for k, v in list(om.items())[:4]))
    return "\n".join(lines)


def cache_key(prompt: str, model: str = CLASSIFY_MODEL) -> str:
    """Content-addressed. Identical facts give an identical key, so nothing is re-asked."""
    return hashlib.sha256((model + prompt).encode()).hexdigest()[:16]


def classify(client, prompt: str, picture: bytes | None = None) -> Naming:
    """One classification call.

    With a picture this is a different question entirely — "what is this" rather than "what
    do these numbers suggest" — so it goes to the dialogue model, which can see. Without
    one it stays on the cheap model, which is all the numbers deserve.
    """
    import base64

    content: list = []
    if picture:
        content.append({"type": "image", "source": {
            "type": "base64", "media_type": "image/png",
            "data": base64.standard_b64encode(picture).decode()}})
    content.append({"type": "text", "text": prompt})
    response = client.messages.parse(
        model=VISION_MODEL if picture else CLASSIFY_MODEL,
        max_tokens=900,
        system=SYSTEM,
        messages=[{"role": "user", "content": content}],
        output_format=Naming,
    )
    return response.parsed_output


def render_of(row) -> bytes | None:
    """Four entity-free views, or None if the world cannot be read right now."""
    import asyncio
    import io

    from assets import Assets
    from render import focus_structure, sheet
    from scan import request_voxels, to_voxels

    async def fetch():
        pad = 2
        return await request_voxels(
            "ws://127.0.0.1:8765", row["dim"],
            [row["min_x"] - pad, row["min_y"] - pad, row["min_z"] - pad],
            [row["max_x"] + pad, row["max_y"] + pad, row["max_z"] + pad])

    try:
        result = asyncio.run(fetch())
    except Exception:
        return None
    if not result.get("ok"):
        return None
    voxels = focus_structure(to_voxels(result), row)
    if not voxels:
        return None
    buffer = io.BytesIO()
    # Object identity comes from its blocks. Occupancy is stored/queryable evidence and must
    # not be painted into the shape where a malformed entity model can redefine the build.
    sheet(voxels, Assets()).save(buffer, format="PNG")
    return buffer.getvalue()


def main(argv: list[str]) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dry-run", action="store_true", help="build prompts, call nothing")
    p.add_argument("--check-tokens", action="store_true",
                   help="compare the local estimate against count_tokens")
    p.add_argument("--force", action="store_true", help="ignore the cache")
    p.add_argument("--blind", action="store_true",
                   help="name from measurements only, without looking")
    p.add_argument("--limit", type=int, help="classify at most this many")
    p.add_argument("--id", action="append", default=[], help="classify only this structure")
    args = p.parse_args(argv)

    store = Store()
    here = Path(__file__).parent
    sessions = sorted(set(list(here.glob("sessions/corpus/*.jsonl"))
                          + list((here / ".." / "server" / "plugins" / "McGod"
                                  / "events").glob("*.jsonl"))))
    index = activity_index(sessions)
    watching_since = instrumented_window(sessions)
    from structures import current
    rows = current(store)
    if args.id:
        wanted = set(args.id)
        rows = [row for row in rows if row["id"] in wanted]
    rows.sort(key=lambda r: -json.loads(r["value"] or "{}").get("blocks", 0))
    if args.limit:
        rows = rows[:args.limit]

    print(f"{BOLD}classify{RESET} {len(rows)} structures with {DIM}{CLASSIFY_MODEL}{RESET}\n")

    if not rows:
        print(f"{DIM}no canonical structures; run a segmentation pass first{RESET}")
        store.close()
        return 0

    if args.dry_run:
        demo = rows[0]
        print(describe(demo, rows, activity_near(demo, index)))
        print(f"\n{DIM}(dry run — nothing sent){RESET}")
        return 0

    if not have_key():
        key = "OPENROUTER_API_KEY" if MODEL_PROVIDER == "openrouter" else "ANTHROPIC_API_KEY"
        print(f"{RED}No {key}.{RESET} Put your key in agent/.env "
              f"(see .env.example) or export it.", file=sys.stderr)
        return 1

    client = model_client()

    if args.check_tokens:
        from context import estimate_tokens
        sample = SYSTEM + "\n\n" + describe(rows[0])
        actual = client.messages.count_tokens(
            model=CLASSIFY_MODEL,
            messages=[{"role": "user", "content": sample}]).input_tokens
        mine = estimate_tokens(sample)
        print(f"  local estimate {mine}, count_tokens {actual}, "
              f"ratio {mine / actual:.2f} ({'conservative' if mine >= actual else 'UNDER'})")
        return 0

    named = cached = 0
    for row in rows:
        # A structure finished before the tap watched activity is told so, rather than
        # handed an empty list that reads as "nothing ever happened here".
        when = json.loads(row["value"] or "{}").get("when") or 0
        instrumented = (watching_since is None or when == 0
                        or when >= watching_since)
        living = None
        held = store.get("relationships", f"holds:{row['id']}")
        if held:
            try:
                living = json.loads(held["object"])
            except (TypeError, ValueError):
                living = None
        prompt = describe(row, rows, activity_near(row, index), instrumented, living)
        key = cache_key(prompt)
        existing = store.structure_name(row["id"])
        try:
            seen = json.loads(existing["source_event_ids"] or "[]") if existing else []
        except (TypeError, ValueError):
            seen = []
        if not args.force and existing and key in seen:
            cached += 1
            continue
        picture = None if args.blind else render_of(row)
        if picture:
            prompt += ("\n\n(you are also shown four entity-free views of it: perspective, "
                       "top, front elevation, and side elevation)"
                       )
            key = cache_key(prompt + "with-eyes", VISION_MODEL)
            existing = store.structure_name(row["id"])
            try:
                seen = json.loads(existing["source_event_ids"] or "[]") if existing else []
            except (TypeError, ValueError):
                seen = []
            if not args.force and existing and key in seen:
                cached += 1
                continue
        try:
            naming = classify(client, prompt, picture)
        except NOT_FOUND_ERRORS:
            print(f"{RED}model {CLASSIFY_MODEL!r} not found.{RESET} Check "
                  f"MCGOD_CLASSIFY_MODEL in .env.", file=sys.stderr)
            return 1
        except RATE_LIMIT_ERRORS:
            print(f"{RED}rate limited; retry shortly{RESET}", file=sys.stderr)
            return 1
        except STATUS_ERRORS as e:
            print(f"{RED}{row['id']}: API error {e.status_code}: {e.message}{RESET}",
                  file=sys.stderr)
            continue
        except CONNECTION_ERRORS as e:
            print(f"{RED}cannot reach the API: {e}{RESET}", file=sys.stderr)
            return 1

        # A name is INFERRED and is stored apart from the DERIVED geometry it describes,
        # so each belief keeps its own provenance. See Store.name_structure.
        store.name_structure(row["id"], naming.label, naming.confidence,
                             naming.rationale, row["verified_at_tick"], key,
                             category=naming.category, saw=bool(picture))
        if picture:
            from render import save_for_review
            canonical = store.structure_name(row["id"])
            save_for_review(picture, row["id"],
                            canonical["object"] if canonical else naming.label,
                            canonical["confidence"] if canonical else naming.confidence)

        gate = (f"{GREEN}assertable{RESET}"
                if store.assertable("relationships", f"named:{row['id']}")
                else f"{YELLOW}hedge only{RESET}")
        print(f"  {row['id'][:26]:28} {BOLD}{naming.label:26}{RESET} "
              f"{DIM}{naming.category:11}{RESET} conf {naming.confidence:.2f}  "
              f"{'saw it' if picture else 'blind '}  {gate}")
        print(f"    {DIM}{naming.rationale}{RESET}")
        if naming.alternative:
            print(f"    {DIM}or: {naming.alternative}{RESET}")
        named += 1

    print(f"\n{named} classified, {cached} already cached")
    store.close()
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
