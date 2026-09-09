#!/usr/bin/env python3
"""The god, running live.

    python3 god.py                 # connect to a running server and watch
    python3 god.py --no-model      # skip the model; use a deterministic quest

Everything below already existed in pieces. This is the loop that connects them: watch the
live event stream, issue a quest when a player arrives, feed the watcher, and when the
watcher trips after a quiet spell, scan the region and judge it. The model writes the words.
It does not decide whether anyone succeeded.

The order of operations matters and is the same as the offline pipeline:

    join -> baseline scan -> issue (spoken) -> watch -> debounce -> confirming scan -> judge

The baseline is taken before the player is told what to build, because judgment is on what
changed. Without it a quest region full of hillside passes `min_blocks` before anyone lifts
a finger.
"""

from __future__ import annotations

import argparse
import asyncio
import collections
import io
import json
import os
import re
import time
import sys
import traceback
from datetime import datetime

from websockets.asyncio.client import connect
from websockets.exceptions import ConnectionClosed, InvalidHandshake

from config import DIALOGUE_MODEL, MODEL_PROVIDER, VISION_MODEL, have_key, thinking_for
from consumer import BOLD, DIM, GREEN, RED, RESET, YELLOW
from model_api import model_client
from quests import (DEED_CONSTRAINTS, Watcher, featureless, judge,
                    judge_semantically, validate)
from scan import request_scan, request_voxels, run_command, speak, to_voxels
from store import Belief, Provenance, Store

URL = "ws://127.0.0.1:8765"

SCHEMA = """\
Reply with JSON only, exactly this shape and no other keys:

{"speech": "<one or two sentences addressed to the player, in your own voice>",
 "quest_spec": {
   "id": "<lowercase_slug>",
   "intent": "<what you want and why, in one sentence — this is what it is judged against>",
   "where": "<where to put it, in the words a person would use: 'on the rise west of your
              pen', 'against the cliff above the rail line'. Leave empty if it does not
              matter. NEVER a coordinate.>",
   "near": "<the id of a structure it should stand by, from the list above, or empty>",
   "constraints": [ ... ],
   "deadline_ticks": 72000}}

**Prefer to ask for a thing, in words, with no constraints at all.** You will be shown what
they build and you will decide whether it is what you wanted, by looking at it. You do not
need to specify a size, a material mix or a place, and specifying them makes the task read
like a form: a player told "within 7 blocks of 146,68,312, at least 55 blocks, 30 percent
cobblestone" has been handed a spreadsheet, not a calling. Say what you want and let them
build it.

Use `constraints` only for things that are naturally counted — a hunt, a haul, a descent —
where a number is the point rather than a specification:

    {"type": "collect", "item": "minecraft:<item>", "count": <int>}
    {"type": "craft", "item": "minecraft:<item>", "count": <int>}
    {"type": "smelt", "item": "minecraft:<item>", "count": <int>}
    {"type": "kill", "entity": "minecraft:<mob>", "count": <int>}
    {"type": "breed", "entity": "minecraft:<animal>", "count": <int>}
    {"type": "reach_depth", "y": <int>}
    {"type": "advancement", "id": "minecraft:<advancement>"}

If you truly need to constrain a build — and you rarely do — these exist, but every one you
add makes the quest smaller:

    {"type": "min_blocks", "value": <int>}   {"type": "max_blocks", "value": <int>}
    {"type": "min_dimensions", "value": [w, h, d]}
    {"type": "material_fraction", "block": "minecraft:<block>", "min": <0-1>}
    {"type": "required", "blocks": [...]}    {"type": "forbidden", "blocks": [...]}

Rules:
- **Never put a coordinate in your speech.** Name places by what is there: their pen, the
  ridge west of the rails, the shore. You know where their works are; talk like it.
- **Ask for the end, never the path to it.** "Breed four chickens" is a quest; "bring back
  sixteen seeds and breed four chickens" is the same quest with an errand bolted on.
- **Vary what you ask.** You are shown what you have already set them. A hunt, a haul, a
  descent, a herd, a statue, a bridge, a lit road, a workshop, a machine — all are quests.
- **Pitch it at them.** You are told the best tools and armour they have ever held. Someone
  in leather with a stone pickaxe should be sent for iron, not asked for a beacon.
- Do not ask for a build on top of something they made.
- If you name a place you think is theirs, say it is your guess. You have never seen it;
  you have only measured it.
- Danger is a place, not a score.
- Keep it doable in ten to fifteen minutes."""


_NOTHING_HAPPENED = re.compile(
    r"\bchanged 0\b|\b0 blocks?\b|\bno blocks?\b|\bnothing changed\b"
    r"|\bno entit(?:y|ies)\b|\bcould not\b|\bunable\b", re.IGNORECASE)


def _changed_nothing(output: str) -> bool:
    """Whether a command that reported success actually did anything.

    Minecraft counts what it touched, so a fill that matched nothing says "Changed 0
    blocks" and calls it a success. That is the difference between "I cleared the ice" and
    "I looked where the ice was not".
    """
    return bool(output) and bool(_NOTHING_HAPPENED.search(output))


def fallback_quest(pos, tick):
    """Used when the model is unavailable or proposes something unusable.

    A quest nobody can evaluate is worse than a dull one, so the deterministic option is
    always available and always valid.
    """
    return {
        "id": f"a_marker_{tick}",
        "intent": "a sturdy landmark that visibly changes this place",
        "region": {"dim": "overworld", "center": [pos[0], pos[1], pos[2]], "radius": 12},
        "constraints": [
            {"type": "min_blocks", "value": 40},
            {"type": "min_dimensions", "value": [3, 3, 3]},
            {"type": "forbidden", "blocks": ["minecraft:dirt"]},
        ],
        "deadline_ticks": 72000,
    }


#: Ground materials. A site is "clear" when nothing but these is on it.
REGION = 64

#: How long a disturbed place must stay quiet before it is looked at again, in ticks. The
#: quest debounce keeps its own longer settle; this one is for the ledger and the
#: landmarks, where the cost of looking early is a model call and the cost of looking late
#: is a god that has not noticed the elephant.
RESEGMENT_SETTLE_TICKS = 200          # 10s

#: Two disturbed cells this close together are covered by one read.
SAME_READ_BLOCKS = 8

#: How many block changes a cell accumulates before its landmarks are re-boxed. Under
#: this, the ledger keeps geometry exact and the look waits for a threshold, a question
#: that refers to the place, or an idle god.
LOOK_THRESHOLD = int(os.environ.get("MCGOD_LOOK_THRESHOLD", "32"))
#: A brand-new unnamed build gets its first look once the builder has walked away, or
#: after this long if they stay: ten quiet seconds after eight blocks is a footing.
FIRST_LOOK_WAIT_TICKS = 1200          # 60s
#: With no block placed anywhere for this long, deferred looks are worked off oldest first.
IDLE_LOOK_TICKS = 12_000              # 10 min
#: How long the god may spend looking and acting before it must answer.
#:
#: The loop can act, observe and correct, which is what makes it useful and also what makes
#: it able to spend forever on a stubborn request. Generous rather than tight, because a
#: build is a real piece of work: the builder may spend minutes writing several hundred
#: commands, they take a moment to land, and looking at the result costs another round. A
#: question that needs none of that never comes near this. Set below the builder's own
#: deadline this was cutting every build off mid-design, which surfaced as a timeout.
ACT_BUDGET_SECONDS = float(os.environ.get("MCGOD_ACT_BUDGET", "480"))

#: A movement sample this old no longer says where someone is.
PRESENCE_TICKS = 600                  # 30s
#: How many edges one read may chase. Village paths cut every window they touch.
FOLLOW_UPS_PER_READ = 3
#: How far a chain of follow-ups may run. Following on is how a build longer than one read
#: gets measured whole, so the chain is allowed; this is only the backstop that keeps a
#: pathological arrangement of changed ground from walking forever.
MAX_FOLLOW_DEPTH = 6

GROUND = {"grass_block", "dirt", "sand", "gravel", "stone", "podzol", "coarse_dirt",
          "short_grass", "tall_grass", "fern", "dead_bush", "snow", "moss_block",
          "rooted_dirt", "clay", "mud", "farmland", "dirt_path"}


def open_sites(voxels: dict, store, dim: str, want: int = 7, limit: int = 4) -> list:
    """Ground with room to build on, measured rather than eyeballed.

    A picture shows the shape of a place but not how much space is in it, and the god
    proved that by siting a silo in a three-block gap between two buildings it had already
    asked for. Free space is arithmetic and belongs in code: for each candidate centre,
    grow a square until it hits something that is not ground, or a structure already on
    record. What comes back is a handful of places something could actually stand.
    """
    surface: dict = {}
    for (x, y, z), state in voxels.items():
        name = state.split("[")[0].replace("minecraft:", "")
        key = (x, z)
        if key not in surface or y > surface[key][0]:
            surface[key] = (y, name)
    if not surface:
        return []

    taken = []
    for row in store.all("structures"):
        if row["dim"] == dim and row["min_x"] is not None:
            taken.append((row["min_x"] - 1, row["min_z"] - 1,
                          row["max_x"] + 1, row["max_z"] + 1))

    def clear(x, z) -> bool:
        top = surface.get((x, z))
        if top is None or top[1] not in GROUND:
            return False
        return not any(x0 <= x <= x1 and z0 <= z <= z1 for x0, z0, x1, z1 in taken)

    xs = [p[0] for p in surface]
    zs = [p[1] for p in surface]
    found = []
    for cx in range(min(xs) + 4, max(xs) - 4, 4):
        for cz in range(min(zs) + 4, max(zs) - 4, 4):
            if not clear(cx, cz):
                continue
            r = 1
            while r < want:
                ring = [(cx + dx, cz + dz)
                        for dx in range(-r - 1, r + 2) for dz in (-r - 1, r + 1)] + \
                       [(cx + dx, cz + dz)
                        for dx in (-r - 1, r + 1) for dz in range(-r, r + 1)]
                if not all(clear(x, z) for x, z in ring):
                    break
                r += 1
            heights = [surface[(cx + dx, cz + dz)][0]
                       for dx in (-r, 0, r) for dz in (-r, 0, r)
                       if (cx + dx, cz + dz) in surface]
            if r >= 3 and heights:
                found.append({"center": [cx, max(heights), cz], "half_width": r,
                              "flat": max(heights) - min(heights) <= 1})
    found.sort(key=lambda f: (-f["half_width"], not f["flat"]))

    # One per neighbourhood, or the list is the same clearing described four times.
    kept: list = []
    for site in found:
        if all(abs(site["center"][0] - k["center"][0]) > 10
               or abs(site["center"][2] - k["center"][2]) > 10 for k in kept):
            kept.append(site)
        if len(kept) >= limit:
            break
    return kept


async def terrain_view(dim: str, pos, span: int = 40):
    """A wide render of the ground around a point, for siting a build.

    Fetched at quest time and thrown away. It is a picture of the world as it is right now,
    which is cheap to take again and would only go stale if kept.
    """
    from assets import Assets
    from render import landscape

    lo = [pos[0] - span, max(-64, pos[1] - 20), pos[2] - span]
    hi = [pos[0] + span, pos[1] + 24, pos[2] + span]
    result = await request_voxels(URL, dim, lo, hi)
    if not result.get("ok"):
        return None, result.get("error")
    voxels = to_voxels(result)
    if not voxels:
        return None, "empty region", []
    image = await asyncio.to_thread(landscape, voxels, Assets())
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue(), f"{len(voxels)} voxels", voxels


async def ask_model(store, actor, pos, tick, dim="overworld", name=None):
    """Asks for a quest. Returns (speech, proposal) or (None, None)."""
    import base64

    from context import Assembler, render
    sections = Assembler(store).build(
        now_tick=tick, actor=actor,
        trigger={"reason": "a player has arrived and you may set them a task",
                 "pos": list(pos), "facts": {"player_at": list(pos)}},
        player_state=None, episodes=[], findings=[])
    who = f"The player is called {name}. " if name else ""
    prompt = (render(sections) + f"\n\n{who}They are standing at {list(pos)}.\n\n"
              + SCHEMA)

    content = []
    picture, _, voxels = await terrain_view(dim, pos)
    if picture:
        content.append({"type": "image", "source": {
            "type": "base64", "media_type": "image/png",
            "data": base64.standard_b64encode(picture).decode()}})
    if voxels:
        sites = open_sites(voxels, store, dim)
        if sites:
            prompt += "\n\nOpen ground nearby, measured — nothing built on it and level "
            prompt += "enough to stand something on:\n"
            for s in sites:
                prompt += (f"  centre {s['center']}, clear for {s['half_width']} blocks "
                           f"in every direction{' (flat)' if s['flat'] else ''}\n")
            prompt += ("Choose one of these unless you have a reason not to. A radius "
                       "larger than the clear space means asking for a build that will "
                       "not fit.\n")
    content.append({"type": "text", "text": prompt})

    client = model_client(asynchronous=True)
    reply = await client.messages.create(
        model=DIALOGUE_MODEL, max_tokens=1600,
        
thinking=thinking_for(DIALOGUE_MODEL),
        messages=[{"role": "user", "content": content}])
    text = "".join(b.text for b in reply.content if b.type == "text")
    start, end = text.find("{"), text.rfind("}")
    data = json.loads(text[start:end + 1])
    return data.get("speech"), data.get("quest_spec")


def sync_world(store, sessions: list | None = None) -> str:
    """Brings the store up to date with every recorded session before watching anything.

    Without this the god knows only what someone remembered to persist by hand, and will
    happily set a quest on ground already occupied by something it has no record of.
    """
    from pathlib import Path as _Path

    from detectors import analyse as _analyse
    from activity import record_activities
    from reconcile import index_regions
    from world import World

    here = _Path(__file__).parent
    # `sessions` is for callers that need a fixed input, chiefly the idempotence test: a
    # server writing its session file between two replays is a different world, not a
    # broken replay.
    candidates = (sorted(_Path(p) for p in sessions) if sessions is not None else
                  sorted(set(list(here.glob("sessions/corpus/*.jsonl"))
                             + list((here / ".." / "server" / "plugins" / "McGod"
                                     / "events").glob("*.jsonl")))))

    # The corpus files are copies of server session files under different names, so folding
    # in both counted every session twice. That doubled recorded playtime, and made every
    # excavation look revisited — which is one of the signals separating a deliberate work
    # from a trace, so nearly everything was promoted to a work. Identity is the content of
    # a session, not the path it happens to sit at.
    chosen, seen = [], set()
    for path in candidates:
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except OSError:
            continue
        if not lines:
            continue
        try:
            key = (json.loads(lines[0])["t"], json.loads(lines[-1])["t"], len(lines))
        except (json.JSONDecodeError, KeyError):
            key = (str(path), len(lines))
        if key in seen:
            continue
        seen.add(key)
        chosen.append(path)

    world = World()
    all_findings = []
    for path in chosen:
        try:
            episodes, findings, events = _analyse([path])
        except SystemExit:
            continue
        if not events:
            continue
        # Rebuild every deterministic layer from the same deduplicated sessions. The
        # earlier sync populated actors and work history but silently skipped both episode
        # memory and the region correction queue, so restarts had a thinner world model
        # than the live process that preceded them.
        store.record_episodes(episodes, str(path))
        index_regions(store, events)
        record_activities(store, events)
        world.observe(events, episodes, findings)
        all_findings.extend(findings)
    # Materialize history from the complete replay, not one session at a time. Incremental
    # writes made the last session's overlap depend on whether a later startup happened to
    # provide another consolidation pass: an empty store held 50 works after startup one
    # and 49 after startup two. The complete evidence set has one deterministic fixed point.
    world.persist_work_events(store, all_findings)
    from structures import (attribute_origins, migrate_player_built_generated,
                            migrate_unknown_generated, normalize_generated_hierarchy)
    attribute_origins(store)
    world.persist(store)
    moved_to_player = migrate_player_built_generated(store)
    migrated = migrate_unknown_generated(store)
    normalize_generated_hierarchy(store)
    if moved_to_player or migrated:
        from render import move_review_render
        for old_id, new_id in moved_to_player + migrated:
            move_review_render(old_id, new_id)
    # The ledger is scan-derived and cannot be rebuilt from events alone, but its links can:
    # every mass is re-anchored to whatever landmark now holds it, and every history row
    # without a more specific place is attached to the mass it happened in.
    from masses import anchor, current as current_masses, relink
    anchor(store)
    relink(store)
    # Count rows, not per-session merges: summing the latter looked like runaway growth.
    structures = len(store.all("structures"))
    works = len(store.all("work_events"))
    return (f"{len(chosen)} sessions ({len(candidates) - len(chosen)} duplicates "
            f"skipped), {structures} current structures over {len(current_masses(store))} "
            f"built masses, {works} work events, {len(world.actors)} actors")


from commands import GUIDANCE as COMMAND_GUIDANCE

ANSWER_SCHEMA = """\
The player has said something to you. Answer them.

Everything you know is above. Answer from it and nothing else: if the context does not say,
say that you do not know rather than inventing. Your own past words are marked ASSERTED —
you may remember saying them, but they are not evidence of anything.

For present-tense claims about a named place, GROUNDING FOR THIS QUESTION is authoritative.
Names remain INFERRED. Do not turn bbox containment into "part of", and do not claim two
objects are connected unless a RELATION explicitly gives physically_connected=true.
If verification failed, geometry is contradicted, or survival is low, say what is uncertain;
do not describe the old record as confirmed current state.

You may also do one thing, if what they said calls for it:
  {"action": "reroll"}   they want a different quest — abandon the current one and set
                         another. Only when they clearly ask for a different task.
  {"action": "abandon"}  they want the current quest dropped and no replacement.
  {"action": null}       anything else. Answering IS the response.

Be brief — two or three sentences unless they asked for detail. Speak in your own voice.
Use a restrained, watchful voice. Do not address the player with modern casual familiarity
unless their established relationship supports it.

Never speak raw coordinates unless the player explicitly asked for coordinates. Describe a
place by its direction, terrain, nearby works, or another in-world landmark instead. Internal
measurements may establish the answer, but database notation is not the god's language.
Follow that rule silently. Never announce that you are withholding coordinates, refusing to
guess, or obeying a policy; simply answer with the evidence available.

For advice about a structure, use its render and its complete list of existing features.
Never propose an already-present feature as though it were absent. You may suggest changing,
enlarging or repeating it only when you explicitly acknowledge what already exists.

""" + COMMAND_GUIDANCE + """

Reply with JSON only:
{"speech": str, "action": null | "reroll" | "abandon", "commands": [str],
 "anchor": null | "<player name>", "every": null | <ticks>, "duration": null | <ticks>,
 "spell": null | "<short name>"}"""


_COORDINATE_TRIPLE = re.compile(
    r"(?<!\d)(?:[\[(]\s*)?-?\d{1,8}\s*,\s*-?\d{1,8}\s*,\s*-?\d{1,8}(?:\s*[\])])?(?!\d)")
_NAMED_COORDINATES = re.compile(
    r"\bx\s*[:=]\s*-?\d{1,8}\s*[,; ]+\s*y\s*[:=]\s*-?\d{1,8}"
    r"\s*[,; ]+\s*z\s*[:=]\s*-?\d{1,8}\b", re.IGNORECASE)
#: Axis-labelled without an equals sign: "X -1,332, Z 5,716", "x 300 z -300". A model told
#: not to give coordinates will often give them this way instead, thousands separators and
#: all, so the axis letters are the thing to match rather than the punctuation.
_AXIS_LABELLED = re.compile(
    r"\bx[\s:=]+-?[\d,]{1,12}\s*(?:,|\s|and)+\s*(?:y[\s:=]+-?[\d,]{1,12}"
    r"\s*(?:,|\s|and)+\s*)?z[\s:=]+-?[\d,]{1,12}", re.IGNORECASE)
#: A placeholder a model leaves where it decided not to give a coordinate. Saying nothing
#: is the instruction; announcing the omission is the failure this removes.
_OMISSION_MARK = re.compile(
    r"[\[(]\s*(?:coordinates?|coords?|location)\s+(?:omitted|withheld|redacted|hidden)"
    r"\s*[\])]|,?\s*\b(?:coordinates?|coords?)\s+(?:omitted|withheld|redacted)\b",
    re.IGNORECASE)
#: A bracketed pair, "[-1332, 5716]" or "(204, -1130)". The triple pattern above misses it
#: because a horizontal position names only two axes, and that is exactly how a model
#: reports somewhere far away. Brackets are required: without them "10,000" is two numbers
#: with a comma between them and every large number would be censored as a coordinate.
_COORDINATE_PAIR = re.compile(
    r"[\[(]\s*-?\d{1,8}\s*,\s*-?\d{1,8}\s*[\])]")
#: A bounding box read out as ranges: "301-308 by -303 to -297". Two ranges joined by
#: "by" are a box, never a distance; "10-17 blocks away" has no "by" and is left alone.
_COORDINATE_RANGES = re.compile(
    r"(?<![\w-])-?\d{1,8}\s*(?:-|to|–)\s*-?\d{1,8}\s+by\s+-?\d{1,8}\s*(?:-|to|–)\s*-?\d{1,8}"
    r"(?:\s+by\s+-?\d{1,8}\s*(?:-|to|–)\s*-?\d{1,8})?", re.IGNORECASE)


def pure_greeting(text: str) -> bool:
    """A social greeting needs no factual grounding or model deliberation."""
    tokens = set(re.findall(r"[a-z]+", text.lower()))
    return bool(tokens & {"hello", "hi", "hail", "greetings", "hey"}) and tokens <= {
        "hello", "hi", "hail", "greetings", "hey", "god", "there", "oh", "o"
    }


def coordinates_requested(text: str) -> bool:
    lower = text.lower()
    return bool(re.search(r"\b(coordinates?|coords?|xyz|exact position|exact location)\b", lower))


def contains_raw_coordinates(text: str) -> bool:
    return bool(_COORDINATE_TRIPLE.search(text) or _NAMED_COORDINATES.search(text)
                or _COORDINATE_RANGES.search(text) or _COORDINATE_PAIR.search(text)
                or _AXIS_LABELLED.search(text))


def remove_raw_coordinates(text: str) -> str:
    """Final invariant: model retries cannot leak an internal coordinate triple."""
    text = re.sub(r"\b(?:at|near|around)\s+" + _COORDINATE_TRIPLE.pattern,
                  "there", text, flags=re.IGNORECASE)
    text = re.sub(r"\b(?:at|near|around)\s+" + _COORDINATE_RANGES.pattern,
                  "there", text, flags=re.IGNORECASE)
    text = re.sub(r"\b(?:at|near|around)\s+" + _COORDINATE_PAIR.pattern,
                  "there", text, flags=re.IGNORECASE)
    text = _COORDINATE_TRIPLE.sub("that place", text)
    text = _NAMED_COORDINATES.sub("that place", text)
    text = _COORDINATE_RANGES.sub("that place", text)
    text = _COORDINATE_PAIR.sub("that place", text)
    text = re.sub(r"\b(?:at|near|around)\s+" + _AXIS_LABELLED.pattern,
                  "there", text, flags=re.IGNORECASE)
    text = _AXIS_LABELLED.sub("that place", text)
    # A placeholder left behind reads as the god refusing rather than answering.
    text = _OMISSION_MARK.sub("", text)
    text = re.sub(r"\b(?:near|around|at)\s+the\s+generated\s+location\b(?=[\s,.])",
                  "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+([,.;:!?])", r"\1", text)
    # Removing a clause can leave its comma stranded against the full stop.
    text = re.sub(r",\s*([.!?])", r"\1", text)
    text = re.sub(r",\s*$", "", text)
    return re.sub(r"\s{2,}", " ", text).strip()


def announces_answer_policy(text: str) -> bool:
    """Whether an answer narrates its restrictions instead of answering the player."""
    return bool(re.search(
        r"\b(?:i (?:will not|won't|refuse to)|do not ask me to|"
        r"i (?:cannot|can't) (?:give|tell|draw|point|provide))\b", text.lower()))


def remove_policy_announcements(text: str) -> str:
    """Strip narration about the answer's own limits, in either form.

    A whole sentence about what the god will not say, and the smaller version: a
    placeholder left where a coordinate would have gone. Both tell the player about the
    rules instead of about the world, and the second is the one a model reaches for when
    it is told not to give coordinates.
    """
    kept = [sentence.strip() for sentence in re.split(r"(?<=[.!?])\s+", text)
            if sentence.strip() and not announces_answer_policy(sentence)]
    text = " ".join(kept)
    text = _OMISSION_MARK.sub("", text)
    text = re.sub(r"\b(?:near|around|at)\s+the\s+generated\s+location\b(?=[\s,.]|$)",
                  "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+([,.;:!?])", r"\1", text)
    # Removing a clause can leave its comma stranded against the full stop.
    text = re.sub(r",\s*([.!?])", r"\1", text)
    text = re.sub(r",\s*$", "", text)
    return re.sub(r"\s{2,}", " ", text).strip()


def trim_chat_answer(text: str, limit: int = 420, sentences: int = 3) -> str:
    """Keep a chat reply readable on the game screen even when the model rambles."""
    parts = [part.strip() for part in re.split(r"(?<=[.!?])\s+", text) if part.strip()]
    chosen = []
    for part in parts[:sentences]:
        if chosen and len(" ".join(chosen + [part])) > limit:
            break
        chosen.append(part)
    answer = " ".join(chosen) or text.strip()
    if len(answer) <= limit:
        return answer
    clipped = answer[:limit].rsplit(" ", 1)[0].rstrip(",;:-")
    return clipped + "…"


MATCH_SYSTEM = """\
You asked a player to build something. They have built something. Decide whether it is the
thing you asked for, and whether it is finished.

You are shown four entity-free views rendered from the world itself—perspective, overhead,
front elevation, and side elevation—plus its measurements. You
did not tell them where to put it or how big to make it — you told them what you wanted, in
words — so judge it as a person would: does this serve what you asked for?

Be generous about form and strict about substance. A tower of a different stone than you
imagined is still the tower. A shape that could not possibly do the job is not, however
neatly it is made. A pile with no interior, no entrance and no working parts is not a
building.

If it is not the thing at all, say so plainly and say what is missing — they may simply not
have finished, or may have built something else entirely.

Name it too, whatever the verdict — you are looking straight at it, and what you can see
is worth remembering whether or not it is the thing you asked for.

Reply as JSON: {"is_it": true|false, "finished": true|false,
                "why": "<one sentence to the player>",
                "craft": "crude|plain|considered|fine",
                "label": "<short noun phrase for what it is, lowercase>",
                "category": "building|excavation|harvest|clearing|path|farm|demolition|other",
                "confidence": <0-1, how sure you are of the name>}"""


LOOK_SYSTEM = """\
You are a god judging a player's build. You are shown four entity-free views rendered from
the world itself: perspective, overhead, front elevation, and side elevation.

Say what it looks like and how well it is made. Judge craft and character, never size or
count — those are measured for you and you must not guess at them from a picture. Be honest;
flattery is worth nothing. Name the one or two things that would most improve it.

You will be told whether the player is still standing there. If they are, they may well not
be finished — speak as though the work is in progress, and make clear that carrying on is
welcome rather than handing down a verdict. If they have gone, speak as though it is done.

Keep it to two or three sentences, in your own voice.

Reply as JSON: {"looks_like": str, "craft": "crude|plain|considered|fine", "notes": str}"""


class God:
    def __init__(self, store, use_model: bool) -> None:
        self.store = store
        self.use_model = use_model
        #: Which model draws structure boundaries. Separate from the dialogue model because
        #: segmentation is a different job — read a picture, read a grid, return exact
        #: coordinates — and it is the one the pipeline makes most often, so what it costs
        #: is worth measuring against what a cheaper one gets wrong.
        self.segment_model = os.environ.get("MCGOD_SEGMENT_MODEL", VISION_MODEL)
        self.quests: dict = {}
        #: UUID is identity; the name is a label that can change, so it is only ever used
        #: for addressing someone.
        self.names: dict = {}
        self.tick = 0
        self.tick_seen_at = time.monotonic()
        self.last_sweep = 0.0
        #: Places where blocks have moved, and when they last did. Keyed coarsely so a
        #: player working across a build produces one place to look at rather than four
        #: hundred. This is what drives re-segmentation: a structure being contradicted
        #: only covers ground the god already knows about, and a build raised on empty
        #: grass contradicts nothing at all.
        self.disturbed: dict = {}
        #: Cells queued because a mass ran past a read edge, and when. Stops a long wall
        #: from being followed back and forth forever.
        self.followed: dict = {}
        #: What the ledger looked like under each cell the last time the model drew boxes
        #: there. An unchanged ledger means unchanged boxes; the look is not repeated.
        self.boxed: dict = {}
        #: Block changes per 24-block cell since that cell was last boxed, and when the
        #: last one happened.
        self.edits: dict = {}
        self.edit_ticks: dict = {}
        #: Cells whose look is owed but deferred: under threshold, or a first look waiting
        #: for the builder to step away. Released by threshold, reference, or idleness.
        self.edited: dict = {}
        #: Last known position per actor, from movement samples.
        self.positions: dict = {}
        self.last_block_tick = 0

    def log(self, message: str) -> None:
        print(f"{DIM}{datetime.now():%H:%M:%S}{RESET} {message}", flush=True)

    async def on_join(self, event):
        actor, pos = event["actor"], event["pos"]
        name = event.get("name")
        self.names[actor] = name or self.names.get(actor)

        # Someone with work outstanding gets reminded, not ignored. This returned silently,
        # so a player who took a quest, logged out and came back was met with nothing at
        # all — indistinguishable from the god being down.
        if actor in self.quests:
            spec, watcher = self.quests[actor]
            who = self.names.get(actor) or actor[:8]
            self.log(f"{BOLD}{who} returned{RESET}; {spec.id} still stands")
            await speak(f"You are seen again, {who}. What I asked of you still stands.",
                        URL)
            await asyncio.sleep(3.2)
            await speak(self.describe(spec), URL)
            return

        self.names[actor] = name or self.names.get(actor)
        who = self.names.get(actor) or actor[:8]
        self.log(f"{BOLD}{who} joined{RESET} at {pos}")
        await speak(f"You are seen, {who}.", URL)

        speech, proposal = None, None
        if self.use_model and have_key():
            try:
                speech, proposal = await ask_model(
                    self.store, actor, pos, event["tick"],
                    dim=event.get("dim", "overworld"), name=self.names.get(actor))
                self.log(f"model proposed: {json.dumps(proposal)[:120]}")
            except Exception as e:
                self.log(f"{YELLOW}model failed ({type(e).__name__}); "
                         f"falling back{RESET}")

        spec = None
        if proposal:
            spec, problems = validate(proposal, actor, event["tick"],
                                      now_t=event.get("t", 0))
            if problems:
                # The gate refuses rather than interprets, so a bad proposal costs a
                # fallback rather than an unjudgeable quest.
                self.log(f"{RED}proposal rejected:{RESET} {problems}")
                spec = None
        if spec is None:
            speech = None
            spec, problems = validate(fallback_quest(pos, event["tick"]), actor,
                                      event["tick"], now_t=event.get("t", 0))
            self.log(f"{YELLOW}using the deterministic quest{RESET}")

        # Baseline BEFORE the player is told. Judgment is on what changes from here.
        lo, hi = spec.region.bbox()
        try:
            base_scan = await request_scan(URL, spec.region.dim, list(lo), list(hi))
            baseline = base_scan.get("materials", {}) if base_scan.get("ok") else {}
        except Exception:
            baseline = {}
        spec, _ = validate(
            json.loads(spec.to_json()) | {"region": {
                "dim": spec.region.dim, "center": list(spec.region.center),
                "radius": spec.region.radius}},
            actor, event["tick"], baseline=baseline, now_t=event.get("t", 0))

        self.quests[actor] = (spec, Watcher(spec))
        self.store.put("quests", {
            "id": spec.id, "actor": actor, "spec": spec.to_json(), "state": "active",
            "issued_tick": spec.issued_tick,
            "deadline_tick": spec.issued_tick + spec.deadline_ticks,
            "resolved_tick": None,
        }, Belief(provenance=Provenance.DERIVED, first_seen_tick=spec.issued_tick))

        if not speech:
            speech = ("Build me something that stands: forty blocks at least, three by "
                      "three by three or better, and no dirt. Near where you are now.")
        await speak(speech, URL)
        self.store.record_utterance(actor, event["tick"], speech, "quest")
        await asyncio.sleep(3.2)
        await speak(self.describe(spec), URL)
        self.log(f"{GREEN}issued{RESET} {spec.id} at {spec.region.center} "
                 f"r{spec.region.radius}")

    @staticmethod
    def _contradicted_at(row) -> int:
        try:
            return int((json.loads(row["value"] or "{}") or {}).get(
                "contradicted_tick") or 0)
        except (TypeError, ValueError):
            return 0

    def _structures_at(self, dim: str, pos) -> list:
        """Every structure whose bounding box contains a point, padded a little."""
        hits = []
        for row in self.store.all("structures", "dim = ?", (dim,)):
            if row["min_x"] is None:
                continue
            if (row["min_x"] - 2 <= pos[0] <= row["max_x"] + 2
                    and row["min_y"] - 2 <= pos[1] <= row["max_y"] + 2
                    and row["min_z"] - 2 <= pos[2] <= row["max_z"] + 2):
                hits.append(row)
        return hits

    def revise(self, event) -> None:
        """Let an event contradict the beliefs it disproves, the moment it happens.

        Decay covers beliefs that merely went quiet. This is the other half: someone kills
        the animals we counted, breaks the blocks we measured, empties the chest we listed.
        Waiting for a half-life would leave the god reciting a flock that has been eaten.

        Nothing is rewritten from an event — that would be deriving a new measurement from
        a partial view. The belief is marked contradicted, which drops its confidence to the
        floor and puts it at the front of the queue to be looked at again.
        """
        from activity import record_activity
        from world import fold_live_player
        fold_live_player(self.store, event)
        record_activity(self.store, event)
        pos = event.get("pos")
        if not pos:
            return
        # The correction queue is not an offline reporting tool. Every observed visit is
        # folded into it as it happens, using the same reducer used during startup replay.
        # This keeps restart and live behavior equivalent without a second implementation.
        from reconcile import index_regions
        index_regions(self.store, [event])
        kind = event["type"]
        dim = event.get("dim", "overworld")

        if kind in ("move", "player_state", "region_enter") and event.get("actor"):
            self.positions[event["actor"]] = (dim, [pos[0], pos[1], pos[2]],
                                              event.get("tick", 0))
        if kind in ("block_place", "block_break") and self._changes_something_built(event):
            cell = (dim, pos[0] // 24, pos[1] // 24, pos[2] // 24)
            self.disturbed[cell] = {"tick": event.get("tick", 0),
                                    "pos": [pos[0], pos[1], pos[2]]}
            self.edits[cell] = self.edits.get(cell, 0) + 1
            self.edit_ticks[cell] = event.get("tick", 0)
            self.last_block_tick = max(self.last_block_tick, event.get("tick", 0))
            if event.get("actor"):
                self.positions[event["actor"]] = (dim, [pos[0], pos[1], pos[2]],
                                                  event.get("tick", 0))

        if kind in ("mob_kill", "breed", "tame", "shear"):
            for row in self._structures_at(dim, pos):
                what = (event.get("entity") or "something").replace("minecraft:", "")
                if self.store.contradict(
                        "relationships", f"holds:{row['id']}",
                        f"{kind} {what} here", event.get("tick", 0)):
                    self.log(f"{DIM}what lives in {row['id']} is no longer what I "
                             f"counted ({kind} {what}){RESET}")

        elif kind == "block_break":
            for row in self._structures_at(dim, pos):
                self.store.contradict("relationships", f"survives:{row['id']}",
                                      "blocks taken out of it", event.get("tick", 0))
                self.store.contradict("structures", row["id"],
                                      "blocks taken out of it", event.get("tick", 0))

        elif kind == "block_place":
            for row in self._structures_at(dim, pos):
                self.store.contradict("structures", row["id"],
                                      "added to since it was measured",
                                      event.get("tick", 0))

        elif kind in ("container_take", "container_put", "explosion"):
            for row in self._structures_at(dim, pos):
                self.store.contradict("relationships", f"holds:{row['id']}",
                                      f"{kind} here", event.get("tick", 0))

    def _changes_something_built(self, event) -> bool:
        """Whether a block change can have altered anything built.

        Breaking one shrub sent the god scanning for three minutes. Any block change marked
        its cell for a fresh look, and vegetation is a block change: pulling up grass,
        picking a flower, chopping leaves. None of it can alter a built mass, because
        vegetation is not in one.

        Placing anything counts, whatever the material, since a dirt pillar is dirt. So does
        breaking anything that is not vegetation. And so does breaking vegetation that sits
        inside a mass on record, because a build made of natural material is still a build
        and taking it down must be noticed.
        """
        from masses import mass_at
        from render import is_natural

        if event["type"] != "block_break":
            return True
        before = event.get("before")
        if not before or not is_natural(before):
            return True
        pos = event.get("pos")
        return mass_at(self.store, event.get("dim", "overworld"), pos) is not None

    async def sweep(self) -> None:
        """Look at one place that needs looking at.

        Somewhere blocks have moved comes first: that is where the world has actually
        changed, and re-reading it settles what is there now — a new building, a floor added
        to an old one, a wall taken down. Only when nothing is disturbed does this fall back
        to confirming a belief that has merely gone quiet.

        Deliberately one at a time, and only where a player has been. Nothing changes where
        nobody is.
        """
        from reconcile import verify, work_queue

        now = self.now_tick()
        settled = [(cell, seen) for cell, seen in self.disturbed.items()
                   if now - seen["tick"] >= RESEGMENT_SETTLE_TICKS]
        if settled:
            # Oldest first, so a place someone has left alone is dealt with before the one
            # they are still working in. A full look outranks a ledger-only one at the
            # same age, and every other settled cell one read will cover comes along.
            cell, seen = min(settled, key=lambda p: (p[1]["tick"], p[1].get("ledger_only",
                                                                            False)))
            del self.disturbed[cell]
            ledger_only = bool(seen.get("ledger_only"))
            for other, near in list(self.disturbed.items()):
                if (other[0] == cell[0]
                        and all(abs(near["pos"][i] - seen["pos"][i]) <= SAME_READ_BLOCKS
                                for i in range(3))):
                    del self.disturbed[other]
                    ledger_only = ledger_only and bool(near.get("ledger_only"))
            self.log(f"{DIM}something changed near {seen['pos']}; "
                     f"{'measuring' if ledger_only else 'looking again'}{RESET}")
            written = await self.resegment(cell[0], seen["pos"], ledger_only=ledger_only,
                                           depth=int(seen.get("depth") or 0))
            if written:
                self.log(f"{DIM}settled {written} structure(s) there{RESET}")
            return

        # A first look waiting for its builder to step away, once they have, or once they
        # have stayed long enough that what stands is probably what they meant.
        for key, entry in sorted(self.edited.items(), key=lambda p: p[1]["since"]):
            if entry.get("reason") != "first":
                continue
            lo, hi = self._window_of(entry["pos"])
            if (not self._someone_within(entry["dim"], lo, hi)
                    or now - entry["since"] >= FIRST_LOOK_WAIT_TICKS):
                del self.edited[key]
                self.log(f"{DIM}the builder has left a new build alone; first look{RESET}")
                await self.resegment(entry["dim"], entry["pos"], force_look=True)
                return

        wrong = [r for r in self.store.all("structures")
                 if r["min_x"] is not None
                 and self.store.contradicted("structures", r["id"])]
        for row in wrong:
            if self.now_tick() - self._contradicted_at(row) < RESEGMENT_SETTLE_TICKS:
                continue
            centre = [(row["min_x"] + row["max_x"]) // 2,
                      (row["min_y"] + row["max_y"]) // 2,
                      (row["min_z"] + row["max_z"]) // 2]
            if self._deferred_near(row["dim"], centre):
                continue          # its look is owed and deferred; the ledger is current
            # One updater for every kind of change.  The former refresh path used a
            # connectivity rule while discovery used model boundaries, so the same house
            # could be represented differently depending on what triggered the look.
            await self.resegment(row["dim"], centre)
            return

        # Nobody has placed a block anywhere for a while: work off the deferred looks,
        # oldest first, while the calls cost nothing in latency.
        if self.edited and now - self.last_block_tick >= IDLE_LOOK_TICKS:
            key, entry = min(self.edited.items(), key=lambda p: p[1]["since"])
            del self.edited[key]
            self.log(f"{DIM}quiet everywhere; paying a deferred look{RESET}")
            await self.resegment(entry["dim"], entry["pos"], force_look=True)
            return

        queue = work_queue(self.store)
        if not queue:
            return
        cell = queue[0]
        if not cell["dirty"] and not self.store.stale("regions", cell["id"], 0.4):
            return
        for row in self.store.all("structures", "dim = ?", (cell["dim"],)):
            if row["min_x"] is None:
                continue
            if (row["min_x"] // REGION, row["min_z"] // REGION) != (cell["cx"], cell["cz"]):
                continue
            if not self.store.stale("structures", row["id"], 0.5):
                continue
            try:
                verdict = await verify(self.store, row["id"], URL)
            except Exception as e:
                self.log(f"{YELLOW}sweep failed on {row['id']} ({type(e).__name__}){RESET}")
                return
            if verdict.get("ok"):
                self.log(f"{DIM}swept {row['id']}: "
                         f"{verdict['survival'] * 100:.0f}% still standing{RESET}")
            return

    async def resegment(self, dim: str, centre, span: int = 26,
                        origin: str | None = None, ledger_only: bool = False,
                        force_look: bool = False, depth: int = 0) -> int:
        """Re-decide what structures stand around a point, from the world itself.

        Boundaries are drawn by the model, not by code. Block-level rules were tried first
        and could not be made to work: cutting at material seams split one cottage into
        three, plain connectivity fused a house and the pen leaning on it into one lump, and
        every rule that fixed one case broke another. "Where does this building end" is not
        a fact about which blocks touch. It is a judgement about what someone meant.

        So the model is given the patch three ways — rendered, sliced into character grids
        with real coordinates so it can be exact, and listed as whatever already stands
        there — and it draws the boxes. Then each box is rendered alone and handed back
        once, because a box drawn from a picture of everything tends to cut things in half.

        Code keeps the parts code is better at: measuring what is actually inside a box, and
        refusing a box that is empty, enormous, or claims an identity twice.
        """
        import base64
        import io as _io

        from assets import Assets
        from render import entities_within, framed, plan, save_for_review, sheet
        from segment import (MIN_STRUCTURE_BLOCKS, REFINE, SEGMENT, built_blocks,
                             describe_cluster, slices)

        x, y, z = centre
        lo = [x - span, max(-64, y - 16), z - span]
        hi = [x + span, y + 30, z + span]
        try:
            result = await request_voxels(URL, dim, lo, hi)
        except Exception as e:
            self.log(f"{YELLOW}could not read the ground ({type(e).__name__}){RESET}")
            return 0
        if not result.get("ok"):
            return 0

        voxels = to_voxels(result)
        # Natural terrain proves the read itself succeeded even when the former build is
        # gone. Do not require a new object before applying measured demolition: that left
        # an isolated, completely removed structure in canonical state forever.
        if not voxels:
            return 0
        # The ledger first, and from code alone: every connected built mass in the read is
        # recorded before any model is asked what deserves a name. Nothing the model then
        # declines to box can vanish, because its existence was never the model's to decide.
        # Terrain the block history proves a player placed — a dirt pillar — counts as built.
        from masses import apply as apply_masses, ledger_built, within as masses_within
        from masses import facts_of as mass_facts
        before = {r["id"]: (bool(mass_facts(r).get("partial")), r["place_id"])
                  for r in masses_within(self.store, dim, lo, hi)}
        built = ledger_built(self.store, dim, voxels, lo, hi)
        ledger = apply_masses(self.store, dim, voxels, built, lo, hi, self.tick)
        if ledger["written"] or ledger["retired"]:
            self.log(f"{DIM}  ledger: {len(ledger['written'])} mass(es) measured"
                     + (f", {len(ledger['retired'])} retired" if ledger["retired"] else "")
                     + (f", {len(ledger['partial'])} cut by the read edge"
                        if ledger["partial"] else "") + RESET)
        # Follow an edge only from ground the player actually changed. A build can be
        # longer than one read, so a chain of follow-ups is how a bridge gets measured
        # whole; but a follow-up that lands on untouched ground has nothing to complete,
        # and following on from there is what walked the scan across a whole village.
        self._follow_partials(dim, ledger["partial"], lo, hi, depth=depth)
        after = masses_within(self.store, dim, lo, hi)
        key = (dim, x // 24, y // 24, z // 24)
        signature = tuple(sorted(
            (r["id"], r["min_x"], r["min_y"], r["min_z"], r["max_x"], r["max_y"],
             r["max_z"], mass_facts(r).get("blocks")) for r in after))
        decision = self._look_decision(dim, key, centre, lo, hi, signature,
                                       ledger_only=ledger_only, force=force_look)
        if decision != "look":
            await self._settle_ledger(dim, lo, hi, voxels, built)
            return 0
        from structures import current, current_generated

        source = (current_generated(self.store, dim)
                  if origin == "world_generated" else current(self.store, dim)
                  if origin else current(self.store, dim) + current_generated(self.store, dim))
        here = [r for r in source
                if r["max_x"] >= lo[0] and r["min_x"] <= hi[0]
                and r["max_z"] >= lo[2] and r["min_z"] <= hi[2]]
        if len(built) < MIN_STRUCTURE_BLOCKS:
            written = self._apply_segmentation(dim, [], here, built, lo, hi, origin)
            await self._settle_ledger(dim, lo, hi, voxels, built)
            return written

        entities = entities_within(result.get("entities"), voxels)
        assets = Assets()
        buffer = _io.BytesIO()
        (await asyncio.to_thread(sheet, voxels, assets)).save(buffer, format="PNG")
        picture = buffer.getvalue()
        # A plan view alongside the three-quarter views. Perspective is what a person sees,
        # but it is the wrong picture for deciding where a building ends: buildings at
        # different depths overlap and the same wall is wider at the near end. Looking
        # straight down separates footprints, which is the judgement being asked for, and
        # because it is orthographic the coordinate grid ruled onto it is exact.
        # Toggleable, because it is not yet proven to help. On the one patch measured so far
        # it cost ~40% more time per pass and did not improve accuracy — so it stays behind
        # a switch until an eval over several patches says otherwise, rather than being
        # kept because the reasoning for it sounded good.
        overhead = None
        if os.environ.get("MCGOD_PLAN_VIEW", "0") != "0":
            buffer = _io.BytesIO()
            (await asyncio.to_thread(plan, voxels, assets, lo, hi)).save(buffer, format="PNG")
            overhead = buffer.getvalue()

        known = []
        for r in here:
            known.append({"id": r["id"],
                          "min": [r["min_x"], r["min_y"], r["min_z"]],
                          "max": [r["max_x"], r["max_y"], r["max_z"]]})

        drawn = await self._draw_boundaries(dim, picture, voxels, lo, hi, known, entities,
                                            overhead)
        observed_lo, observed_hi = lo, hi
        # One general fallback for scale, not a catalogue of special buildings.  An
        # overview is necessary when several structures changed together, but an isolated
        # cottage can become too small to read in it.  If the overview finds nothing,
        # retry the same scanned facts around the disturbance at half scale.
        if not drawn and span > 14:
            close_span = max(12, span // 2)
            close_lo = [x - close_span, lo[1], z - close_span]
            close_hi = [x + close_span, hi[1], z + close_span]
            close = {p: material for p, material in voxels.items()
                     if all(close_lo[i] <= p[i] <= close_hi[i] for i in range(3))}
            if len(built_blocks(close)) >= MIN_STRUCTURE_BLOCKS:
                buffer = _io.BytesIO()
                close_entities = entities_within(entities, close)
                (await asyncio.to_thread(sheet, close, assets)).save(buffer, format="PNG")
                close_plan = None
                if os.environ.get("MCGOD_PLAN_VIEW", "0") != "0":
                    plan_buffer = _io.BytesIO()
                    (await asyncio.to_thread(plan, close, assets, close_lo, close_hi)).save(
                        plan_buffer, format="PNG")
                    close_plan = plan_buffer.getvalue()
                close_known = [item for item in known
                               if item["max"][0] >= close_lo[0]
                               and item["min"][0] <= close_hi[0]
                               and item["max"][2] >= close_lo[2]
                               and item["min"][2] <= close_hi[2]]
                drawn = await self._draw_boundaries(
                    dim, buffer.getvalue(), close, close_lo, close_hi, close_known,
                    close_entities, close_plan)
                observed_lo, observed_hi = close_lo, close_hi
        if not drawn:
            await self._settle_ledger(dim, lo, hi, voxels, built)
            self._looked(dim, key, lo, hi, signature)
            return 0
        drawn = await self._second_look(drawn, voxels, built, observed_lo, observed_hi)

        written = self._apply_segmentation(
            dim, drawn, here, built, observed_lo, observed_hi, origin)
        await self._settle_ledger(dim, lo, hi, voxels, built)
        self._looked(dim, key, lo, hi, signature)
        return written

    def _retire_gone(self, dim: str, built: dict, lo, hi) -> list:
        """Retire a landmark whose blocks a fresh read shows to be gone.

        Demolition must never wait for a deferred look. The rule is the one the boxing pass
        uses: a structure wholly inside the read with none of its blocks left, or under
        fifteen percent of them, is not there. Absence of a mention retires nothing; this
        is measured absence.
        """
        from render import forget_renders
        from structures import current

        retired = []
        for row in current(self.store, dim):
            inside = all(lo[i] < row[low] and row[high] < hi[i]
                         for i, (low, high) in enumerate((
                             ("min_x", "max_x"), ("min_y", "max_y"), ("min_z", "max_z"))))
            if not inside:
                continue
            still = sum(1 for p in built
                        if row["min_x"] <= p[0] <= row["max_x"]
                        and row["min_y"] <= p[1] <= row["max_y"]
                        and row["min_z"] <= p[2] <= row["max_z"])
            try:
                believed = sum(json.loads(row["materials"] or "{}").values())
            except (TypeError, ValueError, AttributeError):
                believed = 0
            if still == 0 or still / max(1, believed) < 0.15:
                name = self._delete_structure(row["id"])
                forget_renders(row["id"])
                retired.append(row["id"])
                self.log(f"{DIM}  retired {row['id']} ({name or 'unnamed'}): not there on "
                         f"a fresh look{RESET}")
        return retired

    def _look_decision(self, dim: str, key, centre, lo, hi, signature,
                       ledger_only: bool = False, force: bool = False) -> str:
        """Whether this read earns a vision pass: ``look``, ``defer`` or ``skip``.

        The ledger is already exact by the time this is asked; what a look buys is names
        and boxes. So it is spent only where names can change: a brand-new build nobody
        has had the chance to name, a place edited past the threshold, a place a question
        refers to, or, when nothing is being built anywhere, whatever is still owed.
        """
        from masses import unnamed_whole

        if force:
            return "look"
        if self.boxed.get(key) == signature:
            # Every mass under this window is exactly as it was when the model last drew
            # boxes here. Asking again would spend two vision calls to be told the same.
            self.edited.pop(key, None)
            self.log(f"{DIM}  nothing in the ledger here has changed since it was last "
                     f"boxed; not asking again{RESET}")
            return "skip"
        fresh = unnamed_whole(self.store, dim, lo, hi)
        if fresh:
            quiet = self.tick - max((self.edit_ticks.get(cell, 0)
                                     for cell in self._cells_under(dim, lo, hi)),
                                    default=0)
            if not self._someone_within(dim, lo, hi) or quiet >= FIRST_LOOK_WAIT_TICKS:
                self.log(f"{DIM}  {len(fresh)} whole player-built mass(es) nobody has "
                         f"named; looking{RESET}")
                return "look"
            self.edited[key] = {"pos": list(centre), "since": self.tick, "reason": "first",
                                "dim": dim}
            self.log(f"{DIM}  a new build nobody has named, but its builder is still "
                     f"here; the first look waits{RESET}")
            return "defer"
        changed = self._edits_under(dim, lo, hi)
        if changed >= LOOK_THRESHOLD and not ledger_only:
            self.log(f"{DIM}  {changed} block changes here since it was last boxed; "
                     f"looking{RESET}")
            return "look"
        self.edited[key] = {"pos": list(centre), "since": self.edited.get(key, {}).get(
            "since", self.tick), "reason": "threshold", "dim": dim, "changed": changed}
        self.log(f"{DIM}  {changed} block change(s) here, under the threshold of "
                 f"{LOOK_THRESHOLD}; the look waits for more, a question, or a quiet "
                 f"hour{RESET}")
        return "defer"

    def _cells_under(self, dim: str, lo, hi) -> list:
        """Every 24-block cell a read window intersects."""
        return [(dim, cx, cy, cz)
                for cx in range(lo[0] // 24, hi[0] // 24 + 1)
                for cy in range(lo[1] // 24, hi[1] // 24 + 1)
                for cz in range(lo[2] // 24, hi[2] // 24 + 1)]

    def _edits_under(self, dim: str, lo, hi) -> int:
        return sum(self.edits.get(cell, 0) for cell in self._cells_under(dim, lo, hi))

    def _someone_within(self, dim: str, lo, hi) -> bool:
        for actor_dim, pos, tick in self.positions.values():
            if actor_dim != dim or self.tick - tick > PRESENCE_TICKS:
                continue
            if all(lo[i] <= pos[i] <= hi[i] for i in range(3)):
                return True
        return False

    def _looked(self, dim: str, key, lo, hi, signature) -> None:
        """After a vision pass: remember what it saw, and clear the debt it paid."""
        from masses import mark_looked

        self.boxed[key] = signature
        mark_looked(self.store, dim, lo, hi, int(time.time() * 1000))
        for cell in self._cells_under(dim, lo, hi):
            self.edits.pop(cell, None)
            self.edited.pop(cell, None)

    def _window_of(self, centre, span: int = 26) -> tuple[list, list]:
        x, y, z = centre
        return [x - span, max(-64, y - 16), z - span], [x + span, y + 30, z + span]

    def _deferred_near(self, dim: str, point) -> list:
        """Deferred looks whose read window would contain a point."""
        out = []
        for key, entry in self.edited.items():
            if entry.get("dim", key[0]) != dim:
                continue
            lo, hi = self._window_of(entry["pos"])
            if all(lo[i] <= point[i] <= hi[i] for i in range(3)):
                out.append((key, entry))
        return out

    async def release_for_question(self, text: str, dim: str, pos, limit: int = 2) -> int:
        """Pay a deferred look before answering a question that refers to its place.

        A question about a structure standing in an edited window, or asked from inside
        one, is the one moment a stale name is certain to be heard. At most two looks a
        question, so a long-deferred settlement cannot stall the answer for minutes.
        """
        from grounding import resolve_references

        owed: dict = {}
        if pos:
            for key, entry in self._deferred_near(dim, pos):
                owed[key] = entry
        try:
            referenced = resolve_references(self.store, text, dim, pos)
        except Exception:  # noqa: BLE001
            referenced = []
        for row in referenced:
            centre = [(row["min_x"] + row["max_x"]) // 2,
                      (row["min_y"] + row["max_y"]) // 2,
                      (row["min_z"] + row["max_z"]) // 2]
            for key, entry in self._deferred_near(row["dim"], centre):
                owed[key] = entry
        looked = 0
        for key, entry in list(owed.items())[:limit]:
            self.log(f"{DIM}the question concerns a place with a look owed; "
                     f"looking first{RESET}")
            self.edited.pop(key, None)
            await self.resegment(dim, entry["pos"], force_look=True)
            looked += 1
        return looked

    def _follow_partials(self, dim: str, partial: list, lo, hi, depth: int = 0) -> None:
        """Queue a look at the ground past each read edge a mass was cut by.

        Half a bridge is not a fact about the world; it is a fact about where we happened
        to look. But following every cut edge walks the world: a village's paths are one
        connected mass hundreds of blocks long, so every read cuts it, and each follow-up
        cut it again somewhere new. One broken shrub became three minutes of scanning
        across a whole village.

        A build can be longer than one read, so following on IS how a long bridge gets
        measured whole, and the chain has to be allowed to continue. What it must not do is
        continue across ground nobody touched. So the test is the player's own work: a read
        with no unspent block changes under it has nothing to complete and does not follow.
        A chain therefore runs the length of what someone built and stops where their work
        does.

        Three further bounds. Only a mass worth completing is followed — the player's own
        work, or something small enough to plausibly be one object, never a terrain-scale
        network with no edge to find. At most a few per read. And a hard depth cap, so no
        arrangement of changed ground can make the chain run forever.
        """
        from reconcile import cell_id

        if depth >= MAX_FOLLOW_DEPTH:
            return
        if not self._edits_under(dim, lo, hi):
            return
        span = [hi[i] - lo[i] + 1 for i in range(3)]
        worth = [o for o in partial
                 if o.get("origin") == "player_built" or (o.get("blocks") or 0) <= 400]
        worth.sort(key=lambda o: (o.get("origin") != "player_built", -(o.get("blocks") or 0)))
        queued = 0
        for o in worth:
            if queued >= FOLLOW_UPS_PER_READ:
                break
            centre = [(o["lo"][i] + o["hi"][i]) // 2 for i in range(3)]
            for face in o["partial"]:
                axis = {"west": 0, "east": 0, "below": 1, "above": 1,
                        "north": 2, "south": 2}[face]
                sign = -1 if face in ("west", "below", "north") else 1
                target = list(centre)
                target[axis] = (lo[axis] if sign < 0 else hi[axis]) + sign * (span[axis] // 3)
                target[1] = max(-64, min(319, target[1]))
                rid, _, _ = cell_id(dim, target[0], target[2])
                if self.store.get("regions", rid) is None:
                    continue
                cell = (dim, target[0] // 24, target[1] // 24, target[2] // 24)
                if cell in self.disturbed:
                    continue
                if self.tick - self.followed.get(cell, -10 ** 9) < 6000:
                    continue
                self.followed[cell] = self.tick
                self.disturbed[cell] = {"tick": self.tick, "pos": target,
                                        "ledger_only": True, "depth": depth + 1}
                queued += 1
                self.log(f"{DIM}  a mass runs past the {face} edge of the read; "
                         f"queued a measurement beyond it{RESET}")
                if queued >= FOLLOW_UPS_PER_READ:
                    break

    async def _settle_ledger(self, dim: str, lo, hi, voxels, built) -> None:
        """After a pass: re-cut the ledger along the landmarks the pass just drew, anchor
        masses to them, attach history, and ask what the unnamed ones are for.

        The ledger was written before the model drew its boxes, so it was cut along the
        boxes that stood before this pass. Boxes may have moved; masses honour landmark
        boundaries, so the window is measured once more against the new ones. No model
        is involved and it costs a fraction of a second.
        """
        from masses import (anchor, apply as apply_masses, classify_roles, pending_roles,
                            refresh_landmarks, relink)

        try:
            apply_masses(self.store, dim, voxels, built, lo, hi, self.tick)
            self._retire_gone(dim, built, lo, hi)
            anchor(self.store, dim)
            refresh_landmarks(self.store, dim, int(time.time() * 1000), self.tick)
            relink(self.store, dim, lo, hi)
        except Exception as e:  # noqa: BLE001
            self.log(f"{YELLOW}could not anchor the ledger ({type(e).__name__}: {e}){RESET}")
            return
        if not (self.use_model and have_key()):
            return
        from config import CLASSIFY_MODEL
        rows = pending_roles(self.store, CLASSIFY_MODEL, dim, lo, hi)
        if not rows:
            return
        done = await classify_roles(self.store, rows, CLASSIFY_MODEL, log=self.log)
        self.log(f"{DIM}  asked what {len(rows)} unnamed mass(es) are for; "
                 f"{done} answered{RESET}")

    def _apply_segmentation(self, dim, drawn, existing, built,
                            observed_lo, observed_hi, origin: str | None = None) -> int:
        """Atomically replace one observed patch of the canonical world view.

        The two model calls above only produce a proposal. Identity matching, writes and
        retirements form one database transaction here. If any object fails validation or
        persistence, every prior object remains exactly as it was before the pass. Renders
        are published only after that transaction commits, so a filesystem error can make
        the render audit fail but cannot leave a half-updated world model.
        """
        from render import forget_renders, save_for_review
        from structures import assign_identities, explained_fraction

        observations = [{**found["_facts"], "dim": dim} for found in drawn]
        matchable = [r for r in existing if all(
            observed_lo[i] < r[low] and r[high] < observed_hi[i]
            for i, (low, high) in enumerate((
                ("min_x", "max_x"), ("min_y", "max_y"), ("min_z", "max_z"))))]
        assigned = assign_identities(matchable, observations)
        written, retired, kept = [], [], set()

        with self.store.transaction():
            for index, found in enumerate(drawn):
                sid = self._write_segment(
                    dim, found["_facts"], found, assigned.get(index), found["_shot"], origin)
                kept.add(sid)
                written.append((sid, found))

            # A row is retired only from measured evidence: either too few of its old
            # blocks remain, or those blocks are already represented by fresh canonical
            # boxes. Merely being omitted by the model is never evidence of demolition.
            for row in existing:
                if row["id"] in kept:
                    continue
                still = sum(1 for p in built
                            if row["min_x"] <= p[0] <= row["max_x"]
                            and row["min_y"] <= p[1] <= row["max_y"]
                            and row["min_z"] <= p[2] <= row["max_z"])
                inside = all(observed_lo[i] < row[low] and row[high] < observed_hi[i]
                             for i, (low, high) in enumerate((
                                 ("min_x", "max_x"), ("min_y", "max_y"),
                                 ("min_z", "max_z"))))
                if not inside:
                    continue
                reason = None
                try:
                    believed = sum(json.loads(row["materials"] or "{}").values())
                except (TypeError, ValueError, AttributeError):
                    believed = 0
                survival = still / max(1, believed)
                if still == 0 or survival < 0.15:
                    reason = "not there on a fresh look"
                elif explained_fraction(row, observations, built) >= 0.80:
                    reason = "duplicate geometry absorbed by fresh segmentation"
                if reason:
                    name = (self._delete_generated(row["id"])
                            if self.store.get("generated_features", row["id"])
                            else self._delete_structure(row["id"]))
                    retired.append((row["id"], name, reason))

        # Landmarks are groupings of masses. Re-anchor as soon as the structures commit so
        # the ledger and the landmark tier never disagree about what belongs to what.
        from masses import anchor
        anchor(self.store, dim)

        # Files are a materialized review view, not canonical state. Publish them after the
        # state transition so an image write can never roll back some objects but not others.
        for sid, found in written:
            named = self.store.structure_name(sid)
            label = named["object"] if named else found.get("label")
            confidence = float(named["confidence"] if named
                               else found.get("confidence") or 0.6)
            if label:
                save_for_review(found["_shot"], sid, label, confidence)
            facts = found["_facts"]
            self.log(f"{DIM}  {sid}: {label or '(unnamed)'} "
                     f"{facts['dims']} {facts['blocks']} blocks{RESET}")
        for sid, name, reason in retired:
            forget_renders(sid)
            self.log(f"{DIM}  retired {sid} ({name or 'unnamed'}): {reason}{RESET}")
        return len(written)

    async def _draw_boundaries(self, dim, picture, voxels, lo, hi, known, entities,
                               overhead=None) -> list:
        """Ask the model where the structures are. Returns boxes with what is in them."""
        import base64

        from segment import (MIN_STRUCTURE_BLOCKS, SEGMENT, SegmentationOutput,
                             built_blocks, slices)

        if not (self.use_model and have_key()):
            return []
        note = ""
        if entities:
            seen: dict = {}
            for e in entities:
                seen[e.get("type", "?")] = seen.get(e.get("type", "?"), 0) + 1
            note = ("Living things are not drawn in the structure images. Exact census: "
                    + ", ".join(f"{n} {k}" for k, n in sorted(seen.items())) + ".")
        work = []
        for row in self.store.all(
                "work_events",
                "dim = ? AND kind = 'placement' AND max_x >= ? AND min_x <= ? "
                "AND max_y >= ? AND min_y <= ? AND max_z >= ? AND min_z <= ?",
                (dim, lo[0], hi[0], lo[1], hi[1], lo[2], hi[2])):
            try:
                value = json.loads(row["value"] or "{}")
            except (TypeError, ValueError):
                value = {}
            work.append({
                "actor": row["actor"],
                "from": [row["min_x"], row["min_y"], row["min_z"]],
                "to": [row["max_x"], row["max_y"], row["max_z"]],
                "placed_blocks": value.get("blocks") or sum(
                    json.loads(row["materials"] or "{}").values()),
                "materials": json.loads(row["materials"] or "{}"),
            })
        # Exact geometry code has already measured: every connected built mass in the
        # patch. The model groups and names these; it does not get to invent extents.
        from masses import facts_of as mass_facts, within as masses_within
        masses_here = []
        for row in masses_within(self.store, dim, lo, hi):
            facts = mass_facts(row)
            masses_here.append({
                "min": [row["min_x"], row["min_y"], row["min_z"]],
                "max": [row["max_x"], row["max_y"], row["max_z"]],
                "blocks": facts.get("blocks"),
                "materials": dict(list(json.loads(row["materials"] or "{}").items())[:4]),
                "builder_known": bool(row["actor"]),
                "cut_by_read_edge": facts.get("partial") or [],
            })
        try:
            client = model_client(asynchronous=True)
            reply = await client.messages.parse(
                model=self.segment_model, max_tokens=8000,
                thinking=thinking_for(self.segment_model), system=SEGMENT,
                output_format=SegmentationOutput,
                messages=[{"role": "user", "content": [
                    {"type": "text", "text":
                        "Canonical views: perspective upper-left, orthographic top "
                        "upper-right, front elevation lower-left, side elevation lower-right."},
                    {"type": "image", "source": {
                        "type": "base64", "media_type": "image/png",
                        "data": base64.standard_b64encode(picture).decode()}},
                    *([{"type": "text", "text":
                        "Plan view from directly overhead, orthographic, north up. The "
                        "grid is ruled in real world coordinates — read boundaries off it."},
                       {"type": "image", "source": {
                           "type": "base64", "media_type": "image/png",
                           "data": base64.standard_b64encode(overhead).decode()}}]
                      if overhead else []),
                    {"type": "text", "text": json.dumps({
                        "patch": {"from": lo, "to": hi}, "note": note,
                        "already_known": known,
                        "player_work_footprints": work,
                        "built_masses": masses_here,
                        "slices": slices(voxels, lo, hi)}, indent=1)}]}])
            if reply.parsed_output is None:
                raise ValueError("model returned no segmentation object")
            proposed = [item.model_dump() for item in reply.parsed_output.structures]
            if os.environ.get("MCGOD_TRACE"):
                self.log(f"{DIM}SEGMENT work={json.dumps(work)} -> "
                         f"{json.dumps(proposed)[:1600]}{RESET}")
        except Exception as e:
            self.log(f"{YELLOW}could not segment ({type(e).__name__}: {e}){RESET}")
            return []

        built = built_blocks(voxels)
        return [b for b in (self._measure(p, built, lo, hi,
                                          floor=MIN_STRUCTURE_BLOCKS)
                            for p in proposed) if b]

    def _measure(self, found: dict, built: dict, lo, hi, floor: int = 8):
        """Measure what is actually inside a drawn box, and refuse a box that is nonsense.

        The model decides where the edges are; it does not get to decide what is inside
        them. A box below the shared evidence floor is not a structure however confidently
        it was drawn; above that floor, whether a compact arrangement has meaning is the
        semantic model's decision.
        """
        from segment import describe_cluster

        try:
            a = [int(v) for v in found["min"]]
            b = [int(v) for v in found["max"]]
        except (KeyError, TypeError, ValueError):
            return None
        a = [max(a[i], lo[i]) for i in range(3)]
        b = [min(b[i], hi[i]) for i in range(3)]
        if any(b[i] < a[i] for i in range(3)):
            return None
        inside = [p for p in built if all(a[i] <= p[i] <= b[i] for i in range(3))]
        if len(inside) < floor:
            return None
        # A read has no information beyond its own boundary. Even if no visible block
        # spills past it, a box touching that boundary may be half a building. Partial
        # observations are useful as a signal to scan nearby, but never safe state.
        if any(a[i] <= lo[i] or b[i] >= hi[i] for i in range(3)):
            return None
        found["_facts"] = describe_cluster(inside, built)
        found["_blocks"] = inside
        return found

    async def _second_look(self, drawn: list, voxels, built, scan_lo, scan_hi) -> list:
        """Render each box alone and hand them back once, to catch the bad cuts.

        A box drawn from a picture of a whole patch is drawn from a picture where everything
        overlaps everything. Seen on its own, a box that swallowed a neighbour or sliced a
        roof off is obvious, and the model can say so.
        """
        import base64
        import io as _io

        from PIL import Image

        from assets import Assets
        from render import sheet

        shots = []
        for found in drawn:
            a, b = found["_facts"]["lo"], found["_facts"]["hi"]
            near = {p: m for p, m in voxels.items()
                    if all(a[i] <= p[i] <= b[i] for i in range(3))}
            buffer = _io.BytesIO()
            (await asyncio.to_thread(sheet, near, Assets())).save(buffer, format="PNG")
            shots.append(buffer.getvalue())
            found["_shot"] = shots[-1]
        if not (self.use_model and have_key()) or not shots:
            return drawn

        from segment import REFINE, RefinementOutput, slices, spilling
        try:
            client = model_client(asynchronous=True)
            content: list = []
            view_names = ("perspective", "orthographic top", "front elevation",
                          "side elevation")
            for i, (found, shot) in enumerate(zip(drawn, shots)):
                facts = found["_facts"]
                beyond = spilling(built, facts["lo"], facts["hi"])
                content.append({"type": "text", "text": json.dumps({
                    # The first-pass label is intentionally withheld. The second look is
                    # independent evidence; showing the guess it is meant to check turns
                    # it into an anchoring exercise instead of verification.
                    "index": i,
                    "box": {"min": facts["lo"], "max": facts["hi"]},
                    "blocks_inside": facts["blocks"],
                    "materials": facts["materials"],
                    "horizontal_slices": slices(near, facts["lo"], facts["hi"]),
                    "built_blocks_just_outside_each_face": beyond or "none"})})
                whole = Image.open(_io.BytesIO(shot))
                half_w, half_h = whole.width // 2, whole.height // 2
                boxes = ((0, 0, half_w, half_h), (half_w, 0, whole.width, half_h),
                         (0, half_h, half_w, whole.height),
                         (half_w, half_h, whole.width, whole.height))
                extents = [facts["hi"][axis] - facts["lo"][axis] for axis in range(3)]
                if extents[1] == 0:
                    selected_views = (1,)  # horizontal plane: orthographic top
                elif extents[2] == 0:
                    selected_views = (2,)  # X/Y plane: front elevation
                elif extents[0] == 0:
                    selected_views = (3,)  # Z/Y plane: side elevation
                else:
                    selected_views = range(4)
                for view_index in selected_views:
                    view_name, box = view_names[view_index], boxes[view_index]
                    panel = _io.BytesIO()
                    whole.crop(box).save(panel, format="PNG")
                    content.append({"type": "text", "text": view_name})
                    content.append({"type": "image", "source": {
                        "type": "base64", "media_type": "image/png",
                        "data": base64.standard_b64encode(panel.getvalue()).decode()}})
            reply = await client.messages.parse(
                model=self.segment_model, max_tokens=6000,
                thinking=thinking_for(self.segment_model), system=REFINE,
                output_format=RefinementOutput,
                messages=[{"role": "user", "content": content}])
            if reply.parsed_output is None:
                raise ValueError("model returned no refinement object")
            fixes = [item.model_dump(exclude_none=True)
                     for item in reply.parsed_output.fixes]
            if os.environ.get("MCGOD_TRACE"):
                self.log(f"{DIM}REFINE -> {json.dumps(fixes)[:1200]}{RESET}")
        except Exception as e:
            self.log(f"{YELLOW}no second look ({type(e).__name__}){RESET}")
            return drawn

        out, dropped = list(drawn), set()
        extra = []
        for fix in fixes:
            i = fix.get("index")
            if not isinstance(i, int) or not 0 <= i < len(out):
                continue
            action = fix.get("action", "keep")
            if action == "drop":
                dropped.add(i)
                self.log(f"{DIM}  dropped [{i}]: {fix.get('why', '')}{RESET}")
                continue
            if action == "keep":
                for key in ("label", "description"):
                    if fix.get(key):
                        out[i][key] = fix[key]
                if fix.get("confidence") is not None:
                    out[i]["confidence"] = fix["confidence"]
            if action in ("grow", "shrink", "split") and fix.get("min"):
                fixed = self._measure({**out[i], **{k: fix[k] for k in ("min", "max")
                                                    if k in fix}}, built,
                                      scan_lo, scan_hi)
                if fixed:
                    for k in ("label", "description"):
                        if fix.get(k):
                            fixed[k] = fix[k]
                    # A corrected box is a better box, not a less certain one. Dropping the
                    # confidence to zero made every fixed structure look like a guess.
                    fixed["confidence"] = fix.get("confidence") or out[i].get(
                        "confidence") or 0.7
                    fixed.pop("_shot", None)
                    out[i] = fixed
                    self.log(f"{DIM}  {action} [{i}]: {fix.get('why', '')}{RESET}")
            for piece in fix.get("also") or []:
                # Split pieces use the same universal floor as first-pass proposals. Their
                # meaning comes from the model's visual judgement, not a type-specific size
                # exception for rails, pixel art, signs, or any future compact structure.
                more = self._measure({**piece, "continues": None,
                                      "confidence": piece.get("confidence") or 0.7},
                                     built, scan_lo, scan_hi, floor=8)
                if more:
                    extra.append(more)
                    self.log(f"{DIM}  split off: {piece.get('label')}{RESET}")
                else:
                    self.log(f"{YELLOW}  lost the split piece "
                             f"{piece.get('label')!r} — nothing measurable in its box"
                             f"{RESET}")

        out = [f for i, f in enumerate(out) if i not in dropped] + extra
        for found in out:
            if "_shot" not in found:
                a, b = found["_facts"]["lo"], found["_facts"]["hi"]
                near = {p: m for p, m in voxels.items()
                        if all(a[i] <= p[i] <= b[i] for i in range(3))}
                buffer = _io.BytesIO()
                (await asyncio.to_thread(sheet, near, Assets())).save(buffer, format="PNG")
                found["_shot"] = buffer.getvalue()
        return out

    def retire(self, sid: str, why: str) -> None:
        """Forget a structure that is no longer there. Renders go too, or they mislead."""
        from render import forget_renders

        name = self._delete_structure(sid)
        forget_renders(sid)
        self.log(f"{DIM}  retired {sid} ({name or 'unnamed'}): {why}{RESET}")

    def _delete_structure(self, sid: str) -> str | None:
        """Delete canonical rows; callers publish filesystem side effects after commit."""
        name = self.store.structure_name(sid)
        self.store.db.execute("DELETE FROM structures WHERE id = ?", (sid,))
        self.store.db.execute("DELETE FROM relationships WHERE subject = ?", (sid,))
        self.store.commit()
        return name["object"] if name else None

    def _delete_generated(self, sid: str) -> str | None:
        """Delete a surveyed generator feature without touching player structures."""
        name = self.store.structure_name(sid)
        self.store.db.execute("DELETE FROM generated_features WHERE id = ?", (sid,))
        self.store.db.execute("DELETE FROM relationships WHERE subject = ?", (sid,))
        self.store.commit()
        return name["object"] if name else None

    def _write_segment(self, dim, facts, found, sid, picture,
                       origin: str | None = None) -> str:
        """Write one structure inside the caller's state transition."""
        from structures import (facts_of, generated_near, generated_observation_value,
                                generated_root, new_generated_id, new_id,
                                observation_value, owner_from_work)

        owner = owner_from_work(self.store, dim, facts)
        generated_parent = generated_near(self.store, dim, facts)
        continues_generated = bool(sid and self.store.get("generated_features", sid))
        # The caller's origin says which surrounding ontology triggered the scan. It is
        # not authorship evidence. Direct placement evidence wins for a new object inside
        # generated terrain; an observation continuing an existing generated identity
        # remains generated even when the player has renovated part of it.
        if owner and not continues_generated:
            origin = "player_built"
        elif origin is None:
            if continues_generated:
                origin = "world_generated"
            elif owner:
                origin = "player_built"
            elif generated_parent:
                origin = "world_generated"

        if origin == "world_generated":
            previous = self.store.get("generated_features", sid) if sid else None
            if previous is None and generated_parent and facts_of(generated_parent).get(
                    "parent"):
                sid = generated_parent["id"]
                previous = generated_parent
            if previous is None:
                sid = new_generated_id(r["id"] for r in
                                       self.store.all("generated_features"))
            observed_ms = int(time.time() * 1000)
            centre = [(facts["lo"][i] + facts["hi"][i]) // 2 for i in range(3)]
            prior_value = facts_of(previous) if previous else {}
            parent_id = prior_value.get("parent")
            if not parent_id and generated_parent and generated_parent["id"] != sid:
                parent_id = generated_root(self.store, generated_parent)["id"]
            self.store.put("generated_features", {
                "id": sid, "dim": dim,
                "kind": ((previous["kind"] if previous else None)
                         or found.get("category") or found.get("label") or "unknown"),
                "x": centre[0], "y": centre[1], "z": centre[2],
                "discovered_from": "survey",
                "min_x": facts["lo"][0], "min_y": facts["lo"][1],
                "min_z": facts["lo"][2], "max_x": facts["hi"][0],
                "max_y": facts["hi"][1], "max_z": facts["hi"][2],
                "materials": json.dumps(facts["materials"]),
            }, Belief(provenance=Provenance.INFERRED,
                      confidence=float(found.get("confidence") or 0.6),
                      verified_at_tick=self.tick, verified_at_ms=observed_ms,
                      first_seen_tick=(previous["first_seen_tick"]
                                       if previous else self.tick),
                      value=generated_observation_value(
                          facts, observed_ms, **({"parent": parent_id} if parent_id else {}),
                          **({"role_evidence": prior_value["role_evidence"]}
                             if prior_value.get("role_evidence") else {}))))
            if found.get("label"):
                self.store.name_structure(
                    sid, found["label"], float(found.get("confidence") or 0.6),
                    found.get("description") or "", self.tick,
                    f"seg:{sid}:{facts['blocks']}",
                    category=found.get("category") or "other", saw=True,
                    description=found.get("description") or "")
            return sid

        previous = self.store.get("structures", sid) if sid else None
        if sid is None:
            sid = new_id(r["id"] for r in self.store.all("structures"))
        observed_ms = int(time.time() * 1000)
        actor = previous["actor"] if previous and previous["actor"] else owner
        prior_facts = {}
        if previous:
            try:
                prior_facts = json.loads(previous["value"] or "{}")
            except (TypeError, ValueError):
                pass
        origin = (origin or ("player_built" if actor
                             else prior_facts.get("origin", "unknown")))
        self.store.put("structures", {
            "id": sid, "actor": actor, "dim": dim, "name": None,
            "min_x": facts["lo"][0], "min_y": facts["lo"][1], "min_z": facts["lo"][2],
            "max_x": facts["hi"][0], "max_y": facts["hi"][1], "max_z": facts["hi"][2],
            "materials": json.dumps(facts["materials"]),
        }, Belief(provenance=Provenance.INFERRED,
                  confidence=float(found.get("confidence") or 0.6),
                  verified_at_tick=self.tick,
                  verified_at_ms=observed_ms,
                  first_seen_tick=(previous["first_seen_tick"] if previous else self.tick),
                  value=observation_value(facts, observed_ms, origin=origin)))
        if found.get("label"):
            self.store.name_structure(
                sid, found["label"], float(found.get("confidence") or 0.6),
                found.get("description") or "", self.tick,
                f"seg:{sid}:{facts['blocks']}",
                category=found.get("category") or "other", saw=True,
                description=found.get("description") or "")
        return sid

    async def refresh_structure(self, row) -> bool:
        """Re-measure a structure that has been added to, and look at it again.

        Marking a belief contradicted says only that it is wrong. This makes it right again.

        The stored bounding box is the box as it was when the census last ran, so a second
        floor or a tower on the roof falls entirely outside it — verifying against that box
        would confirm the old building was still there and never notice the new one. The
        region read here is padded generously, especially upwards, and the extent is taken
        from what is actually standing rather than from what was expected.
        """
        import io as _io

        from assets import Assets
        from render import built_only, framed, save_for_review, sheet
        pad, up = 10, 24
        try:
            result = await request_voxels(
                URL, row["dim"],
                [row["min_x"] - pad, max(-64, row["min_y"] - pad), row["min_z"] - pad],
                [row["max_x"] + pad, row["max_y"] + up, row["max_z"] + pad])
        except Exception as e:
            self.log(f"{YELLOW}could not re-measure {row['id']} "
                     f"({type(e).__name__}){RESET}")
            return False
        if not result.get("ok"):
            return False
        voxels = to_voxels(result)
        built = built_only(voxels)
        if not built:
            return False

        # The padded read takes in trees and whatever else stands nearby, and `framed`
        # only drops ground. Without clustering, one block added to a roof grew a
        # ten-by-nine house into a thirty-by-twenty-nine region of nine thousand blocks —
        # every refresh would swallow more of the world. Keep only the connected group
        # that the structure already occupied.
        built = self._same_structure(built, row)
        if not built:
            return False

        xs = [p[0] for p in built]
        ys = [p[1] for p in built]
        zs = [p[2] for p in built]
        dims = [max(xs) - min(xs) + 1, max(ys) - min(ys) + 1, max(zs) - min(zs) + 1]
        grew = (max(ys) > row["max_y"] or min(ys) < row["min_y"]
                or max(xs) > row["max_x"] or min(xs) < row["min_x"]
                or max(zs) > row["max_z"] or min(zs) < row["min_z"])

        materials = collections.Counter(
            state.split("[")[0].replace("minecraft:", "") for state in built.values())
        facts = {"blocks": len(built), "dims": f"{dims[0]}x{dims[1]}x{dims[2]}",
                 "fill": round(len(built) / max(1, dims[0] * dims[1] * dims[2]), 3),
                 "vertical": dims[1] >= 3, "significance": "work",
                 "when": int(time.time() * 1000)}
        self.store.put("structures", {
            "id": row["id"], "actor": row["actor"], "dim": row["dim"], "name": None,
            "min_x": min(xs), "min_y": min(ys), "min_z": min(zs),
            "max_x": max(xs), "max_y": max(ys), "max_z": max(zs),
            "materials": json.dumps(dict(materials.most_common(12))),
        }, Belief(provenance=Provenance.DERIVED, verified_at_tick=self.tick,
                  first_seen_tick=row["first_seen_tick"], value=json.dumps(facts)))

        # Render the framed region, not the bare blocks: a building floating with no ground
        # under it is harder to read than one sitting in its own terrain.
        shown = {p: m for p, m in voxels.items()
                 if all(min(c[i] for c in built) - 2 <= p[i]
                        <= max(c[i] for c in built) + 2 for i in range(3))}
        buffer = _io.BytesIO()
        (await asyncio.to_thread(sheet, shown or built, Assets())).save(buffer, format="PNG")
        picture = buffer.getvalue()

        # It has changed shape, so what it IS may have changed with it. Ask again.
        naming = None
        if self.use_model and have_key():
            try:
                naming = await self.name_from_sight(row["id"], picture, facts, materials)
            except Exception as e:
                self.log(f"{YELLOW}could not re-name {row['id']} "
                         f"({type(e).__name__}){RESET}")
        label = (naming or {}).get("label") or ""
        if naming:
            self.store.name_structure(
                row["id"], label, float(naming.get("confidence") or 0.7),
                naming.get("rationale") or "", self.tick,
                f"saw:{row['id']}:{len(built)}",
                category=naming.get("category") or "other", saw=True,
                description=naming.get("description") or "")
        # A direct observation or other stronger name can correctly refuse the fresh model
        # inference. The review filename must reflect canonical state, not the rejected
        # proposal, or the filesystem and database immediately disagree.
        canonical_name = self.store.structure_name(row["id"])
        if canonical_name:
            label = canonical_name["object"]
        save_for_review(picture, row["id"], label,
                        float(canonical_name["confidence"] if canonical_name
                              else (naming or {}).get("confidence") or 0) or None)
        self.log(f"{DIM}re-measured {row['id']}: {facts['dims']}, {len(built)} blocks"
                 + (f" (grew) — now \"{label}\"" if grew and label else "") + RESET)
        return True

    @staticmethod
    def _same_structure(built: dict, row) -> dict:
        """The connected group of blocks that is this structure, and not its neighbours."""
        remaining = set(built)
        best: list = []
        while remaining:
            seed = remaining.pop()
            group, frontier = [seed], [seed]
            while frontier:
                cx, cy, cz = frontier.pop()
                near = [p for p in remaining
                        if abs(p[0] - cx) <= 2 and abs(p[1] - cy) <= 2
                        and abs(p[2] - cz) <= 2]
                for p in near:
                    remaining.discard(p)
                    group.append(p)
                    frontier.append(p)
            # The group that overlaps where the structure already was, biggest first.
            overlaps = sum(1 for p in group
                           if row["min_x"] <= p[0] <= row["max_x"]
                           and row["min_y"] <= p[1] <= row["max_y"]
                           and row["min_z"] <= p[2] <= row["max_z"])
            if overlaps and len(group) > len(best):
                best = group
        return {p: built[p] for p in best}

    async def name_from_sight(self, structure_id, picture, facts, materials) -> dict:
        """What is this, looking at it? Same question the batch pass asks."""
        import base64

        from classify import SYSTEM
        client = model_client(asynchronous=True)
        described = {"measurements": facts,
                     "materials": dict(materials.most_common(8))}
        previous = self.store.structure_description(structure_id)
        if previous:
            described["what you said about it before"] = previous
        reply = await client.messages.create(
            model=VISION_MODEL, max_tokens=700,
thinking=thinking_for(VISION_MODEL),
            system=SYSTEM + (
                "\n\nYou may also be given what you said about this place last time. If it "
                "has changed, rewrite the description to match what is there now, keeping "
                "what is still true — you are maintaining a record of one place over time, "
                "not writing it fresh each visit.\n\n"
                "Reply as JSON only: {\"label\": str, \"category\": "
                "\"building|excavation|harvest|clearing|path|farm|demolition|other\", "
                "\"confidence\": <0-1>, \"rationale\": str, "
                "\"description\": \"<two or three sentences on what this place is, what "
                "is in it and what it is for>\"}"),
            messages=[{"role": "user", "content": [
                {"type": "image", "source": {
                    "type": "base64", "media_type": "image/png",
                    "data": base64.standard_b64encode(picture).decode()}},
                {"type": "text", "text": json.dumps(described, indent=1)}]}])
        text = "".join(b.text for b in reply.content if b.type == "text")
        return json.loads(text[text.find("{"):text.rfind("}") + 1])

    def remember_structure(self, actor, row, facts, verdict) -> None:
        """Queue a fresh spatial read; a quest candidate is not canonical geometry."""
        centre = [(row[f"min_{axis}"] + row[f"max_{axis}"]) // 2
                  for axis in ("x", "y", "z")]
        cell = (row["dim"], centre[0] // 24, centre[1] // 24, centre[2] // 24)
        self.disturbed[cell] = {"tick": self.tick, "pos": centre}
        self.log(f"{DIM}queued fresh segmentation for completed work at {centre}{RESET}")

    async def render_structure(self, row) -> bytes | None:
        """Four entity-free views of one structure, framed to the build."""
        import io as _io

        from assets import Assets
        from render import framed, sheet
        pad = 2
        result = await request_voxels(
            URL, row["dim"],
            [row["min_x"] - pad, row["min_y"] - pad, row["min_z"] - pad],
            [row["max_x"] + pad, row["max_y"] + pad, row["max_z"] + pad])
        if not result.get("ok"):
            return None
        voxels = framed(to_voxels(result))
        if not voxels:
            return None
        buffer = _io.BytesIO()
        (await asyncio.to_thread(sheet, voxels, Assets())).save(buffer, format="PNG")
        return buffer.getvalue()

    async def match(self, spec, row, facts) -> dict | None:
        """Is this the thing that was asked for?

        The god described what it wanted in words and left the siting to the player, so it
        cannot check a specification — it has to look. Everything filtered before this point
        is a fact (their build, built after the asking, substantial, still standing); what
        happens here is judgement, and judgement is what the model is for.
        """
        import base64

        picture = await self.render_structure(row)
        if picture is None:
            return None
        described = {
            "you asked for": spec.intent,
            "you suggested it go": spec.where or "(you left the place to them)",
            "measurements": {k: facts.get(k) for k in
                             ("blocks", "dims", "fill", "vertical")},
            "materials": json.loads(row["materials"] or "{}"),
            "where they put it": [row["min_x"], row["min_y"], row["min_z"]],
        }
        client = model_client(asynchronous=True)
        reply = await client.messages.create(
            model=VISION_MODEL, max_tokens=900,
thinking=thinking_for(VISION_MODEL),
            system=MATCH_SYSTEM,
            messages=[{"role": "user", "content": [
                {"type": "image", "source": {
                    "type": "base64", "media_type": "image/png",
                    "data": base64.standard_b64encode(picture).decode()}},
                {"type": "text", "text": json.dumps(described, indent=1)}]}])
        text = "".join(b.text for b in reply.content if b.type == "text")
        verdict = json.loads(text[text.find("{"):text.rfind("}") + 1])
        # Keep the picture the verdict was reached on, so it can be checked afterwards.
        from render import save_for_review
        save_for_review(picture, f"quest_{spec.id}",
                        verdict.get("label") or spec.id,
                        float(verdict.get("confidence") or 0) or None)
        return verdict

    async def settle_by_looking(self, actor, spec, watcher) -> None:
        """Decide a where-in-words quest by looking at what the player built."""
        built = watcher.built_bbox()
        if not built:
            return
        (lo, hi, blocks, materials) = built
        row = {"id": f"candidate_{lo[0]}_{lo[1]}_{lo[2]}", "dim": spec.region.dim,
               "min_x": lo[0], "min_y": lo[1], "min_z": lo[2],
               "max_x": hi[0], "max_y": hi[1], "max_z": hi[2],
               "materials": json.dumps({k.replace("minecraft:", ""): v
                                        for k, v in materials.most_common()})}
        facts = {"blocks": blocks,
                 "dims": f"{hi[0]-lo[0]+1}x{hi[1]-lo[1]+1}x{hi[2]-lo[2]+1}",
                 "fill": round(blocks / max(1, (hi[0]-lo[0]+1) * (hi[1]-lo[1]+1)
                                            * (hi[2]-lo[2]+1)), 3),
                 "vertical": hi[1] - lo[1] >= 2}

        junk, why = featureless({"dims": [hi[i] - lo[i] + 1 for i in range(3)],
                                 "total": blocks,
                                 "placed_net": {k.replace("minecraft:", ""): v
                                                for k, v in materials.items()}})
        if junk:
            self.log(f"{RED}featureless: {why}{RESET}")
            await speak(f"That is not a thing, it is {why}. Try again.", URL)
            return

        self.log(f"{DIM}looking at {blocks} blocks at {lo}{RESET}")
        try:
            verdict = await self.match(spec, row, facts)
        except Exception as e:
            self.log(f"{YELLOW}could not look ({type(e).__name__}: {e}){RESET}")
            return
        if not verdict:
            return

        self.log(f"{DIM}is it? {verdict.get('is_it')} finished? "
                 f"{verdict.get('finished')} — {verdict.get('why')}{RESET}")
        if verdict.get("is_it") and verdict.get("finished"):
            await speak(verdict.get("why") or "It stands.", URL)
            when = int(time.time() * 1000)
            work_id = f"placement_{lo[0]}_{lo[1]}_{lo[2]}"
            self.store.put("work_events", {
                "id": work_id, "kind": "placement", "actor": actor,
                "dim": row["dim"],
                "min_x": row["min_x"], "min_y": row["min_y"], "min_z": row["min_z"],
                "max_x": row["max_x"], "max_y": row["max_y"], "max_z": row["max_z"],
                "materials": row["materials"],
            }, Belief(provenance=Provenance.OBSERVED, verified_at_tick=self.tick,
                      verified_at_ms=when,
                      value=json.dumps({**facts, "when": when, "significance": "work"})))
            self.store.put("relationships", {
                "id": f"answers:{work_id}", "subject": work_id,
                "predicate": "answers", "object": spec.id,
            }, Belief(provenance=Provenance.DERIVED, verified_at_tick=self.tick))
            self.remember_structure(actor, row, facts, verdict)
            self.store.put("quests", {
                "id": spec.id, "actor": actor, "spec": spec.to_json(),
                "state": "complete", "issued_tick": spec.issued_tick,
                "deadline_tick": spec.issued_tick + spec.deadline_ticks,
                "resolved_tick": self.tick,
            }, Belief(provenance=Provenance.DERIVED, verified_at_tick=self.tick))
            if watcher.completed_tick is None:
                watcher.completed_tick = self.tick
        else:
            await speak("Not yet — " + (verdict.get("why") or "keep at it."), URL)

    async def look(self, spec, intent: str, present: bool = False) -> dict | None:
        """Renders the build and asks what it looks like.

        Appearance is the one thing neither the event stream nor a histogram can reach. It is
        also the one thing that must never gate: a player refused for taste has no argument
        available to them. This shapes what the god SAYS, never whether they passed.
        """
        import base64

        from assets import Assets
        from render import framed, sheet
        lo, hi = spec.region.bbox()
        result = await request_voxels(URL, spec.region.dim, list(lo), list(hi))
        if not result.get("ok"):
            return None
        voxels = framed(to_voxels(result))
        if not voxels:
            return None
        image = await asyncio.to_thread(sheet, voxels, Assets())
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        from render import save_for_review
        save_for_review(buffer.getvalue(), f"quest_{spec.id}", spec.id)
        client = model_client(asynchronous=True)
        reply = await client.messages.create(
            model=VISION_MODEL, max_tokens=800,
thinking=thinking_for(VISION_MODEL),
            system=LOOK_SYSTEM,
            messages=[{"role": "user", "content": [
                {"type": "image", "source": {
                    "type": "base64", "media_type": "image/png",
                    "data": base64.standard_b64encode(buffer.getvalue()).decode()}},
                {"type": "text", "text":
                    f"The task was: {intent or spec.id}\n"
                    + ("The player is still standing there."
                       if present else "The player has moved on.")}]}])
        text = "".join(b.text for b in reply.content if b.type == "text")
        start, end = text.find("{"), text.rfind("}")
        return json.loads(text[start:end + 1])

    @staticmethod
    def describe(spec) -> str:
        """The quest, in one line a player can act on.

        A quest described in words says it in words. Falling back to "within 7 blocks of
        146,68,312" would put the coordinates straight back into the thing they were taken
        out of.
        """
        parts = []
        for c in (dict(c) for c in spec.constraints):
            t = c["type"]
            if t == "min_blocks":
                parts.append(f"at least {c['value']} blocks")
            elif t == "max_blocks":
                parts.append(f"no more than {c['value']} blocks")
            elif t == "min_dimensions":
                w, h, d = c["value"]
                parts.append(f"at least {w}x{h}x{d}")
            elif t == "material_fraction":
                parts.append(f"{c['min']:.0%} {c['block'].replace('minecraft:', '')}")
            elif t == "required":
                parts.append("using " + ", ".join(
                    b.replace("minecraft:", "") for b in c["blocks"]))
            elif t == "forbidden":
                parts.append("no " + ", ".join(
                    b.replace("minecraft:", "") for b in c["blocks"]))
            elif t == "collect":
                parts.append(f"bring back {c['count']} "
                             f"{c['item'].replace('minecraft:', '')}")
            elif t == "craft":
                parts.append(f"craft {c['count']} {c['item'].replace('minecraft:', '')}")
            elif t == "smelt":
                parts.append(f"smelt {c['count']} {c['item'].replace('minecraft:', '')}")
            elif t == "kill":
                parts.append(f"kill {c['count']} {c['entity'].replace('minecraft:', '')}")
            elif t == "breed":
                parts.append(f"breed {c['count']} {c['entity'].replace('minecraft:', '')}")
            elif t == "reach_depth":
                parts.append(f"go down to y {c['y']}")
            elif t == "advancement":
                parts.append(f"earn {c['id'].replace('minecraft:', '')}")

        # Nothing to count: the quest is the sentence.
        if not parts:
            return f"[{spec.id}] {spec.intent}" + (f" — {spec.where}" if spec.where else "")

        if spec.where:
            where = f"{spec.where}: "
        elif any(dict(c)["type"] not in DEED_CONSTRAINTS for c in spec.constraints) \
                and spec.region.radius > 1:
            cx, cy, cz = spec.region.center
            where = f"within {spec.region.radius} blocks of {cx},{cy},{cz}: "
        else:
            where = "wherever you find it: "
        return f"[{spec.id}] {where}" + "; ".join(parts) + "."

    def adopt_active_quests(self) -> int:
        """Picks up quests already outstanding in the store.

        Watchers live in memory, so restarting the god orphaned every quest: the store kept
        saying "active" while nothing was watching, and the next join issued a second one on
        top. The player then held two quests and was told about neither.

        Progress made before the restart is gone — a watcher counts events, and those are
        not replayed — so this adopts the newest and abandons any older duplicates rather
        than pretending the count survived.
        """
        adopted = 0
        for actor in {r["actor"] for r in self.store.all("quests", "state = 'active'")}:
            rows = sorted(self.store.all("quests", "actor = ? AND state = 'active'",
                                         (actor,)),
                          key=lambda r: -(r["issued_tick"] or 0))
            for old in rows[1:]:
                self.store.db.execute(
                    "UPDATE quests SET state = 'abandoned' WHERE id = ?", (old["id"],))
            keep = rows[0]
            try:
                proposal = json.loads(keep["spec"])
            except (TypeError, ValueError):
                continue
            spec, problems = validate(proposal, actor, keep["issued_tick"] or 0,
                                      now_t=int(proposal.get("issued_t") or 0))
            if problems:
                self.store.db.execute(
                    "UPDATE quests SET state = 'abandoned' WHERE id = ?", (keep["id"],))
                continue
            self.quests[actor] = (spec, Watcher(spec))
            adopted += 1
        self.store.db.commit()
        return adopted

    async def on_chat(self, event):
        """Answers a player who spoke to the god.

        Context is rebuilt from the store for every question, exactly as for every other
        call. What the god said before comes back through the utterances table marked
        ASSERTED rather than being carried along as conversation, so it can remember its own
        words without being able to cite them as evidence.
        """
        from context import Assembler, render
        from environment import (generated_feature_answer, persist_environment,
                                 render_environment,
                                 render_known_features, request_environment, requested_features,
                                 summarize_voxels, wants_environment)
        from grounding import (advice_conflicts, ground_query, render_grounding,
                               resolve_references)

        actor = event.get("actor")
        text = (event.get("text") or "").strip()
        if not actor or not text:
            return
        who = self.names.get(actor) or actor[:8]
        self.log(f"{BOLD}{who}{RESET} asks: {text}")
        if pure_greeting(text):
            said = "Well met."
            await speak(said, URL, target=self.names.get(actor), reply=True)
            self.store.record_utterance(actor, self.now_tick(), said, "greeting")
            return
        if not (self.use_model and have_key()):
            await speak("I have no voice today.", URL, reply=True)
            return
        try:
            await self.release_for_question(text, event.get("dim", "overworld"),
                                            event.get("pos"))
        except Exception as error:  # noqa: BLE001
            self.log(f"{YELLOW}could not pay a deferred look ({type(error).__name__})"
                     f"{RESET}")

        # Every non-greeting input is semantically inspected by the same evidence-gathering
        # model. Historical questions never pass through phrase gates or deterministic
        # summaries: model-written SELECTs / seed lookups -> model's plain-text narrative.
        from history import answer_if_history
        try:
            history_said = await answer_if_history(self.store, event, text, bridge_url=URL)
        except Exception as error:
            self.log(f"{RED}history query pipeline failed "
                     f"({type(error).__name__}: {error}){RESET}")
            await speak("Ask me again.", URL, target=self.names.get(actor), reply=True)
            return
        if history_said is not None:
            said = history_said.strip()
            if contains_raw_coordinates(said) and not coordinates_requested(text):
                said = remove_raw_coordinates(said)
            # Answer constraints are silent behaviour, not dialogue. This runs whether or
            # not a coordinate was found, because the placeholder a model leaves behind is
            # itself the announcement.
            said = remove_policy_announcements(said)
            await speak(said, URL, target=self.names.get(actor), reply=True)
            self.store.record_utterance(actor, self.now_tick(), said, "history SQL answer")
            return

        import base64

        held = self.quests.get(actor)
        spec = held[0] if held else None
        dim = event.get("dim", "overworld")
        pos = event.get("pos")
        try:
            grounded = await ground_query(self.store, text, dim, pos, URL, actor)
        except Exception as error:
            # Failure to establish a fact is rendered as missing grounding, never converted
            # into evidence that the structure is absent.
            grounded = {"rule": "Grounding failed; make no present-tense place claims.",
                        "resolved_count": 0, "ambiguous": False,
                        "structures": [], "relations": []}
            self.log(f"{YELLOW}grounding failed ({type(error).__name__}){RESET}")
        sections = Assembler(self.store).build(
            now_tick=self.now_tick(), actor=actor,
            trigger={"reason": f"{who} has spoken to you",
                     "pos": pos,
                     "facts": {"they said": text,
                               "their quest": spec.id if spec else "none"}},
            player_state=None, episodes=[], findings=[], immersive=True)

        pictures: list[bytes] = []
        environment_text = ""
        if wants_environment(text) and pos:
            features = requested_features(text)
            try:
                environment = await request_environment(
                    URL, dim, pos, radius=8192 if features else 256,
                    step=16, features=features)
                if environment.get("ok"):
                    persist_environment(self.store, environment)
                    feature_answer = generated_feature_answer(environment, features)
                    if feature_answer:
                        await speak(feature_answer, URL, target=self.names.get(actor),
                                    reply=True)
                        self.store.record_utterance(
                            actor, self.now_tick(), feature_answer, "generated feature answer")
                        return
                    environment_text = "\n\n" + render_environment(environment)
                else:
                    self.log(f"{YELLOW}environment RPC unavailable: "
                             f"{environment.get('error', 'unknown failure')}{RESET}")
                    if features:
                        said = "My sight into the untravelled world is unavailable just now."
                        await speak(said, URL, target=self.names.get(actor), reply=True)
                        self.store.record_utterance(actor, self.now_tick(), said,
                                                    "environment unavailable")
                        return
            except Exception as error:
                self.log(f"{YELLOW}environment query unavailable "
                         f"({type(error).__name__}); using terrain render{RESET}")
                if features:
                    said = "My sight into the untravelled world is unavailable just now."
                    await speak(said, URL, target=self.names.get(actor), reply=True)
                    self.store.record_utterance(actor, self.now_tick(), said,
                                                "environment unavailable")
                    return
            try:
                terrain_picture, _, terrain_voxels = await terrain_view(dim, pos)
                if terrain_picture:
                    pictures.append(terrain_picture)
                if not environment_text and terrain_voxels:
                    environment_text = "\n\n" + summarize_voxels(terrain_voxels, pos)
            except Exception as error:
                self.log(f"{YELLOW}terrain view unavailable ({type(error).__name__}){RESET}")
            known_features = render_known_features(self.store, dim, pos)
            if known_features:
                environment_text += "\n\n" + known_features

        # What they are already holding. Without it the god can only ever add: asked to
        # change a power it does not know the name of, it grants a second one beside the
        # first and both go on firing.
        powers_text = ""
        try:
            from scan import held_powers
            holding = (await held_powers(who, URL)).get("holding") or []
            powers_text = ("\n\nPOWERS THEY HOLD RIGHT NOW: "
                           + (", ".join(f"{name!r}" for name in holding) if holding
                              else "none")
                           + ". To change one, grant it again with the SAME name, which "
                             "replaces it. To take one away, revoke it by name; to take "
                             "them all, revoke with no name.")
        except Exception as error:  # noqa: BLE001
            self.log(f"{YELLOW}could not read held powers "
                     f"({type(error).__name__}){RESET}")

        visual_targets = {
            "current_position": pos,
            "resolved_structures": [
                {"id": item["id"], "bounds": item["bbox"]}
                for item in grounded.get("structures", []) if item.get("bbox")
            ],
        }
        prompt = (render(sections) + "\n\n" + render_grounding(grounded, immersive=True)
                  + environment_text + powers_text
                  + f"\n\nINTERNAL VISUAL TARGETS: {json.dumps(visual_targets)}"
                  + "\nYou have a bounded render_world_area tool. When appearance "
                    "materially determines the answer, call it and inspect the fresh image "
                    "instead of inferring appearance from names or materials. Structure "
                    "renders contain blocks only. If an interpretation depends on living "
                    "things being present, call inspect_world_entities separately before "
                    "making that claim. For voxel "
                    "sculpture, account for every major protrusion and indentation across "
                    "the views before categorizing it; do not classify from body and legs "
                    "while ignoring diagnostic face, ear, tail, horn, wing, or other "
                    "appendage geometry. Choose the single best-supported specific subject, "
                    "hedging that one interpretation if needed rather than listing "
                    "alternatives. Internal coordinates must not appear in speech."
                  + f"\n\n{who} says: {text!r}\n\n" + ANSWER_SCHEMA)

        # What the god has actually done in this exchange. The acting tool appends to it,
        # so the reply can be checked against the world rather than against the draft.
        acted: dict = {"done": [], "failures": [], "spell": None, "attempted": []}

        async def draft_answer(instruction: str) -> dict:
            from evidence import (ACT_TOOL, DESIGN_TOOL, ENTITY_TOOL, POWER_TOOL,
                                  VISUAL_TOOL, assistant_content, complete_message,
                                  inspect_world_entities, render_world_area,
                                  tool_result_block)

            client = model_client(asynchronous=True)
            content = [{"type": "image", "source": {
                "type": "base64", "media_type": "image/png",
                "data": base64.standard_b64encode(picture).decode()}}
                for picture in pictures]
            content.append({"type": "text", "text": instruction})
            messages = [{"role": "user", "content": content}]
            # Enough rounds to look, act, look again and put it right. Acting used to
            # happen after this loop closed, so the god could see before it acted and never
            # after: it laid ice and could not tell whether the ice it then tried to clear
            # was gone. A thing that can act but not observe the result is not doing the
            # job it was asked to do.
            #
            # Bounded by time as well as by rounds, and for the same reason the evidence
            # loop is: eight rounds of a slow provider is minutes, and a player waiting
            # that long has been given no answer whatever arrives eventually. Past the
            # budget the tools are withheld, which is what forces a reply rather than
            # another attempt.
            started = time.monotonic()
            for _ in range(8):
                spent = time.monotonic() - started > ACT_BUDGET_SECONDS
                if spent:
                    messages.append({"role": "user", "content": (
                        "Time is up. Answer now from what you have already done and seen. "
                        "Say plainly what worked and what did not; do not mention time.")})
                reply = await complete_message(
                    client, model=DIALOGUE_MODEL,
                    thinking=thinking_for(DIALOGUE_MODEL),
                    **({} if spent else
                       {"tools": [ACT_TOOL, DESIGN_TOOL, POWER_TOOL, VISUAL_TOOL,
                                 ENTITY_TOOL]}),
                    messages=messages)
                tool_uses = [block for block in reply.content if block.type == "tool_use"]
                if not tool_uses:
                    body = "".join(b.text for b in reply.content if b.type == "text")
                    return json.loads(body[body.find("{"):body.rfind("}") + 1])
                messages.append({"role": "assistant",
                                 "content": assistant_content(reply)})
                results = []
                for block in tool_uses:
                    if block.name == "design_build":
                        result = await self.design_build(block.input or {}, actor)
                        if result.get("error"):
                            # A tool that failed is something the player has to be told.
                            # Without this the exchange ended on the grounding guard's
                            # message, which describes our own machinery rather than what
                            # went wrong, and the player learned nothing.
                            acted["failures"].append(
                                "the builder gave me nothing to work with")
                    elif block.name == "grant_power":
                        result = await self.grant_powers(block.input or {}, actor)
                    elif block.name == "act_on_world":
                        result = await self.run_commands(block.input or {}, actor)
                        acted["done"].extend(result["done"])
                        acted["failures"].extend(result["failures"])
                        acted["attempted"].extend(result.get("attempted") or [])
                        acted["spell"] = result.get("spell") or acted["spell"]
                        acted["duration"] = result.get("duration") or acted.get("duration")
                    elif block.name == "render_world_area":
                        result = await render_world_area(block, URL)
                    elif block.name == "inspect_world_entities":
                        result = await inspect_world_entities(block, URL)
                    else:
                        result = {"error": f"unknown tool: {block.name}"}
                    results.append(tool_result_block(block, result))
                messages.append({"role": "user", "content": results})
            raise RuntimeError("visual evidence loop exhausted its tool budget")

        try:
            data = await draft_answer(prompt)
        except Exception as e:
            self.log(f"{RED}answer failed ({type(e).__name__}: {e}){RESET}")
            await speak("Ask me again.", URL, reply=True)
            return

        requested_action = data.get("action")
        said = str(data.get("speech") or "").strip()
        speech_refused = False
        # The first draft is not spoken yet. Ground every named place and spatial claim it
        # actually chose to make. If that adds evidence, ask for one evidence-aware rewrite;
        # a second draft that introduces a new ungrounded place is refused rather than
        # recursively chasing model prose.
        try:
            draft_grounded = await ground_query(self.store, said, dim, pos, URL, actor)
            initial_ids = {item["id"] for item in grounded["structures"]}
            draft_ids = {item["id"] for item in draft_grounded["structures"]}
            # Only a place the draft actually NAMED needs grounding. A structure whose
            # stored description happens to contain "world" or "remains" is a coincidence
            # of prose, and treating it as a claim spent a rewrite on every reply and then
            # refused answers that named nothing at all.
            named_in_draft = {item["id"] for item in draft_grounded["structures"]
                              if item.get("role") == "referenced"}
            reasons = []
            if named_in_draft - initial_ids or (draft_grounded["relations"] and
                                                draft_grounded["relations"] != grounded["relations"]):
                reasons.append("the draft introduced a place or relation needing grounding")
            if said and contains_raw_coordinates(said) and not coordinates_requested(text):
                reasons.append("the draft exposed raw coordinates instead of landmarks")
            conflicts = advice_conflicts(said, grounded)
            if conflicts:
                reasons.append("the draft proposed features already present: "
                               + ", ".join(conflicts))
            if announces_answer_policy(said):
                reasons.append("the draft narrated an answer policy instead of following it")
            if len(said) > 420 or len(re.findall(r"[.!?](?:\s|$)", said)) > 3:
                reasons.append("the draft is too long for in-game chat")
            needs_rewrite = bool(reasons)
            if said and needs_rewrite:
                rewrite = (prompt + "\n\nUNSPOKEN DRAFT:\n" + said + "\n\n"
                           + render_grounding(draft_grounded, immersive=True)
                           + "\nProblems requiring correction: " + "; ".join(reasons)
                           + "\nRewrite the draft once using these grounded facts. JSON only.")
                data = await draft_answer(rewrite)
                said = str(data.get("speech") or "").strip()
                requested_action = data.get("action")
                allowed = initial_ids | draft_ids
                final_grounded = await ground_query(self.store, said, dim, pos, URL, actor)
                final_ids = {item["id"] for item in final_grounded["structures"]
                             if item.get("role") == "referenced"}
                new_relation = (bool(final_grounded["relations"])
                                and not bool(draft_grounded["relations"]))
                if final_ids - allowed or new_relation:
                    self.log(f"{YELLOW}answer introduced an ungrounded place or relation; "
                             f"refused{RESET}")
                    said = "I cannot ground that answer reliably enough to give it."
                    speech_refused = True
                final_conflicts = advice_conflicts(said, grounded)
                if final_conflicts:
                    self.log(f"{YELLOW}answer repeated contradicted advice; refused{RESET}")
                    said = "I need a closer look before I can suggest an honest improvement."
        except Exception as error:
            self.log(f"{YELLOW}draft grounding failed ({type(error).__name__}); refused{RESET}")
            said = "I cannot verify the places needed to answer that reliably."
        if said and contains_raw_coordinates(said) and not coordinates_requested(text):
            self.log(f"{YELLOW}removed raw coordinates from answer{RESET}")
            said = remove_raw_coordinates(said)
        if said and announces_answer_policy(said):
            self.log(f"{YELLOW}removed answer-policy narration{RESET}")
            said = (remove_policy_announcements(said)
                    or "I cannot answer that from what I can presently see.")
        said = trim_chat_answer(said)

        # Act first, then speak. A god that announces a gift and is then refused by the
        # server has told the player something untrue, and there is no taking it back.
        # Anything the model still put in `commands` runs too, so a simple act need not
        # cost a tool round; anything it did through the tool has already happened.
        if data.get("commands"):
            extra = await self.run_commands(data, actor)
            acted["done"].extend(extra["done"])
            acted["failures"].extend(extra["failures"])
            acted["attempted"].extend(extra.get("attempted") or [])
            acted["spell"] = extra.get("spell") or acted["spell"]
            acted["duration"] = extra.get("duration") or acted.get("duration")
        if acted["attempted"] and not acted["done"]:
            # It tried, looked, tried again and still could not. Appending a correction to
            # a sentence that already claims the gift tells the player two different things.
            self.log(f"{DIM}nothing was done; the claim goes with it{RESET}")
            said = "That I could not do: " + acted["failures"][0].rstrip(".") + "."
        elif acted["done"] and speech_refused:
            # The guard rejects prose that makes unfounded claims about places. It does not
            # decide whether the player may have what they asked for, and it must never
            # leave them holding diamonds beside a sentence saying nothing could be said.
            self.log(f"{DIM}the draft was refused but the act stands; reporting it{RESET}")
            said = self.report_of(acted)
        elif speech_refused and acted["failures"]:
            # Refused prose plus a real failure: say the failure. The guard's own sentence
            # describes our machinery, and a player who asked for a castle and got
            # "I cannot ground that answer" has been told nothing they can use.
            self.log(f"{DIM}the draft was refused; reporting the failure instead{RESET}")
            said = "That I could not do: " + acted["failures"][0].rstrip(".") + "."
        elif acted["failures"]:
            said = ((said + " ") if said else "") + ("Part of it I could not do: "
                                                    + acted["failures"][0].rstrip(".") + ".")

        if said:
            await speak(said, URL, target=self.names.get(actor), reply=True)
            self.store.record_utterance(actor, self.now_tick(), said, "answer")

        # The model proposes an action; code decides whether it happens.
        action = requested_action
        if action in ("reroll", "abandon") and held:
            self.store.db.execute(
                "UPDATE quests SET state = ?, resolved_tick = ? WHERE id = ?",
                ("abandoned", self.now_tick(), spec.id))
            self.store.db.commit()
            del self.quests[actor]
            self.log(f"{YELLOW}{spec.id} abandoned at their request{RESET}")
            if action == "reroll":
                await asyncio.sleep(0.5)
                await self.on_join({"type": "player_join", "actor": actor,
                                    "tick": self.tick, "t": event.get("t", 0),
                                    "dim": event.get("dim", "overworld"),
                                    "pos": event.get("pos") or [0, 64, 0],
                                    "name": self.names.get(actor)})
        elif action in ("reroll", "abandon"):
            await speak("You have nothing of mine to set aside.", URL, reply=True)

    async def run_commands(self, data, actor: str | None = None) -> dict:
        """Run what the model drafted: one command, a thousand, or a lasting effect.

        Nothing here caps how much is asked for. The agent checks each command so a
        forbidden draft is refused instantly and the model told why; the server checks again
        and is the one that decides, and paces the work so the tick survives.

        The player is the anchor by default. Almost everything anyone asks for is relative
        to where they are standing — fire above them, ice beneath them, a bridge from here —
        and a command run from the console is relative to the world origin instead, which is
        never what was meant.
        """
        from commands import clean, refusal

        drafted = data.get("commands") if isinstance(data, dict) else data
        if isinstance(drafted, str):
            drafted = [drafted]
        if not drafted:
            return {"done": [], "failures": [], "queued": 0, "spell": None}

        anchor = self.names.get(actor) if actor else None
        if isinstance(data, dict) and data.get("anchor"):
            anchor = str(data["anchor"])
        every = int((data or {}).get("every") or 0) if isinstance(data, dict) else 0
        duration = int((data or {}).get("duration") or 0) if isinstance(data, dict) else 0
        spell = (data or {}).get("spell") if isinstance(data, dict) else None

        ready, failures = [], []
        for raw in drafted:
            command = clean(str(raw))
            refused = refusal(command, anchored=bool(anchor))
            if refused:
                self.log(f"{YELLOW}refused /{command}: {refused}{RESET}")
                failures.append(refused)
            else:
                ready.append(command)
        if not ready:
            return {"done": [], "failures": failures, "queued": 0, "spell": None}

        try:
            result = await run_command(ready, URL, anchor=anchor, every=every,
                                       duration=duration, spell=spell)
        except Exception as error:  # noqa: BLE001 - the bridge may be down
            self.log(f"{YELLOW}could not act ({type(error).__name__}){RESET}")
            return {"done": [], "failures": failures + ["the world did not answer"],
                    "queued": 0, "spell": None}
        if not result.get("ok"):
            reason = result.get("error") or "it did not work"
            self.log(f"{YELLOW}refused: {reason}{RESET}")
            return {"done": [], "failures": failures + [reason], "queued": 0,
                    "spell": None}

        failures.extend(result.get("refused") or [])
        cast = result.get("spell")
        # A spell answers on acceptance because it runs for minutes. A one-shot batch
        # answers with what each command actually did, so nothing is claimed unseen.
        ran = result.get("ran") or []
        done, idle = [], []
        for item in ran:
            output = (item.get("output") or "").strip()
            if not item.get("ok"):
                trouble = output or "it did not work"
                self.log(f"{YELLOW}/{item['command']} failed: {trouble}{RESET}")
                failures.append(trouble)
                continue
            # A command can succeed and change nothing: a fill whose region held none of
            # the block it was replacing reports "Changed 0 blocks" and reports it as
            # success. Left unmarked, the god reads that as done and says the ice is gone.
            if _changed_nothing(output):
                self.log(f"{YELLOW}/{item['command']} changed nothing{RESET}")
                idle.append({"command": item["command"], "output": output})
                continue
            done.append(item["command"])
        if not ran:
            done = list(ready)
        self.log(f"{GREEN}{len(done)} command(s) ran{RESET}"
                 + (f" {DIM}at {anchor}{RESET}" if anchor else "")
                 + (f" {DIM}as {cast}, every {every} for {duration} ticks{RESET}"
                    if cast else ""))
        report = {"done": done, "failures": failures, "spell": cast,
                  "every": every, "duration": duration, "anchor": anchor,
                  "attempted": ready}
        if idle:
            report["changed_nothing"] = idle
            report["note"] = ("some commands succeeded but changed nothing; whatever they "
                              "were aimed at was not there. Look, then try elsewhere or "
                              "wider rather than reporting it done")
        return report

    async def design_build(self, request: dict, actor: str | None = None) -> dict:
        """Ask the builder for a shape. It returns commands; running them stays the loop's
        job, so the god still looks at what it made.

        The builder is given the ground first. It used to be told only what to build and a
        single anchor point, which is not enough to put a thing down: on a slope it sets one
        foot in the air and buries the other, and it cannot know whether it is about to
        write a statue through somebody's roof.
        """
        from builder import SITE_RADIUS, design, survey

        what = str(request.get("what") or "").strip()
        if not what:
            return {"error": "say what to build"}
        anchor = [int(request.get("x", 0)), int(request.get("y", 64)),
                  int(request.get("z", 0))]
        dim = str(request.get("dim") or "overworld")

        site, standing = None, []
        try:
            # The server refuses a voxel read taller than 48, and asking for 24 either side
            # of the anchor is 49. That one block put every build back to blind, and the
            # refusal was swallowed, so the log said "blind" and never said why.
            lo = [anchor[0] - SITE_RADIUS, max(-64, anchor[1] - 24),
                  anchor[2] - SITE_RADIUS]
            hi = [anchor[0] + SITE_RADIUS, lo[1] + 47, anchor[2] + SITE_RADIUS]
            read = await request_voxels(URL, dim, lo, hi)
            if read.get("ok"):
                site = survey(to_voxels(read), anchor)
            else:
                self.log(f"{YELLOW}could not survey the site: "
                         f"{read.get('error')}; building blind{RESET}")
        except Exception as error:  # noqa: BLE001 - a blind build beats no build
            self.log(f"{YELLOW}could not survey the site "
                     f"({type(error).__name__}: {error}); building blind{RESET}")
        try:
            from grounding import box_of, current, current_generated
            for row in list(current(self.store, dim)) + list(
                    current_generated(self.store, dim)):
                lo_b, hi_b = box_of(row)
                if not lo_b or not hi_b:
                    continue
                # Anything whose box overlaps the surveyed square, however it is placed.
                # Comparing centres missed a village wall running through the site.
                if (lo_b[0] <= anchor[0] + SITE_RADIUS
                        and hi_b[0] >= anchor[0] - SITE_RADIUS
                        and lo_b[2] <= anchor[2] + SITE_RADIUS
                        and hi_b[2] >= anchor[2] - SITE_RADIUS):
                    # structure_name returns the whole belief row, not a string, and a
                    # sqlite3.Row does not serialise. That raised a TypeError from inside
                    # json.dumps which surfaced as "the builder did not answer", so four
                    # retries chased a provider that was never the problem. The label is
                    # the row's object; its value holds the description.
                    named = self.store.structure_name(row["id"])
                    label, about = "unnamed", ""
                    if named:
                        label = str(named["object"] or "unnamed")
                        try:
                            about = str(json.loads(named["value"] or "{}")
                                        .get("description") or "")
                        except (ValueError, TypeError):
                            about = ""
                    here = {"name": label,
                            "from": [int(v) for v in lo_b],
                            "to": [int(v) for v in hi_b]}
                    if about:
                        here["description"] = about
                    standing.append(here)
        except Exception as error:  # noqa: BLE001
            self.log(f"{YELLOW}could not list what stands here "
                     f"({type(error).__name__}){RESET}")

        # A build is minutes, not seconds. Silence for that long is indistinguishable from
        # the god having ignored them, and they were told as much.
        if actor and self.names.get(actor):
            try:
                await speak("Hold on. I am making it.", URL,
                            target=self.names.get(actor), reply=True)
            except Exception:  # noqa: BLE001
                pass
        self.log(f"{DIM}asking the builder for {what!r} at {anchor}"
                 f"{' with a survey' if site else ' blind'}"
                 f"{f', {len(standing)} thing(s) already here' if standing else ''}"
                 f"{RESET}")
        result = await design(what, anchor, str(request.get("facing") or ""),
                              str(request.get("materials") or ""),
                              site=site, standing=standing or None)
        if result.get("commands"):
            self.log(f"{GREEN}the builder returned {result['count']} command(s){RESET}"
                     f" {DIM}{result.get('describes')}{RESET}")
        else:
            self.log(f"{YELLOW}the builder gave nothing: {result.get('error')}{RESET}")
        return result

    async def grant_powers(self, request: dict, actor: str | None = None) -> dict:
        """Give or take an ability the model has just invented.

        Nothing here decides what a power may be. The vocabulary it is built from lives in
        the plugin, its commands pass the plugin's own gate, and anything unrecognised comes
        back named so the next draft can use a word that exists.
        """
        from scan import grant_power

        player = str(request.get("player") or self.names.get(actor) or "").strip()
        # A missing name means "all of them" when taking away, and only when granting does
        # it need a placeholder. Defaulting it to "power" on both paths made "remove all my
        # superpowers" ask for one called "power", match nothing, and report success.
        name = str(request.get("name") or "").strip()
        if not player:
            return {"error": "no player named"}
        if not name and not request.get("revoke"):
            name = "power"
        scripts = request.get("scripts") or {}
        if not isinstance(scripts, dict):
            scripts = {}
        try:
            result = await grant_power(
                player, name, URL,
                scripts=scripts,
                switches=request.get("switches") or [],
                on=request.get("on") or [],
                projectile=request.get("projectile") or None,
                speed=float(request.get("speed") or 0),
                impulse=request.get("impulse") or None,
                power=float(request.get("power") or 0),
                beam=request.get("beam") or None,
                range=int(request.get("range") or 0),
                damage=float(request.get("damage") or 0),
                every=int(request.get("every") or 0),
                duration=int(request.get("duration") or 0),
                cooldown_ms=int(request.get("cooldown_ms") or 200),
                revoke=bool(request.get("revoke")))
        except Exception as error:  # noqa: BLE001
            return {"error": f"the world did not answer ({type(error).__name__})"}
        if result.get("ok") and request.get("revoke"):
            took = result.get("revoked") or []
            self.log(f"{GREEN}took {took or 'nothing'} from {player}{RESET}"
                     f" {DIM}still holding {result.get('holding')}{RESET}")
        elif result.get("ok"):
            verb = "changed" if result.get("replaced") else "gave"
            thrown = result.get("projectile")
            self.log(f"{GREEN}{verb} {player} '{result.get('granted')}'{RESET}"
                     f" {DIM}{result.get('triggers')} {result.get('switches')}"
                     f"{' throws ' + thrown if thrown else ''}{RESET}")
            if result.get("unknown") or result.get("refused"):
                self.log(f"{YELLOW}not understood: {result.get('unknown')} "
                         f"{result.get('refused')}{RESET}")
        return result

    @staticmethod
    def report_of(result: dict) -> str:
        """A plain sentence for what the god just did.

        Used when the drafted prose is thrown away. Never invented, and never claiming more
        than was accepted: the work is paced by the server, so what is true at this moment
        is that it is under way.
        """
        if result.get("spell"):
            seconds = max(1, int(result.get("duration") or 0) // 20)
            return f"It is done, and will hold for {seconds} seconds."
        count = len(result.get("done") or [])
        if count > 1:
            return "It is done."
        return "It is done."

    async def on_event(self, event):
        self.tick = max(self.tick, event.get("tick", 0))
        self.tick_seen_at = time.monotonic()
        kind = event["type"]
        # Index every event before dispatch. Chat and join used to return early and only
        # appear in history after the next restart replay, which broke current-session
        # boundaries and made locative follow-ups impossible to resolve live.
        try:
            self.revise(event)
        except Exception:
            traceback.print_exc()
        if kind == "player_join":
            await self.on_join(event)
            return
        if kind == "chat":
            await self.on_chat(event)
            return

        for _, (_, watcher) in list(self.quests.items()):
            watcher.observe(event)
        await self.evaluate()

    def now_tick(self) -> int:
        """The current tick, estimated when nothing is arriving.

        The event stream is entirely player-driven, so it stops the moment the last player
        logs out. Advancing the clock only on events meant a player who finished a build and
        immediately quit was never judged: the debounce could not elapse because nothing
        arrived to notice that it had. The server keeps ticking whether or not anyone is
        watching, so the clock is extrapolated from wall time.
        """
        if not self.tick:
            return 0
        return self.tick + int((time.monotonic() - self.tick_seen_at) * 20)

    async def evaluate(self):
        """Checks every live quest. Safe to call on an event or on a timer."""
        now = self.now_tick()
        for actor, (spec, watcher) in list(self.quests.items()):
            if watcher.completed_tick is None and watcher.expired(now):
                self.log(f"{RED}{spec.id} expired{RESET}")
                await speak("Your time for that is gone.", URL)
                self.store.db.execute(
                    "UPDATE quests SET state = 'expired', resolved_tick = ? WHERE id = ?",
                    (now, spec.id))
                self.store.db.commit()
                del self.quests[actor]
                continue
            if not watcher.ready(now):
                continue
            watcher.judged_at_tick = watcher.last_activity_tick

            # A quest described in words is settled by looking at what they built, not by
            # checking a specification. Everything filtered before this is a fact: their
            # blocks, placed after the asking, enough of them to be a thing.
            if not spec.constraints:
                await self.settle_by_looking(actor, spec, watcher)
                continue

            # Stage two. The watcher was necessary, never sufficient.
            self.log(f"watcher tripped for {spec.id}; scanning to confirm")
            lo, hi = spec.region.bbox()
            scan = await request_scan(URL, spec.region.dim, list(lo), list(hi))
            verdict = judge(spec, scan, watcher)
            self.log(f"measured: {verdict.summary}")

            # Two tiers. Code decides the gates — that something substantial was built and
            # that it still stands — and no interpretation may overturn them. Everything
            # else is evidence a model weighs against the stated intent, reading only
            # numbers that code produced. It never sees the world.
            complete, reason = verdict.complete, verdict.summary

            # Deterministic, and the only aesthetic-adjacent thing that may refuse anyone:
            # a solid single-material cuboid is measurably featureless, and no picture is
            # needed to say so.
            junk, why = featureless(verdict.measurements)
            if junk:
                self.log(f"{RED}featureless build: {why}{RESET}")
                await speak(f"That is not a thing, it is {why}. Try again.", URL)
                continue

            if verdict.gates_passed and self.use_model and have_key():
                try:
                    call = await judge_semantically(spec, verdict, DIALOGUE_MODEL)
                    complete = bool(call.get("met"))
                    reason = call.get("reason") or reason
                    self.log(f"{DIM}semantic verdict: {complete} — {reason}{RESET}")
                except Exception as e:
                    self.log(f"{YELLOW}semantic judge failed ({type(e).__name__}); "
                             f"falling back to the literal checks{RESET}")
            elif not verdict.gates_passed:
                self.log(f"{RED}hard gates not met; not asking anyone's opinion{RESET}")

            revisit = watcher.completed_tick is not None
            if complete:
                await speak(("You have been at it again. " if revisit else "It stands. ")
                            + verdict.summary, URL)
                self.store.put("quests", {
                    "id": spec.id, "actor": actor, "spec": spec.to_json(),
                    "state": "complete", "issued_tick": spec.issued_tick,
                    "deadline_tick": spec.issued_tick + spec.deadline_ticks,
                    "resolved_tick": watcher.completed_tick or now,
                }, Belief(provenance=Provenance.DERIVED, verified_at_tick=now))
                # The watcher stays registered so that later work on the same build is
                # noticed and remarked on. A completion is never revoked — they did build
                # it, and that is history; what can still change is whether it stands.
                if watcher.completed_tick is None:
                    watcher.completed_tick = now

                # What the god thinks of the craftsmanship. Flavour, never a gate.
                if self.use_model and have_key():
                    try:
                        look = await self.look(spec, spec.intent, present=watcher.present)
                        if look:
                            self.log(f"{DIM}looks like: {look.get('looks_like')}{RESET}")
                            await asyncio.sleep(3.2)
                            await speak(look.get("notes", ""), URL)
                    except Exception as e:
                        self.log(f"{YELLOW}render/look failed ({type(e).__name__}: {e})"
                                 f"{RESET}")
                if watcher.present:
                    await asyncio.sleep(3.2)
                    await speak("Keep working it if you are not finished — "
                                "I will look again.", URL)
            elif revisit:
                # It passed once and no longer does. Say so without taking it back.
                await speak("What you built there has changed — " + reason, URL)
            else:
                await speak("Not yet — " + reason, URL)


async def run(use_model: bool) -> int:
    store = Store()
    print(f"{DIM}syncing world state…{RESET}", flush=True)
    print(f"{DIM}  {sync_world(store)}{RESET}", flush=True)
    god = God(store, use_model)
    adopted = god.adopt_active_quests()
    if adopted:
        print(f"{DIM}  picked up {adopted} quest(s) already outstanding{RESET}", flush=True)
    active_model = (f"{MODEL_PROVIDER}/{DIALOGUE_MODEL}; vision={VISION_MODEL}"
                    if use_model and have_key() else "off")
    print(f"{BOLD}god{RESET} watching {URL}  "
          f"{DIM}(model: {active_model}){RESET}",
          flush=True)

    while True:
        try:
            await _watch(god)
        except (OSError, ConnectionClosed, InvalidHandshake) as e:
            print(f"{YELLOW}lost the bridge ({type(e).__name__}); reconnecting{RESET}",
                  flush=True)
        except asyncio.CancelledError:
            break
        await asyncio.sleep(2)
    return 0


#: Events a player is waiting on an answer to. Everything else can wait for them.
URGENT_EVENTS = frozenset({"chat", "player_join"})


def is_urgent(event) -> bool:
    return event.get("type") in URGENT_EVENTS


async def _watch(god) -> None:
    """One connection's lifetime.

    Reading and thinking are separate tasks on purpose. A model call takes the better part
    of a minute, and while the god was awaiting one it stopped draining its socket — the
    server watched its outbound buffer fill, decided the consumer had stalled, and
    disconnected it. The reader now always drains; a queue absorbs the burst.
    """
    from scan import bind_rpc, dispatch_rpc, unbind_rpc

    queue: asyncio.Queue = asyncio.Queue(maxsize=4096)
    # Someone who speaks is waiting for a reply. Everything else can wait for them.
    #
    # One worker drained one queue, so a question sat behind whatever the loop was already
    # doing: a quest being judged, a build being segmented, an ordinary event whose
    # handling ran long. Each is seconds to a minute of awaited model work, and a player
    # who asked a question and got nothing for minutes has been given no answer.
    #
    # Speech gets its own worker. Both still index through the same store on the same
    # thread, so nothing races; they simply no longer wait on each other.
    spoken: asyncio.Queue = asyncio.Queue(maxsize=64)

    async def work(inbox: asyncio.Queue):
        while True:
            event = await inbox.get()
            try:
                await god.on_event(event)
            except Exception:
                traceback.print_exc()
            finally:
                inbox.task_done()

    async def think():
        await work(queue)

    async def listen():
        await work(spoken)

    async def heartbeat():
        """Keeps judging when nothing is arriving.

        The event stream is entirely player-driven and stops dead when the last player logs
        out — exactly when a finished build most wants judging.
        """
        while True:
            await asyncio.sleep(5)
            try:
                await god.evaluate()
            except Exception:
                traceback.print_exc()

    async def reconcile_loop():
        """Confirms beliefs: the contradicted ones promptly, the merely quiet ones slowly.

        A belief an event has disproved is worth confirming within seconds — someone who
        just killed your chickens will notice if the god still speaks of a flock — while a
        belief that has only gone quiet can wait. The debounce means a player mid-demolition
        is not scanned once per block.
        """
        while True:
            await asyncio.sleep(2)
            try:
                pending = bool(god.disturbed) or bool(god.store.db.execute(
                    "SELECT 1 FROM structures WHERE value LIKE '%contradicted%' "
                    "LIMIT 1").fetchone())
                due = 5 if pending else 120
                if time.monotonic() - god.last_sweep > due:
                    god.last_sweep = time.monotonic()
                    await god.sweep()
            except Exception:
                traceback.print_exc()

    # Voxel RPC replies share this connection and can legitimately exceed the WebSocket
    # library's 1 MiB default, while the server-side scan limits keep them bounded.
    async with connect(URL, max_queue=8192, ping_interval=20,
                       max_size=64 * 1024 * 1024) as ws:
        bind_rpc(ws)
        print(f"{GREEN}connected. join the server.{RESET}", flush=True)
        workers = [asyncio.create_task(think()), asyncio.create_task(listen()),
                   asyncio.create_task(heartbeat()),
                   asyncio.create_task(reconcile_loop())]
        try:
            async for frame in ws:
                event = json.loads(frame)
                if "rpc" in event:
                    dispatch_rpc(event)
                    continue
                if is_urgent(event):
                    # Never dropped, and never behind the backlog.
                    if spoken.full():
                        await spoken.get()
                        spoken.task_done()
                    spoken.put_nowait(event)
                    continue
                if queue.full():
                    # Never block the reader. Heartbeat samples are the ones worth losing:
                    # dropping one costs a little precision, stalling costs the connection.
                    if event["type"] in ("move", "player_state", "region_enter"):
                        continue
                    await queue.get()
                    queue.task_done()
                queue.put_nowait(event)
        finally:
            unbind_rpc(ws)
            for w in workers:
                w.cancel()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--no-model", action="store_true",
                   help="skip the model and use the deterministic quest")
    args = p.parse_args()
    try:
        sys.exit(asyncio.run(run(not args.no_model)))
    except KeyboardInterrupt:
        sys.exit(130)
