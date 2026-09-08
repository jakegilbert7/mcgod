#!/usr/bin/env python3
"""Quests: the model sets the goal, code decides whether it was met.

    python3 quests.py --demo sessions/corpus/01-house.jsonl

Non-negotiable #6 is the whole design here. A quest is a constraint spec — a machine-checkable
description of a thing that must exist — and completion is decided by evaluating it against a
region scan. The model chooses what to ask for and narrates the verdict. It never decides
whether the player succeeded, and it never writes to the store.

Three rules from CLAUDE.md shape the watchers:

  Two-stage.  The incremental predicate is necessary, not sufficient. A watcher trips, a
              region scan confirms, and only then is the model woken. Counting placements
              cannot tell you a tower is 20 blocks tall or that no dirt was used where it
              matters; only reading the world can.
  Debounce.   Fire after a quiet period in the region, not the instant a threshold is
              crossed, or the god interrupts someone mid-build to congratulate them.
  Unregister. On completion, failure or timeout. Otherwise every block placed fans out to
              thousands of dead predicates.

Watchers are O(1) per event: a region test and a counter bump.
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

#: How long the region must be quiet before a build is looked at.
#:
#: One short setting for everyone. A longer grace for players still standing in their build
#: was tempting — they are often fetching materials — but the person most likely to be
#: waiting on a reply is the one who just finished and is standing there watching for it.
#: Three minutes of silence reads as absence, not patience.
#:
#: Judging early is cheap because it is not final: the watcher re-arms whenever more work
#: happens, so carrying on simply earns another look. Presence is still tracked, but it
#: shapes what the god SAYS rather than how long it waits — someone still standing in their
#: build is told that continuing is welcome, rather than being handed a verdict.
SETTLE_TICKS = 900            # 45s

#: How far outside the quest region a player counts as having left.
AWAY_MARGIN = 12

DEFAULT_DEADLINE_TICKS = 72000  # 1 hour


# --- the spec -------------------------------------------------------------------------

@dataclass(frozen=True)
class Region:
    dim: str
    center: tuple
    radius: int

    def bbox(self) -> tuple:
        x, y, z = self.center
        r = self.radius
        return (x - r, y - r, z - r), (x + r, y + r, z + r)

    def contains(self, pos) -> bool:
        x, y, z = self.center
        return (abs(pos[0] - x) <= self.radius
                and abs(pos[1] - y) <= self.radius
                and abs(pos[2] - z) <= self.radius)


#: Every constraint the system understands. A model proposing anything else is rejected
#: rather than interpreted — an unknown constraint that silently passes is a quest the
#: player can complete by doing nothing.
#:
#: Two families. The first describes a thing that must exist in the world and is settled by
#: a region scan. The second describes something the player must DO, and is settled by
#: counting events since the quest was issued — which is stricter than checking an
#: inventory, because it cannot be satisfied by what they already had.
CONSTRAINTS = {
    # about a structure, judged against the region
    "min_blocks": ("value",),
    "max_blocks": ("value",),
    "min_dimensions": ("value",),
    "material_fraction": ("block", "min"),
    "required": ("blocks",),
    "forbidden": ("blocks",),
    # about the player, judged against the event stream
    "collect": ("item", "count"),
    "craft": ("item", "count"),
    "smelt": ("item", "count"),
    "kill": ("entity", "count"),
    "breed": ("entity", "count"),
    "reach_depth": ("y",),
    "advancement": ("id",),
}

#: Constraints that need no region and no scan.
DEED_CONSTRAINTS = {"collect", "craft", "smelt", "kill", "breed", "reach_depth",
                    "advancement"}

#: What each animal is bred with. A short, stable domain fact, not a heuristic.
BREEDING_FOOD = {
    "minecraft:chicken": {"minecraft:wheat_seeds", "minecraft:beetroot_seeds",
                          "minecraft:melon_seeds", "minecraft:pumpkin_seeds",
                          "minecraft:torchflower_seeds"},
    "minecraft:cow": {"minecraft:wheat"}, "minecraft:sheep": {"minecraft:wheat"},
    "minecraft:mooshroom": {"minecraft:wheat"},
    "minecraft:pig": {"minecraft:carrot", "minecraft:potato", "minecraft:beetroot"},
    "minecraft:rabbit": {"minecraft:carrot", "minecraft:golden_carrot",
                         "minecraft:dandelion"},
    "minecraft:horse": {"minecraft:golden_apple", "minecraft:golden_carrot"},
    "minecraft:wolf": {"minecraft:beef", "minecraft:mutton", "minecraft:chicken"},
    "minecraft:cat": {"minecraft:cod", "minecraft:salmon"},
    "minecraft:bee": {"minecraft:dandelion", "minecraft:poppy"},
}


def _strip_means(checked: list) -> list:
    """Drops constraints that only describe how to satisfy another one.

    A quest should name the end, not the path to it. "Bring back 16 wheat seeds; breed 4
    chickens" asks a player who already has seeds to go and fetch more for no reason — the
    seeds were never the point. Removing the input rather than rejecting the whole proposal
    keeps the quest and loses only the busywork.
    """
    wanted = {c["entity"] for c in checked if c["type"] == "breed"}
    inputs: set = set()
    for entity in wanted:
        inputs |= BREEDING_FOOD.get(entity, set())
    if not inputs:
        return checked
    return [c for c in checked
            if not (c["type"] == "collect" and c.get("item") in inputs)]


@dataclass(frozen=True)
class QuestSpec:
    id: str
    actor: str
    region: Region
    constraints: tuple
    #: Where to build it, in words a player would use — "on the rise west of your pen",
    #: not a coordinate and a radius.
    #:
    #: Coordinates and percentages are how you describe a thing to a compiler. A player told
    #: "within 7 blocks of 146,68,312: at least 55 blocks, 30% cobblestone" is being handed
    #: a specification, and a god that can only recognise its own specification is not
    #: seeing the world, it is checking a form. The site is found afterwards by looking at
    #: what they actually built.
    where: str = ""
    #: A structure this one should stand near, by id, so "beside your workshop" can be
    #: grounded without pinning a coordinate.
    near: str = ""
    deadline_ticks: int = DEFAULT_DEADLINE_TICKS
    issued_tick: int = 0
    #: What the quest is actually for, in words. Constraints are a description of the intent,
    #: never a substitute for it — see `judge` and `HARD_GATES`.
    intent: str = ""
    #: Wall clock at issue. Ticks restart with the server and cannot say what came after.
    issued_t: int = 0
    #: Material histogram of the region as it stood when the quest was issued.
    #:
    #: Judgment compares against this, never against the raw scan. A quest region is mostly
    #: ground: asking for a cobblestone tower inside a 25-block radius and then measuring
    #: cobblestone as a fraction of everything solid in that box gives 1%, because the box
    #: contains seven thousand blocks of hillside. Non-negotiable #4 applies to quests as
    #: much as to episodes — what matters is what changed, not what is there.
    baseline: tuple = ()

    def to_json(self) -> str:
        return json.dumps({
            "id": self.id, "actor": self.actor, "intent": self.intent,
            "region": {"dim": self.region.dim, "center": list(self.region.center),
                       "radius": self.region.radius},
            "where": self.where, "near": self.near,
            "constraints": [dict(c) for c in self.constraints],
            "deadline_ticks": self.deadline_ticks,
            "issued_tick": self.issued_tick, "issued_t": self.issued_t,
        })


def validate(proposal: dict, actor: str, now_tick: int, baseline: dict | None = None,
             now_t: int = 0) -> tuple:
    """Turns a model's proposal into a spec, or explains why it cannot be one.

    This is the gate. Model output is a proposal, never a fact, and a quest that reaches the
    store unvalidated is a rule the player is judged against that nobody checked. Observed in
    practice: asked to suggest a quest, the model invented its own schema — objectives, a
    reward block, a `where` clause — none of which this system can evaluate. Interpreting
    that generously would mean judging a player against a rule we made up on their behalf.
    """
    problems = []

    quest_id = str(proposal.get("id") or "").strip()
    if not quest_id.replace("_", "").replace("-", "").isalnum():
        problems.append("id must be a non-empty alphanumeric slug")

    # The intent is what judgment grades against, so a quest without one is judged against
    # its own slug. One proposal shipped with an empty intent and validation accepted it.
    if not str(proposal.get("intent") or "").strip():
        problems.append("intent must say, in words, what the quest is actually for")

    raw = proposal.get("constraints")
    if raw is None:
        raw = []
    if not isinstance(raw, list):
        problems.append("constraints must be a list")
        raw = []

    checked = []
    needs_region = False
    for i, c in enumerate(raw):
        if not isinstance(c, dict):
            problems.append(f"constraint {i} is not an object")
            continue
        kind = c.get("type")
        if kind not in CONSTRAINTS:
            problems.append(f"constraint {i}: unknown type {kind!r}; "
                            f"known types are {sorted(CONSTRAINTS)}")
            continue
        missing = [f for f in CONSTRAINTS[kind] if f not in c]
        if missing:
            problems.append(f"constraint {i} ({kind}): missing {missing}")
            continue
        if kind == "min_dimensions" and not (
                isinstance(c["value"], list) and len(c["value"]) == 3):
            problems.append(f"constraint {i}: min_dimensions value must be [w, h, d]")
            continue
        if kind == "material_fraction" and not (0 < c["min"] <= 1):
            problems.append(f"constraint {i}: material_fraction min must be in (0, 1]")
            continue
        if kind not in DEED_CONSTRAINTS:
            needs_region = True
        checked.append(dict(c))

    # Only a quest about a structure needs somewhere to put it. A quest to kill ten
    # skeletons or bring back iron is about the player, not a place.
    region = proposal.get("region") or {}
    center = region.get("center")
    radius = region.get("radius")
    if needs_region:
        if not (isinstance(center, list) and len(center) == 3
                and all(isinstance(v, (int, float)) for v in center)):
            problems.append("region.center must be [x, y, z] for a build")
        if not isinstance(radius, (int, float)) or not (1 <= radius <= 64):
            problems.append("region.radius must be a number from 1 to 64")
    elif not (isinstance(center, list) and len(center) == 3):
        center, radius = [0, 0, 0], 1

    deadline = proposal.get("deadline_ticks", DEFAULT_DEADLINE_TICKS)
    if not isinstance(deadline, int) or not (1200 <= deadline <= 24 * 72000):
        problems.append("deadline_ticks must be between 1200 and 1728000")

    where = str(proposal.get("where") or "").strip()[:200]
    near = str(proposal.get("near") or "").strip()[:64]

    # A quest is either a thing to build, described in words, or a set of countable deeds.
    # Counting is natural for "kill three skeletons" and reads as homework for "a tower" —
    # so a build quest is allowed to carry no constraints at all, and is judged by looking
    # at what appeared.
    if not checked and not where:
        problems.append("say either what to build, in `where` and `intent`, "
                        "or what to do, in `constraints`")

    if problems:
        return None, problems
    checked = _strip_means(checked)
    return QuestSpec(
        id=quest_id, actor=actor,
        region=Region(dim=region.get("dim", "overworld"),
                      center=tuple(int(v) for v in center),
                      radius=int(radius or 1)),
        constraints=tuple(tuple(sorted(c.items())) for c in checked),
        deadline_ticks=int(deadline), issued_tick=now_tick, issued_t=now_t,
        where=where, near=near,
        intent=str(proposal.get("intent") or "").strip()[:400],
        baseline=tuple(sorted((baseline or {}).items()))), []


def _constraints(spec: QuestSpec) -> list:
    return [dict(c) for c in spec.constraints]


# --- stage one: the watcher -----------------------------------------------------------

@dataclass
class Watcher:
    """An O(1) incremental predicate over the event stream.

    Necessary, never sufficient. It can count blocks placed inside a region; it cannot know
    how tall the result is, whether it is hollow, or whether the dirt someone placed is part
    of the structure or the path they walked in on. Its only job is to decide when a scan is
    worth doing.
    """

    spec: QuestSpec
    placed: collections.Counter = field(default_factory=collections.Counter)
    broken: collections.Counter = field(default_factory=collections.Counter)
    #: Tight bounds of the blocks actually placed, updated in O(1) per event. A scan returns
    #: a histogram and no positions — by design, since block arrays never leave the server —
    #: so the only honest source for "how big is the thing they built" is where the placement
    #: events happened. Composition is judged from the world; extent from what was observed.
    lo: list = field(default_factory=lambda: [None, None, None])
    hi: list = field(default_factory=lambda: [None, None, None])
    #: Every block they put down, with what it was. A bounding box cannot tell a building
    #: from a building plus the path someone walked in on; the positions can.
    spots: list = field(default_factory=list)
    last_activity_tick: int = 0
    tripped_tick: int | None = None
    failed_reason: str | None = None
    live: bool = True
    #: Where the player was last seen, and whether they were near the build.
    present: bool = True
    #: What the player has DONE since the quest was issued, keyed by constraint.
    #:
    #: Counted from events rather than read off an inventory: "bring me thirty iron" should
    #: mean thirty iron won since I asked, not thirty iron you happened to be carrying.
    deeds: collections.Counter = field(default_factory=collections.Counter)
    deepest: int = 999
    earned: set = field(default_factory=set)

    #: Set once the quest has been judged complete. The watcher stays alive afterwards so
    #: that further work on the same build is noticed, but a completion is never revoked:
    #: they did build it, and that is history. What can change is whether it still stands.
    completed_tick: int | None = None
    judged_at_tick: int = 0

    def observe(self, event: dict) -> None:
        if not self.live:
            return
        tick = event["tick"]
        if (self.completed_tick is None
                and tick - self.spec.issued_tick > self.spec.deadline_ticks):
            self.live = False
            self.failed_reason = "deadline passed"
            return
        if event.get("actor") != self.spec.actor:
            return

        # Presence is tracked from the movement samples that already exist, so knowing
        # whether someone wandered off costs nothing extra. Depth is read from the same
        # samples — returning here before recording it left "go down to y=-40" impossible
        # to satisfy, because the only events that carry a y coordinate never got that far.
        if event["type"] in ("move", "player_state", "region_enter"):
            pos = event.get("pos")
            if pos:
                self.present = self.spec.region.contains(
                    pos) or self._near(pos, AWAY_MARGIN)
                self.deepest = min(self.deepest, pos[1])
            return
        if event["type"] == "player_quit":
            self.present = False
            return
        # Deeds are counted wherever they happen. A quest to bring back iron is about the
        # player, not about a place, so confining these to the region would make most of
        # them impossible.
        kind = event["type"]
        if kind in ("item_pickup", "craft", "smelt") and event.get("item"):
            self.deeds[(kind, event["item"])] += event.get("count", 1)
            self.last_activity_tick = tick
        elif kind in ("mob_kill", "breed") and event.get("entity"):
            self.deeds[(kind, event["entity"])] += 1
            self.last_activity_tick = tick
        elif kind == "advancement" and event.get("advancement"):
            self.earned.add(event["advancement"])
            self.last_activity_tick = tick

        if kind not in ("block_place", "block_break"):
            return
        pos = event.get("pos")
        if not pos:
            return
        # A quest that named no place watches the player, not a box. They were told what to
        # build and left to choose where, so anywhere they build is where it might be.
        if self.spec.region.radius > 1 and not self.spec.region.contains(pos):
            return

        self.last_activity_tick = tick
        if event["type"] == "block_place":
            self.placed[event["after"]] += 1
            self.spots.append((pos[0], pos[1], pos[2], event["after"]))
            for i in range(3):
                self.lo[i] = pos[i] if self.lo[i] is None else min(self.lo[i], pos[i])
                self.hi[i] = pos[i] if self.hi[i] is None else max(self.hi[i], pos[i])
        else:
            self.broken[event["before"]] += 1

    @property
    def net(self) -> collections.Counter:
        n = collections.Counter(self.placed)
        n.subtract(self.broken)
        return collections.Counter({k: v for k, v in n.items() if v > 0})

    def deed_progress(self, c: dict) -> tuple:
        """How far along one deed constraint is, as (done, needed).

        Counted from events since the quest was issued, never from an inventory: "bring me
        thirty iron" should mean thirty iron won since I asked, not thirty you happened to
        be carrying when I asked.
        """
        kind = c["type"]
        if kind == "collect":
            done = sum(v for (k, item), v in self.deeds.items()
                       if k in ("item_pickup", "craft", "smelt") and item == c["item"])
            return done, c["count"]
        if kind in ("craft", "smelt"):
            return self.deeds.get((kind, c["item"]), 0), c["count"]
        if kind in ("kill", "breed"):
            key = "mob_kill" if kind == "kill" else "breed"
            return self.deeds.get((key, c["entity"]), 0), c["count"]
        if kind == "reach_depth":
            return (1, 1) if self.deepest <= c["y"] else (0, 1)
        if kind == "advancement":
            return (1, 1) if c["id"] in self.earned else (0, 1)
        return 0, 1

    def necessary(self) -> bool:
        """Could the quest plausibly be complete? Cheap, optimistic, never conclusive."""
        # A quest described in words has nothing to count against. The bar is simply that
        # they built something worth looking at; whether it is the right thing is decided
        # by looking, not here.
        if not self.spec.constraints:
            built = self.built_bbox()
            return bool(built and built[2] >= CANDIDATE_FLOOR)
        for c in _constraints(self.spec):
            if c["type"] in DEED_CONSTRAINTS:
                done, needed = self.deed_progress(c)
                if done < needed:
                    return False
        total = sum(self.net.values())
        building = [c for c in _constraints(self.spec)
                    if c["type"] not in DEED_CONSTRAINTS]
        for c in building:
            if c["type"] == "min_blocks" and total < c["value"]:
                return False
            if c["type"] == "required":
                for block in c["blocks"]:
                    if not self.net.get(block):
                        return False
            if c["type"] == "material_fraction":
                if total and self.net.get(c["block"], 0) / total < c["min"] * 0.8:
                    return False
        # A quest with nothing to build is satisfied by deeds alone.
        return total > 0 if building else True

    def _near(self, pos, margin: int) -> bool:
        cx, cy, cz = self.spec.region.center
        r = self.spec.region.radius + margin
        return (abs(pos[0] - cx) <= r and abs(pos[1] - cy) <= r
                and abs(pos[2] - cz) <= r)

    def ready(self, now_tick: int) -> bool:
        """Has the player stopped for long enough that the build can be called finished?

        Debounced on the session boundary rather than on the threshold crossing, or the god
        congratulates someone in the middle of building. Once judged, the watcher only
        becomes ready again if there has been new work since — which is what lets a player
        keep improving something and be told about it, without being re-judged for standing
        still.
        """
        if not self.live or self.last_activity_tick <= 0:
            return False
        if self.last_activity_tick <= self.judged_at_tick:
            return False
        if not self.necessary() and self.completed_tick is None:
            return False
        return now_tick - self.last_activity_tick >= SETTLE_TICKS

    def expired(self, now_tick: int) -> bool:
        return now_tick - self.spec.issued_tick > self.spec.deadline_ticks

    def built_bbox(self) -> tuple | None:
        """Tight bounds of the largest thing they built, and its materials.

        For a quest with no named place this is how the god finds what to look at: cluster
        what they actually put down and take the biggest group. A player who lays a path on
        the way to the site should not have the path judged as the building.
        """
        if not self.spots:
            return None
        points = {(x, y, z): m for x, y, z, m in self.spots}
        groups, remaining = [], set(points)
        while remaining:
            seed = remaining.pop()
            group, frontier = [seed], [seed]
            while frontier:
                cx, cy, cz = frontier.pop()
                near = [p for p in remaining
                        if abs(p[0] - cx) <= 6 and abs(p[1] - cy) <= 6
                        and abs(p[2] - cz) <= 6]
                for p in near:
                    remaining.discard(p)
                    group.append(p)
                    frontier.append(p)
            groups.append(group)
        biggest = max(groups, key=len)
        xs, ys, zs = zip(*biggest)
        materials = collections.Counter(points[p] for p in biggest)
        return ((min(xs), min(ys), min(zs)), (max(xs), max(ys), max(zs)),
                len(biggest), materials)

    def built_dims(self) -> list:
        """Extent of what was placed, or zeros if nothing was."""
        if self.lo[0] is None:
            return [0, 0, 0]
        return [self.hi[i] - self.lo[i] + 1 for i in range(3)]


# --- finding what they actually built ---------------------------------------------------

#: A candidate has to be at least this many blocks to be worth looking at, whatever else is
#: true of it. Below this, someone stacked a few blocks; nothing was built.
CANDIDATE_FLOOR = 25


def candidates(store, spec: QuestSpec, now_tick: int = 0) -> list:
    """Structures that could be the answer to a build quest.

    The god no longer says where to build, so it has to find out. Everything that filters
    here is a fact and none of it is a judgement: built by the player who was asked, built
    after the asking, substantial rather than a handful of blocks, and not already claimed
    by another quest. Deciding which of the survivors is the thing — or that none of them
    is — is the model's job, done by looking at them.
    """
    claimed = set()
    for row in store.all("relationships"):
        if row["predicate"] == "answers":
            claimed.add(row["subject"])

    out = []
    for row in store.all("work_events", "actor = ? AND kind = 'placement'", (spec.actor,)):
        if row["id"] in claimed:
            continue
        try:
            facts = json.loads(row["value"] or "{}")
        except (TypeError, ValueError):
            continue
        if (facts.get("blocks") or 0) < CANDIDATE_FLOOR:
            continue
        # Built since the asking. Ticks restart with the server, so the wall clock decides.
        if spec.issued_t and (facts.get("when") or 0) < spec.issued_t:
            continue
        out.append((row, facts))
    out.sort(key=lambda p: -(p[1].get("when") or 0))
    return out


# --- stage two: deterministic judgment ------------------------------------------------

#: Gates that code decides and a model may never overturn.
#:
#: Non-negotiable #6 said judgment is wholly deterministic. In practice that produced quests
#: like "at least 55 blocks; no more than 200; at least 5x5x5; 30% cobblestone" — a
#: spreadsheet, which punishes a player who builds something better than what was asked for
#: in a way nobody would defend. So the rule is narrowed rather than abandoned: these two
#: facts are measured and binding, and everything else becomes evidence a model weighs.
#:
#: The distinction that keeps this honest is that the model never perceives the world. It
#: reads numbers that code produced from a scan and an event stream. It cannot invent a
#: block that is not there, only decide whether what IS there satisfies the intent.
#:
#: The gates exist so no interpretation can conclude that nothing is something:
#:   substantive  the player really placed a meaningful amount in the region
#:   standing     it is still there when checked
HARD_GATES = ("substantive", "still_standing")

#: A build must reach at least this fraction of the requested block count to be considered
#: at all. Generous on purpose: the gate is against nothing, not against creativity.
SUBSTANTIVE_FRACTION = 0.5


def featureless(measurements: dict) -> tuple:
    """Is this a solid block of one material with no shape to it?

    Deterministic on purpose. "Clearly rubbish and gaming the system" turns out to be
    measurable — a filled cuboid of a single material has a fill ratio near one, one
    material, and no interior — whereas "is it pretty" does not. Keeping this in code means
    the model is never asked to be the bouncer, only the critic, and a player cannot be
    refused on taste.
    """
    dims = measurements.get("dims") or [0, 0, 0]
    total = measurements.get("total", 0)
    materials = measurements.get("placed_net") or {}
    volume = max(1, dims[0] * dims[1] * dims[2])
    fill = total / volume
    reasons = []
    if len(materials) <= 1 and fill > 0.85:
        reasons.append("a solid mass of a single material")
    if dims[1] <= 1 and fill > 0.85:
        reasons.append("a flat slab one block thick")
    if total and max(materials.values(), default=0) / total > 0.98 and fill > 0.9:
        reasons.append("effectively one material, filled solid")
    return bool(reasons), "; ".join(reasons)


@dataclass
class Verdict:
    complete: bool
    checks: list
    summary: str
    measurements: dict = field(default_factory=dict)

    @property
    def gates_passed(self) -> bool:
        return all(c["ok"] for c in self.checks if c["type"] in HARD_GATES)


def judge(spec: QuestSpec, scan: dict, watcher: "Watcher") -> Verdict:
    """Decides whether the quest was met. Never asks a model.

    Two sources, each used for what it is actually good at.

    **The event stream says what the player built.** Every constraint about composition,
    count and extent is evaluated against blocks the player was observed placing. This is
    precise and attributable, and immune to everything the world does on its own.

    **The scan says whether it is still standing.** That is the only question events cannot
    answer, and it is the whole reason scanning exists: a player who builds a tower and then
    removes it has not built a tower.

    Judging composition from the scan was wrong and failed a real quest. The region is mostly
    hillside, so the raw scan was hopeless; the delta against a baseline was better but still
    counted the world's own side effects as the player's work. Placing a block on grass turns
    the grass underneath into dirt through a block update that fires no placement event — so
    a player who placed 61 cobblestone, 105 oak planks and no dirt whatsoever was told
    "found dirt" and refused.
    """
    def bare(name: str) -> str:
        return str(name).replace("minecraft:", "")

    # What the player put there, minus what they took back out.
    materials = {bare(k): v for k, v in watcher.net.items()}
    placed_ever = {bare(k): v for k, v in watcher.placed.items()}
    solid = sum(materials.values())
    dims = list(watcher.built_dims())

    checks = []
    for c in _constraints(spec):
        kind = c["type"]
        # Deeds are settled by counting events, with no scan involved: the world does not
        # record that someone killed ten skeletons, only the event stream does.
        if kind in DEED_CONSTRAINTS:
            done, needed = watcher.deed_progress(c)
            what = bare(str(c.get("item") or c.get("entity") or c.get("id")
                            or c.get("y", "")))
            ok = done >= needed
            checks.append({"type": kind, "ok": ok,
                           "detail": (f"{what}: {done} of {needed}" if needed > 1
                                      else f"{what}: {'done' if ok else 'not yet'}")})
            continue
        if kind == "min_blocks":
            ok, detail = solid >= c["value"], f"{solid} solid blocks, need {c['value']}"
        elif kind == "max_blocks":
            ok, detail = solid <= c["value"], f"{solid} solid blocks, limit {c['value']}"
        elif kind == "min_dimensions":
            need = c["value"]
            ok = all(dims[i] >= need[i] for i in range(3))
            detail = f"{dims[0]}x{dims[1]}x{dims[2]}, need {need[0]}x{need[1]}x{need[2]}"
        elif kind == "material_fraction":
            have = materials.get(bare(c["block"]), 0)
            frac = have / solid if solid else 0.0
            ok = frac >= c["min"]
            detail = f"{bare(c['block'])} is {frac:.0%} of it, need {c['min']:.0%}"
        elif kind == "required":
            absent = [bare(b) for b in c["blocks"] if not placed_ever.get(bare(b))]
            ok, detail = not absent, ("all used" if not absent else f"never used {absent}")
        elif kind == "forbidden":
            # What the player CHOSE to place. "Do not use dirt" is about their hands, not
            # about what the world put under their foundation.
            used = [bare(b) for b in c["blocks"] if placed_ever.get(bare(b))]
            ok, detail = not used, ("none used" if not used else f"used {used}")
        else:
            # Unreachable through `validate`, but a quest must never pass by accident.
            ok, detail = False, f"unknown constraint {kind!r}"
        checks.append({"type": kind, "ok": ok, "detail": detail})

    # A quest with nothing to build has nothing to stand or fall down.
    if not any(c["type"] not in DEED_CONSTRAINTS for c in _constraints(spec)):
        complete = all(c["ok"] for c in checks)
        failed = [c["detail"] for c in checks if not c["ok"]]
        return Verdict(complete, checks,
                       "all constraints met" if complete else "; ".join(failed),
                       measurements={"deeds": {f"{k[0]}:{k[1]}": v
                                               for k, v in watcher.deeds.items()},
                                     "total": 0, "dims": [0, 0, 0], "survival": 1.0,
                                     "placed_net": {}, "placed_ever": {}})

    # Survival: the blocks the player placed must actually still be in the world. This is
    # the one thing events cannot tell us, and the only reason a scan is taken at all.
    present = {bare(k): v for k, v in (scan.get("materials") or {}).items()}
    expected = sum(materials.values())
    standing = sum(min(v, present.get(k, 0)) for k, v in materials.items())
    survival = standing / expected if expected else 0.0
    checks.append({
        "type": "still_standing", "ok": survival >= 0.9,
        "detail": f"{survival:.0%} of what you placed is still there",
    })

    # The one gate that is about effort rather than compliance.
    wanted = next((dict(c)["value"] for c in spec.constraints
                   if dict(c)["type"] == "min_blocks"), 1)
    floor = max(8, int(wanted * SUBSTANTIVE_FRACTION))
    checks.append({
        "type": "substantive", "ok": solid >= floor,
        "detail": f"{solid} blocks placed and kept, floor is {floor}",
    })

    complete = all(c["ok"] for c in checks)
    failed = [c["detail"] for c in checks if not c["ok"]]
    return Verdict(complete, checks,
                   "all constraints met" if complete else "; ".join(failed),
                   measurements={
                       "placed_net": materials, "placed_ever": placed_ever,
                       "total": solid, "dims": dims, "survival": round(survival, 3),
                   })


# --- demo ------------------------------------------------------------------------------

def main(argv: list[str]) -> int:
    import json as _json
    from consumer import BOLD, DIM, GREEN, RED, RESET

    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("session", type=Path)
    p.add_argument("--spec", type=Path, help="a proposal to validate and run")
    args = p.parse_args(argv)

    events = [_json.loads(l) for l in args.session.open() if l.strip()]
    actor = next(e["actor"] for e in events if e.get("actor") != "world")

    proposal = _json.loads(args.spec.read_text()) if args.spec else {
        "id": "a_place_to_live",
        "intent": "a durable shelter fit to live in",
        "region": {"dim": "overworld", "center": [175, 65, 297], "radius": 12},
        "constraints": [
            {"type": "min_blocks", "value": 80},
            {"type": "min_dimensions", "value": [5, 4, 5]},
            {"type": "material_fraction", "block": "minecraft:cobblestone", "min": 0.25},
            {"type": "forbidden", "blocks": ["minecraft:netherrack"]},
        ],
        "deadline_ticks": 72000,
    }

    spec, problems = validate(proposal, actor, now_tick=events[0]["tick"])
    if problems:
        print(f"{RED}rejected{RESET}")
        for problem in problems:
            print(f"  - {problem}")
        return 1
    print(f"{BOLD}issued{RESET} {spec.id} to {actor[:8]} "
          f"in {spec.region.dim} within {spec.region.radius} of {spec.region.center}")

    watcher = Watcher(spec)
    tripped_at = None
    for e in events:
        watcher.observe(e)
        if tripped_at is None and watcher.ready(e["tick"]):
            tripped_at = e["tick"]
    if tripped_at is None:
        # A watcher outlives the session that created it. Polling only while events arrive
        # models it wrongly: a player who finishes a build and immediately logs off has
        # still finished it, and the debounce simply elapses while they are away.
        after = events[-1]["tick"] + SETTLE_TICKS
        if watcher.ready(after) and not watcher.expired(after):
            tripped_at = after
            print(f"  {DIM}(session ended before the debounce elapsed; the watcher "
                  f"persists and fires at tick {after}){RESET}")
    print(f"  watcher saw {sum(watcher.placed.values())} placed / "
          f"{sum(watcher.broken.values())} broken in region; net "
          f"{sum(watcher.net.values())}")
    if tripped_at is None:
        print(f"  {DIM}never tripped — no scan would have been spent{RESET}")
        return 0
    print(f"  {GREEN}tripped at tick {tripped_at}{RESET} "
          f"{DIM}(after the region settled){RESET}")
    print(f"  {DIM}stage two would now scan {spec.region.bbox()} to confirm{RESET}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))


SEMANTIC_PROMPT = """\
A player was set this task and has stopped working. Decide whether they satisfied it.

You are NOT looking at the world. Everything below was measured by code from a region scan
and from the blocks this player was observed placing. You cannot see anything else, and you
must not assume anything the measurements do not show.

Judge the INTENT, not the letter. The numeric constraints describe what was asked for; they
are not a checklist to tick. A player who builds something better than what was specified
has satisfied it. A player who technically hits every number with a featureless box has not.
Missing a threshold narrowly is not a failure if the thing they made clearly serves the
intent; missing it because they did not really try is.

Reply with JSON only: {"met": true|false, "reason": "<one sentence to the player>"}"""


async def judge_semantically(spec: QuestSpec, verdict: Verdict, model: str) -> dict:
    """Asks a model whether the measurements satisfy the intent.

    Only ever called once the hard gates have passed, so the worst a wrong answer can do is
    be generous about a real build or harsh about one. It can never invent a structure: the
    only thing it sees is arithmetic that code performed.
    """
    facts = {
        "intent": spec.intent or "(none stated)",
        "asked_for": [dict(c) for c in spec.constraints],
        "measured": verdict.measurements,
        "deterministic_checks": verdict.checks,
    }
    from model_api import model_client
    client = model_client(asynchronous=True)
    reply = await client.messages.create(
        model=model, max_tokens=600, 
thinking=thinking_for(model),
        system=SEMANTIC_PROMPT,
        messages=[{"role": "user", "content": json.dumps(facts, indent=1)}])
    text = "".join(b.text for b in reply.content if b.type == "text")
    start, end = text.find("{"), text.rfind("}")
    return json.loads(text[start:end + 1])
