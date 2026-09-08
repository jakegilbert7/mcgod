#!/usr/bin/env python3
"""World state, derived from the episode pipeline.

    python3 world.py sessions/corpus/*.jsonl          # derive and print
    python3 world.py sessions/corpus/*.jsonl --store  # derive and persist

This is the model of the world, not a report about it. Events become episodes, episodes and
findings become state: who exists, what they can do, where they operate, what stands in the
world and who put it there. The god's prompt is one view of this; a query, a quest predicate
or a dashboard are others. Nothing here formats for a language model.

Everything is DERIVED — deterministic computation over OBSERVED events. No classification,
no guessing. The one place a model's opinion enters is a structure's name, which is stored
separately as INFERRED and never merged into these rows.
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

from consumer import BOLD, DIM, RESET
from detectors import analyse
from store import NON_PLAYER_ACTORS, Belief, Provenance, Store

#: Material progression. Small, stable domain fact rather than a heuristic: these are the
#: tiers Minecraft itself defines, and "what is this player equipped to do" is unanswerable
#: without them. It is not a list of interesting things and does not grow.
TIERS = ["wooden", "leather", "stone", "copper", "chainmail", "golden", "iron",
         "diamond", "netherite"]
TOOL_KINDS = ("pickaxe", "axe", "shovel", "hoe", "sword")
ARMOR_KINDS = ("helmet", "chestplate", "leggings", "boots")

REGION = 64


def _tier_of(item: str) -> int:
    base = item.replace("minecraft:", "").split("_")[0]
    return TIERS.index(base) if base in TIERS else -1


def fold_live_player(store: Store, event: dict) -> bool:
    """Apply one observed event to the durable player view without waiting for restart.

    Startup replay remains the canonical reconstruction. This reducer only closes the gap
    between an event arriving and the next replay, and is deliberately idempotent for
    cumulative vanilla statistics.
    """
    actor = event.get("actor")
    if not actor or actor in NON_PLAYER_ACTORS:
        return False
    row = store.get("players", actor)
    try:
        profile = json.loads(row["profile"] or "{}") if row else {}
    except (TypeError, ValueError):
        profile = {}
    kind = event.get("type")
    pos = event.get("pos")
    if pos:
        profile["last_pos"] = list(pos)
    if kind == "player_state":
        for key in ("health", "food", "level"):
            if key in event:
                profile[key] = event[key]
        profile["equipment"] = {key: event[key] for key in
                                ("held", "helmet", "chestplate", "leggings", "boots")
                                if event.get(key)}
        travel = dict(profile.get("travel_m") or {})
        for vehicle, centimetres in (event.get("travel_cm") or {}).items():
            try:
                travel[vehicle] = max(float(travel.get(vehicle, 0)),
                                      round(int(centimetres) / 100.0, 1))
            except (TypeError, ValueError):
                continue
        profile["travel_m"] = travel
        if event.get("vehicle"):
            profile["vehicle"] = {
                key: event[key] for key in (
                    "vehicle", "vehicle_id", "vehicle_passengers",
                    "vehicle_in_water", "vehicle_surface")
                if key in event
            }
        else:
            # A sampled absence is fresh state, not missing information. Retaining the old
            # vehicle here would make the durable player view claim a ride never ended.
            profile.pop("vehicle", None)
    elif kind == "vehicle_enter":
        vehicle = str(event.get("vehicle") or "unknown").replace("minecraft:", "")
        entries = dict(profile.get("vehicle_entries") or {})
        entries[vehicle] = int(entries.get(vehicle, 0)) + 1
        profile["vehicle_entries"] = entries
        profile["vehicle"] = {
            key: event[key] for key in (
                "vehicle", "vehicle_id", "passengers", "vehicle_in_water",
                "vehicle_surface")
            if key in event
        }
    elif kind == "vehicle_exit":
        profile.pop("vehicle", None)
    elif kind == "mob_kill" and event.get("entity"):
        kills = dict(profile.get("kills") or {})
        kills[event["entity"]] = int(kills.get(event["entity"], 0)) + 1
        profile["kills"] = kills
    elif kind == "player_death":
        profile["deaths"] = int(profile.get("deaths") or 0) + 1
    elif kind == "block_place":
        profile["placed"] = int(profile.get("placed") or 0) + 1
    elif kind == "block_break":
        profile["broken"] = int(profile.get("broken") or 0) + 1

    store.put("players", {
        "id": actor, "name": row["name"] if row else event.get("name"),
        "dim": event.get("dim") or (row["dim"] if row else "?"),
        "last_seen_tick": event.get("tick"), "profile": json.dumps(profile),
    }, Belief(provenance=Provenance.DERIVED,
              first_seen_tick=(row["first_seen_tick"] if row else event.get("tick")),
              verified_at_tick=event.get("tick"), verified_at_ms=event.get("t")))
    return True


@dataclass
class Actor:
    """What is known about one player, as state rather than history."""

    id: str
    first_seen_t: int = 0
    last_seen_t: int = 0
    playtime_ms: int = 0
    sessions: int = 0

    # where
    last_pos: tuple | None = None
    last_dim: str = "?"
    home_cell: tuple | None = None
    cells_visited: set = field(default_factory=set)
    y_min: int = 999
    y_max: int = -999
    range_bbox: tuple | None = None

    # condition, as of the most recent sample
    health: float | None = None
    food: int | None = None
    level: int | None = None
    equipment: dict = field(default_factory=dict)

    # what they have done
    placed: collections.Counter = field(default_factory=collections.Counter)
    broken: collections.Counter = field(default_factory=collections.Counter)
    crafted: collections.Counter = field(default_factory=collections.Counter)
    kills: collections.Counter = field(default_factory=collections.Counter)
    deaths: int = 0
    #: The last few places they died, so danger can be spoken of as a place rather than as
    #: a lifetime score. A total alone made every quest a shelter.
    recent_deaths: list = field(default_factory=list)
    damage_taken: float = 0.0
    health_floor: float = 20.0
    advancements: list = field(default_factory=list)
    #: Cumulative vanilla travel counters, keyed by vehicle. These survive plugin upgrades
    #: and therefore recover journeys that happened before vehicle events were captured.
    travel_cm: collections.Counter = field(default_factory=collections.Counter)
    vehicle_entries: collections.Counter = field(default_factory=collections.Counter)
    materials_seen: set = field(default_factory=set)
    best_tool: str | None = None
    best_armor: str | None = None

    # what they have made that still stands
    built: list = field(default_factory=list)
    episodes: int = 0

    @property
    def build_ratio(self) -> float:
        """Above 0.5 is a builder, below is a miner. A fact, not a label."""
        total = sum(self.placed.values()) + sum(self.broken.values())
        return sum(self.placed.values()) / total if total else 0.0


class World:
    """Derives and holds the state of everything known."""

    def __init__(self) -> None:
        self.actors: dict[str, Actor] = {}

    def _actor(self, actor_id: str) -> Actor:
        return self.actors.setdefault(actor_id, Actor(id=actor_id))

    def observe(self, events: list[dict], episodes, findings) -> None:
        """Folds one session into the world. Call repeatedly; state accumulates."""
        by_actor = collections.defaultdict(list)
        for e in events:
            if e.get("actor") and e["actor"] != "world":
                by_actor[e["actor"]].append(e)

        for actor_id, own in by_actor.items():
            a = self._actor(actor_id)
            a.sessions += 1
            a.first_seen_t = min(a.first_seen_t or own[0]["t"], own[0]["t"])
            a.last_seen_t = max(a.last_seen_t, own[-1]["t"])
            a.playtime_ms += own[-1]["t"] - own[0]["t"]

            cells = collections.Counter()
            for e in own:
                kind = e["type"]
                pos = e.get("pos")
                if pos:
                    a.y_min = min(a.y_min, pos[1])
                    a.y_max = max(a.y_max, pos[1])
                    if kind in ("move", "player_state"):
                        cells[(pos[0] // REGION, pos[2] // REGION)] += 1
                if kind == "block_place":
                    a.placed[e["after"]] += 1
                    a.materials_seen.add(e["after"])
                elif kind == "block_break":
                    a.broken[e["before"]] += 1
                    a.materials_seen.add(e["before"])
                elif kind in ("craft", "smelt", "item_pickup"):
                    if e.get("item"):
                        a.materials_seen.add(e["item"])
                        if kind == "craft":
                            a.crafted[e["item"]] += e.get("count", 1)
                elif kind == "mob_kill":
                    a.kills[e["entity"]] += 1
                elif kind == "player_death":
                    a.deaths += 1
                    if pos:
                        a.recent_deaths = ([{"pos": list(pos), "t": e["t"]}]
                                           + a.recent_deaths)[:4]
                elif kind == "damage":
                    a.damage_taken += e.get("amount", 0)
                    a.health_floor = min(a.health_floor, e.get("health", 20))
                elif kind == "advancement":
                    a.advancements.append(e["advancement"])
                elif kind == "vehicle_enter":
                    vehicle = str(e.get("vehicle") or "unknown").replace("minecraft:", "")
                    a.vehicle_entries[vehicle] += 1

                if kind == "player_state":
                    for vehicle, centimetres in (e.get("travel_cm") or {}).items():
                        try:
                            a.travel_cm[vehicle] = max(a.travel_cm[vehicle], int(centimetres))
                        except (TypeError, ValueError):
                            continue

            # Home is simply where this actor spends time. Nothing is assumed about houses.
            if cells:
                a.home_cell = cells.most_common(1)[0][0]
                a.cells_visited |= set(cells)

            latest = next((e for e in reversed(own) if e["type"] == "player_state"), None)
            if latest:
                a.last_pos = tuple(latest["pos"])
                a.last_dim = latest["dim"]
                a.health = latest.get("health")
                a.food = latest.get("food")
                a.level = latest.get("level")
                a.equipment = {k: latest[k] for k in
                               ("held", "helmet", "chestplate", "leggings", "boots")
                               if latest.get(k)}

            # Capability: the best tier this actor has ever been seen holding. Answers
            # "what are they equipped to do", which raw material lists do not.
            for item in a.materials_seen | set(a.equipment.values()):
                tier = _tier_of(item)
                if tier < 0:
                    continue
                name = item.replace("minecraft:", "")
                if any(k in name for k in TOOL_KINDS):
                    if a.best_tool is None or tier > _tier_of(a.best_tool):
                        a.best_tool = item
                if any(k in name for k in ARMOR_KINDS):
                    if a.best_armor is None or tier > _tier_of(a.best_armor):
                        a.best_armor = item

        for ep in episodes:
            self._actor(ep.actor).episodes += 1
            a = self._actor(ep.actor)
            if ep.bbox:
                lo, hi = ep.bbox
                if a.range_bbox is None:
                    a.range_bbox = (lo, hi)
                else:
                    (x0, y0, z0), (x1, y1, z1) = a.range_bbox
                    a.range_bbox = ((min(x0, lo[0]), min(y0, lo[1]), min(z0, lo[2])),
                                    (max(x1, hi[0]), max(y1, hi[1]), max(z1, hi[2])))

        for f in findings:
            if f.detector == "structure_census":
                self._actor(f.actor).built.append({
                    "kind": f.kind, "bbox": f.facts["bbox"], "blocks": f.facts["blocks"],
                    "dims": f.facts["dims"], "dominant": f.facts["dominant"],
                    "tick": f.tick, "t": f.t,
                })

    @staticmethod
    def _works(store: Store, actor: str) -> int:
        """How many deliberate works this actor has to their name."""
        n = 0
        for row in store.all("work_events"):
            if row["actor"] != actor:
                continue
            try:
                if json.loads(row["value"] or "{}").get("significance") == "work":
                    n += 1
            except (TypeError, ValueError):
                continue
        return n

    def persist(self, store: Store) -> int:
        """Writes actors to the store as DERIVED state."""
        for a in self.actors.values():
            store.put("players", {
                "id": a.id,
                "name": None,
                "dim": a.last_dim,
                "last_seen_tick": None,
                "profile": json.dumps({
                    "sessions": a.sessions,
                    "playtime_min": round(a.playtime_ms / 60000, 1),
                    "episodes": a.episodes,
                    "last_pos": a.last_pos, "home_cell": a.home_cell,
                    "cells_visited": len(a.cells_visited),
                    "y_range": [a.y_min, a.y_max],
                    "health": a.health, "food": a.food, "level": a.level,
                    "equipment": a.equipment,
                    "best_tool": a.best_tool, "best_armor": a.best_armor,
                    "placed": sum(a.placed.values()), "broken": sum(a.broken.values()),
                    "build_ratio": round(a.build_ratio, 2),
                    "kills": dict(a.kills), "deaths": a.deaths,
                    "recent_deaths": a.recent_deaths,
                    "damage_taken": round(a.damage_taken, 1),
                    "health_floor": a.health_floor,
                    "advancements": a.advancements,
                    "travel_m": {vehicle: round(cm / 100.0, 1)
                                 for vehicle, cm in a.travel_cm.items() if cm > 0},
                    "vehicle_entries": dict(a.vehicle_entries),
                    "distinct_materials": len(a.materials_seen),
                    # Counted from the store's own works, not from raw census findings.
                    # Counting every finding had the god telling a player that "sixty-four
                    # structures stand to your name" when twenty-four did and the rest were
                    # caving traces and pre-merge duplicates.
                    "structures_built": self._works(store, a.id),
                }),
            }, Belief(provenance=Provenance.DERIVED,
                      first_seen_tick=None, verified_at_tick=None))
        return len(self.actors)

    def persist_work_events(self, store: Store, findings) -> int:
        """Writes historical census findings without claiming they still stand.

        Two problems, both of which showed up as soon as builds were rendered.

        This had only ever been run by hand, so the store silently fell behind the corpus:
        21 of 38 structures had never been written, and the god proposed a build directly on
        top of a roller coaster it had no record of.

        And the census clusters WITHIN an episode and never across them, so a house built
        over three sessions became three structures at overlapping bounding boxes — the same
        building rendered three times, once with its roof and twice without. Anything a
        player returns to is built across episodes, so findings that occupy the same space
        are merged here into one structure. Placements and removals are never merged with
        each other: a build and its later demolition share a footprint and are different
        facts.
        """
        groups = []
        for f in findings:
            if f.detector != "structure_census":
                continue
            lo, hi = f.facts["bbox"].split("..")
            lo = [int(v) for v in lo.strip("()").split(",")]
            hi = [int(v) for v in hi.strip("()").split(",")]
            groups.append({"kind": f.kind, "actor": f.actor, "dim": f.dim,
                           "lo": lo, "hi": hi,
                           "materials": collections.Counter(f.facts["materials"]),
                           "blocks": f.facts["blocks"],
                           "first": f.facts.get("first_tick") or 0,
                           "last": f.facts.get("last_tick") or 0,
                           # Wall clock as well as tick. Ticks restart with the server, so
                           # they cannot say whether a structure predates a schema change.
                           "when": f.t or 0,
                           "vertical": f.facts.get("vertical"),
                           "sources": {f.episode_id}, "sessions": 1})
        merged = self._merge_overlapping(groups, store)

        # A merged structure takes its id from the union's low corner, which may differ from
        # the id a consumed row was stored under. Without this the old row survives beside
        # the merged one and the same build appears twice, differing only in framing.
        for g in merged:
            lo = g["lo"]
            new_id = f"{g['kind']}_{lo[0]}_{lo[1]}_{lo[2]}"
            for old in g.get("absorbed", ()):
                if old != new_id:
                    # The render goes with the row. Leaving it behind put a merged-away
                    # structure in the renders folder with nothing backing it, which is
                    # indistinguishable from a real one to anything reading that folder.
                    from render import forget_renders
                    forget_renders(old)
                    store.db.execute("DELETE FROM work_events WHERE id = ?", (old,))
                    store.db.execute(
                        "UPDATE relationships SET subject = ? WHERE subject = ?",
                        (new_id, old))
        store.db.commit()

        placements = [g for g in merged if g["kind"] == "placement"]
        for g in merged:
            lo, hi = g["lo"], g["hi"]
            dims = [hi[i] - lo[i] + 1 for i in range(3)]
            weight = self.significance(g, placements)
            store.put("work_events", {
                "id": f"{g['kind']}_{lo[0]}_{lo[1]}_{lo[2]}",
                "kind": g["kind"], "actor": g["actor"], "dim": g["dim"],
                "min_x": lo[0], "min_y": lo[1], "min_z": lo[2],
                "max_x": hi[0], "max_y": hi[1], "max_z": hi[2],
                "materials": json.dumps(dict(g["materials"].most_common())),
            }, Belief(provenance=Provenance.DERIVED,
                      first_seen_tick=g["first"], verified_at_tick=g["last"],
                      verified_at_ms=g.get("when") or None,
                      source_event_ids=tuple(sorted(g.get("sources", ()))),
                      value=json.dumps({
                          "blocks": g["blocks"],
                          "dims": f"{dims[0]}x{dims[1]}x{dims[2]}",
                          "bbox": f"({lo[0]},{lo[1]},{lo[2]})..({hi[0]},{hi[1]},{hi[2]})",
                          "fill": round(g["blocks"] / max(1, dims[0] * dims[1] * dims[2]), 3),
                          "vertical": bool(g["vertical"]),
                          "first_tick": g["first"], "last_tick": g["last"],
                          "sessions": g.get("sessions", 1),
                          "when": g.get("when", 0),
                          "significance": weight})))
        return len(merged)

    # Compatibility for external scripts.  The name is intentionally boring: callers made
    # before schema v2 still preserve history, but no longer pollute current world state.
    def persist_structures(self, store: Store, findings) -> int:
        return self.persist_work_events(store, findings)

    @staticmethod
    def _merge_overlapping(groups: list, store: Store) -> list:
        """Unions findings that occupy the same space, including ones already stored.

        Seeded from the store so a session recorded later merges into a structure written
        by an earlier one, rather than starting a second copy beside it.
        """
        seeds = []
        for row in store.all("work_events"):
            kind = row["kind"]
            if kind not in ("placement", "removal") or row["min_x"] is None:
                continue
            try:
                facts = json.loads(row["value"] or "{}")
            except (TypeError, ValueError):
                facts = {}
            try:
                sources = set(json.loads(row["source_event_ids"] or "[]"))
            except (TypeError, ValueError):
                sources = set()
            seeds.append({
                "kind": kind, "actor": row["actor"], "dim": row["dim"],
                "lo": [row["min_x"], row["min_y"], row["min_z"]],
                "hi": [row["max_x"], row["max_y"], row["max_z"]],
                "materials": collections.Counter(json.loads(row["materials"] or "{}")),
                "blocks": facts.get("blocks", 0),
                "first": row["first_seen_tick"] or 0,
                "last": row["verified_at_tick"] or 0,
                "vertical": facts.get("vertical"),
                "when": facts.get("when") or 0,
                "sources": sources, "sessions": facts.get("sessions") or 1,
                "stored": True, "id": row["id"],
            })

        def absorb(left: dict, right: dict) -> None:
            for i in range(3):
                left["lo"][i] = min(left["lo"][i], right["lo"][i])
                left["hi"][i] = max(left["hi"][i], right["hi"][i])
            # Max, not sum: replaying evidence must never manufacture blocks.
            for material, count in right["materials"].items():
                left["materials"][material] = max(left["materials"][material], count)
            left["blocks"] = max(left["blocks"], right["blocks"])
            left["first"] = min(left["first"] or right["first"], right["first"])
            left["last"] = max(left["last"], right["last"])
            left["vertical"] = left["vertical"] or right["vertical"]
            left["when"] = max(left.get("when", 0), right.get("when", 0))
            left.setdefault("sources", set()).update(right.get("sources", ()))
            left["sessions"] = max(left.get("sessions", 1), right.get("sessions", 1),
                                   len(left["sources"]))
            left.setdefault("absorbed", set()).update(right.get("absorbed", ()))
            if right.get("id"):
                left["absorbed"].add(right["id"])

        out = []
        for item in seeds + groups:
            merged = dict(item)
            merged["lo"] = list(item["lo"])
            merged["hi"] = list(item["hi"])
            merged["materials"] = collections.Counter(item["materials"])
            merged["sources"] = set(item.get("sources", ()))
            merged["absorbed"] = set(item.get("absorbed", ()))
            if item.get("id"):
                merged["absorbed"].add(item["id"])

            # Union the whole connected component. A new bridge can make two previously
            # separate boxes one work, so stopping at the first match made consolidation
            # require a second daemon restart.
            changed = True
            while changed:
                changed = False
                for existing in list(out):
                    if (existing["kind"] == merged["kind"]
                            and existing["dim"] == merged["dim"]
                            and World._touching(existing, merged)):
                        out.remove(existing)
                        absorb(merged, existing)
                        changed = True
            out.append(merged)
        return out

    @staticmethod
    def significance(g: dict, placements: list) -> str:
        """Is this a work someone made, or a trace of passing through?

        Caving leaves hundreds of small removal clusters that are not structures at all —
        they are the shape of someone walking through stone. Counting them as works made
        the god tell a player they had "cut fourteen shafts into this land" when they had
        dug one pit and gone spelunking.

        Decided by code, not by a model, from three signals that need no interpretation:
        something built on the spot, a return visit, or a deliberate excavation at the
        surface. Traces are still recorded and still queryable — they are facts, and
        deleting them for being dull is how a store starts lying — but they are summarised
        rather than named.
        """
        if g["kind"] == "placement":
            return "work"
        if g.get("sessions", 1) >= 2:
            return "work"            # they came back to it
        for p in placements:
            if World._touching(g, p):
                return "work"        # they built something here
        surface = g["hi"][1] >= 55
        if surface and g["blocks"] >= 60:
            return "work"            # a deliberate pit or quarry, not a tunnel
        return "trace"

    @staticmethod
    def _touching(a: dict, b: dict) -> bool:
        """Is this the same structure, continued?

        Not mere proximity. Adjacency alone merged a watch-post with the roller coaster
        beside it, and a farm with the cottage next to it, because their boxes touched.

        The test is a shared FOOTPRINT with vertical continuity: a floor laid in one session
        and the walls raised above it in another occupy the same ground and stack, whereas
        two buildings side by side do not, however close they stand.
        """
        overlap = []
        for i in (0, 2):
            lo = max(a["lo"][i], b["lo"][i])
            hi = min(a["hi"][i], b["hi"][i])
            if hi < lo:
                return False
            overlap.append(hi - lo + 1)
        shared = overlap[0] * overlap[1]
        area_a = (a["hi"][0] - a["lo"][0] + 1) * (a["hi"][2] - a["lo"][2] + 1)
        area_b = (b["hi"][0] - b["lo"][0] + 1) * (b["hi"][2] - b["lo"][2] + 1)
        if shared / max(1, min(area_a, area_b)) < 0.5:
            return False
        # Vertically overlapping, or stacked directly on top of one another.
        return a["lo"][1] - 1 <= b["hi"][1] and b["lo"][1] - 1 <= a["hi"][1]



    def render(self) -> str:
        out = []
        for a in sorted(self.actors.values(), key=lambda x: -x.playtime_ms):
            out.append(f"{BOLD}{a.id[:8]}{RESET}  {a.playtime_ms/60000:.0f} min over "
                       f"{a.sessions} session(s), {a.episodes} episodes")
            out.append(f"  now      {a.last_pos} in {a.last_dim}, {a.health}hp "
                       f"food {a.food} level {a.level}")
            out.append(f"  carrying {a.equipment}")
            out.append(f"  capable  best tool {a.best_tool}, best armor {a.best_armor}, "
                       f"{len(a.materials_seen)} distinct materials handled")
            out.append(f"  range    y {a.y_min}..{a.y_max}, {len(a.cells_visited)} region "
                       f"cells, home {a.home_cell}")
            out.append(f"  work     placed {sum(a.placed.values())} broke "
                       f"{sum(a.broken.values())} (build ratio {a.build_ratio:.2f}), "
                       f"{len(a.built)} structures")
            out.append(f"  risk     {a.deaths} deaths, {a.damage_taken:.0f} damage taken, "
                       f"floor {a.health_floor}hp, kills {dict(a.kills)}")
            out.append(f"  earned   {len(a.advancements)} advancements")
        return "\n".join(out)


def main(argv: list[str]) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("sessions", nargs="+", type=Path)
    p.add_argument("--store", action="store_true", help="persist to the belief store")
    args = p.parse_args(argv)

    world = World()
    # One session at a time, in recording order, so state accumulates the way it would live.
    for path in sorted(args.sessions):
        episodes, findings, events = analyse([path])
        world.observe(events, episodes, findings)

    print(world.render())
    if args.store:
        store = Store()
        n = world.persist(store)
        m = 0
        for path in sorted(args.sessions):
            _, findings, _ = analyse([path])
            m += world.persist_work_events(store, findings)
        store.close()
        print(f"\n{DIM}persisted {n} actor(s) and {m} work events as DERIVED{RESET}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
