#!/usr/bin/env python3
"""L2: primitives that turn episodes into facts.

    python3 detectors.py sessions/corpus/*.jsonl

Detectors emit FACTS and never labels. "18 x 6 x 15, 63% cobblestone, 214 blocks, hollow" is
a fact. "A house" is a label, and labelling is the model's job, done lazily and cached, so
that a wrong guess never becomes a row in the store.

There are three here. The rest of the ~20 are mechanical once the interface is fixed: a
detector takes episodes plus whatever history it needs and returns Findings.
"""

from __future__ import annotations

import argparse
import collections
import json
import statistics
import sys
from dataclasses import dataclass, field
from pathlib import Path

from episodes import Episode, segment
from store import Provenance

# --- clustering ----------------------------------------------------------------------

CLUSTER_RADIUS = 6      # blocks; placements this close are the same structure
MIN_CLUSTER = 8         # a structure has to be at least this many blocks


def cluster(points: list[tuple[int, int, int]],
            radius: int = CLUSTER_RADIUS) -> list[list[tuple[int, int, int]]]:
    """Single-linkage clustering with a Chebyshev radius.

    Structure census must cluster rather than take a raw min/max bounding box. Measured on
    03-hole-idle: one stray block 170 blocks from the dig site inflated a naive bbox
    twentyfold, to a volume that was 0% filled. Outliers become their own small clusters and
    get dropped by the size floor instead of poisoning the real one.
    """
    remaining = set(points)
    clusters = []
    while remaining:
        seed = remaining.pop()
        group = [seed]
        frontier = [seed]
        while frontier:
            cx, cy, cz = frontier.pop()
            near = [p for p in remaining
                    if abs(p[0] - cx) <= radius
                    and abs(p[1] - cy) <= radius
                    and abs(p[2] - cz) <= radius]
            for p in near:
                remaining.discard(p)
                group.append(p)
                frontier.append(p)
        clusters.append(group)
    return sorted(clusters, key=len, reverse=True)


# --- findings ------------------------------------------------------------------------

@dataclass(frozen=True)
class Finding:
    detector: str
    kind: str
    actor: str
    dim: str
    episode_id: str
    tick: int
    facts: dict
    #: Wall-clock milliseconds. `tick` is server uptime and resets to zero every time the
    #: server restarts, so it cannot order findings across sessions — sorting a five-session
    #: corpus by tick interleaves them into nonsense. This is the global ordering.
    t: int = 0
    confidence: float = 1.0
    provenance: Provenance = Provenance.DERIVED

    def render(self) -> str:
        body = "  ".join(f"{k}={v}" for k, v in self.facts.items())
        return (f"  [{self.provenance.name}] {self.kind:16} tick {self.tick:<7} "
                f"ep {self.episode_id[:8]}  {body}")


class Detector:
    """Uniform interface. A detector sees episodes in order and returns findings."""

    name = "detector"

    def run(self, episodes: list[Episode]) -> list[Finding]:
        raise NotImplementedError


# --- 1. structure census -------------------------------------------------------------

class StructureCensus(Detector):
    """Describes what was built, without saying what it is."""

    name = "structure_census"

    def run(self, episodes: list[Episode]) -> list[Finding]:
        findings = []
        for ep in episodes:
            # One clustering primitive, run over what was put down and over what was taken
            # out. Two names, both purely mechanical: "placement" and "removal" describe the
            # operation and nothing else. Calling the removal side "excavation" was already
            # a label in disguise — it reads as digging, and it was being applied to a player
            # chopping down trees. The facts (dims, fill, materials) let the model decide
            # whether a removal is a mine, a quarry or a cleared forest.
            for kind, blocks in (("placement", ep.placements), ("removal", ep.removals)):
                if len(blocks) < MIN_CLUSTER:
                    continue
                by_pos = {(x, y, z): m for x, y, z, m, _ in blocks}
                ticks = {(x, y, z): t for x, y, z, _, t in blocks}
                for group in cluster(list(by_pos)):
                    if len(group) < MIN_CLUSTER:
                        continue
                    when = [ticks[p] for p in group]
                    xs, ys, zs = zip(*group)
                    w = max(xs) - min(xs) + 1
                    h = max(ys) - min(ys) + 1
                    d = max(zs) - min(zs) + 1
                    here = collections.Counter(by_pos[p] for p in group)
                    dominant, dominant_n = here.most_common(1)[0]
                    findings.append(Finding(
                        detector=self.name, kind=kind, actor=ep.actor, dim=ep.dim,
                        episode_id=ep.id, tick=ep.tick_end, t=ep.t_end,
                        facts={
                            "blocks": len(group),
                            "dims": f"{w}x{h}x{d}",
                            "bbox": f"({min(xs)},{min(ys)},{min(zs)}).."
                                    f"({max(xs)},{max(ys)},{max(zs)})",
                            # Low fill in a large box means walls and a roof rather than
                            # a solid mass; high fill in a flat box means a floor or field.
                            "fill": round(len(group) / (w * h * d), 3),
                            "dominant": dominant.replace("minecraft:", ""),
                            "dominant_share": round(dominant_n / len(group), 2),
                            "distinct_materials": len(here),
                            "materials": {k.replace("minecraft:", ""): v
                                          for k, v in here.most_common(6)},
                            "vertical": h >= 3,
                            # Per-block times. A build and its demolition inside one
                            # episode are indistinguishable without these.
                            "first_tick": min(when),
                            "last_tick": max(when),
                        },
                    ))
        return findings


# --- 2. rate anomaly -----------------------------------------------------------------

class RateAnomaly(Detector):
    """Flags activity far faster than this actor's own recent normal.

    Deliberately relative, never absolute. A player who always builds fast is not remarkable
    for building fast, and a farm assembled casually across three real days should not trip
    this — that one is the structure census's job. Bursts are what this catches.
    """

    name = "rate_anomaly"

    #: Same-kind episodes of prior history required before any judgement is made at all.
    MIN_HISTORY = 3

    #: Spread multiplier. The threshold is median + DEVIATIONS x MAD, not a multiple of the
    #: median, because these distributions are bimodal and a median multiple is meaningless
    #: on them. Measured over 54 episodes of real play: placement rates split into episodes
    #: that build (20-40/min) and episodes that incidentally set a torch while caving
    #: (1-3/min). The median lands at 3.5 in the incidental cluster, so "twice the median"
    #: is 7/min and fires on 44% of everything — a detector that fires on half of all play
    #: reports nothing. Scaling by spread instead brings that to 11%.
    DEVIATIONS = 5.0

    #: Floor on the median multiple, for the case where MAD collapses because early history
    #: happens to be uniform.
    MIN_RATIO = 2.0

    #: Absolute floor. A four-second episode that places three blocks is 45/min and means
    #: nothing; anomalies need enough work behind them to be worth waking anyone for.
    MIN_BLOCKS = 30

    def run(self, episodes: list[Episode]) -> list[Finding]:
        findings = []
        # Placing and breaking are separate activities with different natural speeds, and
        # comparing a mining rate against a baseline that includes building episodes
        # compares nothing to anything. Episodes with none of a given kind of work
        # contribute no sample to that baseline rather than a zero, which would otherwise
        # drag every median toward the floor and make all work look anomalous.
        history: dict[tuple[str, str], list[float]] = collections.defaultdict(list)
        for ep in episodes:
            minutes = max(ep.duration_s / 60.0, 1 / 60.0)
            for kind, gross in (("placed", ep.gross_placed), ("broken", ep.gross_broken)):
                if not gross:
                    continue
                rate = gross / minutes
                past = history[(ep.actor, kind)]
                if len(past) >= self.MIN_HISTORY and gross >= self.MIN_BLOCKS:
                    baseline = statistics.median(past)
                    spread = statistics.median([abs(r - baseline) for r in past])
                    threshold = max(baseline + self.DEVIATIONS * spread,
                                    baseline * self.MIN_RATIO)
                    if baseline > 0 and rate >= threshold:
                        findings.append(Finding(
                            detector=self.name, kind="rate_anomaly", actor=ep.actor,
                            dim=ep.dim, episode_id=ep.id, tick=ep.tick_end,
                            facts={
                                "work": kind,
                                "rate_per_min": round(rate, 1),
                                "baseline_per_min": round(baseline, 1),
                                "threshold_per_min": round(threshold, 1),
                                "ratio": round(rate / baseline, 2),
                                "blocks": gross,
                                "history_episodes": len(past),
                            },
                        ))
                history[(ep.actor, kind)].append(rate)
        return findings


# --- 3. firsts -----------------------------------------------------------------------

class Firsts(Detector):
    """The first time an actor ever does or touches something.

    Minecraft's own advancements cover some of this; these cover the rest, and unlike
    advancements they extend to anything the schema records.
    """

    name = "firsts"

    def __init__(self, raw_events: list[dict]) -> None:
        self.events = raw_events

    def run(self, episodes: list[Episode]) -> list[Finding]:
        spans = [(ep.tick_start, ep.tick_end, ep.id) for ep in episodes]

        def episode_for(tick: int) -> str:
            for start, end, eid in spans:
                if start <= tick <= end:
                    return eid
            return ""

        findings = []
        seen: set[tuple[str, str, str]] = set()
        for e in self.events:
            actor = e.get("actor")
            if not actor or actor == "world":
                continue
            for field_name, kind in (("after", "first_placed"),
                                     ("before", "first_broken"),
                                     ("item", "first_item"),
                                     ("entity", "first_entity"),
                                     ("advancement", "first_advancement")):
                value = e.get(field_name)
                if not value or value == "minecraft:air":
                    continue
                key = (actor, kind, value)
                if key in seen:
                    continue
                seen.add(key)
                findings.append(Finding(
                    detector=self.name, kind=kind, actor=actor,
                    dim=e.get("dim", "?"), episode_id=episode_for(e["tick"]),
                    tick=e["tick"], t=e.get("t", 0),
                    facts={"value": value.replace("minecraft:", ""), "via": e["type"]},
                ))
        return findings


# --- 4. context profile ------------------------------------------------------------

class ContextProfile(Detector):
    """Where and when an episode happened.

    The game emits no event when night falls, when it starts raining, or when a player walks
    into a swamp — those are state, sampled by `player_state`, and until now nothing read
    them. "Dug for five minutes" and "dug for five minutes at 3am in a thunderstorm in
    pitch dark" are different stories, and only this tells them apart.
    """

    name = "context_profile"

    #: 13000-23000 of the 24000-tick day is night.
    NIGHT = (13000, 23000)

    def __init__(self, raw_events: list[dict]) -> None:
        self.states = [e for e in raw_events if e["type"] == "player_state"]

    def run(self, episodes: list[Episode]) -> list[Finding]:
        findings = []
        for ep in episodes:
            inside = [s for s in self.states
                      if ep.tick_start <= s["tick"] <= ep.tick_end
                      and s.get("actor") == ep.actor]
            if not inside:
                continue
            biomes = collections.Counter(
                s["biome"].replace("minecraft:", "") for s in inside if s.get("biome"))
            weather = collections.Counter(s.get("weather") for s in inside if s.get("weather"))
            night = sum(1 for s in inside
                        if self.NIGHT[0] <= s.get("time", 0) <= self.NIGHT[1])
            lights = [s["light"] for s in inside if s.get("light") is not None]
            skies = [s["sky_light"] for s in inside if s.get("sky_light") is not None]
            ys = [s["pos"][1] for s in inside if s.get("pos")]
            findings.append(Finding(
                detector=self.name, kind="context", actor=ep.actor, dim=ep.dim,
                episode_id=ep.id, tick=ep.tick_end, t=ep.t_end,
                facts={
                    "samples": len(inside),
                    "biomes": dict(biomes.most_common(3)),
                    "weather": dict(weather.most_common(2)),
                    "night_fraction": round(night / len(inside), 2),
                    "median_light": statistics.median(lights) if lights else None,
                    # Sky light, not block light. Block light counts the torches the player
                    # placed themselves, so a well-lit mineshaft at y=-39 read as "not
                    # underground" — a contradiction the god spotted in its own context and
                    # correctly refused to resolve. Sky light is 0 under any roof whatever
                    # you have lit, which is the thing actually being asked.
                    "median_sky_light": statistics.median(skies) if skies else None,
                    "underground": bool(skies) and statistics.median(skies) == 0,
                    "y_range": f"{min(ys)}..{max(ys)}" if ys else None,
                },
            ))
        return findings


# --- 5. risk profile ---------------------------------------------------------------

class RiskProfile(Detector):
    """What an episode cost the player.

    Health is reported as what each hit *left* them on, so the floor is how close they came
    to dying — not how much damage they absorbed. A player who took 40 damage across a
    session and never dropped below 15 hearts was never in danger; one who took 6 and hit
    half a heart nearly died.
    """

    name = "risk_profile"

    def __init__(self, raw_events: list[dict]) -> None:
        self.events = raw_events

    def run(self, episodes: list[Episode]) -> list[Finding]:
        findings = []
        for ep in episodes:
            inside = [e for e in self.events
                      if ep.tick_start <= e["tick"] <= ep.tick_end
                      and e.get("actor") == ep.actor]
            hits = [e for e in inside if e["type"] == "damage"]
            deaths = [e for e in inside if e["type"] == "player_death"]
            if not hits and not deaths:
                continue
            causes = collections.Counter()
            for h in hits:
                label = h.get("cause", "?")
                if h.get("source"):
                    label += f"({h['source'].replace('minecraft:', '')})"
                causes[label] += 1
            floor = min((h.get("health", 20) for h in hits), default=20)
            findings.append(Finding(
                detector=self.name, kind="risk", actor=ep.actor, dim=ep.dim,
                episode_id=ep.id, tick=ep.tick_end, t=ep.t_end,
                facts={
                    "hits": len(hits),
                    "total_damage": round(sum(h.get("amount", 0) for h in hits), 1),
                    "causes": dict(causes.most_common(4)),
                    "health_floor": floor,
                    "near_death": floor <= 6,
                    "deaths": len(deaths),
                    "ate": sum(1 for e in inside if e["type"] == "eat"),
                    # Where it happened, so risk can be read as "this route is dangerous"
                    # rather than as a lifetime tally the god reaches for every time.
                    "where": [d.get("pos") for d in deaths if d.get("pos")][:3],
                },
            ))
        return findings


# --- driver --------------------------------------------------------------------------

def analyse(paths: list[Path]) -> tuple[list[Episode], list[Finding], list[dict]]:
    """Runs every detector over sessions in order, as one continuous history."""
    episodes: list[Episode] = []
    events: list[dict] = []
    for path in paths:
        session = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines()
                   if l.strip()]
        events.extend(session)
        episodes.extend(segment(session))

    findings: list[Finding] = []
    for detector in (StructureCensus(), RateAnomaly(), Firsts(events),
                     ContextProfile(events), RiskProfile(events)):
        findings.extend(detector.run(episodes))
    return episodes, findings, events


def main(argv: list[str]) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("sessions", nargs="+", type=Path)
    p.add_argument("--only", help="run one detector by name")
    args = p.parse_args(argv)

    episodes, findings, _ = analyse(args.sessions)
    if args.only:
        findings = [f for f in findings if f.detector == args.only]

    print(f"{len(args.sessions)} sessions -> {len(episodes)} episodes -> "
          f"{len(findings)} findings")
    by_detector = collections.defaultdict(list)
    for f in findings:
        by_detector[f.detector].append(f)
    for name, group in by_detector.items():
        print(f"\n{name}  ({len(group)})")
        for f in group[:40]:
            print(f.render())
        if len(group) > 40:
            print(f"  ... {len(group) - 40} more")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
