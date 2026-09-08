#!/usr/bin/env python3
"""L6: the trigger router.

    python3 router.py sessions/corpus/*.jsonl

Findings arrive constantly. Almost none of them are worth speaking about, and a god that
comments on every felled tree is worse than one that says nothing. The router decides which
of three things happens to each finding:

    interrupt  wake the model now and speak. Rare, hard-capped per hour.
    queued     worth saying at the next natural pause, not now.
    ambient    recorded, retrievable, never volunteered.

Three mechanisms keep it quiet. Salience scores each finding on its own merits. Novelty decay
divides that by how many similar things happened recently, so the sixth tree felling scores a
fraction of the first. And a hard budget caps interrupts per hour regardless of how exciting
the world gets — without it a good day underground would produce a monologue.

Debouncing is inherited rather than implemented: findings are attached to episodes, and an
episode only closes after the activity that defines it has stopped. Nothing here can fire
mid-build because nothing is scored until the build ends.
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
from dataclasses import dataclass
from pathlib import Path

from consumer import BOLD, DIM, GREEN, RED, RESET, YELLOW
from detectors import Finding, analyse

TICKS_PER_HOUR = 72000
HOUR_MS = 3_600_000

#: Salience needed to interrupt, and to queue. Everything else is ambient.
INTERRUPT_AT = 0.80
QUEUE_AT = 0.45

#: Hard ceiling on interrupts per hour of play, whatever the world does.
INTERRUPTS_PER_HOUR = 2

#: How long a finding counts as "recent" when deciding how novel the next one is, in ms.
NOVELTY_WINDOW = 1_800_000


@dataclass
class Trigger:
    lane: str
    salience: float
    reason: str
    finding: Finding

    def render(self) -> str:
        colour = {"interrupt": RED, "queued": YELLOW, "ambient": DIM}[self.lane]
        return (f"  {colour}{self.lane:9}{RESET} {self.salience:4.2f}  "
                f"tick {self.finding.tick:<7} {self.finding.kind:14} {self.reason}")


def _advancement_times(events: list[dict]) -> list[int]:
    return [e["t"] for e in events if e["type"] == "advancement"]


#: How close two things must be to count as part of the same moment.
MOMENT_MS = 20_000

#: Nothing interrupts during this much of the start of a stream.
#:
#: "First" means first in the recorded history, not first ever, so the opening seconds of any
#: recording are wall-to-wall firsts. On the corpus this made `adventure/kill_a_mob` score
#: five simultaneous firsts and interrupt fifteen seconds in. The real fix is firsts backed
#: by the store rather than the current stream — P10 territory — and until then the honest
#: position is that novelty cannot be told from ignorance this early.
WARMUP_MS = 120_000


class Router:
    """Scores findings and assigns each a lane."""

    def __init__(self, events: list[dict]) -> None:
        self.advancements = _advancement_times(events)
        self.start_t = events[0]["t"] if events else 0
        self._firsts: list[int] = []

    def _moment_size(self, when: int) -> int:
        """How many other first-time things happened at the same moment.

        Advancements are Minecraft's own milestone list, but they are not equally weighted:
        `story/obtain_armor` and `story/mine_diamond` are not the same event, and scoring
        every advancement flat meant a diamond vein could never outrank a helmet. Rather
        than hardcode which advancements matter — the edge-case spiral this design exists to
        avoid — weight by how much else was new at that instant. Striking diamond brought
        three simultaneous firsts with it; putting on a helmet brought one.
        """
        return sum(1 for t in self._firsts if abs(t - when) < MOMENT_MS)

    def _base_salience(self, f: Finding) -> tuple[float, str, str]:
        """What this finding is worth before novelty and budget are applied.

        Deliberately computed from the facts rather than from a table of interesting things.
        A hardcoded list of important blocks is the edge-case spiral this architecture exists
        to avoid; "the player nearly died" and "the game itself marked this a milestone" are
        general signals that work for anything.

        Returns the score, a human reason, and a novelty key. The key decides which earlier
        findings count as "the same thing happening again", and it has to come from the
        scorer: a flat key per detector made every risk finding identical, so a scrape early
        in a session decayed a death later in it to a fraction of its worth.
        """
        facts = f.facts
        if f.detector == "risk_profile":
            if facts.get("deaths"):
                return 1.0, f"died ({', '.join(list(facts.get('causes', {}))[:2])})", "death"
            if facts.get("near_death"):
                return 0.75, f"survived on {facts.get('health_floor')} hp", "near_death"
            return 0.15, f"took {facts.get('total_damage')} damage", "damage"

        if f.detector == "firsts":
            # Minecraft's own advancement system is the game's judgement of what counts as
            # a milestone. Borrowing it beats inventing a list of blocks we think matter.
            near = any(abs(a - f.t) < MOMENT_MS for a in self.advancements)
            if f.kind == "first_advancement":
                size = self._moment_size(f.t)
                score = min(0.98, 0.78 + 0.06 * size)
                return (score,
                        f"advancement: {facts['value']}"
                        + (f" (+{size} firsts at once)" if size else ""),
                        f"adv:{facts['value']}")
            if near:
                return (0.70, f"first {facts['value']} (alongside an advancement)",
                        f"first_near_adv")
            # A first reached late took longer to reach, so it is likelier to be rare.
            depth = min(1.0, (f.t - self.start_t) / HOUR_MS)
            return 0.20 + 0.25 * depth, f"first {facts['value']}", "first"

        if f.detector == "structure_census":
            blocks = facts.get("blocks", 0)
            return (min(0.65, 0.10 + blocks / 500.0),
                    f"{f.kind} {facts.get('dims')} of {facts.get('dominant')}",
                    f"built:{f.kind}:{facts.get('dominant')}")

        if f.detector == "rate_anomaly":
            # Settled empirically in P6: focused excavation really is anomalous for this
            # player, and it still is not worth interrupting them over.
            return (0.35, f"{facts['work']} {facts['rate_per_min']}/min "
                          f"vs {facts['baseline_per_min']} baseline",
                    f"rate:{facts['work']}")

        return 0.05, f.kind, f.detector

    def route(self, findings: list[Finding]) -> list[Trigger]:
        seen: dict[str, list[int]] = collections.defaultdict(list)
        interrupts: list[int] = []
        triggers: list[Trigger] = []

        # Moment size is measured over firsts, so they must all be known before scoring.
        self._firsts = [f.t for f in findings
                        if f.detector == "firsts" and f.kind != "first_advancement"]

        # Ordered by wall clock, never by tick: tick restarts at zero every server launch.
        for f in sorted(findings, key=lambda x: x.t):
            base, reason, key = self._base_salience(f)

            # Novelty decay. The sixth tree felling is not news.
            recent = [t for t in seen[key] if f.t - t < NOVELTY_WINDOW]
            seen[key] = recent + [f.t]
            salience = base / (1.0 + len(recent))
            if recent:
                reason += f" (x{len(recent) + 1} recently)"

            # The bar rises with each interrupt already spent this hour. A flat threshold
            # is spent greedily by whatever clears it first: on the baseline session two
            # minor advancements consumed the budget before the diamond vein arrived. A
            # rising bar cannot see the future either, but it makes each further
            # interruption cost more, which is the behaviour we actually want.
            # Both sides of this comparison are wall-clock milliseconds. Mixing a tick in
            # here silently disables the whole budget: every stored interrupt looks an
            # eternity old, `live` is always empty, and the cap never binds.
            live = [t for t in interrupts if f.t - t < HOUR_MS]
            interrupts = live
            bar = INTERRUPT_AT + (1.0 - INTERRUPT_AT) * (len(live) / INTERRUPTS_PER_HOUR)
            warming = (f.t - self.start_t) < WARMUP_MS

            if salience >= bar and len(live) < INTERRUPTS_PER_HOUR and not warming:
                interrupts.append(f.t)
                lane = "interrupt"
            elif salience >= INTERRUPT_AT:
                lane = "queued"
                reason += (" [warming up, novelty not yet distinguishable from ignorance]"
                           if warming else
                           f" [needed {bar:.2f}, {len(live)} interrupt(s) already spent]"
                           if len(live) < INTERRUPTS_PER_HOUR else " [budget spent]")
            elif salience >= QUEUE_AT:
                lane = "queued"
            else:
                lane = "ambient"

            triggers.append(Trigger(lane, round(salience, 3), reason, f))
        return triggers


def main(argv: list[str]) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("sessions", nargs="+", type=Path)
    p.add_argument("--lane", help="show only one lane")
    args = p.parse_args(argv)

    episodes, findings, events = analyse(args.sessions)
    triggers = Router(events).route(findings)

    counts = collections.Counter(t.lane for t in triggers)
    span_h = (events[-1]["t"] - events[0]["t"]) / HOUR_MS
    print(f"{BOLD}router{RESET} {len(findings)} findings over {span_h:.1f}h of play")
    for lane in ("interrupt", "queued", "ambient"):
        n = counts[lane]
        print(f"  {lane:10} {n:5}  ({n / max(1, len(triggers)) * 100:4.1f}%)"
              + (f"   {n / max(span_h, 0.01):.1f}/hour" if lane != "ambient" else ""))

    for lane in (("interrupt", "queued") if not args.lane else (args.lane,)):
        shown = [t for t in triggers if t.lane == lane]
        if not shown:
            continue
        print(f"\n{BOLD}{lane}{RESET}")
        for t in shown[:25]:
            print(t.render())
        if len(shown) > 25:
            print(f"  {DIM}... {len(shown) - 25} more{RESET}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
