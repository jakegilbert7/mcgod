#!/usr/bin/env python3
"""L1: group an event stream into episodes.

    python3 episodes.py sessions/corpus/03-hole-idle.jsonl

An episode is a stretch of one actor's activity with no significant break in it. Episodes are
NOT fixed-length time chunks — a four-minute dig is one episode and a twenty-minute build is
one episode. Boundaries fall where something deterministically changed:

  * a gap of ~90s with nothing substantive happening
  * activity resuming far from where it stopped (walking to another site is a real boundary
    even with no pause)
  * a dimension change, always

Segmentation deliberately does NOT depend on classifying what the player was doing. That
would make model output define the boundaries that later become facts in the store, which is
the failure non-negotiable #5 exists to prevent, and one bad classification would poison
every boundary after it. Change-point detection on observable facts has no such feedback.

Episodes emit facts only, never labels. "gross_placed 0, gross_broken 331, 5 blocks deep" is
an episode; calling it a hole is the model's job, done later and cached.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

# The 2s position sample and the 10s context snapshot are heartbeats: they fire whether or not
# the player is doing anything, so counting them would mean a session never contains a gap.
# region_enter is derived from the same sampler and is excluded for the same reason.
HEARTBEAT = {"move", "player_state", "region_enter"}

GAP_TICKS = 1800     # 90s at 20 tps
JUMP_BLOCKS = 24     # block work resuming this far away starts a new episode
# 24 was chosen by sweeping the corpus against remembered sessions, not picked a priori.
# 48 and 40 left 01-house as a single episode spanning y=29 to y=72, silently merging two
# cave trips into the house build. 16 shattered the same session into ten. At 24 the six
# episodes line up with what the player said they did, and 04-farm splits cleanly into
# gathering and building.

# Events that mark a place the player was *working*, as opposed to merely standing. Only
# these can trigger a spatial boundary: taking fall damage 60 blocks along a walk is not a
# change of activity, it is a thing that happened during one.
BLOCK_WORK = {"block_place", "block_break"}

# An episode has to clear one of these to stand on its own; otherwise it is folded into the
# episode before it. Without a floor, one stray event mid-walk shatters a single coherent
# stretch of wandering into several fragments joined by nothing.
MIN_BLOCK_WORK = 5
MIN_DURATION_TICKS = 600   # 30s


@dataclass
class Episode:
    actor: str
    dim: str
    tick_start: int
    tick_end: int
    t_start: int
    t_end: int
    placed: collections.Counter = field(default_factory=collections.Counter)
    broken: collections.Counter = field(default_factory=collections.Counter)
    tools: collections.Counter = field(default_factory=collections.Counter)
    kinds: collections.Counter = field(default_factory=collections.Counter)
    # (x, y, z, material, tick) for each block put down and taken out. Kept apart because a
    # structure is made of what was placed and an excavation of what was removed, and one
    # clustering primitive over either gives both without a second detector. The tick is
    # carried per block, not per episode: a build and its demolition inside one episode
    # share every episode-level timestamp, and only the block times separate them.
    placements: list = field(default_factory=list)
    removals: list = field(default_factory=list)
    path: float = 0.0

    @property
    def positions(self) -> list:
        return [p[:3] for p in self.placements] + [p[:3] for p in self.removals]

    @property
    def id(self) -> str:
        """Stable across re-runs: same input bytes give the same id."""
        seed = f"{self.actor}:{self.dim}:{self.tick_start}:{self.tick_end}"
        return hashlib.sha1(seed.encode()).hexdigest()[:12]

    @property
    def duration_s(self) -> float:
        return (self.tick_end - self.tick_start) / 20.0

    @property
    def gross_placed(self) -> int:
        return sum(self.placed.values())

    @property
    def gross_broken(self) -> int:
        return sum(self.broken.values())

    @property
    def net(self) -> collections.Counter:
        n = collections.Counter(self.placed)
        n.subtract(self.broken)
        return collections.Counter({k: v for k, v in n.items() if v})

    @property
    def net_total(self) -> int:
        return self.gross_placed - self.gross_broken

    @property
    def bbox(self):
        if not self.positions:
            return None
        xs, ys, zs = zip(*self.positions)
        return (min(xs), min(ys), min(zs)), (max(xs), max(ys), max(zs))

    @property
    def substantive(self) -> int:
        return sum(v for k, v in self.kinds.items() if k not in HEARTBEAT)


def _dist(a, b) -> float:
    """Chebyshev distance in three dimensions.

    Y matters as much as X and Z here. A player mining forty blocks below their house is
    somewhere else entirely, even though the two sites sit on top of each other on a map.
    """
    return max(abs(a[0] - b[0]), abs(a[1] - b[1]), abs(a[2] - b[2]))


def segment(events: list[dict], gap_ticks: int = GAP_TICKS,
            jump: int = JUMP_BLOCKS) -> list[Episode]:
    """Splits a session into episodes covering every event, in order."""
    if not events:
        return []

    # 1. Cluster the substantive events. These anchor the episodes.
    spans: list[list[dict]] = []
    current: list[dict] = []
    last_work = None  # position of the most recent block work, whatever came after it
    for e in events:
        if e["type"] in HEARTBEAT:
            continue
        if current:
            previous = current[-1]
            # The work site is tracked separately from the event stream. Crafting and
            # smelting between mining and building keep the stream continuous, but the
            # player still moved from a cave to a hilltop, and that is a real boundary.
            moved = (e["type"] in BLOCK_WORK and last_work is not None
                     and e.get("pos") and _dist(e["pos"], last_work) > jump)
            broke = (
                e["tick"] - previous["tick"] > gap_ticks
                or e.get("dim") != previous.get("dim")
                or moved
            )
            if broke:
                spans.append(current)
                current = []
                last_work = None
        current.append(e)
        if e["type"] in BLOCK_WORK and e.get("pos"):
            last_work = tuple(e["pos"])
    if current:
        spans.append(current)

    # 2. Turn each cluster into a window, then fill the holes between them so that every
    #    moment of the session belongs to some episode. A stretch with no substantive events
    #    is still an episode — it is how "they wandered for four minutes" gets recorded
    #    instead of vanishing.
    first, last = events[0], events[-1]
    windows: list[tuple[int, int]] = []
    cursor = first["tick"]
    for span in spans:
        if span[0]["tick"] > cursor:
            windows.append((cursor, span[0]["tick"] - 1))
        windows.append((span[0]["tick"], span[-1]["tick"]))
        cursor = span[-1]["tick"] + 1
    if cursor <= last["tick"]:
        windows.append((cursor, last["tick"]))
    if not windows:
        windows = [(first["tick"], last["tick"])]

    # 3. Fold every event, heartbeats included, into its window.
    episodes = []
    for start, end in windows:
        inside = [e for e in events if start <= e["tick"] <= end]
        if not inside:
            continue
        actors = collections.Counter(e.get("actor") for e in inside if e.get("actor") != "world")
        ep = Episode(
            actor=actors.most_common(1)[0][0] if actors else "world",
            dim=next((e["dim"] for e in inside if e.get("dim")), "?"),
            tick_start=start, tick_end=end,
            t_start=inside[0]["t"], t_end=inside[-1]["t"],
        )
        previous_pos = None
        for e in inside:
            ep.kinds[e["type"]] += 1
            if e["type"] == "block_place":
                ep.placed[e["after"]] += 1
            elif e["type"] == "block_break":
                ep.broken[e["before"]] += 1
            if e.get("tool"):
                ep.tools[e["tool"]] += 1
            pos = e.get("pos")
            if pos:
                # Only block work defines the worked area; walking past does not.
                if e["type"] == "block_place":
                    ep.placements.append((pos[0], pos[1], pos[2], e["after"], e["tick"]))
                elif e["type"] == "block_break":
                    ep.removals.append((pos[0], pos[1], pos[2], e["before"], e["tick"]))
                if e["type"] == "move":
                    if previous_pos:
                        ep.path += _dist(pos, previous_pos)
                    previous_pos = tuple(pos)
        episodes.append(ep)
    return _absorb_fragments(episodes)


def _absorb_fragments(episodes: list[Episode]) -> list[Episode]:
    """Folds insignificant episodes into the one before them.

    A boundary should mean the player's activity actually changed. A lone damage event
    partway through a long walk is not that, and left alone it splits one coherent stretch
    into three.

    Two tests, and an episode must fail both to be folded in. It survives if it contains real
    block work, or if it is long enough to be a phase rather than an instant. But two
    ADJACENT stretches that both lack block work are one stretch however long they run —
    nothing structural happened to separate them, and reporting "wandered, then wandered"
    is worse than useless.
    """
    if not episodes:
        return []
    kept = [episodes[0]]
    for ep in episodes[1:]:
        previous = kept[-1]
        work = ep.gross_placed + ep.gross_broken
        previous_work = previous.gross_placed + previous.gross_broken
        long_enough = ep.tick_end - ep.tick_start >= MIN_DURATION_TICKS
        stands_alone = work >= MIN_BLOCK_WORK or (
            long_enough and previous_work >= MIN_BLOCK_WORK)
        if stands_alone or ep.dim != previous.dim:
            kept.append(ep)
            continue
        previous.tick_end = ep.tick_end
        previous.t_end = ep.t_end
        previous.placed.update(ep.placed)
        previous.broken.update(ep.broken)
        previous.tools.update(ep.tools)
        previous.kinds.update(ep.kinds)
        previous.placements.extend(ep.placements)
        previous.removals.extend(ep.removals)
        previous.path += ep.path
    return kept


def render(ep: Episode, index: int) -> str:
    bbox = ep.bbox
    if bbox:
        (x0, y0, z0), (x1, y1, z1) = bbox
        shape = (f"{x1 - x0 + 1}x{y1 - y0 + 1}x{z1 - z0 + 1} at "
                 f"({x0},{y0},{z0})..({x1},{y1},{z1})")
    else:
        shape = "no block work"
    lines = [
        f"  episode {index}  [{ep.id}]  {ep.duration_s / 60:5.1f} min  "
        f"ticks {ep.tick_start}..{ep.tick_end}",
        f"    placed {ep.gross_placed:4d}   broken {ep.gross_broken:4d}   "
        f"net {ep.net_total:+5d}   travelled {ep.path:.0f} blocks",
        f"    area   {shape}",
    ]
    if ep.net:
        top = ", ".join(f"{v:+d} {k.replace('minecraft:', '')}"
                        for k, v in ep.net.most_common(4))
        lines.append(f"    net by material: {top}")
    busy = ", ".join(f"{k}x{v}" for k, v in ep.kinds.most_common(6)
                     if k not in HEARTBEAT)
    lines.append(f"    events: {busy or '(movement only)'}")
    return "\n".join(lines)


def main(argv: list[str]) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("session", type=Path)
    p.add_argument("--gap", type=int, default=GAP_TICKS, help=f"gap in ticks (default {GAP_TICKS})")
    p.add_argument("--jump", type=int, default=JUMP_BLOCKS,
                   help=f"spatial jump in blocks (default {JUMP_BLOCKS})")
    args = p.parse_args(argv)

    events = [json.loads(l) for l in args.session.open() if l.strip()]
    eps = segment(events, args.gap, args.jump)
    print(f"{args.session}: {len(events)} events -> {len(eps)} episodes "
          f"(gap {args.gap} ticks, jump {args.jump} blocks)")
    for i, ep in enumerate(eps, 1):
        print(render(ep, i))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
