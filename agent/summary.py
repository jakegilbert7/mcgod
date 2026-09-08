#!/usr/bin/env python3
"""Summarise a recorded session, and flag anything that looks wrong with the recording.

    python3 summary.py sessions/session_02.jsonl

Meant to be run on each session as you record it. A corpus flaw found now costs one replay;
found in P6 it costs a detector you cannot trust and no obvious reason why.

Named `summary` rather than `inspect` deliberately — the latter shadows a stdlib module.
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

BOLD, DIM, RESET = "\033[1m", "\033[2m", "\033[0m"
GREEN, YELLOW, RED = "\033[32m", "\033[33m", "\033[31m"

TICKS_PER_SECOND = 20

# A return to a previously-left cell within this window reads as boundary jitter rather
# than as travel. 30s is comfortably longer than any real crossing-and-returning.
JITTER_TICKS = 600


def load(path: Path) -> list[dict]:
    events = []
    for number, line in enumerate(path.open(encoding="utf-8"), start=1):
        line = line.strip()
        if line:
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise SystemExit(f"{RED}{path}:{number}: malformed JSON: {e}{RESET}")
    return events


def header(title: str) -> None:
    print(f"\n{BOLD}{title}{RESET}")


def check(ok: bool, good: str, bad: str) -> str:
    return f"{GREEN}{good}{RESET}" if ok else f"{RED}{bad}{RESET}"


def main(argv: list[str]) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("session", type=Path)
    args = p.parse_args(argv)
    if not args.session.exists():
        raise SystemExit(f"{RED}no such session: {args.session}{RESET}")

    ev = load(args.session)
    if not ev:
        raise SystemExit(f"{RED}{args.session} is empty{RESET}")

    kinds = collections.Counter(e["type"] for e in ev)
    # World-level events (weather, explosions) carry actor "world", not a player UUID.
    actors = {e["actor"] for e in ev if e.get("actor") and e["actor"] != "world"}
    wall = (ev[-1]["t"] - ev[0]["t"]) / 1000.0
    ticks = ev[-1]["tick"] - ev[0]["tick"]

    print(f"{BOLD}{args.session}{RESET}")
    print(f"  {len(ev)} events, {len(actors)} actor(s), {wall / 60:.1f} min")

    # --- recording integrity -------------------------------------------------
    header("integrity")
    # Game time versus real time. If the server stalled, ticks fall behind the wall clock.
    implied_tps = ticks / wall if wall else 0
    drift = abs(implied_tps - TICKS_PER_SECOND) / TICKS_PER_SECOND * 100 if wall else 0
    print(f"  implied TPS      {implied_tps:5.2f}  ({drift:.1f}% from 20)  "
          + check(drift < 2, "server kept up", "server fell behind"))

    out_of_order = sum(1 for a, b in zip(ev, ev[1:]) if b["tick"] < a["tick"])
    print(f"  tick ordering    {out_of_order} inversions  "
          + check(out_of_order == 0, "ordered", "OUT OF ORDER"))

    moves = [e for e in ev if e["type"] == "move"]
    if len(moves) > 2:
        gaps = collections.Counter(b["tick"] - a["tick"] for a, b in zip(moves, moves[1:]))
        period, hits = gaps.most_common(1)[0]
        clean = hits / (len(moves) - 1)
        print(f"  move cadence     {period} ticks in {clean * 100:.1f}% of gaps  "
              + check(clean > 0.95, "steady", "irregular — check for stalls"))
        odd = {g: n for g, n in gaps.items() if g != period}
        if odd:
            print(f"                   {DIM}other gaps: {dict(sorted(odd.items()))}{RESET}")

    # --- what happened -------------------------------------------------------
    header("events")
    for kind, count in kinds.most_common():
        print(f"  {count:6d}  {kind}")

    placed = collections.Counter()
    broken = collections.Counter()
    picked = collections.Counter()
    for e in ev:
        if e["type"] == "block_place":
            placed[e["after"]] += 1
        elif e["type"] == "block_break":
            broken[e["before"]] += 1
        elif e["type"] == "item_pickup":
            picked[e["item"]] += e.get("count", 0)

    if placed or broken:
        header("net delta (placed - broken)")
        gross = sum(placed.values()) + sum(broken.values())
        net = sum(placed.values()) - sum(broken.values())
        for m in sorted(set(placed) | set(broken), key=lambda m: -(placed[m] - broken[m])):
            d = placed[m] - broken[m]
            print(f"  {d:+5d}  {m}  {DIM}(+{placed[m]} -{broken[m]}){RESET}")
        print(f"\n  gross {gross} block actions, net {net:+d}")
        if gross and abs(net) / gross < 0.1:
            print(f"  {YELLOW}net is near zero — this looks like a build-then-demolish "
                  f"session{RESET}")

    dropped_items = collections.Counter()
    for e in ev:
        if e["type"] == "item_drop":
            dropped_items[e["item"]] += e.get("count", 0)

    if picked or dropped_items:
        header("items (net = picked up - dropped)")
        for item in sorted(set(picked) | set(dropped_items),
                           key=lambda i: -(picked[i] - dropped_items[i])):
            net = picked[item] - dropped_items[item]
            tail = f"  {DIM}(+{picked[item]} -{dropped_items[item]}){RESET}" if dropped_items[item] else ""
            print(f"  {net:6d}  {item}{tail}")

    # --- the player's own story ---------------------------------------------
    crafted = collections.Counter()
    smelted = collections.Counter()
    kills = collections.Counter()
    damage = []
    for e in ev:
        if e["type"] == "craft":
            crafted[e["item"]] += e.get("count", 0)
        elif e["type"] == "smelt":
            smelted[e["item"]] += e.get("count", 0)
        elif e["type"] == "mob_kill":
            kills[e["entity"]] += 1
        elif e["type"] == "damage":
            damage.append(e)

    if crafted or smelted:
        header("crafted / smelted")
        for item, n in crafted.most_common():
            print(f"  {n:6d}  {item}  {DIM}(crafted){RESET}")
        for item, n in smelted.most_common():
            print(f"  {n:6d}  {item}  {DIM}(smelted){RESET}")

    if kills:
        header("kills")
        for entity, n in kills.most_common():
            print(f"  {n:6d}  {entity}")

    if damage:
        header("damage taken")
        by_cause = collections.Counter()
        for d in damage:
            label = d.get("cause", "?")
            if d.get("source"):
                label += f" ({d['source'].replace('minecraft:', '')})"
            by_cause[label] += d.get("amount", 0)
        for cause, total in by_cause.most_common():
            hits = sum(1 for d in damage
                       if (d.get("cause", "?") + (f" ({d['source'].replace('minecraft:', '')})"
                                                  if d.get("source") else "")) == cause)
            print(f"  {total:6.1f} hp over {hits:3d} hits  {cause}")
        closest = min((d.get("health", 20) for d in damage), default=20)
        colour = RED if closest <= 4 else YELLOW if closest <= 8 else GREEN
        print(f"  lowest health this hit left them on: {colour}{closest} hp{RESET}")

    dealt = [e for e in ev if e["type"] == "damage_dealt"]
    if dealt:
        header("damage dealt")
        by_target = collections.Counter()
        for d in dealt:
            by_target[d.get("entity", "?")] += d.get("amount", 0)
        for target, total in by_target.most_common():
            hits = sum(1 for d in dealt if d.get("entity") == target)
            print(f"  {total:6.1f} hp over {hits:3d} hits  {target}")

    states = [e for e in ev if e["type"] == "player_state"]
    if states:
        header("world context (sampled)")
        biomes = collections.Counter(s.get("biome") for s in states if s.get("biome"))
        weather = collections.Counter(s.get("weather") for s in states if s.get("weather"))
        night = sum(1 for s in states if 13000 <= s.get("time", 0) <= 23000)
        low = min((s.get("health", 20) for s in states), default=20)
        hungry = min((s.get("food", 20) for s in states), default=20)
        print(f"  {len(states)} samples")
        print(f"  biomes:  {', '.join(f'{b.replace('minecraft:', '')} x{n}' for b, n in biomes.most_common(5))}")
        print(f"  weather: {', '.join(f'{w} x{n}' for w, n in weather.most_common())}")
        print(f"  night:   {night}/{len(states)} samples ({night / len(states) * 100:.0f}% of session)")
        print(f"  lowest health {low} hp, lowest food {hungry}")
        xp = [s.get("level", 0) for s in states]
        if xp:
            print(f"  xp level: {xp[0]} -> {xp[-1]} (peak {max(xp)})")

    said = [e for e in ev if e["type"] in ("chat", "sign_change")]
    if said:
        header("what the player wrote")
        for e in said:
            print(f"  tick {e['tick']:<8} {e['type']:12} {e.get('text', '')!r}")

    put = collections.Counter()
    took = collections.Counter()
    for e in ev:
        if e["type"] == "container_put":
            put[e["item"]] += e.get("count", 0)
        elif e["type"] == "container_take":
            took[e["item"]] += e.get("count", 0)
    if put or took:
        header("storage (net = put in - taken out)")
        for item in sorted(set(put) | set(took), key=lambda i: -(put[i] - took[i])):
            net = put[item] - took[item]
            print(f"  {net:+6d}  {item}  {DIM}(in {put[item]}, out {took[item]}){RESET}")

    milestones = [e for e in ev if e["type"] == "advancement"]
    if milestones:
        header("advancements")
        for m in milestones:
            print(f"  tick {m['tick']:<8} {m['advancement']}")

    for kind, label in (("sleep", "bed attempts"), ("eat", "ate"), ("container_open", "opened containers"),
                        ("xp_change", "xp pickups"), ("level_change", "level changes"),
                        ("potion_effect", "potion effects"), ("item_break", "tools broken"),
                        ("respawn", "respawns"), ("explosion", "explosions"),
                        ("shear", "sheared"), ("command", "commands"),
                        ("dimension_change", "changed dimension"), ("villager_trade", "traded"),
                        ("enchant", "enchanted"), ("breed", "bred"), ("tame", "tamed"),
                        ("fish", "fished"), ("portal_create", "made a portal")):
        n = kinds.get(kind, 0)
        if n:
            print(f"  {DIM}{label}: {n}{RESET}")

    # --- movement ------------------------------------------------------------
    entries = [e for e in ev if e["type"] == "region_enter"]
    if entries:
        header("regions")
        cells = [(e["pos"][0] // 64, e["pos"][2] // 64) for e in entries]
        # Returning to a cell you left is normal play — walking between a mine and a build
        # does it all day. It is only jitter if the round trip was fast, which is what
        # distinguishes a player straddling a boundary from one actually going somewhere.
        jitter = sum(
            1
            for i in range(2, len(cells))
            if cells[i] == cells[i - 2]
            and entries[i]["tick"] - entries[i - 2]["tick"] < JITTER_TICKS
        )
        print(f"  {len(entries)} entries across {len(set(cells))} distinct cells")
        print(f"  quick returns (<{JITTER_TICKS // 20}s)  {jitter}  "
              + check(jitter == 0, "none — no boundary jitter", "jitter — raise region-hysteresis"))
        print(f"  {DIM}{' -> '.join(f'{c[0]},{c[1]}' for c in cells)}{RESET}")

    deaths = [e for e in ev if e["type"] == "player_death"]
    if deaths:
        header("deaths")
        for d in deaths:
            print(f"  tick {d['tick']:<8} {d['dim']} {d['pos']}")

    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
