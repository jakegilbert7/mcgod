#!/usr/bin/env python3
"""P3: replay a recorded session through the same consumers the live bridge drives.

    python3 replay.py sessions/session_02.jsonl --speed 50
    python3 replay.py session.jsonl --speed 0            # as fast as possible
    python3 replay.py session.jsonl --hide move

The point is substitutability. `bridge.py` and `replay.py` are two drivers of one
`EventConsumer` interface, so everything built above L0 can be developed and tested against
a recording instead of a running server and a person playing it.

That only holds if replay is faithful, which is why the default rendering carries no timing
column: the output becomes a pure function of the event stream, and a replay of a session
prints byte-identical text to the live run that recorded it. `--latency` turns the column
back on for eyeballing pacing, at the cost of that guarantee.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from consumer import BOLD, DIM, RED, RESET, EventConsumer, Fanout, JsonlTee, Printer, now_ms


def load(path: Path) -> list[dict]:
    """Reads a session file, failing loudly on a bad line rather than skipping it.

    A recorded session is test data for every later phase. A silently dropped line here
    becomes an unreproducible discrepancy much further downstream.
    """
    events: list[dict] = []
    with path.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise SystemExit(f"{RED}{path}:{number}: malformed JSON: {e}{RESET}")
    return events


def replay(events: list[dict], consumer: EventConsumer, speed: float) -> None:
    """Feeds events to a consumer, pacing on the recorded wall clock divided by `speed`.

    `speed <= 0` means no pacing at all. Pacing uses `t` (real milliseconds) rather than
    `tick`, so a session recorded through a lag spike replays with that spike intact.

    Each event is scheduled against an absolute deadline derived from the first event rather
    than by sleeping the inter-event gap. Sleeping gaps accumulates the overhead of every
    sleep, which on a long session at high speed drifts by tens of percent; deadlines absorb
    it. If the consumer falls behind the deadline the replay simply stops sleeping and
    catches up, so slow consumers cost fidelity of pacing, never of ordering or content.
    """
    if not events:
        consumer.on_close()
        return

    base = events[0].get("t")
    origin = time.monotonic()
    for event in events:
        stamp = event.get("t")
        if speed > 0 and stamp is not None and base is not None:
            deadline = origin + (stamp - base) / 1000.0 / speed
            delay = deadline - time.monotonic()
            if delay > 0:
                time.sleep(delay)
        consumer.on_event(event, now_ms())
    consumer.on_close()


def main(argv: list[str]) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("session", type=Path, help="recorded session JSONL")
    p.add_argument("--speed", type=float, default=1.0,
                   help="multiplier; 1 = real time, 50 = 50x, 0 = unpaced (default 1)")
    p.add_argument("--out", type=Path, help="also tee events to this file")
    p.add_argument("--show", nargs="+", metavar="TYPE", help="only these event types")
    p.add_argument("--hide", nargs="+", metavar="TYPE", default=[], help="suppress these types")
    p.add_argument("--latency", action="store_true",
                   help="show a timing column (breaks byte-identity with a live run)")
    p.add_argument("--quiet", action="store_true", help="do not print events")
    args = p.parse_args(argv)

    if not args.session.exists():
        raise SystemExit(f"{RED}no such session: {args.session}{RESET}")

    events = load(args.session)
    span = (events[-1]["t"] - events[0]["t"]) / 1000.0 if len(events) > 1 else 0.0
    pace = "unpaced" if args.speed <= 0 else f"{args.speed}x"
    print(
        f"{BOLD}replay{RESET} {args.session}  {DIM}{len(events)} events, "
        f"{span / 60:.1f} min recorded, {pace}{RESET}",
        file=sys.stderr,
    )

    consumers: list[EventConsumer] = []
    if not args.quiet:
        consumers.append(
            Printer(
                show=set(args.show) if args.show else None,
                hide=set(args.hide),
                show_latency=args.latency,
            )
        )
    if args.out:
        consumers.append(JsonlTee(args.out))

    started = time.time()
    try:
        replay(events, Fanout(*consumers), args.speed)
    except KeyboardInterrupt:
        print(f"\n{DIM}interrupted{RESET}", file=sys.stderr)
        return 130
    print(f"{DIM}replayed {len(events)} events in {time.time() - started:.1f}s{RESET}",
          file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
