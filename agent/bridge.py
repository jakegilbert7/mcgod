#!/usr/bin/env python3
"""P2: consume the live L0 event stream from the McGod plugin.

    python bridge.py                     # connect, print, tee to sessions/
    python bridge.py --hide move         # quieter: drop the 2s position samples
    python bridge.py --show block_break  # only block breaks

Reconnects on its own, so restarting the server does not mean restarting this.
Ctrl-C prints the latency summary the P2 acceptance test is judged on.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import signal
import sys
from datetime import datetime
from pathlib import Path

from websockets.asyncio.client import connect
from websockets.exceptions import ConnectionClosed, InvalidHandshake

from consumer import BOLD, DIM, GREEN, RED, RESET, Fanout, JsonlTee, LatencyMeter, Printer, now_ms

DEFAULT_URL = "ws://127.0.0.1:8765"


async def pump(url: str, consumer, first_connect: list[bool]) -> None:
    """One connection's lifetime: read frames until the socket closes."""
    async with connect(url, max_queue=4096) as ws:
        first_connect[0] = False
        print(f"{GREEN}connected to {url}{RESET}", file=sys.stderr)
        async for frame in ws:
            received = now_ms()
            try:
                event = json.loads(frame)
            except json.JSONDecodeError:
                print(f"{RED}malformed frame, skipped: {frame[:120]!r}{RESET}", file=sys.stderr)
                continue
            consumer.on_event(event, received)


async def run(args: argparse.Namespace) -> int:
    consumers = [Printer(show=args.show, hide=args.hide, show_latency=not args.no_latency),
                 LatencyMeter()]
    tee = None
    if not args.no_tee:
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        tee = JsonlTee(Path(args.out) if args.out else Path("sessions") / f"live-{stamp}.jsonl")
        consumers.append(tee)
    consumer = Fanout(*consumers)

    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, stop.set)

    print(f"{BOLD}McGod bridge consumer{RESET}  {DIM}{args.url}{RESET}", file=sys.stderr)
    if tee:
        print(f"{DIM}teeing to {tee.path}{RESET}", file=sys.stderr)

    first_connect = [True]
    backoff = 0.5
    while not stop.is_set():
        try:
            pumping = asyncio.create_task(pump(args.url, consumer, first_connect))
            waiting = asyncio.create_task(stop.wait())
            done, pending = await asyncio.wait(
                {pumping, waiting}, return_when=asyncio.FIRST_COMPLETED
            )
            for task in pending:
                task.cancel()
            if pumping in done:
                pumping.result()  # surface any error
                backoff = 0.5
        except asyncio.CancelledError:
            break
        except (OSError, ConnectionClosed, InvalidHandshake) as e:
            if first_connect[0]:
                print(
                    f"{RED}cannot reach {args.url}{RESET} ({type(e).__name__}). "
                    f"Is the server running with bridge.enabled: true?",
                    file=sys.stderr,
                )
            else:
                print(f"{DIM}disconnected ({type(e).__name__}); retrying{RESET}", file=sys.stderr)
        if stop.is_set():
            break
        await asyncio.sleep(backoff)
        backoff = min(backoff * 2, 5.0)

    consumer.on_close()
    return 0


def parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--url", default=DEFAULT_URL, help=f"bridge URL (default {DEFAULT_URL})")
    p.add_argument("--out", help="tee target (default sessions/live-<stamp>.jsonl)")
    p.add_argument("--no-tee", action="store_true", help="print only, do not write a file")
    p.add_argument("--show", nargs="+", metavar="TYPE", help="only these event types")
    p.add_argument("--hide", nargs="+", metavar="TYPE", default=[], help="suppress these event types")
    p.add_argument("--no-latency", action="store_true",
                   help="omit the timing column, making output directly diffable against replay.py")
    args = p.parse_args(argv)
    args.show = set(args.show) if args.show else None
    args.hide = set(args.hide)
    return args


if __name__ == "__main__":
    try:
        sys.exit(asyncio.run(run(parse_args(sys.argv[1:]))))
    except KeyboardInterrupt:
        sys.exit(130)
