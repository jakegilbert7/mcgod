"""Consumers for the L0 event stream.

The bridge (P2) and the replay harness (P3) both drive this same interface, which is the
point: if replay can produce output identical to a live run, everything built on top of it
can be tested without a running server.

A consumer sees decoded events one at a time, in order. It never sees the transport.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

RESET = "\033[0m"
DIM = "\033[2m"
BOLD = "\033[1m"
RED = "\033[31m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
BLUE = "\033[34m"
MAGENTA = "\033[35m"
CYAN = "\033[36m"

# Colour per event type, so a block event is findable in a scrolling terminal at a glance.
TYPE_COLOUR = {
    "block_place": GREEN,
    "block_break": RED,
    "move": DIM,
    "region_enter": CYAN,
    "item_pickup": YELLOW,
    "item_drop": YELLOW,
    "player_join": BOLD + BLUE,
    "player_quit": BLUE,
    "player_death": BOLD + MAGENTA,
    "damage": MAGENTA,
    "mob_kill": RED,
    "craft": BOLD + GREEN,
    "smelt": GREEN,
    "eat": YELLOW,
    "sleep": BLUE,
    "advancement": BOLD + YELLOW,
    "dimension_change": BOLD + CYAN,
    "container_open": DIM + CYAN,
    "enchant": BOLD + MAGENTA,
    "villager_trade": CYAN,
    "breed": GREEN,
    "tame": GREEN,
    "fish": CYAN,
    "projectile_hit": RED,
    "bucket_use": CYAN,
    "portal_create": BOLD + MAGENTA,
    "damage_dealt": RED,
    "xp_change": YELLOW,
    "level_change": BOLD + YELLOW,
    "player_state": DIM,
    "chat": BOLD + CYAN,
    "sign_change": CYAN,
    "command": DIM + CYAN,
    "potion_effect": MAGENTA,
    "shear": GREEN,
    "explosion": BOLD + RED,
    "weather_change": DIM + BLUE,
    "thunder_change": BOLD + BLUE,
    "respawn": BOLD + BLUE,
    "gamemode_change": BOLD + YELLOW,
    "item_break": RED,
    "container_put": YELLOW,
    "container_take": BOLD + YELLOW,
}


def describe(event: dict) -> str:
    """The type-specific half of a printed line.

    Kept in one place so `bridge.py` and `replay.py` render identically — that equality is
    the P3 acceptance test, and it quietly breaks if each driver formats its own.
    """
    kind = event.get("type")
    item, count = event.get("item"), event.get("count")
    if kind == "block_place":
        return event.get("after", "")
    if kind == "block_break":
        return event.get("before", "")
    if kind in ("container_put", "container_take"):
        return f"{count}x {item} {DIM}({event.get('container', '?')}){RESET}"
    if kind in ("item_pickup", "item_drop", "craft", "smelt", "villager_trade"):
        return f"{count}x {item}" if count is not None else str(item or "")
    if kind in ("eat", "bucket_use"):
        return " ".join(str(v) for v in (item, event.get("action")) if v)
    if kind in ("mob_kill", "breed", "tame"):
        return event.get("entity", "")
    if kind == "damage":
        src = f" from {event['source']}" if event.get("source") else ""
        return f"{event.get('cause', '?')}{src} -{event.get('amount', '?')} -> {event.get('health', '?')} hp"
    if kind == "damage_dealt":
        return f"{event.get('entity', '?')} -{event.get('amount', '?')} ({event.get('cause', '?')})"
    if kind == "xp_change":
        return f"+{event.get('amount', '?')} xp"
    if kind == "level_change":
        return f"level {event.get('from', '?')} -> {event.get('to', '?')}"
    if kind in ("chat", "sign_change", "command"):
        return repr(event.get("text", ""))
    if kind == "potion_effect":
        return f"{event.get('action', '?')} {event.get('effect', '?')} ({event.get('reason', '?')})"
    if kind == "explosion":
        return f"{event.get('entity', '?')} destroyed {event.get('blocks', '?')} blocks"
    if kind in ("weather_change", "thunder_change"):
        return event.get("weather", "")
    if kind == "sleep":
        return event.get("result", "")
    if kind == "item_break":
        return event.get("item", "")
    if kind == "gamemode_change":
        return f"{event.get('from', '?')} -> {event.get('to', '?')}"
    if kind == "shear":
        return event.get("entity", "")
    if kind == "player_state":
        return (f"{event.get('health', '?')}hp {event.get('food', '?')}f "
                f"lvl{event.get('level', '?')} {event.get('weather', '')} "
                f"{str(event.get('biome', '')).replace('minecraft:', '')}")
    if kind == "player_death":
        return event.get("cause", "")
    if kind == "advancement":
        return event.get("advancement", "")
    if kind == "dimension_change":
        return f"{event.get('from', '?')} -> {event.get('to', '?')}"
    if kind == "container_open":
        return event.get("container", "")
    if kind == "enchant":
        return f"{item} (lvl {event.get('cost', '?')})"
    if kind == "fish":
        return event.get("entity") or event.get("state", "")
    if kind == "projectile_hit":
        return f"{event.get('projectile', '?')} -> {event.get('entity') or event.get('block') or '?'}"
    if kind == "portal_create":
        return f"{event.get('reason', '?')} ({event.get('blocks', '?')} blocks)"
    return ""


class EventConsumer:
    """Base class. Override the hooks you care about."""

    def on_event(self, event: dict, received_ms: float) -> None:
        """One decoded event. `received_ms` is the wall clock at receipt, epoch millis."""

    def on_close(self) -> None:
        """Stream ended. Flush and summarise."""


class Fanout(EventConsumer):
    """Drives several consumers from one stream, in order."""

    def __init__(self, *consumers: EventConsumer) -> None:
        self._consumers = [c for c in consumers if c is not None]

    def on_event(self, event: dict, received_ms: float) -> None:
        for consumer in self._consumers:
            consumer.on_event(event, received_ms)

    def on_close(self) -> None:
        for consumer in self._consumers:
            consumer.on_close()


class JsonlTee(EventConsumer):
    """Writes every event back out as JSONL.

    Re-serialising rather than echoing the raw frame is deliberate: it proves the event
    survived a decode/encode round trip, so a malformed frame fails here rather than silently
    poisoning a recorded session.
    """

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = self.path.open("a", encoding="utf-8")
        self.written = 0

    def on_event(self, event: dict, received_ms: float) -> None:
        self._file.write(json.dumps(event, separators=(",", ":")) + "\n")
        self.written += 1
        self._file.flush()

    def on_close(self) -> None:
        self._file.close()
        print(f"{DIM}tee: wrote {self.written} events to {self.path}{RESET}", file=sys.stderr)


class Printer(EventConsumer):
    """Human-readable one line per event.

    `show_latency` controls the leading timing column. With it off, the rendering is a pure
    function of the event, so a live run and a replay of that run's recording print
    byte-identical text. That equivalence is the P3 acceptance test, so it needs to be
    mechanically checkable rather than eyeballed.
    """

    def __init__(
        self,
        show: set[str] | None = None,
        hide: set[str] | None = None,
        show_latency: bool = True,
    ) -> None:
        self.show = show
        self.hide = hide or set()
        self.show_latency = show_latency

    def on_event(self, event: dict, received_ms: float) -> None:
        kind = event.get("type", "?")
        if kind in self.hide or (self.show is not None and kind not in self.show):
            return

        latency = received_ms - event.get("t", received_ms)
        colour = TYPE_COLOUR.get(kind, "")
        actor = str(event.get("actor", "?"))[:8]
        pos = event.get("pos")
        where = f"{pos[0]},{pos[1]},{pos[2]}" if pos else "-"

        detail = describe(event)

        stamp = f"{_latency_colour(latency)}[{latency:6.0f}ms]{RESET} " if self.show_latency else ""
        print(
            f"{stamp}"
            f"{DIM}tick {event.get('tick', 0):<8}{RESET}"
            f"{colour}{kind:<13}{RESET} "
            f"{DIM}{actor}{RESET} "
            f"{detail:<28} {DIM}@ {where}{RESET}",
            flush=True,
        )


class LatencyMeter(EventConsumer):
    """Measures plugin-emit to Python-receipt, the P2 acceptance criterion.

    Both ends read the same machine clock, so the difference is meaningful. It bundles the
    plugin's queue wait, dispatch, and the socket hop.
    """

    def __init__(self) -> None:
        self.samples: list[float] = []

    def on_event(self, event: dict, received_ms: float) -> None:
        if "t" in event:
            self.samples.append(received_ms - event["t"])

    def on_close(self) -> None:
        if not self.samples:
            print(f"{DIM}latency: no samples{RESET}", file=sys.stderr)
            return
        ordered = sorted(self.samples)
        n = len(ordered)

        def pct(p: float) -> float:
            return ordered[min(n - 1, int(n * p))]

        worst = ordered[-1]
        verdict = f"{GREEN}PASS{RESET}" if worst < 100 else f"{RED}FAIL{RESET}"
        print(
            f"\n{BOLD}latency over {n} events{RESET} "
            f"(plugin emit -> python receipt)\n"
            f"  p50 {pct(0.50):6.1f}ms\n"
            f"  p95 {pct(0.95):6.1f}ms\n"
            f"  max {worst:6.1f}ms\n"
            f"  P2 requires max < 100ms: {verdict}",
            file=sys.stderr,
        )


def _latency_colour(ms: float) -> str:
    if ms < 100:
        return GREEN
    if ms < 500:
        return YELLOW
    return RED


def now_ms() -> float:
    return time.time() * 1000.0
