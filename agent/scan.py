#!/usr/bin/env python3
"""Ask a running server to scan a region, and store the result.

    python3 scan.py --min 173 60 273 --max 191 75 301
    python3 scan.py --min 173 60 273 --max 191 75 301 --slices

Scans travel on the same socket as events but on a separate logical channel: the request
and reply both carry an "rpc" key, the reply goes only to this connection, and neither ever
reaches the session JSONL. The event stream has to stay byte-identical to the recorded file
or the replay harness stops being a faithful stand-in for it.

A scan is a direct read of the world, so what it returns is stored with provenance SCANNED —
the strongest evidence the system has.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import uuid

from websockets.asyncio.client import connect

from consumer import BOLD, DIM, GREEN, RED, RESET
from store import Belief, Provenance, Store

DEFAULT_URL = "ws://127.0.0.1:8765"

# The live god already owns a durable bridge connection. RPCs use that same transport and
# are matched by request ID; command-line tools, which have no event connection, fall back
# to one short-lived socket. This avoids a new WebSocket handshake for every sentence while
# preserving the standalone scan and render utilities.
_bound_ws = None
_pending_rpc: dict[str, tuple[str, asyncio.Future]] = {}


def bind_rpc(ws) -> None:
    """Attach RPC routing to the live event connection for its lifetime."""
    global _bound_ws
    if _bound_ws is not None and _bound_ws is not ws:
        unbind_rpc(_bound_ws, ConnectionError("bridge connection replaced"))
    _bound_ws = ws


def unbind_rpc(ws, error: Exception | None = None) -> None:
    """Detach a closing connection and fail commands that cannot receive a reply."""
    global _bound_ws
    if _bound_ws is not ws:
        return
    _bound_ws = None
    failure = error or ConnectionError("bridge connection closed")
    for _, future in list(_pending_rpc.values()):
        if not future.done():
            future.set_exception(failure)
    _pending_rpc.clear()


def dispatch_rpc(message: dict) -> bool:
    """Deliver one RPC response read by the event loop to its waiting command."""
    pending = _pending_rpc.get(message.get("id"))
    if not pending:
        return False
    expected, future = pending
    # Older plugins may report an unknown command as a generic failure response. Surface
    # that immediately instead of leaving the requester waiting for another response type.
    if message.get("rpc") == expected or message.get("ok") is False:
        if not future.done():
            future.set_result(message)
        return True
    return False


async def request_rpc(url: str, payload: dict, expected: str, timeout: float,
                      max_size: int | None = None) -> dict:
    """Issue an RPC over the live bridge, with a standalone-client fallback."""
    request_id = payload.setdefault("id", uuid.uuid4().hex[:12])
    if _bound_ws is not None:
        future = asyncio.get_running_loop().create_future()
        _pending_rpc[request_id] = (expected, future)
        try:
            await _bound_ws.send(json.dumps(payload))
            return await asyncio.wait_for(future, timeout=timeout)
        finally:
            _pending_rpc.pop(request_id, None)

    options = {"max_queue": 4096}
    if max_size is not None:
        options["max_size"] = max_size
    async with connect(url, **options) as ws:
        await ws.send(json.dumps(payload))
        deadline = asyncio.get_running_loop().time() + timeout
        while True:
            remaining = deadline - asyncio.get_running_loop().time()
            if remaining <= 0:
                raise TimeoutError(f"no {expected} before timeout")
            message = json.loads(await asyncio.wait_for(ws.recv(), timeout=remaining))
            if (message.get("id") == request_id
                    and (message.get("rpc") == expected or message.get("ok") is False)):
                return message


async def request_scan(url: str, dim: str, lo: list[int], hi: list[int],
                       timeout: float = 60.0) -> dict:
    return await request_rpc(url, {
        "rpc": "scan_region", "dim": dim, "min": lo, "max": hi,
    }, "scan_result", timeout)


async def request_voxels(url: str, dim: str, lo: list[int], hi: list[int],
                         timeout: float = 180.0) -> dict:
    """Fetches a region as a material palette plus one byte per block.

    Agent-side data for the renderer. The model is shown the resulting image, never this.
    """
    import base64

    message = await request_rpc(url, {
        "rpc": "voxels", "dim": dim, "min": lo, "max": hi,
    }, "voxel_result", timeout, max_size=64 * 1024 * 1024)
    if message.get("ok"):
        message["blocks"] = base64.b64decode(message["data"])
    return message


def to_voxels(result: dict) -> dict:
    """Unpacks a voxel result into {(x, y, z): material}, air omitted."""
    palette = result["palette"]
    sx, sy, sz = result["size"]
    ox, oy, oz = result["origin"]
    raw = result["blocks"]
    out = {}
    i = 0
    for y in range(sy):
        for z in range(sz):
            for x in range(sx):
                index = raw[i]
                i += 1
                if index:
                    out[(ox + x, oy + y, oz + z)] = palette[index]
    return out


#: The triggers and switches an ability may be built from. The plugin is authoritative;
#: this is the vocabulary the model is shown, so an invented word comes back named.
TRIGGERS = ("on_use", "on_sneak", "on_move", "on_land", "on_attack", "on_damaged",
            "on_hit", "every")
IMPULSES = ("look", "up", "back", "bounce", "stop")
SWITCHES = ("fly", "no_fall", "glow", "immune:", "walk_speed:")


async def grant_power(player: str, name: str, url: str = DEFAULT_URL, *,
                      scripts: dict | None = None, switches=(), on=(),
                      projectile: str | None = None, speed: float = 0,
                      impulse: str | None = None, power: float = 0,
                      beam: str | None = None, range: int = 0, damage: float = 0,
                      every: int = 0, duration: int = 0,
                      cooldown_ms: int = 200, revoke: bool = False,
                      timeout: float = 20.0) -> dict:
    """Give or take an ability the god has invented.

    An ability is commands bound to a gesture plus a few switches no command can express.
    The scripted commands pass the server's own gate, exactly as a direct command does, so
    binding one to a right-click is not a way around what may be run. ``duration`` is in
    ticks; zero holds until taken away.
    """
    if revoke:
        return await request_rpc(url, {
            "rpc": "revoke_power", "player": player, "name": name or None,
        }, "power_result", timeout)
    return await request_rpc(url, {
        "rpc": "grant_power", "player": player, "name": name,
        "scripts": {str(k): [str(c) for c in v] for k, v in (scripts or {}).items()},
        "switches": [str(s) for s in switches],
        "on": [str(t) for t in on],
        "projectile": projectile, "speed": float(speed),
        "impulse": impulse, "power": float(power),
        "beam": beam, "range": int(range), "damage": float(damage),
        "every": int(every), "duration": int(duration), "cooldown_ms": int(cooldown_ms),
    }, "power_result", timeout)


async def held_powers(player: str, url: str = DEFAULT_URL,
                      timeout: float = 10.0) -> dict:
    """What a player is holding right now, named.

    The god cannot change or take back what it cannot see. Asked to make an existing power
    automatic while blind to its name, it invented a second power beside the first and left
    both running.
    """
    return await request_rpc(url, {"rpc": "held_powers", "player": player},
                             "power_result", timeout)


async def stop_commands(url: str = DEFAULT_URL, spell: str | None = None,
                        timeout: float = 10.0) -> dict:
    """Stop one repeating effect, or everything the god has running."""
    return await request_rpc(url, {"rpc": "stop_commands", "spell": spell},
                             "stop_result", timeout)


async def run_command(commands, url: str = DEFAULT_URL, anchor: str | None = None,
                      every: int = 0, duration: int = 0, spell: str | None = None,
                      timeout: float = 30.0) -> dict:
    """Ask the server to run one command.

    The agent does not decide what may run. It sends what the model drafted and the plugin
    answers, because the thing drafting is a language model and the thing that has to live
    with the result is the world.

    ``anchor`` names a player to run from, which is what makes ``~`` mean what it means in
    game. ``every`` and ``duration``, both in ticks, make the commands repeat by themselves
    until they have run their course; the reply comes back at once and the effect carries on
    without anyone waiting for it.

    The reply says what was accepted, not what it did: a thousand-block build takes longer
    than a sentence should wait, and each command reports itself through the event stream as
    it runs.
    """
    if isinstance(commands, str):
        commands = [commands]
    return await request_rpc(url, {
        "rpc": "run_command", "commands": list(commands), "anchor": anchor,
        "every": int(every), "duration": int(duration), "spell": spell,
    }, "command_result", timeout)


async def speak(text: str, url: str = DEFAULT_URL, target: str | None = None,
                timeout: float = 15.0, reply: bool = False) -> dict:
    """Says something in chat. The server decides whether it is allowed to.

    The budget, the cooldown and the length cap all live in the plugin, not here. An agent
    that has gone wrong is exactly the case a client-side limit fails to cover.
    """
    payload = {"rpc": "speak", "text": text, "reply": reply}
    if target:
        payload["target"] = target
    return await request_rpc(url, payload, "speak_result", timeout)


def render(result: dict, show_slices: bool) -> None:
    if not result.get("ok"):
        print(f"{RED}scan failed: {result.get('error')}{RESET}")
        return
    box = result["bbox"]
    print(f"{BOLD}scan{RESET} {result['dim']} "
          f"{tuple(box['min'])}..{tuple(box['max'])}  tick {result['tick']}")
    print(f"  volume {result['volume']}, solid {result['solid']}, air {result['air']}, "
          f"{result['distinct_materials']} distinct materials")
    print(f"\n{BOLD}materials{RESET}")
    for material, count in result["materials"].items():
        print(f"  {count:6d}  {material}")
    if result.get("materials_truncated"):
        print(f"  {DIM}+{result['materials_truncated']} in materials beyond the top 40{RESET}")
    if result.get("block_entities"):
        print(f"\n{BOLD}block entities{RESET}")
        for kind, count in result["block_entities"].items():
            print(f"  {count:6d}  {kind}")
    if show_slices:
        for sl in result.get("slices", []):
            print(f"\n{BOLD}slice y={sl['y']}{RESET} "
                  f"{DIM}origin {tuple(sl['origin'])} step {tuple(sl['step'])}{RESET}")
            for row in sl["rows"]:
                print("  " + row)
            print("  " + DIM + "  ".join(f"{c}={m.replace('minecraft:', '')}"
                                         for c, m in sl["legend"].items()) + RESET)


def main(argv: list[str]) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--url", default=DEFAULT_URL)
    p.add_argument("--dim", default="overworld")
    p.add_argument("--min", nargs=3, type=int, required=True, metavar=("X", "Y", "Z"))
    p.add_argument("--max", nargs=3, type=int, required=True, metavar=("X", "Y", "Z"))
    p.add_argument("--slices", action="store_true", help="print the character grids")
    p.add_argument("--store", metavar="ID", help="record as a structure with this id")
    p.add_argument("--json", action="store_true", help="dump the raw reply")
    args = p.parse_args(argv)

    try:
        result = asyncio.run(request_scan(args.url, args.dim, args.min, args.max))
    except (OSError, TimeoutError) as e:
        print(f"{RED}cannot scan: {e}{RESET}. Is the server running?", file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps(result, indent=2))
    else:
        render(result, args.slices)

    if args.store and result.get("ok"):
        box = result["bbox"]
        store = Store()
        store.put("structures", {
            "id": args.store, "actor": None, "dim": result["dim"], "name": None,
            "min_x": box["min"][0], "min_y": box["min"][1], "min_z": box["min"][2],
            "max_x": box["max"][0], "max_y": box["max"][1], "max_z": box["max"][2],
            "materials": json.dumps(result["materials"]),
        }, Belief(provenance=Provenance.SCANNED,
                  verified_at_tick=result["tick"],
                  value=json.dumps({"solid": result["solid"], "air": result["air"]})))
        store.close()
        print(f"\n{GREEN}stored as structures/{args.store} with provenance SCANNED{RESET}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
