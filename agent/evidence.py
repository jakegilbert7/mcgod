#!/usr/bin/env python3
"""Shared model evidence tools that are not specific to one dialogue route."""

from __future__ import annotations

import asyncio
import base64
import io
import json
import os
import time
from pathlib import Path

MAX_RENDER_AXIS = 64
MAX_RENDER_VOLUME = 220_000
MODEL_OUTPUT_TOKENS = 4096
MODEL_RETRY_TOKENS = 8192

#: How long one model call may take before it is abandoned and asked again.
#:
#: Measured on this setup, a call is 1-9 seconds whatever its shape: with tools, with an
#: image, with two hundred kilobytes of evidence, or all three. Then one ordinary call took
#: **609 seconds** and the player waited ten minutes for a sentence. Nothing about the
#: request explained it and repeating it was fast, so it was the provider, and no amount of
#: prompt or model choice prevents that recurring.
#:
#: A stalled call is not a slow answer, it is a silent god, which from the player's side is
#: indistinguishable from the thing being down. So a call gets a deadline and one more
#: attempt; providers route each request separately, so the retry usually lands elsewhere.
#: Set at twenty-five seconds because typical is one to nine: generous enough that a merely
#: slow call is not thrown away, short enough that a stalled one is not paid for twice over.
MODEL_TIMEOUT_SECONDS = float(os.environ.get("MCGOD_MODEL_TIMEOUT", "25"))
LATEST_EVIDENCE_RENDER = (Path(__file__).parent.parent / "renders" / "evidence"
                          / "latest-world-area.png")

ACT_TOOL = {
    "name": "act_on_world",
    "description": (
        "Do something to the world by running Minecraft commands. Any number of them, in "
        "order. Set `anchor` to a player's name and every command runs from where they "
        "stand, so ~ and ^ mean what they mean in game. Set `every` and `duration` in ticks "
        "to make them repeat by themselves for a while.\n"
        "You get back what each command actually did, including the server's own words when "
        "one fails. If something did not work, look at the world and try a different way "
        "rather than reporting failure: a wrong item syntax, a region that missed, a block "
        "that was not where you assumed are all things you can see and correct.\n"
        "A command can succeed and change nothing. \"Changed 0 blocks\" means whatever you "
        "aimed at was not there, and anything listed under `changed_nothing` did not happen "
        "however cleanly it ran. Treat that as a miss: look, widen the area, or aim "
        "somewhere else. Do not tell the player it is done until something actually changed."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "commands": {"type": "array", "items": {"type": "string"}},
            "anchor": {"type": "string"},
            "every": {"type": "integer", "minimum": 1},
            "duration": {"type": "integer", "minimum": 1},
            "spell": {"type": "string"},
        },
        "required": ["commands"],
        "additionalProperties": False,
    },
}

DESIGN_TOOL = {
    "name": "design_build",
    "description": (
        "Ask the builder for the commands that make something. Use it whenever the player "
        "asks you to BUILD or SCULPT anything whose shape matters — a statue, a house, an "
        "arch, a tower, a bridge with any character to it.\n"
        "Describe what is wanted and where in plain words and give the anchor point. The "
        "build is PLACED FOR YOU and you get back what it was and how much of it landed; "
        "do not run anything yourself for it, just look at the result if you want to "
        "check it. The "
        "builder holds the shape in mind block by block, which is a different skill from "
        "conversation and one you should not attempt yourself for anything figurative.\n"
        "It is surveyed the ground around the anchor for you, so you do not need to "
        "describe the terrain, and it takes a while: a real build is minutes of work, not "
        "seconds. Ask once and wait for it rather than giving up and writing the commands "
        "yourself."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "what": {"type": "string",
                     "description": "what to build, in plain words, with any detail the "
                                    "player asked for"},
            "x": {"type": "integer"},
            "y": {"type": "integer"},
            "z": {"type": "integer"},
            "facing": {"type": "string",
                       "description": "which way it should face: north, south, east, west"},
            "materials": {"type": "string",
                          "description": "any materials the player named"},
            "dim": {"type": "string",
                    "description": "the dimension to build in; overworld unless told "
                                   "otherwise"},
        },
        "required": ["what", "x", "y", "z"],
        "additionalProperties": False,
    },
}

POWER_TOOL = {
    "name": "grant_power",
    "description": (
        "Invent an ability and give it to a player. There is no list of powers to pick "
        "from: you write what the power DOES, and the server binds it to a gesture.\n"
        "TRIGGERS, which say when it happens: on_use (either mouse button), on_sneak, "
        "on_move, on_land (they hit the ground after a fall), on_attack (they hit "
        "something), on_damaged, on_hit (something they threw or shot arrived), every "
        "(with `every` set to a tick interval).\n"
        "An ability is made of any combination of these:\n"
        "  scripts  - commands keyed by trigger. They run anchored to the player, so "
        "`~ ~ ~` is where they stand and `^ ^ ^5` is five blocks ahead of where they are "
        "LOOKING. The exception is on_hit, whose commands run where the thrown or shot "
        "thing arrived, which is the only way to act somewhere the player is not.\n"
        "  switches - what no command can say: fly (flight without creative mode, so "
        "they keep inventory, hunger and damage), no_fall, glow, immune:<cause> (fire, "
        "lava, explosion, drowning, or all), walk_speed:<n> (0.2 normal, 0.6 very fast)."
        "\n"
        "  projectile - a thing thrown along the line of sight, aimed where they look, "
        "which a summon cannot do. Any entity id (small_fireball, arrow, snowball, tnt, "
        "even a cow) or any BLOCK, thrown falling, so anvil works. `speed` sets how hard."
        "\n"
        "  beam - a HITSCAN ray: instant, straight, stopping at the first block or "
        "creature, up to `range` blocks (default 64, max 256). The value is the particle "
        "drawn along it: flame, electric_spark, end_rod, soul_fire_flame, crit, dust. "
        "`damage` is hearts dealt to whatever it strikes. Use this for lasers and beams "
        "rather than writing a run of particle commands: those stop where the list "
        "stops rather than where the beam hits, pass through walls, and land their "
        "damage on whoever is nearest, including the player holding it. A beam never "
        "hits its own shooter. Pair with on_hit for what happens at the far end.\n"
        "  impulse - MOVEMENT, which no command can give. `tp` puts somebody somewhere; "
        "it cannot give them momentum. Modes: look (flung where they face), up, back, "
        "bounce (the speed they landed at, sent back the way it came), stop (all motion "
        "killed). `power` scales it, 1.0 by default.\n"
        "  on - which triggers work the projectile, beam and impulse. Defaults to "
        "on_use. This is separate from the script keys because the good ones have no "
        "script: a bouncy player is on: [on_land] with impulse bounce and nothing else.\n"
        "Examples of the shape, not a menu:\n"
        "  a web shooter -> scripts {on_use: [\"setblock ^ ^ ^4 cobweb\"]}\n"
        "  frozen wake -> scripts {on_move: [\"setblock ~ ~-1 ~ packed_ice\"]}\n"
        "  bouncy -> on [on_land], impulse bounce, power 1.1, switches [no_fall]. The "
        "no_fall matters: bouncing without it kills them on the second landing.\n"
        "  a laser -> beam electric_spark, range 120, damage 6\n"
        "  a rocket jump -> on [on_sneak], impulse up, power 1.6, switches [no_fall]\n"
        "  exploding pigs -> projectile pig, scripts {on_hit: [\"summon tnt ~ ~ ~ "
        "{fuse:1s}\"]}\n"
        "  meteor storm -> every 10, scripts {every: [\"summon fireball ~ ~30 ~ "
        "{power:[0.0,-1.0,0.0]}\"]}\n"
        "Prefer this over creative mode and a stack of items. Creative is not a "
        "superpower, it is a different game, and a fire charge you have to throw by hand "
        "is not what anyone means by throwing fire.\n"
        "CHANGING AND REMOVING. `name` identifies the power. Granting a name that is "
        "already held REPLACES it, and that is how you edit one: grant it again under "
        "its EXISTING name carrying the whole definition you want. Never invent a second "
        "name beside it, or the old one keeps firing. You are told which powers the "
        "player is holding; use those names exactly. To remove one, set revoke with its "
        "name; to remove everything, set revoke with no name at all.\n"
        "`duration` is in ticks (20 a second); leave it out to hold until taken away. "
        "`cooldown_ms` throttles a trigger that would otherwise fire many times a "
        "second; on_move and on_use fire constantly, so set it unless you want a dense "
        "trail, and set it to 0 for anything meant to be fully automatic.\n"
        "Tell the player which gesture works it."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "player": {"type": "string"},
            "name": {"type": "string"},
            "scripts": {
                "type": "object",
                "additionalProperties": {"type": "array", "items": {"type": "string"}},
            },
            "switches": {"type": "array", "items": {"type": "string"}},
            "on": {"type": "array", "items": {"type": "string"}},
            "projectile": {"type": "string"},
            "speed": {"type": "number"},
            "beam": {"type": "string"},
            "range": {"type": "integer", "minimum": 1, "maximum": 256},
            "damage": {"type": "number", "minimum": 0},
            "impulse": {"type": "string",
                        "enum": ["look", "up", "back", "bounce", "stop"]},
            "power": {"type": "number"},
            "every": {"type": "integer", "minimum": 1},
            "duration": {"type": "integer", "minimum": 1},
            "cooldown_ms": {"type": "integer", "minimum": 0},
            "revoke": {"type": "boolean"},
        },
        "required": ["player"],
        "additionalProperties": False,
    },
}

VISUAL_TOOL = {
    "name": "render_world_area",
    "description": (
        "Read and render a bounded live area of the world, returning a PNG you can see. "
        "Use structure view for a tight build/object box and landscape view for terrain "
        "or spatial context. Choose bounds from retrieved evidence."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "dim": {"type": "string"},
            "min_x": {"type": "integer"}, "min_y": {"type": "integer"},
            "min_z": {"type": "integer"}, "max_x": {"type": "integer"},
            "max_y": {"type": "integer"}, "max_z": {"type": "integer"},
            "view": {"type": "string", "enum": ["structure", "landscape"]},
        },
        "required": ["dim", "min_x", "min_y", "min_z",
                     "max_x", "max_y", "max_z", "view"],
        "additionalProperties": False,
    },
}

ENTITY_TOOL = {
    "name": "inspect_world_entities",
    "description": (
        "Return an exact live census of entity types inside bounded world coordinates. "
        "Structure renders intentionally contain blocks only. Call this separately when "
        "occupancy or purpose matters—for example, after geometry suggests a pen—or when "
        "the player asks what creatures are present."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "dim": {"type": "string"},
            "min_x": {"type": "integer"}, "min_y": {"type": "integer"},
            "min_z": {"type": "integer"}, "max_x": {"type": "integer"},
            "max_y": {"type": "integer"}, "max_z": {"type": "integer"},
        },
        "required": ["dim", "min_x", "min_y", "min_z",
                     "max_x", "max_y", "max_z"],
        "additionalProperties": False,
    },
}


def _bounded_world_request(block) -> tuple[str, list[int], list[int]]:
    """Validate bounds shared by live visual and entity evidence tools."""
    values = block.input or {}
    lo = [int(values[f"min_{axis}"]) for axis in "xyz"]
    hi = [int(values[f"max_{axis}"]) for axis in "xyz"]
    spans = [hi[index] - lo[index] + 1 for index in range(3)]
    if any(span <= 0 for span in spans):
        raise ValueError("world evidence bounds must have min <= max")
    if any(span > MAX_RENDER_AXIS for span in spans):
        raise ValueError(f"each world evidence axis is limited to {MAX_RENDER_AXIS} blocks")
    if spans[0] * spans[1] * spans[2] > MAX_RENDER_VOLUME:
        raise ValueError(f"world evidence volume is limited to {MAX_RENDER_VOLUME} blocks")
    return str(values["dim"]).removeprefix("minecraft:"), lo, hi


def assistant_content(reply) -> list[dict]:
    """Turn SDK response blocks back into portable message parameters.

    Thinking blocks carry an API signature and must be replayed unchanged before tool
    results. Dropping them makes a real multi-tool exchange fail even though text-only fakes
    appear healthy.
    """
    content = []
    for block in reply.content:
        if hasattr(block, "model_dump"):
            content.append(block.model_dump(mode="json", exclude_none=True))
        elif block.type == "text":
            content.append({"type": "text", "text": block.text})
        elif block.type == "tool_use":
            content.append({"type": "tool_use", "id": block.id,
                            "name": block.name, "input": block.input})
    return content


def tool_result_block(block, result: dict) -> dict:
    """Format text or image evidence as one Anthropic client-tool result."""
    tool_result = {"type": "tool_result", "tool_use_id": block.id}
    if result.get("_image_base64"):
        metadata = {key: value for key, value in result.items()
                    if key != "_image_base64"}
        tool_result["content"] = [
            {"type": "image", "source": {
                "type": "base64", "media_type": "image/png",
                "data": result["_image_base64"],
            }},
            {"type": "text", "text": json.dumps(
                metadata, default=str, separators=(",", ":"))},
        ]
    else:
        tool_result["content"] = json.dumps(
            result, default=str, separators=(",", ":"))
    if result.get("error"):
        tool_result["is_error"] = True
    return tool_result


async def complete_message(client, timeout: float | None = None, **request):
    """Return a complete model turn, retrying rather than accepting cut-off prose.

    Adaptive thinking and visible text share ``max_tokens``. A difficult visual/tool turn can
    therefore exhaust a seemingly ample answer budget during reasoning and leave half a word
    as its final text. Partial output is not evidence and must never reach chat or memory.

    A caller's own ``max_tokens`` is honoured. It used to be overwritten with the dialogue
    default, which quietly broke the one caller that needs a much larger answer: a statue is
    several hundred setblock commands, so the builder asked for sixteen thousand tokens, got
    four, hit the ceiling on both attempts and raised. Every build failed, and the failure
    read as a timeout.

    ``timeout`` likewise, because how long is too long is a property of the job. Twenty-five
    seconds is right for a reply somebody is waiting on in chat and wrong for a build.
    """
    request = dict(request)
    ceiling = int(request.pop("max_tokens", 0) or MODEL_OUTPUT_TOKENS)
    larger = max(ceiling * 2, MODEL_RETRY_TOKENS)
    request["max_tokens"] = ceiling
    reply = await _within_deadline(client, request, timeout=timeout)
    if getattr(reply, "stop_reason", None) != "max_tokens":
        return reply
    print(f"model output reached {ceiling} tokens; retrying with {larger}", flush=True)
    request["max_tokens"] = larger
    reply = await _within_deadline(client, request, timeout=timeout)
    if getattr(reply, "stop_reason", None) == "max_tokens":
        raise RuntimeError("model output remained incomplete after a larger retry")
    return reply


#: The longest we will sit on a provider's own Retry-After. Past this it is not a pause,
#: it is an outage, and saying so beats a player watching nothing happen for ten minutes.
MAX_RETRY_AFTER_SECONDS = 150.0


def _retry_after(error) -> float:
    """How long the provider asked us to wait, if it asked at all.

    Read from the response header first and from the error body second, because the same
    refusal arrives both ways depending on which layer raised it.
    """
    headers = getattr(getattr(error, "response", None), "headers", None) or {}
    value = headers.get("Retry-After") or headers.get("retry-after")
    if value is None:
        body = getattr(error, "body", None)
        if isinstance(body, dict):
            inner = (body.get("error") or {}).get("metadata") or {}
            value = (inner.get("headers") or {}).get("Retry-After")
    try:
        return min(float(value), MAX_RETRY_AFTER_SECONDS) if value else 0.0
    except (TypeError, ValueError):
        return 0.0


async def _within_deadline(client, request: dict, attempts: int = 2,
                           timeout: float | None = None):
    """One model call, abandoned and retried if it stalls.

    The retry is a fresh request rather than a wait: a provider that has not answered in
    forty-five seconds is not about to, and the same prompt sent again is usually served by
    a different one in a couple of seconds.
    """
    last = None
    for attempt in range(attempts):
        started = time.monotonic()
        try:
            return await asyncio.wait_for(client.messages.create(**request),
                                          timeout=timeout or MODEL_TIMEOUT_SECONDS)
        except asyncio.TimeoutError as error:
            last = error
            print(f"model call exceeded {timeout or MODEL_TIMEOUT_SECONDS:.0f}s "
                  f"(attempt {attempt + 1} of {attempts}); "
                  f"{'asking again' if attempt + 1 < attempts else 'giving up'}",
                  flush=True)
        except Exception as error:  # noqa: BLE001 - a transient provider failure
            last = error
            if attempt + 1 >= attempts:
                raise
            # A provider that says "wait" means it. Retrying a rate or budget refusal
            # immediately spends the second attempt on the same answer: one build died to
            # a 402 carrying Retry-After 120 that we asked again 0.4 seconds later.
            wait = _retry_after(error)
            print(f"model call failed after {time.monotonic() - started:.1f}s "
                  f"({type(error).__name__}); "
                  f"{f'waiting {wait:.0f}s then asking again' if wait else 'asking again'}",
                  flush=True)
            if wait:
                await asyncio.sleep(wait)
    raise TimeoutError(
        f"model did not answer within {timeout or MODEL_TIMEOUT_SECONDS:.0f}s across "
        f"{attempts} attempts") from last


async def render_world_area(block, bridge_url: str | None) -> dict:
    """Return an entity-free bounded live render as an image-bearing tool result."""
    if not bridge_url:
        return {"error": "live world rendering is unavailable"}

    from assets import Assets
    from render import framed, landscape, sheet
    from scan import request_voxels, to_voxels

    try:
        dim, lo, hi = _bounded_world_request(block)
        values = block.input or {}
        view = str(values["view"])
        result = await request_voxels(bridge_url, dim, lo, hi)
        if not result.get("ok"):
            raise RuntimeError(result.get("error") or "voxel read failed")
        voxels = to_voxels(result)
        if not voxels:
            raise ValueError("the requested area contains no visible blocks")
        # The software renderer is seconds of pure CPU on a large patch. Left on the event
        # loop it stops everything else the god is doing, including answering whoever asked
        # for the picture in the first place.
        if view == "structure":
            shown = framed(voxels)
            picture = await asyncio.to_thread(sheet, shown, Assets())
        elif view == "landscape":
            shown = voxels
            picture = await asyncio.to_thread(landscape, voxels, Assets())
        else:
            raise ValueError("view must be structure or landscape")
        buffer = io.BytesIO()
        picture.save(buffer, format="PNG")
        png = buffer.getvalue()
        LATEST_EVIDENCE_RENDER.parent.mkdir(parents=True, exist_ok=True)
        LATEST_EVIDENCE_RENDER.write_bytes(png)
        encoded = base64.standard_b64encode(png).decode()
        print(f"world evidence render ({view}, {len(shown)} visible voxels): "
              f"{dim} {lo}..{hi}", flush=True)
        return {
            "ok": True, "view": view, "requested_bounds": {"min": lo, "max": hi},
            "visible_voxels": len(shown),
            "entities_in_image": False,
            "tile_order": (["perspective", "top", "front elevation", "side elevation"]
                           if view == "structure" else ["landscape"]),
            "note": ("Fresh live block-only render; image is attached before this metadata. "
                     "Use inspect_world_entities separately if occupancy matters."),
            "_image_base64": encoded,
        }
    except Exception as error:
        return {"error": f"{type(error).__name__}: {error}"}


async def inspect_world_entities(block, bridge_url: str | None) -> dict:
    """Return exact entity counts without drawing approximate creature geometry."""
    if not bridge_url:
        return {"error": "live entity inspection is unavailable"}

    from collections import Counter
    from scan import request_voxels

    try:
        dim, lo, hi = _bounded_world_request(block)
        result = await request_voxels(bridge_url, dim, lo, hi)
        if not result.get("ok"):
            raise RuntimeError(result.get("error") or "world read failed")
        entities = [entity for entity in result.get("entities") or []
                    if entity.get("pos") and all(
                        lo[index] <= entity["pos"][index] < hi[index] + 1
                        for index in range(3))]
        counts = Counter(str(entity.get("type") or "unknown") for entity in entities)
        return {
            "ok": True,
            "requested_bounds": {"min": lo, "max": hi},
            "total_entities": len(entities),
            "entity_counts": dict(sorted(counts.items())),
            "note": "Exact live census; entities were not added to the structure render.",
        }
    except Exception as error:
        return {"error": f"{type(error).__name__}: {error}"}
