#!/usr/bin/env python3
"""The builder: a model whose whole job is turning a description into blocks.

Building is not conversation. Holding a shape in mind, deciding where its shoulders and its
roofline go, and writing the several hundred commands that put them there is a different
skill from answering a question well, and the statues showed it: asked for a humanoid figure,
a dialogue model produced something nobody would recognise as one.

So it is a separate model behind a separate tool, configured by ``MCGOD_BUILD_MODEL`` (or
``MCGOD_OPENROUTER_BUILD_MODEL``). The dialogue model decides what to build and where; this
decides what blocks that is. Neither has to be good at the other's job, and either can be
swapped without touching the other.
"""

from __future__ import annotations

import json
import re

from config import BUILD_MODEL, thinking_for
from evidence import complete_message
from model_api import model_client

#: A build is written once and then looked at, so it may be large. This is generous rather
#: than tuned: the cost of a truncated build is a half-finished statue in the world.
MAX_BUILD_TOKENS = 16000

SYSTEM = """\
You build things in Minecraft, block by block, and you are given the job because shape is
what you are good at.

You will be told what to build, where its anchor point is, and sometimes which way it faces
and what it is made of. Reply with the commands that build it and nothing else.

How to build well:
- Work in absolute coordinates from the anchor you were given. The anchor is the point the
  thing stands on, at its centre unless told otherwise.
- Prefer `setblock` for anything whose shape matters and `fill` for slabs, walls and solid
  runs. A figure made of fills looks like boxes, because it is boxes.
- Build in the order a person would: footing, then mass, then the details that make it
  recognisable. Put the recognisable parts in — a face, hands, a roofline, a doorway — since
  those are what tell someone what they are looking at.
- Use the block palette to shade. Different stones, wools, terracottas and concretes read as
  light and shadow, and a single material reads as a lump.
- Anything humanoid or animal needs proportion above all: head roughly an eighth of the
  height, shoulders wider than the head, limbs that reach where limbs reach. Get the
  silhouette right before any detail.
- Clear the space first if the ground would swallow it.

Reply as JSON only:
{"commands": ["setblock 100 64 100 minecraft:stone", ...],
 "describes": "<one short sentence naming what you built>"}"""


def _plain(reply) -> str:
    return "".join(block.text for block in reply.content if block.type == "text").strip()


async def design(what: str, anchor, facing: str = "", materials: str = "",
                 model: str | None = None) -> dict:
    """Ask the builder for the commands that make one thing.

    Returns ``{"commands": [...], "describes": str}``, or ``{"error": ...}``. Nothing here
    validates the commands: they go through the same gate as anything else the god runs, and
    a build that names a forbidden command is refused there like any other.
    """
    model = model or BUILD_MODEL
    client = model_client(asynchronous=True)
    request = {
        "build": what,
        "anchor": [int(anchor[0]), int(anchor[1]), int(anchor[2])],
        "facing": facing or "unspecified",
        "materials": materials or "your choice",
    }
    try:
        reply = await complete_message(
            client, model=model, thinking=thinking_for(model), system=SYSTEM,
            max_tokens=MAX_BUILD_TOKENS,
            messages=[{"role": "user", "content": json.dumps(request, indent=1)}])
    except Exception as error:  # noqa: BLE001 - a builder failure is not a world failure
        return {"error": f"the builder did not answer ({type(error).__name__})"}
    text = _plain(reply)
    try:
        body = json.loads(text[text.find("{"):text.rfind("}") + 1])
    except (ValueError, json.JSONDecodeError):
        return {"error": "the builder did not return commands"}
    commands = [str(c) for c in (body.get("commands") or []) if str(c).strip()]
    if not commands:
        return {"error": "the builder returned no commands"}
    return {"commands": commands, "describes": str(body.get("describes") or what),
            "model": model, "count": len(commands)}
