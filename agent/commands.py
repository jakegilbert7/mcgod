#!/usr/bin/env python3
"""What the god may do to the world, said twice.

The gate that matters is in the plugin: ``CommandService`` decides what runs, and it decides
in the server because the thing drafting the command is a language model. Nothing here can
permit anything; this is the same opinion held on the agent's side so that a refusal is
immediate and well worded, and so that a model that drafts something forbidden is told why
without a round trip.

Because a mirror that drifts is worse than no mirror — it would explain a rule the server
does not have — the regression suite parses the Java and fails if these lists disagree.
"""

from __future__ import annotations

import re

#: Commands the god may run. Kept in step with CommandService.ALLOWED.
ALLOWED = frozenset({
    "give", "summon", "effect", "particle", "playsound", "stopsound", "title",
    "time", "weather", "xp", "experience", "enchant", "setblock", "fill",
    "clone", "tp", "teleport", "spawnpoint", "gamemode", "clear", "item",
    "ride", "damage", "attribute", "loot", "place", "fillbiome", "spreadplayers",
})

#: Never, whatever else changes. Kept in step with CommandService.FORBIDDEN.
FORBIDDEN = frozenset({
    "op", "deop", "ban", "ban-ip", "banlist", "pardon", "pardon-ip", "kick",
    "whitelist", "stop", "restart", "reload", "rl", "save-all", "save-off",
    "save-on", "execute", "function", "datapack", "debug", "jfr", "perf",
    "plugins", "pl", "version", "ver", "seed", "say", "tell", "msg", "w",
    "tellraw", "me", "trigger", "gamerule", "difficulty", "setidletimeout",
    "setworldspawn", "worldborder", "kill", "publish", "transfer",
})

#: The game's own fill limit, not a smaller one. A bridge across a chasm is a big fill, and
#: a god that cannot build one is not the thing being built.
MAX_BLOCKS = 32768
MAX_LENGTH = 512

_RELATIVE = re.compile(r"(?<![\w.])[~^]")
_NUMBER = re.compile(r"-?\d+")


def refusal(raw: str, anchored: bool = False) -> str | None:
    """Why this command may not run, or None if it may. Mirrors CommandService.inspect.

    ``anchored`` says whether a player was named to run the command from. Relative
    coordinates are meaningless from the console, where ``~`` is the world origin, and
    exactly right when the command runs at somebody: ice under your feet and fire above
    your head are relative by nature.
    """
    command = (raw or "").strip()
    while command.startswith("/"):
        command = command[1:].strip()
    if not command:
        return "no command given"
    if len(command) > MAX_LENGTH:
        return f"command longer than {MAX_LENGTH} characters"
    if "\n" in command or "\r" in command:
        return "a command is one line"
    root = command.split()[0].lower().removeprefix("minecraft:")
    if ":" in root:
        return "plugin commands are not available to the god"
    if root in FORBIDDEN:
        return f"{root} is never available"
    if root not in ALLOWED:
        return f"{root} is not one of the commands the god may run"
    if not anchored and _RELATIVE.search(command):
        return ("~ and ^ need somebody to be relative to: name an anchor player, or use "
                "absolute coordinates")
    if root in ("fill", "clone", "fillbiome") and not _RELATIVE.search(command):
        values = [int(v) for v in _NUMBER.findall(command)[:6]]
        if len(values) < 6:
            return "expected two absolute corners"
        volume = 1
        for axis in range(3):
            volume *= abs(values[axis + 3] - values[axis]) + 1
            if volume > MAX_BLOCKS:
                return f"that would change more than {MAX_BLOCKS} blocks at once"
    return None


def clean(raw: str) -> str:
    """The command as it will be sent: no leading slash, no surrounding space."""
    command = (raw or "").strip()
    while command.startswith("/"):
        command = command[1:].strip()
    return command


#: What the model is told it can do.
#:
#: Written to describe a capability rather than a permission list, because the failure to
#: avoid is a god that could have done something and did not think to. The two structural
#: facts it must know are the anchor (which is what makes "around me" expressible) and the
#: repeat (which is what makes "for a minute" expressible); without them it writes one-shot
#: absolute commands and quietly does a smaller thing than it was asked for.
GUIDANCE = f"""\
You can act on the world by running Minecraft commands, whenever the player asks you to do
something. There is no limit on how many you may run at once: a bridge is a thousand blocks
and a storm is a command every few ticks, and asking for either is normal.

For anything whose SHAPE matters — a statue, a house, an arch, anything figurative — ask
`design_build` first and run what it gives you. A builder holds a shape in mind block by
block; writing a humanoid figure yourself produces something nobody recognises.

For an ability rather than an object — anything the player should be able to DO, again and
again, by making a gesture — use `grant_power`. You invent the power there rather than
choosing one: you say which gesture triggers it and which commands it runs, so a web
shooter, a wake of ice, a thunder-clap on sneaking are all writable. Creative mode and a
stack of fire charges is not what anyone means by a superpower.

Use the `act_on_world` tool. It tells you what each command actually did, so you can look at
the world afterwards with `render_world_area`, see whether it worked, and put it right before
you say anything. Do that whenever the result is worth checking — clearing something, building
something, anything where "I tried" and "it worked" are different. For a simple act you are
sure of, `commands` in your final answer runs them without a tool round.

Anchoring. Set `anchor` to the player's name and every command runs from where they stand,
so ~ and ^ mean what they mean in game: `~ ~10 ~` is ten blocks above their head, `^ ^ ^3` is
three blocks in front of them. Without an anchor you are running from the console, where ~ is
the world origin and almost never what anyone meant. Anchor by default.

Lasting effects. Set `every` (ticks between repeats, 20 ticks is a second) and `duration`
(total ticks) and the commands repeat by themselves, from the anchor, following the player.
"Fire raining down on me for a minute" is `every: 5, duration: 1200`. "Ice under my feet for
five minutes" is `every: 2, duration: 6000`. The server runs it; you do not wait for it, and
it keeps going while you talk. Name it in `spell` if the player might want it stopped.

You may use: {', '.join(sorted(ALLOWED))}.

What is refused, and only this: anything that grants permission, removes a player, changes
the server's own settings, or prints to chat. Speech is what you say; a command is what you
do. `execute` is refused because anchoring already does what it is for.

Write commands as they would be typed, without the leading slash. Say plainly what you are
doing in `speech`; do not read the commands out."""
