# McGod

An LLM given a Minecraft world to know and to act on: a Paper plugin that watches everything
that happens, a Python agent that turns it into a world model, and a god that can answer
questions about anything you have ever done, then change the world in return.

It is not a chatbot with a Minecraft skin. The model never sees raw world data, cannot write
to the belief store, and cannot decide what is true. Deterministic code measures; the model
interprets; code validates what comes back. Most of the engineering here is about that line
and where it has to sit.

```
you: what did I build here
god: You built a cobblestone elephant sculpture with gold-block eyes and oak-button
     details, then set torches around it.

you: how many villages are within 10000 blocks of me
god: There are 272 villages within 10,000 blocks of you, including the one you're in.

you: give me superpowers, I want to fly and throw fire
god: [grants flight, fireball, fire immunity, a flame trail]
```

## How it works

Seven layers, each of which only trusts the one below it.

| | |
|---|---|
| **L0 capture** | 45 event types from a Paper plugin: block work, movement, combat, containers, vehicles, chat. Listeners push to a bounded queue and return; a daemon thread serialises and ships. A full queue drops events rather than stalling the server. |
| **L1 episodes** | Deterministic clustering of activity in space and time. A boundary falls where something measurably changed, never where a model said so. |
| **L2 primitives** | About twenty generic detectors that emit facts and never labels. Structure census, rate anomaly against a rolling baseline, firsts, risk profile. |
| **L3 store** | SQLite. Every row carries `value, confidence, provenance, first_seen, verified_at, source_event_ids`. |
| **L4 reconciliation** | Confidence decays on read by half-life; events contradict the beliefs they disprove; a named place is verified before it is spoken about. |
| **L5 context** | Hard token budget per section, enforced by truncation. Retrieval by salience, never dumping. |
| **L6 the god** | Trigger router with interrupt, queued and ambient lanes. Cheap model for triage, strong model for dialogue and judgment. |
| **L7 actions** | Commands, granted powers, and builds — gated in the server, never in the agent. |

### Provenance is the whole design

Every belief records where it came from, and the order is load-bearing:

| | | assertable as fact? |
|---|---|---|
| `SCANNED` | a direct world read | yes |
| `OBSERVED` | the event stream | yes |
| `DERIVED` | deterministic computation | yes |
| `INFERRED` | a model's judgement | only above a confidence threshold |
| `ASSERTED` | something the god itself said | **never** |

`ASSERTED` can never be promoted, enforced by a database trigger rather than by convention, so
no future caller can route around it. Without that rule the god reads back its own speculation
as evidence and one confident guess becomes permanent canon.

### What the god knows

Structures are decided in two tiers. A **ledger** (`masses`) is deterministic and complete:
every connected mass of built blocks in ground that has been read is a row, whatever its size
and whoever made it — houses, scaffolding pillars, torch lines, village walls. Authorship is
established by replaying the latest block mutation at every coordinate against what stands
there now. A **landmark** (`structures`) is a named grouping of masses, drawn by a vision model
from rendered and coordinate-bearing views.

The model decides what deserves a name. It never decides what exists.

Questions go through one model-directed evidence loop: the model writes bounded read-only SQL
against a table allowlist, may render the world or query the seed, and narrates what it found.
There are no phrase gates and no canned answers.

### What the god does not need to visit

Minecraft's generation is a pure function of the seed, so the whole world is computable without
going there. `seedmap.py` binds [cubiomes](https://github.com/Cubitect/cubiomes) (MIT, vendored)
carried to 26.2 by a port of the game's own biome search read out of the server jar's bytecode.
Measured against the live server: 46,315 sample points across all three dimensions, zero
mismatches. That is what makes "how many villages within 10,000 blocks" answerable in
milliseconds when the server itself can only find the nearest one.

### Rendering

A software renderer with perspective, z-buffering and textures read from the local client jar.
Block geometry comes from the game's own model JSON rather than from block names, so a fence is
a post with arms and a stair roof reads as a roof. Entity geometry is extracted by walking the
bytecode of the game's `createBodyLayer()` methods, because entity models ship as Java rather
than as data.

## Running it

Needs Java 25, Python 3.11+, and a Paper 26.2 server.

```bash
cd plugin && ./gradlew deploy          # builds and copies the jar into server/plugins
cd ../server && java -Xms4G -Xmx4G -jar paper.jar nogui

cd agent
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.lock
cp .env.example .env                   # then add your key
.venv/bin/python god.py
```

Then talk to it in chat.

### The corpus

Forty-two of the tests replay recorded sessions, and those recordings are not in this
repository: a session file is a play-by-play of somebody's game with their player UUID on
every line. Without one those tests skip and the other 252 run. Play with the plugin
installed, then copy a file from `server/plugins/McGod/events/` into
`agent/sessions/corpus/`, and they all run against your own world.

```bash
.venv/bin/python test_regression.py    # 252 offline tests; no server, no API key
.venv/bin/python seedmap.py --count village --at 300 -300 --radius 10000
.venv/bin/python masses.py --audit
.venv/bin/python render_structures.py --check
```

## Notes on the engineering

A few decisions that were not obvious and cost something to learn:

- **Net deltas, not gross counts.** A player who places 500 blocks and breaks 490 has built
  nothing. Gross counting is the likeliest source of a god confidently describing a structure
  that is not there.
- **Absence of evidence is refused.** A scan that finds nothing where a structure was believed
  to stand is a failed read, not a demolition. The same trap appears in half a dozen places.
- **Where a pipeline and its measurement share state, the measurement flatters the pipeline.**
  The segmentation evaluator seeded its truth from the store and so scored itself; it reported
  a perfect box for doing nothing.
- **A model that gates existence will delete things.** Segmentation once decided what was real,
  so anything it declined to name had no present-tense record at all, and 47% of block events
  could not be attached to any place.
- **Never block the tick.** Commands are queued and drained a bounded number per tick; the
  request is unbounded, the pace is not.

## Layout

```
agent/      the Python side: capture consumers, world model, retrieval, rendering, the god
  cubiomes/ vendored cubiomes (MIT), patched to 26.2 — see its PROVENANCE.md
plugin/     the Paper plugin: capture, scans, seed queries, commands, powers
server/     local server (not in this repository)
```

`HANDOFF-2026-09-07.md` is the operational snapshot. `SESSION-2026-09-07.md` is the development
log, and is the honest one: it records what broke and why, which is most of what was learned.

## Licence

MIT, except `agent/cubiomes/`, which is MIT from
[Cubitect/cubiomes](https://github.com/Cubitect/cubiomes) and carries its own licence file.

Not affiliated with Mojang or Microsoft. No server jar or world data is included here.
