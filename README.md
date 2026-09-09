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

you: give me the power of launching pigs
god: [writes one: right-click throws a pig where you look]
you: make them explode on impact
god: [rewrites the same power: on_hit summons primed tnt where the pig lands]
you: now make me bouncy
god: [on_land, impulse bounce, and no fall damage so the second landing
      does not kill you]
```

## How it works

```
Paper events ─► bounded queue ─► JSONL session files
                     │
                     ▼
              one WebSocket ──► live ingestion ──► event_history + indexes
                     │
                     └──► RPC: scans, voxels, seed queries, speech,
                              commands, powers
                                        │
                                        ▼
                              SQLite belief store
                                        │
                     ┌──────────────────┴──────────────────┐
                     ▼                                     ▼
              chat evidence loop                    dirty-region upkeep
              SQL / tools / narration               scan, segment, publish
```

Capture is dumb and fast. Listeners build a record, push it to a bounded queue, and return;
a daemon thread serialises and ships. A full queue drops events rather than stalling the
server, because the game stuttering is always worse than losing an event. Over forty event
types are captured: block work, movement, combat, containers, crafting, vehicles, chat, and
the god's own acts.

### Provenance is the whole design

Every row in the store carries `value, confidence, provenance, first_seen_tick,
verified_at_tick, verified_at_ms, source_event_ids`, and the provenance order is
load-bearing:

| | | assertable as fact? |
|---|---|---|
| `SCANNED` | a direct world read | yes |
| `OBSERVED` | the event stream | yes |
| `DERIVED` | deterministic computation | yes |
| `INFERRED` | a model's judgement | only above a confidence threshold |
| `ASSERTED` | something the god itself said | **never** |

`ASSERTED` can never be promoted, enforced by a database trigger rather than by convention,
so no future caller can route around it. Without that rule the god reads back its own
speculation as evidence and one confident guess becomes permanent canon.

Sixteen tables hold the world. The important separation is between what *happened*
(`event_history`, `work_events`, `episodes`, `activities`), what *stands* (`masses`,
`structures`, `generated_features`), and what the god merely *said* (`utterances`).
Confidence decays on read by a half-life chosen per kind of fact — twenty minutes for a
count of animals in a pen, fourteen days for a building's geometry — and decay never
rewrites what was measured.

### What stands: two tiers

A **ledger** (`masses`) is deterministic and complete. Every connected mass of built blocks
in ground that has been read is a row, whatever its size and whoever made it: houses,
scaffolding pillars, torch lines, village walls. Authorship comes from replaying the latest
block mutation at every coordinate and comparing it with what stands there now, so a mass
nobody was watching being built has an unknown builder and says so.

A **landmark** (`structures`) is a named grouping of masses, with boundaries drawn by a
vision model from rendered views, coordinate-bearing slice grids, and the measured geometry
code already extracted. Code clips and measures every proposal, refuses one that is empty or
runs off the edge of what was read, and matches identity to prior objects by geometry rather
than by name.

The model decides what deserves a name. It never decides what exists.

### Asking it things

One model-directed evidence loop answers everything. The model decides whether a question is
about the player's past, about what the world contains, or neither; writes bounded read-only
SQL against a table allowlist; may render the world, inspect entities, or query the seed; and
narrates what it found. There are no phrase gates and no canned answers. Both the loop and
every individual model call are bounded by wall clock, because a stalled provider is
indistinguishable from the thing being down.

### Acting on it

Three tools, gated in the server rather than in the agent, because the thing drafting is a
language model:

- **`act_on_world`** runs Minecraft commands, any number of them, optionally anchored to a
  player so `~` means what it means in game, and optionally repeating for a duration. An
  allowlist decides what may run; nothing that grants permission, removes a player, changes
  server settings or prints to chat is on it. Work is drained a bounded number per tick, so
  the request is unbounded but the pace is not.
- **`grant_power`** invents an ability rather than picking one from a list. The model writes
  what gesture triggers it and what commands it runs — right-click, sneak, move, land, attack,
  be hurt, a tick interval, or the moment something they threw arrives — and those commands run
  anchored to the player, so `^ ^ ^4` is four blocks along their line of sight. Beside them sit
  the primitives no command can express: flight without creative mode, immunity to a damage
  cause, walk speed, an entity or block thrown where they are looking, a traced hitscan beam
  that stops at the first thing it meets and never at its own shooter, and an impulse, since
  teleporting somebody somewhere cannot give them momentum and being bouncy is entirely
  momentum. Granting a name that is already held replaces it, which is how a power gets edited
  rather than duplicated. The scripted commands pass the same allowlist a direct command does,
  so binding one to a right-click is not a way around it.
- **`design_build`** asks a separate build model for the commands that make a shape, because
  holding a figure in mind is a different skill from conversation. It is given the site as a
  contour grid of surface heights and a second grid of surface materials rather than a
  picture: a render cannot say that this column is three blocks lower than that one, and a
  builder that cannot read the slope sets a statue's feet in the air on one side and buries
  them on the other.

All three live inside the same loop as the observation tools, so the god can act, look at
what happened, and put it right before it says anything. `/mcgod stop` cancels everything.

### What it knows without going there

Minecraft's generation is a pure function of the seed, so the whole world is computable
without visiting it. `seedmap.py` binds [cubiomes](https://github.com/Cubitect/cubiomes)
(MIT, vendored) carried to 26.2 by porting the game's own biome search out of the server
jar's bytecode. Measured against a live server: 46,315 sample points across all three
dimensions, zero mismatches. That is what makes "how many villages within 10,000 blocks"
answerable in milliseconds when the server itself can only find the nearest one.

### Rendering

A software renderer with perspective, z-buffering and textures read from the local client
jar. Block geometry comes from the game's own model JSON rather than from block names, so a
fence is a post with arms and a stair roof reads as a roof. Entity geometry is extracted by
walking the bytecode of the game's `createBodyLayer()` methods, because entity models ship as
Java rather than as data.

## The code

**Agent** (`agent/`)

| | |
|---|---|
| `god.py` | the live loop: startup sync, chat, the act-and-observe loop, quests, upkeep |
| `store.py` | SQLite schema, belief envelopes, provenance, decay, migrations |
| `history.py` | model-directed read-only SQL and the evidence tool loop |
| `masses.py` | the ledger of every built thing: measure, attribute, identity, roles |
| `structures.py` | canonical landmark identity, authorship, generated-world separation |
| `segment.py` | what counts as a structure, decided from the world not the event log |
| `grounding.py` | resolving what a question refers to, and spatial facts about it |
| `context.py` | context assembly under hard per-section token budgets |
| `activity.py` `episodes.py` `world.py` `detectors.py` | the deterministic layers: deed index, activity clustering, derived player state, and five fact-emitting detectors |
| `reconcile.py` | scan queue, dirty regions, verify-before-speaking |
| `seedmap.py` `environment.py` | the world from the seed, offline and via the server |
| `commands.py` `builder.py` `evidence.py` | what may be run, the build model, the shared tools |
| `quests.py` `router.py` | quest specs and watchers; interrupt / queued / ambient triage |
| `render.py` `assets.py` `entity_models.py` `render_structures.py` | the renderer and its inputs |
| `scan.py` `bridge.py` `consumer.py` `replay.py` | the wire: RPC, live stream, replay harness |
| `model_api.py` `config.py` | provider boundary for Anthropic and OpenRouter |
| `test_regression.py` | 310 offline tests |
| `cubiomes/` | vendored cubiomes, patched to 26.2 — see its `PROVENANCE.md` |

Also `classify.py`, `eval_segment.py`, `compare_models.py`, `survey.py`, `summary.py`,
`narrate.py` and `fakegod.py`: evaluation and one-off tools, not on the live path.

**Plugin** (`plugin/`) — a Paper plugin in about thirty classes. Listeners for blocks,
entities, players, inventories, vehicles, trades, chat and the world; `EventQueue` and
`EventDispatcher` behind them; `JsonlSink` and `WebSocketSink` in front; `ScanService` for
region and voxel reads, `EnvironmentService` for seed queries, `CommandService` and
`CommandRunner` for acting, `PowerService` for abilities.

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

Then talk to it in chat. Models are configured in `agent/.env`; the dialogue, classification,
vision, build and segmentation roles are separate variables so each can be swapped and
measured on its own.

```bash
.venv/bin/python test_regression.py    # 268 offline tests; no server, no API key
.venv/bin/python seedmap.py --count village --at 300 -300 --radius 10000
.venv/bin/python masses.py --audit
.venv/bin/python render_structures.py --check
```

### The corpus

Forty-two of the 310 tests replay recorded sessions, and those recordings are not in this
repository: a session file is a play-by-play of somebody's game with their player UUID on
every line. Without one those tests skip and the other 268 run. Play with the plugin
installed, then copy a file from `server/plugins/McGod/events/` into
`agent/sessions/corpus/`, and the whole suite runs against your own world.

## Layout

```
agent/      the Python side
  cubiomes/ vendored cubiomes (MIT), patched to 26.2
plugin/     the Paper plugin
server/     your local server — not in this repository
```

`HANDOFF-2026-09-07.md` is the operational snapshot and the place to start.
`SESSION-2026-09-07.md` is the development log, and is the honest one: it records what broke
and why, which is most of what was learned.

## Licence

MIT, except `agent/cubiomes/`, which is MIT from
[Cubitect/cubiomes](https://github.com/Cubitect/cubiomes) and carries its own licence file.

Not affiliated with Mojang or Microsoft. No server jar or world data is included here.
