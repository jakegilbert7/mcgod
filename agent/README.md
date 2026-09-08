# agent

Python side of the McGod harness.

## Setup

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.lock
cp .env.example .env       # then configure either provider below
```

`.env` is gitignored. Anthropic remains the default: fill `ANTHROPIC_API_KEY` and leave
`MCGOD_MODEL_PROVIDER=anthropic`. To use OpenRouter instead:

```dotenv
MCGOD_MODEL_PROVIDER=openrouter
OPENROUTER_API_KEY=sk-or-...
MCGOD_OPENROUTER_CLASSIFY_MODEL=openai/gpt-5.6-luna
MCGOD_OPENROUTER_DIALOGUE_MODEL=openai/gpt-5.6-luna
MCGOD_OPENROUTER_VISION_MODEL=google/gemini-3.7-flash
MCGOD_OPENROUTER_REASONING_EFFORT=low
```

Restart the Python agent after switching. The Minecraft server and plugin do not need to be
restarted. `MCGOD_CLASSIFY_MODEL` and `MCGOD_DIALOGUE_MODEL` remain the Anthropic model
settings; the separate OpenRouter names prevent one provider's model IDs from breaking the
other. The vision model handles structure boundary/name passes while Luna remains the cheap
dialogue and classification model. `MCGOD_SEGMENT_MODEL` is still a one-run override for
comparative evaluation. Exported environment values take precedence over `.env`.

For an environment created before OpenRouter support was added, install the updated locked
requirements with `python -m pip install -r requirements.lock`. You can always bypass an
accidentally active shell environment and run the project environment directly with
`.venv/bin/python god.py`.

## Tools

| Script | Purpose |
|---|---|
| `bridge.py` | P2 — consume the live event stream from a running server |
| `replay.py` | P3 — replay a recorded session through the same consumers |
| `summary.py` | Summarise a recorded session and flag recording problems |
| `episodes.py` | P4 — group a session into episodes (L1) |
| `store.py` | P5 — the SQLite belief store (L3) |
| `scan.py` | P5 — ask a running server to scan a region |
| `detectors.py` | P6 — L2 primitives: structure census, rate anomaly, firsts |
| `context.py` | P7 — assemble the god's context under hard token budgets (L5) |
| `grounding.py` | resolve chat references and derive/scan spatial relationships |
| `fakegod.py` | P7 — budget audit and sufficiency test over the corpus |
| `classify.py` | name structures by looking at them; `--blind` for measurements only |
| `config.py` | reads `.env` |
| `model_api.py` | provider boundary for Anthropic Messages and OpenRouter chat completions |
| `segment.py` | what counts as a structure, decided spatially from the world |
| `masses.py` | the complete ledger of built masses: `--list`, `--audit`, `--rebuild`, `--roles` |
| `seedmap.py` | the world as the seed generates it, offline: `--nearest`, `--count`, `--biome`, `--check` |
| `commands.py` | what the god may run, mirroring the plugin's gate |
| `builder.py` | the build model: a description in, the commands that make it out |
| `cubiomes/` | vendored cubiomes (MIT), patched to 26.2; see `cubiomes/PROVENANCE.md` |
| `world.py` | Derived world state: actors, capability, ownership (L3) |
| `router.py` | P9 — L6 trigger router: interrupt / queued / ambient |
| `narrate.py` | Ask the god to describe the world from its own context |
| `reconcile.py` | L4 — scan queue, dirty regions, verify-on-reference |
| `quests.py` | P9 — constraint specs, O(1) watchers, deterministic judgment |
| `assets.py` | block geometry and textures from the client jar |
| `render.py` | software renderer: perspective, textured, shaded |
| `render_structures.py` | archive stale images and render canonical current structures |
| `god.py` | the live loop — watches, issues quests, judges, speaks, answers chat |
| `test_regression.py` | 294 offline tests over the corpus — no server, no API key |

### bridge.py — live

```bash
python3 bridge.py                      # print every event, tee to sessions/
python3 bridge.py --hide move          # drop the 2s position samples
python3 bridge.py --no-latency         # canonical rendering, diffable against replay
```

Ctrl-C prints the latency summary (plugin emit -> Python receipt). Reconnects on its own.

### replay.py — recorded

```bash
python3 replay.py sessions/corpus/01-house.jsonl --speed 50
python3 replay.py session.jsonl --speed 0      # unpaced
```

Pacing follows the recorded wall clock, scheduled against absolute deadlines so long
sessions at high speed do not drift.

### summary.py — check a recording

```bash
python3 summary.py sessions/corpus/01-house.jsonl
```

Reports implied TPS, tick ordering, move cadence, event counts, net block delta,
pickups, region entries with a flapping check, and deaths. Run it on every session as
you record it.

### episodes.py — L1

```bash
python3 episodes.py sessions/corpus/01-house.jsonl
python3 episodes.py session.jsonl --gap 1800 --jump 24
```

Groups a session into variable-length episodes. Boundaries fall on a 90s gap in substantive
activity, block work resuming more than 24 blocks away, or a dimension change — never on
what the activity looked like, since classifying it is the model's job and letting it drive
segmentation would make model output define facts.

Emits facts only: bbox, tick range, gross placed and broken, net delta by material, tool
profile, distance travelled, and a stable id. Naming an episode "a house" happens later.

### store.py / scan.py — L3

One local SQLite file, `mcgod.db`. No server, no account, no network.

```bash
python3 scan.py --min 170 60 270 --max 195 75 305 --slices
python3 scan.py --min 170 60 270 --max 195 75 305 --store house_01
```

Every row carries provenance: SCANNED, OBSERVED, DERIVED, INFERRED, ASSERTED. `ASSERTED` is
what the god itself said and can never be promoted — enforced by a database trigger, not by
convention. `store.assertable(table, id)` is the gate to check before the god states
something as fact.

Scans return a histogram, a block-entity census and character-grid slices. Never block
arrays: one chunk is ~98,000 block states.

## Talking to the god

With `god.py` running, type in Minecraft chat. It answers from everything it knows:

```
what have i built?
what quests do i have?
why did you ask me for that?
give me a different quest
```

Built for testing. The quickest way to find out what the god actually believes, and it will
say plainly when it does not know.

Historical questions use a model-directed evidence loop. The model writes read-only SQL over
the permanent event ledger and may follow the results into generated places, player-built
structures, relationships, cached terrain, or the live seed-backed environment service before
answering. When appearance matters, it may also choose bounds for a live world render and see
the resulting PNG before answering. The ordinary dialogue path has the same visual tool; no
keyword router decides when it is available. Coordinates and database mechanics stay out of
in-game speech.

## Tests

```bash
.venv/bin/python test_regression.py  # 170 tests, offline, no API calls
```

No server, no API key, no network. The corpus is the fixture. Run it before and after any
change to `episodes.py`, `detectors.py`, `context.py` or `store.py`.

## Canonical structure renders

`structures` contains only objects observed in the current world. Historical placement and
removal clusters live in `work_events`; they are useful evidence, but are never presented as
things that still stand.

```bash
python3 render_structures.py --list       # no server required
python3 render_structures.py --clean      # server required to regenerate the sheets
python3 render_structures.py --check      # DB/render invariants; no server required
```

The cleanup is recoverable: obsolete PNGs move under `renders/archive/<timestamp>/`. A
successful run leaves one top-level PNG per current structure. Open those sheets first when
judging segmentation quality.

## Why the two drivers matter

`bridge.py` and `replay.py` are two drivers of one `EventConsumer` interface, so every
layer above L0 can be developed against a recording rather than a running server and a
person playing it.

That substitution is only sound if replay is faithful, so the default rendering carries
no timing column and is a pure function of the event stream. The P3 acceptance test is
therefore mechanical — feed one session down both paths and diff:

```bash
python3 bridge.py --no-latency > live.txt      # against a running server
python3 replay.py <that session> --speed 0 > replay.txt
diff live.txt replay.txt                       # must be empty
```

Verified: identical rendered output, identical teed streams, output independent of
`--speed`, and a replay tee byte-identical to the plugin's own JSONL.

## Layout

- `consumer.py` — `EventConsumer` and implementations: `Printer`, `JsonlTee`,
  `LatencyMeter`, `Fanout`. Transport-agnostic on purpose.
- `bridge.py`, `replay.py`, `summary.py` — entry points.
- `sessions/` — teed output from `bridge.py`.
- `sessions/corpus/` — the recorded corpus. This is the test suite for every phase
  after P3; treat it as source, not scratch.
