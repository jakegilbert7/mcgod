#!/usr/bin/env python3
"""Model-directed, read-only retrieval over action and world evidence.

There is one production path for historical questions: the dialogue model chooses which
read-only evidence tools to call, writes every SQL query itself, and then narrates the tool
results. Code only enforces read-only access and resource bounds. No phrase router or
deterministic deed renderer participates.
"""

from __future__ import annotations

import json
import os
import re
import time
import sqlite3

from config import DIALOGUE_MODEL, thinking_for
from evidence import (ENTITY_TOOL, VISUAL_TOOL, assistant_content, complete_message,
                      inspect_world_entities, render_world_area, tool_result_block)
from model_api import model_client

MAX_QUERY_ROWS = 1500
MAX_RESULT_CHARS = 300_000
MAX_TOOL_ROUNDS = 6

#: How long the whole evidence loop may spend gathering before it must answer.
#:
#: Per-call deadlines bound one request; six rounds of them still add up to minutes. A
#: player who asks a question and waits that long has been given no answer, whatever
#: arrives eventually. Past this the model is asked once more without tools, so it must
#: answer from the evidence it already has.
ANSWER_BUDGET_SECONDS = float(os.environ.get("MCGOD_ANSWER_BUDGET", "40"))
MAX_ENVIRONMENT_RADIUS = 4096

QUERYABLE_TABLES = frozenset({
    "activities", "actor_presence", "episodes", "event_history", "generated_features",
    "masses", "players", "relationships", "structures", "terrain_regions", "work_events",
})

HISTORY_SYSTEM = """\
You retrieve evidence and answer questions about a Minecraft player's lived history.
Decide semantically whether CURRENT_QUESTION asks about something the player previously did,
experienced, changed, obtained, visited, made, fought, rode, or that just happened to them.
Recent player messages are supplied so fragments and follow-ups inherit their meaning.

You also answer questions about what the WORLD contains, including ground nobody has ever
visited: how many villages lie within some distance, where every ancient city near here is,
what biome a far-off place has, where the strongholds or world spawn are. Those are answered
from the seed with query_seed_map, not from the tables, because the tables only hold what
has been found so far.

Return exactly NOT_HISTORY, without calling a tool, when the question is neither about the
player's past nor about what the world contains: a greeting, a request for your opinion or
advice, or a question about the single thing they are standing in or looking at this moment,
which other machinery answers.

What someone has BUILT or DONE is yours, however it is phrased and however local it is.
"What did I build here", "everything within 100 blocks", "how much have I mined" are all
historical, and the retrieval rules below say how to answer them.

Above all, return NOT_HISTORY for anything ASKING YOU TO DO SOMETHING. "Give me a sword",
"make it rain", "build me a bridge", "put ice under my feet" are requests to act, and acting
is done elsewhere with your own hands. They are not questions about the past merely because
you could rephrase them as "have I been given a sword"; a player asking for a thing wants the
thing, not an account of whether they already have one. Recognise the request and step aside
immediately, without looking anything up.

If the question is historical, you MUST call query_database at least once and retrieve
event_history evidence before answering. If it is about the world, you MUST retrieve seed or
environment evidence before answering. You may make several tool calls. Choose and sequence the available
sources yourself until you have the best supported picture; do not stop at the first plausible
rows when a place, structure, terrain, or generated-world lookup would resolve their meaning.
Use SQL joins, subqueries, and CTEs to retrieve evidence whose relationship is already known
in one call, and issue independent tool calls together. Spend another sequential round only
when a result reveals an ID, time boundary, or location that was not knowable beforehand.
render_world_area is visual evidence, not decoration. Use it whenever the question asks what
a newly built or modified object is, how it looks, whether it resembles something, or requires
another judgment that block events and labels cannot settle. Derive its bounds from database
evidence and look before naming the build. You may also call it for any other place where a
fresh view would materially improve the answer.
Structure renders contain blocks only. If the geometry suggests a pen, stable, occupied
building, or another interpretation whose purpose depends on living things, call
inspect_world_entities separately before making that claim. Do not infer occupancy from a
creature merely standing near the requested bounds.
For a structure view, bound the subject by its primary structural materials and attached
details. Do not enlarge the subject to include a peripheral torch ring, paths, landscaping,
or unrelated nearby work; request a separate landscape view if that context matters. When a
render supports a recognizable subject, name it specifically. If it is genuinely ambiguous,
say what it resembles rather than replacing the visual judgment with dimensions.
For voxel sculpture, compare the silhouette, proportions, and distinctive appendages across
all rendered views before categorizing it. First account internally for every major protrusion
and indentation; do not classify a quadruped from its body and legs while ignoring its face,
ears, tail, horns, wings, or another diagnostic part. Distinguish appendages by geometry: for
example, a snout projects mostly outward at head height, while a trunk descends from the face.
Then name the most likely specific subject. If one subject is favored but not certain, say it
is subject-like or subject-shaped. Choose the single best-supported interpretation rather
than listing alternatives. Use a generic class such as "animal" only when the image truly
favors no more specific interpretation.

Database schema:
  event_history(id, actor, dim, tick, event_t, kind, place_id, x, y, z, payload)
    lossless chronological observed events; payload is the complete JSON event. Rows with
    actor='world' are things that happened rather than things the player did: explosions,
    weather, and mob_death. Rows with actor='god' are commands YOU ran, carrying command,
    reason, ok and the server's own output; they have no meaningful position. If asked what
    you did or gave, that is where to look, and it is observed fact rather than something
    you merely said.
  generated_features(id, dim, kind, x, y, z, min_x, min_y, min_z,
                     max_x, max_y, max_z, materials, value)
    generator-authored villages and their scanned component buildings. JSON value can include
    parent, role_evidence, and origin. This is separate from player-built structures.
  structures(id, actor, dim, min_x, min_y, min_z, max_x, max_y, max_z,
             materials, value)
    current player-built segmented structures only. JSON value includes comprises: the
    ids of the masses it is made of.
  masses(id, dim, actor, place_id, min_x, min_y, min_z, max_x, max_y, max_z,
         materials, value)
    the complete ledger of every connected built thing standing in ground that has been
    read: named landmarks and also pillars, bridges, scaffolds, torch lines, paths, and
    generated buildings. Complete materials. actor is the builder when block history
    supports one. place_id is the structure or generated feature it belongs to, else NULL;
    a mass with a non-null place_id is PART of that place, not a thing beside it.
    JSON value includes blocks, dims, fill, levels (blocks per height), footing, origin
    (player_built | world_generated | unknown), builders (actor -> share), observed_share,
    first_built_ms, last_edited_ms, edits (separate visits), placed, broken, partial
    (faces cut by the read edge), lineage. relationships role:<id> says what an unnamed
    mass seems to be for (landmark|utility|decoration|debris|unknown) with a noun.
  relationships(id, subject, predicate, object, value)
    names, containment, modification and other semantic relations for either kind of place.
  terrain_regions(id, dim, center_x, center_z, radius, step, biomes, value)
    cached seed-backed biome samples.
  activities(id, actor, dim, tick, event_t, kind, place_id, x, y, z, detail)
    optional interpreted index; event_history remains authoritative.
  episodes(id, actor, dim, tick_start, tick_end, t_start, t_end, min_x, min_y, min_z,
           max_x, max_y, max_z, gross_placed, gross_broken, net_total, net_by_material,
           tools, event_counts, path_blocks, value)
  work_events(id, kind, actor, dim, min_x, min_y, min_z, max_x, max_y, max_z,
              materials, value)
  players(id, name, dim, last_seen_tick, profile, value)
  actor_presence(id, actor, dim, place_id, event_t, value)

Retrieval rules:
- CURRENT_ACTOR identifies the player. Constrain the player's event rows to that actor. Rows
  with actor='world' may be added only to explain a tightly related time, place, or vehicle.
- CURRENT_POSITION is internal evidence for here, this place, nearby, or a locative follow-up.
- Unless the words impose a time bound, search all recorded time. Never silently substitute
  the current session or a recency window. For "just happened", use CURRENT_TIME_MS for a
  short interval; for "ever" or "everything", search the full history.
- Usually retrieve payload, event_t, kind, place_id and coordinates for action rows, ordered
  by event_t. Include move/player_state when route, vehicle, terrain, health, inventory, or
  elapsed state matters; do not discard them merely because they are low-level.
- generated_features holds two different things. A row with no parent in its JSON value is a
  generated PLACE (a village site, a temple) that the seed service happened to locate for an
  earlier question; it is not a survey, so a count of such rows is "how many I have located
  so far", never "how many exist". A row with a parent is a BUILDING inside a place (a
  church, a blacksmith) and is never counted as a place. To count or find generated places
  over an area, use inspect_seed_environment; if its reach is smaller than the area asked
  about, say so instead of counting rows.
- If the question names or describes a build or place ("my elephant", "the tower"), look it
  up before concluding it is unrecorded: relationships with predicate 'named' (object LIKE
  the words), then structures and masses by id. A thing on record that the player asks
  "where is" has an answer in its box, described by direction and distance.
- To enumerate what someone has built somewhere — "what have I built here", "everything
  within N blocks" — the answer is the structures in that area plus the masses there whose
  place_id IS NULL. Those two sets are disjoint and together they are everything that
  stands. Do not also enumerate from block_place rows: they are history, they include what
  was later torn down, and every object they suggest is already one of the rows above, so
  listing both names the same thing twice. Use event_history for when a thing was built,
  who built it, and how it changed, never for what is there.
- Name things by what they are, not what they are made of. A structure's stored name and a
  mass's role noun are the words to use; fall back to materials only when neither exists.
- Distances are measured from CURRENT_POSITION. A radius is a circle: filter to it rather
  than reporting everything in the square box the SQL selected.
- structures.materials and masses.materials are re-measured from the world on every read,
  so they are the current contents of a build; a question about what is in or on a thing
  now reads them, and may render the area to see it.
- Resolve meaningful non-null place_id values. Join or follow them into generated_features,
  structures or masses and relationships. For "who built", "when", "how big", "what is it
  made of", or "how many times edited" about any built thing, masses is the authority; a
  question about an unnamed thing (a pillar, a bridge) resolves through masses and its role. An event inside a generated child building should be described
  using its generator role when known. A generated kind or role_evidence such as village_church
  is evidence of original function; an inferred visual name describes current appearance and
  must not erase that role. Player edits remain deeds, never authorship of the generated place.
- A seed-map result that coincides with a place already on record carries known_as, and the
  one the player is standing in is marked you_are_here. The nearest village to someone
  standing in a village is that village: when they ask for a different one, skip the marked
  entry rather than reporting it as another place a few blocks away.
- query_seed_map answers questions about ground nobody has visited: how many of a structure
  exist within a distance, all of them, the nearest of a type at any range, the biome at a
  far coordinate, the nearest patch of a named biome, which biome dominates an area, world
  spawn, strongholds. Its range far exceeds inspect_seed_environment's, so "not found
  within N blocks" from that tool is a reason to ask this one, never an answer on its own. It is computed from the seed, so it describes
  generation and never the present, and it is the only way to COUNT. Do not count rows in
  generated_features to answer "how many exist"; those rows are only what has been located
  or visited so far.
- inspect_seed_environment can supply biomes and generator locations around an evidence
  coordinate. Use it when terrain or an unrecorded generated place matters. Biome alone does
  not prove that a particular vehicle was on land or water; prefer captured surface state.
- Coordinates, UUIDs, timestamps, SQL, and telemetry are internal evidence and must never be
  spoken unless the player explicitly asks for coordinates. That covers every notation: a
  bracketed pair, "X -1332, Z 5716", "at 300, -300", or a range. Say where a place is by
  direction and distance from the player, and by what stands near it. Do this silently:
  never write that coordinates are omitted, withheld, or unavailable, and never leave a
  placeholder where one would have gone.
- A render shows current appearance, not necessarily the historical appearance at event time.
  Combine it with the event interval; do not attribute later changes to the earlier builder.

Evidence semantics:
- Infer ordinary causal sequences without inventing missing acts: a block break followed by
  its pickup means the player took it; gathered ingredients followed by crafting can be one
  sequence.
- container_open merchant proves only looking at wares; only villager_trade proves a trade.
- Material names in payloads carry their namespace and may carry a block state:
  "minecraft:chest", "minecraft:oak_stairs[facing=east,half=bottom]". Equality against a
  bare name therefore matches nothing and quietly returns an empty set rather than an
  error. Match with LIKE '%chest%', or strip both sides before comparing.
- Taking from your own chest is not stealing. A container_take carries the CONTAINER's
  coordinates, so a container is the player's own when a block_place by that same actor
  exists at those exact x, y and z with an `after` that is a container (chest, barrel,
  shulker box, hopper, dispenser, furnace). A loot_table on the event marks a generated
  container, which is never theirs. For stole, looted, robbed, or took from someone, count
  only takings from containers they did not place; for "what have I collected", all of them
  count. Most takings are usually from the player's own storage, so answering with the raw
  container_take total overstates theft several times over.
- damage_dealt proves a fight; only mob_kill proves a kill BY THE PLAYER.
- mob_death is a death the player did not cause: `killer` names the creature responsible
  when there was one, `cause` names how it died otherwise, and `name` is its custom name if
  it had one. A creeper blast beside the player also kills the sheep standing there, so when
  describing an explosion or a fight, check for mob_death at the same time and place; it is
  what the blast cost. Its actor is 'world', so it is not one of the player's deeds.
- only sleep with result=ok proves successful sleep.
- vehicle_passenger_enter or a vehicle occupancy snapshot proves a mob rode in a vehicle.
  A player and mob sharing vehicle_id connects their rides. vehicle_in_water and
  vehicle_surface distinguish land from water; biome does not.

Once evidence is sufficient, answer in human deeds, not event names or database language.
Merge repetition, preserve every distinct meaningful act requested, and distinguish what the
player did from what happened to them. Do not announce limitations or describe your lookup.
Return only plain speech, no markdown or preamble, at most three compact sentences and 800
characters.
"""

TOOLS = [
    {
        "name": "query_database",
        "description": (
            "Execute one read-only SQLite SELECT over the approved action and world-state "
            "tables. Write SQL tailored to the player's question. Call again to follow place "
            "IDs, structure bounds, vehicle IDs, time intervals, or other evidence."
        ),
        "input_schema": {
            "type": "object",
            "properties": {"sql": {"type": "string"}},
            "required": ["sql"],
            "additionalProperties": False,
        },
    },
    {
        "name": "inspect_seed_environment",
        "description": (
            "Inspect seed-backed biomes and generated structure locations around an internal "
            "world coordinate. Use evidence coordinates, not guesses. This does not inspect "
            "player-built structures and does not by itself prove a precise surface crossing."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "dim": {"type": "string"},
                "x": {"type": "integer"},
                "z": {"type": "integer"},
                "radius": {"type": "integer", "minimum": 16,
                           "maximum": MAX_ENVIRONMENT_RADIUS},
                "step": {"type": "integer", "minimum": 4, "maximum": 64},
                "features": {"type": "array", "items": {"type": "string"},
                             "maxItems": 8},
            },
            "required": ["dim", "x", "z"],
            "additionalProperties": False,
        },
    },
    {
        "name": "query_seed_map",
        "description": (
            "Compute the world as the seed generates it, without anyone having visited it "
            "and without the server. This is the only source that can answer how MANY of "
            "something exist over a wide area, or list them all, rather than where the "
            "nearest one is. Use it for questions about unexplored ground: how many "
            "villages are within some distance, where every ancient city near here is, "
            "what biome some far-off coordinate has, where the nearest patch of some biome "
            "is at any range, which biome is most common around a point, where the "
            "strongholds are, where the world spawn is.\n"
            "Its reach is not the reach of inspect_seed_environment: that tool asks the "
            "running server and stops at a few thousand blocks, while this computes "
            "directly from the seed and searches tens of thousands. When the server-backed "
            "tool finds nothing within its range, come here rather than reporting that "
            "nothing is there.\n"
            "It describes generation, never the present: it does not know what anyone has "
            "built, mined, looted or destroyed. Every result carries an accuracy field; "
            "when it is not 'exact', confirm a specific position with "
            "inspect_seed_environment before stating that position as fact."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "mode": {"type": "string",
                         "enum": ["nearest", "all", "count", "biome", "nearest_biome",
                                  "biome_distribution", "spawn", "strongholds"]},
                "structure": {"type": "string",
                              "description": "for nearest/all/count, e.g. village, "
                                             "ancient_city, mansion, fortress, end_city"},
                "biome": {"type": "string",
                          "description": "for nearest_biome, e.g. ice_spikes, "
                                         "mushroom_fields, cherry_grove"},
                "step": {"type": "integer", "minimum": 4, "maximum": 256,
                         "description": "sampling interval in blocks for biome searches; "
                                        "smaller finds smaller patches and costs more"},
                "x": {"type": "integer"},
                "z": {"type": "integer"},
                "y": {"type": "integer", "description": "for mode=biome only; default 63"},
                "dim": {"type": "string",
                        "description": "for mode=biome; overworld, the_nether or the_end"},
                "radius": {"type": "integer", "minimum": 16, "maximum": 50000},
                "limit": {"type": "integer", "minimum": 1, "maximum": 40,
                          "description": "how many positions to list; the count is always "
                                         "complete regardless"},
            },
            "required": ["mode"],
            "additionalProperties": False,
        },
    },
    VISUAL_TOOL,
    ENTITY_TOOL,
]


class UnsafeHistoryQuery(ValueError):
    pass


def _response_text(reply) -> str:
    return "".join(block.text for block in reply.content if block.type == "text").strip()


def clean_select(raw: str, *, require_history: bool = True) -> str:
    """Extract and validate one read-only SELECT generated by the model."""
    text = raw.strip()
    if text == "NOT_HISTORY":
        return text
    if text.startswith("```"):
        text = re.sub(r"^```(?:sql)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text).strip()
    text = text.rstrip().rstrip(";").strip()
    if ";" in text:
        raise UnsafeHistoryQuery("multiple SQL statements are not allowed")
    if not re.match(r"^(?:SELECT|WITH)\b", text, re.IGNORECASE):
        raise UnsafeHistoryQuery("history query must be SELECT or WITH ... SELECT")
    forbidden = re.search(
        r"\b(?:INSERT|UPDATE|DELETE|REPLACE|DROP|ALTER|CREATE|ATTACH|DETACH|"
        r"PRAGMA|VACUUM|REINDEX|ANALYZE|LOAD_EXTENSION)\b", text, re.IGNORECASE)
    if forbidden:
        raise UnsafeHistoryQuery(f"forbidden SQL operation: {forbidden.group(0)}")
    if require_history and not re.search(r"\bevent_history\b", text, re.IGNORECASE):
        raise UnsafeHistoryQuery("historical SQL must read event_history")
    return text


def execute_select(store, sql: str, *, require_history: bool = False) -> dict:
    """Execute generated SQL with a table allowlist, row cap, and instruction budget."""
    safe = clean_select(sql, require_history=require_history)
    if safe == "NOT_HISTORY":
        raise UnsafeHistoryQuery("NOT_HISTORY is not executable")
    wrapped = f"SELECT * FROM ({safe}) AS generated_history_query LIMIT {MAX_QUERY_ROWS}"
    calls = 0

    def progress():
        nonlocal calls
        calls += 1
        return 1 if calls > 10_000 else 0

    allowed_actions = {sqlite3.SQLITE_SELECT, sqlite3.SQLITE_READ, sqlite3.SQLITE_FUNCTION,
                       sqlite3.SQLITE_RECURSIVE}

    def authorize(action, arg1, _arg2, _database, _source):
        if action not in allowed_actions:
            return sqlite3.SQLITE_DENY
        if action == sqlite3.SQLITE_READ and arg1 not in QUERYABLE_TABLES:
            return sqlite3.SQLITE_DENY
        return sqlite3.SQLITE_OK

    store.db.set_authorizer(authorize)
    store.db.set_progress_handler(progress, 1_000)
    try:
        cursor = store.db.execute(wrapped)
        names = [column[0] for column in cursor.description]
        raw_rows = [dict(zip(names, row)) for row in cursor.fetchall()]
    finally:
        store.db.set_progress_handler(None, 0)
        store.db.set_authorizer(None)

    kept = []
    used = 0
    for row in raw_rows:
        encoded = json.dumps(row, default=str, separators=(",", ":"))
        if kept and used + len(encoded) > MAX_RESULT_CHARS:
            break
        kept.append(row)
        used += len(encoded)
    return {"sql": safe, "rows": kept, "returned_rows": len(raw_rows),
            "rows_supplied": len(kept), "truncated": len(kept) < len(raw_rows)}


def recent_player_messages(store, actor: str, now_ms: int, limit: int = 5) -> list[str]:
    rows = store.all(
        "event_history", "actor = ? AND kind = 'chat' AND event_t <= ? "
        "ORDER BY event_t DESC LIMIT ?", (actor, now_ms, limit))
    messages = []
    for row in reversed(rows):
        try:
            text = json.loads(row["payload"]).get("text")
        except (TypeError, json.JSONDecodeError):
            text = None
        if text:
            messages.append(text)
    return messages


async def _run_tool(store, block, bridge_url: str | None,
                    actor: str | None = None) -> tuple[dict, bool]:
    """Return a tool result and whether this call consulted the action ledger."""
    if block.name == "query_database":
        sql = str((block.input or {}).get("sql") or "")
        uses_history = bool(re.search(r"\bevent_history\b", sql, re.IGNORECASE))
        try:
            result = execute_select(store, sql)
            print(f"history SQL ({result['returned_rows']} rows, "
                  f"{result['rows_supplied']} supplied): {result['sql']}", flush=True)
            return result, uses_history
        except Exception as error:
            return {"error": f"{type(error).__name__}: {error}"}, False

    if block.name == "inspect_seed_environment":
        if not bridge_url:
            return {"error": "seed environment service is unavailable"}, False
        from environment import FEATURE_ALIASES, request_environment

        values = block.input or {}
        try:
            radius = max(16, min(MAX_ENVIRONMENT_RADIUS, int(values.get("radius", 256))))
            step = max(4, min(64, int(values.get("step", 16))))
            features = []
            for item in values.get("features", []):
                key = str(item).removeprefix("minecraft:").replace(" ", "_")
                features.extend(FEATURE_ALIASES.get(key, (key,)))
            features = list(dict.fromkeys(features))[:8]
            dim = str(values["dim"]).removeprefix("minecraft:")
            result = await request_environment(
                bridge_url, dim,
                [int(values["x"]), 0, int(values["z"])],
                radius=radius, step=step, features=features,
            )
            return result, False
        except Exception as error:
            return {"error": f"{type(error).__name__}: {error}"}, False

    if block.name == "query_seed_map":
        return _run_seed_map(store, block.input or {}, actor), False

    if block.name == "render_world_area":
        return await render_world_area(block, bridge_url), False

    if block.name == "inspect_world_entities":
        return await inspect_world_entities(block, bridge_url), False

    return {"error": f"unknown evidence tool: {block.name}"}, False


#: Listing every structure over a wide radius is cheap to compute and expensive to read.
#: A model handed nine hundred coordinates will summarise them badly; a count plus the
#: nearest handful is the same fact in a form it can actually use.
SEED_MAP_LISTED = 40


#: How close a computed position must be to a generated site already on record to be the
#: same place. Village attempts are at least nine chunks apart by construction, so this
#: cannot merge two neighbours; in practice the match is exact, because both numbers come
#: from the same placement arithmetic.
SAME_SITE_BLOCKS = 64


def _annotate_known_places(store, result: dict, actor: str | None) -> dict:
    """Link computed positions to the places the god already knows.

    Asked for "the closest village that isn't the one I'm in", the god answered with the
    village the player was standing in, 43 blocks away. The computation was right and the
    model had no way to tell: a seed-map position is just a coordinate, and the fact that
    this particular one is the site the player currently occupies lives in the store.

    So code supplies the link. Every result that coincides with a known generated site
    carries that site's identity, and the one the player is standing in is marked. What to
    do about it stays the model's decision.
    """
    found = result.get("found")
    if not found:
        return result
    sites = []
    for row in store.all("generated_features"):
        if row["x"] is None or row["z"] is None:
            continue
        try:
            value = json.loads(row["value"] or "{}") or {}
        except (TypeError, ValueError):
            value = {}
        if value.get("parent"):
            continue        # a component building, not a site
        sites.append(row)
    if not sites:
        return result
    names = {r["subject"]: r["object"] for r in
             store.all("relationships", "predicate = 'named'")}
    here = None
    if actor:
        presence = store.get("actor_presence", actor)
        here = presence["place_id"] if presence else None
    for item in found:
        for row in sites:
            if (abs(row["x"] - item["x"]) <= SAME_SITE_BLOCKS
                    and abs(row["z"] - item["z"]) <= SAME_SITE_BLOCKS):
                item["known_as"] = {"id": row["id"], "kind": row["kind"]}
                if names.get(row["id"]):
                    item["known_as"]["name"] = names[row["id"]]
                if here and row["id"] == here:
                    item["you_are_here"] = True
                    result["standing_in"] = row["id"]
                break
    if result.get("standing_in"):
        result["note_here"] = (
            "one of these is the place the player is standing in, marked you_are_here; if "
            "they asked for a different one, skip it")
    return result


def _run_seed_map(store, values: dict, actor: str | None = None) -> dict:
    """One seed-map query. Offline, deterministic, and never written to the store."""
    import seedmap

    mode = str(values.get("mode") or "").strip()
    try:
        seed = seedmap.world_seed(store)
        if mode == "spawn":
            with seedmap.World(seed, "overworld") as world:
                return {"ok": True, "mode": mode, "spawn": world.spawn(),
                        "version": seedmap.version_name()}
        if mode == "strongholds":
            with seedmap.World(seed, "overworld") as world:
                return {"ok": True, "mode": mode,
                        "strongholds": world.strongholds(int(values.get("limit") or 3)),
                        "version": seedmap.version_name()}
        if mode == "nearest_biome":
            dim = str(values.get("dim") or "overworld").removeprefix("minecraft:")
            with seedmap.World(seed, dim) as world:
                return world.nearest_biome(
                    str(values.get("biome") or "").removeprefix("minecraft:"),
                    int(values.get("x", 0)), int(values.get("z", 0)),
                    y=int(values.get("y", 63)),
                    max_radius=int(values.get("radius") or 10_000),
                    step=int(values.get("step") or 32))
        if mode == "biome_distribution":
            dim = str(values.get("dim") or "overworld").removeprefix("minecraft:")
            with seedmap.World(seed, dim) as world:
                return world.biome_counts(
                    int(values.get("x", 0)), int(values.get("z", 0)),
                    int(values.get("radius") or 1000), y=int(values.get("y", 63)),
                    step=int(values.get("step") or 16))
        if mode == "biome":
            dim = str(values.get("dim") or "overworld").removeprefix("minecraft:")
            with seedmap.World(seed, dim) as world:
                return {"ok": True, "mode": mode, "dim": dim,
                        "x": int(values["x"]), "y": int(values.get("y", 63)),
                        "z": int(values["z"]),
                        "biome": world.biome(int(values["x"]), int(values.get("y", 63)),
                                             int(values["z"])),
                        "version": seedmap.version_name(),
                        "as_generated": "biome as generated; terrain may have been altered"}
        structure = str(values.get("structure") or "").removeprefix("minecraft:")
        centre = [int(values.get("x", 0)), int(values.get("z", 0))]
        radius = int(values.get("radius") or 4096)
        # Only `nearest` may stop early. A count that stopped at the listing cap reported
        # "at least 200" for a number it could have known exactly, which is the kind of
        # hedge that reads as ignorance rather than precision.
        result = seedmap.find(structure, seed, centre, radius,
                              limit=1 if mode == "nearest" else 0)
        listed = max(1, min(SEED_MAP_LISTED, int(values.get("limit") or SEED_MAP_LISTED)))
        if len(result["found"]) > listed:
            result["listed"] = listed
            result["total_found"] = result["count"]
            result["note"] = (f"count is complete; only the {listed} nearest are listed")
            result["found"] = result["found"][:listed]
        return _annotate_known_places(store, result, actor)
    except seedmap.SeedMapUnavailable as error:
        return {"error": f"the seed map cannot answer: {error}"}
    except Exception as error:  # noqa: BLE001
        return {"error": f"{type(error).__name__}: {error}"}


def _is_not_history(text: str) -> bool:
    return re.sub(r"[^A-Z]", "", (text or "").upper()) == "NOTHISTORY"


async def answer_if_history(store, event: dict, text: str,
                            bridge_url: str | None = None) -> str | None:
    """Let one model gather evidence iteratively, then return its plain-text narrative."""
    actor = event.get("actor")
    now_ms = int(event.get("t") or 0)
    recent = recent_player_messages(store, actor, now_ms)
    position = event.get("pos") or [None, None, None]
    context = (
        f"CURRENT_ACTOR: {actor}\n"
        f"CURRENT_DIMENSION: {event.get('dim', 'overworld')}\n"
        f"CURRENT_POSITION: {position}\n"
        f"CURRENT_TIME_MS: {now_ms}\n"
        f"RECENT_PLAYER_MESSAGES: {json.dumps(recent)}\n"
        f"CURRENT_QUESTION: {text!r}"
    )
    client = model_client(asynchronous=True)
    messages: list[dict] = [{"role": "user", "content": context}]
    queried_history = False
    # A question about ground nobody has visited has no history to retrieve; its evidence
    # is the generator. Tracking the two separately keeps the rule that a claim about what
    # the player did must come from the action ledger, while allowing an answer about what
    # the world contains to come from the seed.
    queried_world = False
    started = time.monotonic()

    for round_number in range(MAX_TOOL_ROUNDS + 1):
        # Out of time, but holding evidence: answer from it. Withholding the tools is what
        # forces that, rather than asking politely and being handed another query.
        out_of_time = (time.monotonic() - started > ANSWER_BUDGET_SECONDS
                       and (queried_history or queried_world))
        if out_of_time:
            messages.append({"role": "user", "content": (
                "Time is up. Answer now, in plain speech, from the evidence you already "
                "have. Do not mention that you were short of time.")})
        reply = await complete_message(
            client, model=DIALOGUE_MODEL,
            thinking=thinking_for(DIALOGUE_MODEL), system=HISTORY_SYSTEM,
            **({} if out_of_time else {"tools": TOOLS}), messages=messages,
        )
        tool_uses = [block for block in reply.content if block.type == "tool_use"]
        plain = _response_text(reply)

        if not tool_uses:
            # Models spell the sentinel loosely: "NOT HISTORY", "not_history.", a trailing
            # newline. Spoken aloud, any of those is a bug the player hears.
            #
            # Checked whatever was retrieved first. The model sometimes looks something up,
            # decides the question was not for it after all, and says the sentinel; gating
            # this on having retrieved nothing meant that answer was spoken verbatim.
            if _is_not_history(plain):
                return None
            if not (queried_history or queried_world):
                if round_number >= MAX_TOOL_ROUNDS:
                    raise UnsafeHistoryQuery("model retrieved no evidence")
                messages.append({"role": "assistant", "content": assistant_content(reply)})
                messages.append({"role": "user", "content": (
                    "You are answering, but no evidence has been retrieved yet. Query "
                    "event_history for what the player did, or query_seed_map for what "
                    "the world contains, before answering."
                )})
                continue
            if not plain:
                raise UnsafeHistoryQuery("history model returned no plain-text answer")
            return plain

        messages.append({"role": "assistant", "content": assistant_content(reply)})
        results = []
        for block in tool_uses:
            result, used_history = await _run_tool(store, block, bridge_url,
                                                   event.get("actor"))
            queried_history = queried_history or used_history
            if block.name in ("query_seed_map", "inspect_seed_environment"):
                queried_world = queried_world or bool(result.get("ok"))
            results.append(tool_result_block(block, result))
        messages.append({"role": "user", "content": results})

    raise UnsafeHistoryQuery("history evidence loop exhausted its tool budget")
