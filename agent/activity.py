#!/usr/bin/env python3
"""Durable, place-aware recent activity.

Episodes answer what a sustained period of work amounted to. This ledger answers a different
question: what exactly did the player do a minute ago, and where? It stores observed actions
without coordinates in prose, then joins them to canonical player structures or generator-
authored features. Raw positions remain internal evidence.
"""

from __future__ import annotations

import collections
import hashlib
import json
import math
import re
from dataclasses import dataclass

from store import NON_PLAYER_ACTORS, Belief, Provenance

IGNORED = {"move", "player_state", "region_enter", "chat", "player_join", "player_quit",
           "xp_change", "damage_dealt"}
HISTORY_BACKGROUND = {"move", "player_state", "region_enter", "chat", "player_join",
                      "player_quit", "xp_change", "level_change", "command"}
HERE_RADIUS = 64
CORE = {"tick", "t", "type", "actor", "dim", "pos"}
GENERATED_RADIUS = 128
SUBFEATURE_RADIUS = 12
WORKSTATION_ROLES = {
    "grindstone": "blacksmith",
    "smithing_table": "blacksmith",
    "blast_furnace": "blacksmith",
    "smoker": "butcher",
    "composter": "farmhouse",
    "cartography_table": "cartographer",
    "fletching_table": "fletcher",
    "lectern": "library",
    "stonecutter": "mason",
    "brewing_stand": "church",
}


def _plain(value: str) -> str:
    return str(value or "").replace("minecraft:", "").replace("_", " ")


def _generated_near(store, dim: str, pos):
    if not pos:
        return None
    candidates = []
    for row in store.all("generated_features", "dim = ?", (dim,)):
        if row["x"] is None or row["z"] is None:
            continue
        value = _belief_value(row)
        component = bool(value.get("parent"))
        if component and row["min_x"] is not None:
            dx = max(row["min_x"] - pos[0], 0, pos[0] - row["max_x"])
            dz = max(row["min_z"] - pos[2], 0, pos[2] - row["max_z"])
            distance = math.hypot(dx, dz)
            inside = (row["min_y"] - 3 <= pos[1] <= row["max_y"] + 3
                      and distance <= 3)
        else:
            distance = math.hypot(row["x"] - pos[0], row["z"] - pos[2])
            inside = distance <= (SUBFEATURE_RADIUS if component else GENERATED_RADIUS)
        if inside:
            # A measured component outranks a broad site even when its anchor is farther.
            candidates.append((0 if component else 1, distance, row))
    return min(candidates, key=lambda item: item[:2])[2] if candidates else None


def _belief_value(row) -> dict:
    try:
        return json.loads(row["value"] or "{}") if row else {}
    except (TypeError, json.JSONDecodeError):
        return {}


def _generated_root(store, place):
    """Return the broad generated site containing a generated component."""
    seen = set()
    while place and place["id"] not in seen:
        seen.add(place["id"])
        parent_id = _belief_value(place).get("parent")
        parent = store.get("generated_features", parent_id) if parent_id else None
        if not parent:
            break
        place = parent
    return place


def _player_structure_at(store, dim: str, pos):
    if not pos:
        return None
    for row in store.all("structures", "dim = ?", (dim,)):
        if (row["min_x"] - 3 <= pos[0] <= row["max_x"] + 3
                and row["min_y"] - 3 <= pos[1] <= row["max_y"] + 3
                and row["min_z"] - 3 <= pos[2] <= row["max_z"] + 3):
            return row
    return None


def _exact_place(store, dim: str, pos):
    """The most specific thing standing at a point: a landmark, else the mass itself."""
    row = _player_structure_at(store, dim, pos)
    if row is not None:
        return row
    from masses import place_of
    return place_of(store, dim, pos)


def _resolve_place(generated, exact, role: str | None = None):
    """Exact containment beats a radius. A measured generated building or a workstation
    clue keeps the generated place; a broad village site yields to the pillar, hall or
    mass an event actually happened in."""
    if generated is not None and (role or generated["min_x"] is not None or exact is None):
        return generated
    return exact


def _role_from_event(event: dict) -> str | None:
    loot = _plain(event.get("loot_table"))
    for role in ("weaponsmith", "toolsmith", "armorer"):
        if role in loot:
            return "blacksmith"
    material = _plain(event.get("before") or event.get("after"))
    return next((role for block, role in WORKSTATION_ROLES.items()
                 if _plain(block) in material), None)


def _subfeature(store, parent, event: dict, role: str):
    """Represent a role-bearing generated building separately from its parent village."""
    pos = event.get("pos")
    root = _generated_root(store, parent)
    if not pos or not root or "village" not in (root["kind"] or ""):
        return parent
    parent_value = _belief_value(parent)
    if parent_value.get("parent"):
        # The visual pipeline already measured this exact generated building. A workstation
        # supplies its semantic role; it does not create a second nested building.
        updated = dict(parent_value)
        updated.update(origin="world_generated", parent=root["id"], role_evidence=role)
        store.put("generated_features", {
            key: parent[key] for key in (
                "id", "dim", "x", "y", "z", "discovered_from",
                "min_x", "min_y", "min_z", "max_x", "max_y", "max_z", "materials")
        } | {"kind": f"village_{role}"},
                  Belief(provenance=Provenance.INFERRED,
                         confidence=max(.85, float(parent["confidence"])),
                         first_seen_tick=parent["first_seen_tick"],
                         verified_at_tick=event.get("tick"), verified_at_ms=event.get("t"),
                         source_event_ids=tuple(json.loads(
                             parent["source_event_ids"] or "[]"))
                         + (f"event:{event.get('tick')}",), value=json.dumps(updated)))
        return store.get("generated_features", parent["id"])
    # Roles are not unique within a village.  Reuse a nearby component with the same role,
    # otherwise give this spatially distinct building its own identity.
    for candidate in store.all("generated_features", "dim = ? AND kind = ?",
                               (event.get("dim"), f"village_{role}")):
        if (_belief_value(candidate).get("parent") == root["id"]
                and math.hypot(candidate["x"] - pos[0], candidate["z"] - pos[2])
                <= SUBFEATURE_RADIUS):
            return candidate
    identity = hashlib.sha1(
        f"{root['id']}:{role}:{pos[0]}:{pos[2]}".encode()).hexdigest()[:16]
    sid = f"generated_sub:{identity}"
    existing = store.get("generated_features", sid)
    store.put("generated_features", {
        "id": sid, "dim": event.get("dim"), "kind": f"village_{role}",
        "x": pos[0], "y": pos[1], "z": pos[2],
        "discovered_from": "observed workstation or loot table",
        "min_x": existing["min_x"] if existing else None,
        "min_y": existing["min_y"] if existing else None,
        "min_z": existing["min_z"] if existing else None,
        "max_x": existing["max_x"] if existing else None,
        "max_y": existing["max_y"] if existing else None,
        "max_z": existing["max_z"] if existing else None,
        "materials": existing["materials"] if existing else None,
    }, Belief(provenance=Provenance.INFERRED, confidence=0.85,
              first_seen_tick=event.get("tick"), verified_at_tick=event.get("tick"),
              verified_at_ms=event.get("t"),
              source_event_ids=(f"event:{event.get('tick')}",),
              value=json.dumps({"origin": "world_generated", "parent": root["id"],
                                "role_evidence": role})))
    # A role can become known after the player opened its chest or began working on the
    # building.  Reconcile those already-observed nearby actions onto the newly identified
    # component; do not leave history dependent on the order clues happened to arrive.
    store.db.execute(
        "UPDATE activities SET place_id = ? WHERE place_id = ? AND dim = ? "
        "AND x BETWEEN ? AND ? AND z BETWEEN ? AND ?",
        (sid, root["id"], event.get("dim"),
         pos[0] - SUBFEATURE_RADIUS, pos[0] + SUBFEATURE_RADIUS,
         pos[2] - SUBFEATURE_RADIUS, pos[2] + SUBFEATURE_RADIUS))
    store.db.execute(
        "UPDATE event_history SET place_id = ? WHERE place_id = ? AND dim = ? "
        "AND x BETWEEN ? AND ? AND z BETWEEN ? AND ?",
        (sid, root["id"], event.get("dim"),
         pos[0] - SUBFEATURE_RADIUS, pos[0] + SUBFEATURE_RADIUS,
         pos[2] - SUBFEATURE_RADIUS, pos[2] + SUBFEATURE_RADIUS))
    store.commit()
    return store.get("generated_features", sid)


def _mark_modified(store, place, event: dict) -> None:
    # Identity format is an implementation detail: seed-located sites, inferred components,
    # and visually measured components deliberately have different ID shapes. Ontology
    # membership is the stable test for whether this is generator-authored.
    if not place or not store.get("generated_features", place["id"]):
        return
    actor = event.get("actor")
    rid = f"modified_by:{place['id']}:{actor}"
    store.put("relationships", {
        "id": rid, "subject": place["id"], "predicate": "modified_by", "object": actor,
    }, Belief(provenance=Provenance.OBSERVED, verified_at_tick=event.get("tick"),
              verified_at_ms=event.get("t"),
              source_event_ids=(f"event:{event.get('tick')}",),
              value=json.dumps({"last_kind": event.get("type")})))


def record_activity(store, event: dict) -> bool:
    """Index one event. Returns true only when a durable activity row was written."""
    actor, kind = event.get("actor"), event.get("type")
    dim, pos = event.get("dim", "overworld"), event.get("pos")
    generated = _generated_near(store, dim, pos)
    exact = _exact_place(store, dim, pos)
    initial_place = _resolve_place(generated, exact)
    raw = json.dumps(event, sort_keys=True, separators=(",", ":"))
    history_id = "event_history:" + hashlib.sha1(raw.encode()).hexdigest()[:24]
    store.put("event_history", {
        "id": history_id, "actor": actor, "dim": dim, "tick": event.get("tick"),
        "event_t": event.get("t"), "kind": kind,
        "place_id": initial_place["id"] if initial_place else None,
        "x": pos[0] if pos else None, "y": pos[1] if pos else None,
        "z": pos[2] if pos else None, "payload": raw,
    }, Belief(provenance=Provenance.OBSERVED, first_seen_tick=event.get("tick"),
              verified_at_tick=event.get("tick"), verified_at_ms=event.get("t"),
              source_event_ids=(f"event:{event.get('tick')}",)))
    if not actor or actor in NON_PLAYER_ACTORS:
        return False

    if kind in ("move", "region_enter"):
        # Arrivals belong to the broad site, not each building crossed within it.  Presence
        # changes even when the player leaves for ordinary terrain, making a later return a
        # new visit without inventing arbitrary cooldowns.
        generated = _generated_root(store, generated)
        presence = store.get("actor_presence", actor)
        previous = presence["place_id"] if presence else None
        current = generated["id"] if generated else None
        if previous == current:
            return False
        store.put("actor_presence", {
            "id": actor, "actor": actor, "dim": dim, "place_id": current,
            "event_t": event.get("t"),
        }, Belief(provenance=Provenance.OBSERVED,
                  verified_at_tick=event.get("tick"), verified_at_ms=event.get("t"),
                  source_event_ids=(f"event:{event.get('tick')}",)))
        if generated is None:
            return False
        kind = "visit_generated"
        place = generated
        detail = {"feature": generated["kind"]}
    else:
        if kind in IGNORED:
            return False
        role = _role_from_event(event)
        if generated is not None and role:
            generated = _subfeature(store, generated, event, role)
        place = _resolve_place(generated, exact, role)
        detail = {key: value for key, value in event.items() if key not in CORE}
        if role:
            detail["place_role"] = role

    stable = json.dumps([actor, event.get("t"), event.get("tick"), kind, pos, detail],
                        sort_keys=True, separators=(",", ":"))
    aid = "activity:" + hashlib.sha1(stable.encode()).hexdigest()[:20]
    store.put("activities", {
        "id": aid, "actor": actor, "dim": dim, "tick": event.get("tick"),
        "event_t": event.get("t"), "kind": kind,
        "place_id": place["id"] if place else None,
        "x": pos[0] if pos else None, "y": pos[1] if pos else None,
        "z": pos[2] if pos else None, "detail": json.dumps(detail),
    }, Belief(provenance=Provenance.OBSERVED, first_seen_tick=event.get("tick"),
              verified_at_tick=event.get("tick"), verified_at_ms=event.get("t"),
              source_event_ids=(f"event:{event.get('tick')}",)))
    if event.get("type") in ("block_place", "block_break"):
        _mark_modified(store, place, event)
    return True


def record_activities(store, events: list[dict]) -> int:
    written = 0
    with store.transaction():
        for event in events:
            written += record_activity(store, event)
    return written


def is_activity_question(text: str) -> bool:
    lower = text.lower()
    first_person_history = (re.search(r"\b(?:i|i['’]?ve|ive|me|my)\b", lower)
                            and re.search(
                                r"\b(?:did|done|doings|happened|built|broke|broken|"
                                r"placed|made|crafted|took|taken|gathered|killed|"
                                r"slept|visited|found|history)\b", lower))
    request_verb = re.search(r"\b(?:what|which|name|list|tell|recall|remember)\b", lower)
    return bool((re.search(r"\b(last|recent|just)\b", lower)
                 and re.search(r"\b(did|done|doing|things?|minutes?|seconds?|hours?)\b", lower))
                or re.search(r"what (?:have )?i (?:just )?(?:done|did)", lower)
                or re.search(r"what have i done", lower)
                or ("since" in lower and re.search(r"\b(i|me|my)\b", lower))
                or (request_verb and first_person_history))


def history_query_text(store, actor: str, now_ms: int, text: str) -> str | None:
    """Resolve a direct history question or a short locative follow-up to one.

    User chat is observed input, unlike the god's ASSERTED replies, so it is safe to use for
    conversational reference resolution. The current chat row has already been indexed;
    only earlier rows are considered here.
    """
    if is_activity_question(text):
        return text
    lower = text.lower()
    refinement = bool(re.search(
        r"\b(?:specifically|only|just)\b.*\b(?:here|location|place|area|world)\b", lower)
        or re.search(r"\b(?:here|this (?:location|place|area))\b", lower))
    if not refinement:
        return None
    recent = store.all(
        "event_history", "actor = ? AND kind = 'chat' AND event_t < ? "
        "AND event_t >= ? ORDER BY event_t DESC LIMIT 5",
        (actor, now_ms, now_ms - 2 * 60_000))
    for row in recent:
        try:
            previous = json.loads(row["payload"]).get("text", "")
        except (TypeError, json.JSONDecodeError):
            continue
        if is_activity_question(previous):
            return f"{previous}. More specifically: {text}"
    return None


# Compatibility for callers/tests written when this could only retrieve a recency window.
is_recent_activity_question = is_activity_question


def requested_window_ms(text: str) -> int:
    match = re.search(r"\b(?:last\s+)?(\d+)\s*(seconds?|minutes?|hours?|days?)\b", text.lower())
    if not match:
        return 10 * 60_000
    count, unit = int(match.group(1)), match.group(2)
    scale = (1000 if unit.startswith("second") else 60_000
             if unit.startswith("minute") else 3_600_000
             if unit.startswith("hour") else 24 * 3_600_000)
    return min(count * scale, 365 * 24 * 3_600_000)


def _join_words(parts: list[str]) -> str:
    if len(parts) < 2:
        return "".join(parts)
    if len(parts) == 2:
        return " and ".join(parts)
    return ", ".join(parts[:-1]) + ", and " + parts[-1]


def _numbered_items(counter: collections.Counter) -> str:
    return _join_words([f"{count} {_plain(item)}" for item, count in counter.items()])


def _feature_label(kind: str) -> str:
    plain = _plain(kind)
    if plain.startswith("village ") and plain.removeprefix("village ") in {
            "plains", "desert", "savanna", "snowy", "taiga"}:
        variant = plain.removeprefix("village ")
        return f"{variant} village"
    return plain


def _place_label(store, place_id: str | None) -> str:
    if not place_id:
        return "that place"
    generated = store.get("generated_features", place_id)
    if generated:
        return _feature_label(generated["kind"])
    structure = store.get("structures", place_id)
    if structure:
        named = store.get("relationships", f"named:{place_id}")
        return named["object"] if named else (structure["name"] or "structure")
    return "that place"


def _descendants(store, root_id: str) -> set[str]:
    found, changed = {root_id}, True
    while changed:
        changed = False
        for row in store.all("generated_features"):
            if row["id"] not in found and _belief_value(row).get("parent") in found:
                found.add(row["id"])
                changed = True
    return found


def _mentioned_place(store, actor: str, text: str, current_pos=None,
                     current_dim: str = "overworld"):
    """Resolve a location constraint from the question, preferring where the player is."""
    lower = text.lower()
    place_words = {"village", "church", "blacksmith", "temple", "house", "farm"}
    wanted = place_words & set(re.findall(r"[a-z]+", lower))
    if not wanted:
        return None

    if current_pos:
        nearby = _generated_near(store, current_dim, current_pos)
        if nearby:
            root = _generated_root(store, nearby)
            haystack = " ".join((_plain(nearby["kind"]), _plain(root["kind"])))
            if any(word in haystack or word == "village" and "village" in haystack
                   for word in wanted):
                return root if "village" in wanted else nearby

    candidates = []
    for row in store.all("generated_features"):
        name = store.structure_name(row["id"])
        haystack = " ".join((_plain(row["kind"]),
                             (name["object"].lower() if name else "")))
        if any(word in haystack for word in wanted):
            latest = store.all("event_history", "actor = ? AND place_id = ?",
                               (actor, row["id"]))
            seen = max((r["event_t"] or 0 for r in latest), default=0)
            candidates.append((seen, row))
    if not candidates:
        return None
    place = max(candidates, key=lambda item: item[0])[1]
    return _generated_root(store, place) if "village" in wanted else place


@dataclass(frozen=True)
class HistorySelection:
    """The evidence slice chosen from the permanent low-level event history."""

    rows: tuple[dict, ...]
    after_ms: int
    before_ms: int
    place_id: str | None
    reason: str
    center: tuple[int, int] | None = None
    radius: int | None = None


def select_history(store, actor: str, now_ms: int, text: str,
                   current_pos=None, current_dim: str = "overworld",
                   limit: int = 10_000) -> HistorySelection:
    """Select events by the constraints in the question, not by one global recency rank."""
    lower = text.lower()
    here = bool(re.search(r"\b(?:here|this (?:place|location|area)|around here)\b", lower))
    target = _mentioned_place(store, actor, text, current_pos, current_dim)
    target_ids = _descendants(store, target["id"]) if target else set()
    explicit_time = bool(re.search(r"\b\d+\s*(?:seconds?|minutes?|hours?|days?)\b", lower))
    all_time = bool(re.search(
        r"\b(?:ever|everything|all[- ]time|whole history|entire history)\b", lower))
    if explicit_time:
        after = now_ms - requested_window_ms(text)
        reason = "explicit time window"
    elif all_time:
        after = 0
        reason = "all recorded history"
    else:
        joins = store.all("event_history", "actor = ? AND kind = 'player_join' "
                          "AND event_t <= ?", (actor, now_ms))
        after = max((row["event_t"] or 0 for row in joins), default=now_ms - 10 * 60_000)
        reason = "current play session"
        if "since" in lower and target:
            visits = store.all("activities", "actor = ? AND kind = 'visit_generated' "
                               "AND event_t <= ?", (actor, now_ms))
            arrivals = []
            for row in visits:
                place = store.get("generated_features", row["place_id"])
                root = _generated_root(store, place)
                if root and root["id"] == _generated_root(store, target)["id"]:
                    arrivals.append(row["event_t"] or 0)
            # A server restart while still inside the place is also a valid beginning for
            # "since being here"; do not drag yesterday's visit into today's answer.
            after = max(after, max(arrivals, default=0))
            reason = "since entering the named place in this play session"

    where = "actor = ? AND event_t >= ? AND event_t <= ?"
    params: list = [actor, after, now_ms]
    center = None
    radius = None
    if here and current_pos:
        center = (int(current_pos[0]), int(current_pos[2]))
        radius = HERE_RADIUS
        where += " AND dim = ? AND x BETWEEN ? AND ? AND z BETWEEN ? AND ?"
        params.extend((current_dim, center[0] - radius, center[0] + radius,
                       center[1] - radius, center[1] + radius))
        reason += f", within the local {radius}-block area"
    elif target_ids:
        placeholders = ",".join("?" for _ in target_ids)
        where += f" AND place_id IN ({placeholders})"
        params.extend(sorted(target_ids))
    placeholders = ",".join("?" for _ in HISTORY_BACKGROUND)
    where += f" AND kind NOT IN ({placeholders})"
    params.extend(sorted(HISTORY_BACKGROUND))
    # Apply semantic constraints before the safety limit. Otherwise thousands of movement
    # samples elsewhere could crowd the requested building's deeds out of the result.
    raw_rows = store.all(
        "event_history", where + " ORDER BY event_t DESC, tick DESC LIMIT ?",
        tuple(params + [limit]))
    raw_rows.reverse()
    selected = []
    for row in raw_rows:
        if target_ids and not center:
            place = store.get("generated_features", row["place_id"]) if row["place_id"] else None
            root = _generated_root(store, place)
            if not place or (place["id"] not in target_ids
                             and (not root or root["id"] not in target_ids)):
                continue
        event = json.loads(row["payload"])
        event["_history_id"] = row["id"]
        event["_place_id"] = row["place_id"]
        selected.append(event)
    return HistorySelection(tuple(selected), after, now_ms,
                            target["id"] if target else None, reason, center, radius)


@dataclass(frozen=True)
class Deed:
    t: int
    text: str
    salience: int = 1


def _format_count(count: int, item: str) -> str:
    name = _plain(item)
    mass = {"bread", "wheat", "obsidian", "cobblestone", "granite",
            "deepslate", "cobbled deepslate"}
    if name == "hay block":
        name = "hay bale" if count == 1 else "hay bales"
    elif count != 1 and name not in mass and not name.endswith("s"):
        name += "s"
    article = "an" if name[:1] in "aeiou" else "a"
    return (f"1 {name}" if count == 1 and name in mass else f"{article} {name}"
            if count == 1 else f"{count} {name}")


def _history_deeds(store, selection: HistorySelection, text: str,
                   max_deeds: int | None = 4) -> list[Deed]:
    events = list(selection.rows)
    deeds: list[Deed] = []
    consumed: set[str] = set()

    # A broken block followed by its pickup is acquisition, not merely destruction. Match
    # by item, place and time so this works for workstations, crops, ore, logs, and anything
    # added later without a catalogue of special event cases.
    breaks = [event for event in events if event.get("type") == "block_break"]
    pickups = [event for event in events if event.get("type") == "item_pickup"]
    gathered: list[tuple[int, str, str | None, int]] = []
    for pickup in pickups:
        matching_breaks = [broken for broken in breaks
                           if broken.get("before") == pickup.get("item")
                           and 0 <= pickup.get("t", 0) - broken.get("t", 0) <= 120_000
                           and pickup.get("_place_id") == broken.get("_place_id")]
        if not matching_breaks:
            continue
        for broken in matching_breaks:
            consumed.add(broken["_history_id"])
        consumed.add(pickup["_history_id"])
        gathered.append((min(broken.get("t", 0) for broken in matching_breaks),
                         pickup.get("item"), pickup.get("_place_id"),
                         int(pickup.get("count") or 1)))

    gathered_by_place: dict[tuple[str, str | None], tuple[int, int]] = {}
    for when, item, place_id, count in gathered:
        key = (item, place_id)
        first, total = gathered_by_place.get(key, (when, 0))
        gathered_by_place[key] = (min(first, when), total + count)

    crafts = collections.Counter()
    craft_time = {}
    craft_inputs = collections.Counter()
    craft_places = set()
    for event in events:
        if event.get("type") != "craft" or not event.get("item"):
            continue
        crafts[event["item"]] += int(event.get("count") or 1)
        craft_time.setdefault(event["item"], event.get("t", 0))
        craft_places.add(event.get("_place_id"))
        for item, count in (event.get("inputs") or {}).items():
            craft_inputs[item] += int(count)
        consumed.add(event["_history_id"])

    # Gathered materials and the crafts immediately following them form one work chain.
    first_craft = min(craft_time.values(), default=0)
    chain_gathered = {
        key: value for key, value in gathered_by_place.items()
        if first_craft and key[1] in craft_places
        and 0 <= first_craft - value[0] <= 30_000
    }
    if chain_gathered and crafts:
        gathered_text = _join_words([
            _format_count(count, item) for (item, _), (_, count) in chain_gathered.items()])
        crafted_text = _join_words([_format_count(count, item)
                                    for item, count in crafts.items()
                                    if _plain(item) != "crafting table"])
        if crafted_text:
            causal = bool(craft_inputs and any(item in craft_inputs
                                               for item, _ in chain_gathered))
            link = "and made" if causal else "and later crafted"
            deeds.append(Deed(min(first for first, _ in chain_gathered.values()),
                              f"gathered {gathered_text}, {link} {crafted_text}", 4))
            for key in chain_gathered:
                del gathered_by_place[key]

    for (item, place_id), (when, count) in gathered_by_place.items():
        place = _place_label(store, place_id)
        source = f" from the {place}" if place != "that place" else ""
        deeds.append(Deed(when, f"took {_format_count(count, item)}{source}", 4))

    container_groups = collections.OrderedDict()
    for event in events:
        if event.get("type") == "container_take" and event.get("item"):
            group = container_groups.setdefault(
                event.get("_place_id"), [event.get("t", 0), collections.Counter()])
            group[0] = min(group[0], event.get("t", 0))
            group[1][event["item"]] += int(event.get("count") or 1)
            consumed.add(event["_history_id"])
    for place_id, (when, items) in container_groups.items():
        place = _place_label(store, place_id)
        source = f" from the {place}'s chest" if place != "that place" else " from a chest"
        contents = _join_words([_format_count(count, item)
                                for item, count in items.items()])
        deeds.append(Deed(when, f"looted {contents}{source}", 4))

    for event in events:
        kind = event.get("type")
        hid = event.get("_history_id")
        if hid in consumed:
            continue
        if kind == "container_take" and event.get("item"):
            place = _place_label(store, event.get("_place_id"))
            source = f" from the {place}'s chest" if place != "that place" else " from a chest"
            deeds.append(Deed(event.get("t", 0),
                              f"looted {_format_count(int(event.get('count') or 1), event['item'])}{source}", 4))
        elif kind == "mob_kill" and event.get("entity"):
            deeds.append(Deed(event.get("t", 0),
                              f"killed {_format_count(1, event['entity'])}", 5))
        elif kind == "sleep" and event.get("result") == "ok":
            place = store.get("generated_features", event.get("_place_id"))
            root = _generated_root(store, place)
            where = " in a villager's bed" if root and "village" in root["kind"] else ""
            deeds.append(Deed(event.get("t", 0), f"slept{where}", 4))
        elif kind == "craft" and event.get("item"):
            deeds.append(Deed(event.get("t", 0),
                              f"crafted {_format_count(int(event.get('count') or 1), event['item'])}", 3))

    # Remaining block deltas are actual edits. Group them by semantic place and report net
    # change so scaffolding placed and removed inside the selection cancels out.
    changes: dict[str | None, tuple[collections.Counter, collections.Counter, int]] = {}
    for event in events:
        if event.get("_history_id") in consumed or event.get("type") not in {
                "block_place", "block_break"}:
            continue
        place_id = event.get("_place_id")
        placed, broken, first = changes.setdefault(
            place_id, (collections.Counter(), collections.Counter(), event.get("t", 0)))
        material = event.get("after") if event["type"] == "block_place" else event.get("before")
        (placed if event["type"] == "block_place" else broken)[material] += 1
    for place_id, (placed, broken, when) in changes.items():
        for material in set(placed) & set(broken):
            cancel = min(placed[material], broken[material])
            placed[material] -= cancel
            broken[material] -= cancel
        if sum(placed.values()) + sum(broken.values()) == 0:
            continue
        place = _place_label(store, place_id)
        target = f"the {place}" if place != "that place" else place
        generated = store.get("generated_features", place_id) if place_id else None
        if (generated and _belief_value(generated).get("parent")
                and not target.endswith(" building")):
            target += " building"
        deeds.append(Deed(when, f"modified {target}", 3))

    # Retrieval decides what is eligible; salience keeps the in-game answer readable while
    # the complete evidence remains queryable in event_history.
    # Preserve chronology among the deeds retained.
    if max_deeds is not None and len(deeds) > max_deeds:
        deeds = sorted(deeds, key=lambda deed: (-deed.salience, -deed.t))[:max_deeds]
    return sorted(deeds, key=lambda deed: deed.t)


def render_history_evidence(store, selection: HistorySelection, text: str,
                            gap_ms: int = 20 * 60_000) -> str:
    """Turn the selected SQL rows into chronological evidence chunks for narration.

    Positions were used by the query and are intentionally absent here. The narrator sees
    observed actions, their order, and their semantic places—not database coordinates.
    """
    if not selection.rows:
        return f"[OBSERVED QUERY] {selection.reason}; no matching action events."
    chunks: list[list[dict]] = []
    for event in selection.rows:
        if (not chunks or event.get("t", 0) - chunks[-1][-1].get("t", 0) > gap_ms):
            chunks.append([])
        chunks[-1].append(event)

    handled = {"block_place", "block_break", "item_pickup", "craft",
               "container_take", "mob_kill", "sleep"}
    lines = [f"[OBSERVED QUERY] {selection.reason}; {len(selection.rows)} action events "
             f"in {len(chunks)} chronological visit(s)."]
    for number, rows in enumerate(chunks, 1):
        sub = HistorySelection(tuple(rows), rows[0].get("t", 0), rows[-1].get("t", 0),
                               selection.place_id, selection.reason,
                               selection.center, selection.radius)
        kinds = collections.Counter(event.get("type") for event in rows)
        tally = ", ".join(f"{count} {kind}" for kind, count in sorted(kinds.items()))
        lines.append(f"[OBSERVED visit {number}] event tally: {tally}")

        # Material-level rollups let the narrator distinguish building, harvesting and
        # dismantling without flooding it with one row per block.
        for kind, field, verb in (("block_place", "after", "blocks placed"),
                                  ("block_break", "before", "blocks broken"),
                                  ("item_pickup", "item", "items picked up"),
                                  ("craft", "item", "items crafted")):
            by_place: dict[str, collections.Counter] = collections.defaultdict(
                collections.Counter)
            for event in rows:
                if event.get("type") == kind and event.get(field):
                    place = _place_label(store, event.get("_place_id"))
                    amount = int(event.get("count") or 1)
                    by_place[place][_plain(event[field])] += amount
            for place, materials in by_place.items():
                detail = ", ".join(f"{count} {material}"
                                   for material, count in materials.most_common())
                lines.append(f"[OBSERVED visit {number}] {verb} in {place}: {detail}")
        for deed in _history_deeds(store, sub, text, max_deeds=None):
            lines.append(f"[OBSERVED visit {number}] {deed.text}")

        # Preserve less-common event types without teaching the retriever every possible
        # Minecraft narrative. Exact payload fields remain available for model reasoning.
        other = collections.Counter()
        for event in rows:
            if event.get("type") in handled:
                continue
            detail = {key: value for key, value in event.items()
                      if key not in CORE and not key.startswith("_")}
            other[(event.get("type"), json.dumps(detail, sort_keys=True))] += 1
        for (kind, detail), count in other.items():
            suffix = f" x{count}" if count > 1 else ""
            lines.append(f"[OBSERVED visit {number}] {kind}: {detail}{suffix}")
    return "\n".join(lines)


def summarize_recent(store, actor: str, now_ms: int, text: str, current_pos=None,
                     current_dim: str = "overworld") -> str:
    """Compatibility name for query-scoped history retrieval and deed assembly."""
    selection = select_history(store, actor, now_ms, text, current_pos, current_dim)
    deeds = _history_deeds(store, selection, text)

    # A discovery immediately before an explicit window is useful context for deeds inside
    # it. It is not used for place-scoped/current-session questions where "found" would be
    # an unhelpful restatement of the filter.
    if re.search(r"\b\d+\s*(?:seconds?|minutes?|hours?)\b", text.lower()):
        arrivals = store.all(
            "activities", "actor = ? AND kind = 'visit_generated' AND event_t >= ? "
            "AND event_t <= ? ORDER BY event_t DESC", (actor, selection.after_ms - 60_000,
                                                        selection.before_ms))
        if arrivals:
            place = store.get("generated_features", arrivals[0]["place_id"])
            root = _generated_root(store, place)
            if root:
                deeds.insert(0, Deed(arrivals[0]["event_t"] or 0,
                                     f"found a {_feature_label(root['kind'])}", 5))
    if not deeds:
        place = f" in the {_place_label(store, selection.place_id)}" if selection.place_id else ""
        return f"I found no recorded deeds from you{place} in that span."
    phrases = [deed.text for deed in deeds]
    return "You " + "; ".join(phrases[:-1]) + (("; then " if len(phrases) > 1 else "")
                                                + phrases[-1]) + "."


def render_history_for_context(store, actor: str, now_ms: int, text: str,
                               current_pos=None) -> list[str]:
    """Compact, provenance-labelled evidence for the ordinary dialogue context pipeline."""
    if not is_activity_question(text):
        return []
    selection = select_history(store, actor, now_ms, text, current_pos)
    lines = [f"[OBSERVED] selected event history: {selection.reason}"]
    lines.extend(f"[OBSERVED] {deed.text}" for deed in _history_deeds(store, selection, text))
    return lines
