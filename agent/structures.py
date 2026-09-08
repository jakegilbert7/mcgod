#!/usr/bin/env python3
"""Canonical present-tense structure identity.

The event stream can prove that work happened; it cannot prove what stands now.  This
module owns the smaller, stricter contract for current world objects:

* one row in ``structures`` is one object visible in a fresh voxel read;
* model output may propose boundaries and names, never database identity;
* identity is matched one-to-one from geometry and material composition;
* missing model output never means demolition; measured absence does.

Keeping these rules in one module prevents the live loop, survey tool and evaluator from
quietly inventing different definitions of a structure.
"""

from __future__ import annotations

import collections
import json
import math
import uuid

from store import Belief, Provenance

SCHEMA_VERSION = 2
KIND = "current_structure"
GENERATED_KIND = "generated_feature"
MATCH_FLOOR = 0.34
GENERATED_SITE_RADIUS = 128
GENERATED_COMPONENT_RADIUS = 12
PLAYER_ORIGIN_SHARE = 0.50


def facts_of(row) -> dict:
    try:
        value = json.loads(row["value"] or "{}")
    except (TypeError, ValueError):
        value = {}
    return value if isinstance(value, dict) else {}


def is_current(row) -> bool:
    """Whether a row obeys the present-tense structure contract.

    ``s_`` accepts the four scan-derived rows made before the schema marker existed.  The
    store migration removes placement/removal history, so this compatibility branch can be
    deleted after old databases have made one successful pass.
    """
    return facts_of(row).get("kind") == KIND or row["id"].startswith("s_")


def current(store, dim: str | None = None) -> list:
    rows = store.all("structures", "dim = ?", (dim,)) if dim else store.all("structures")
    return [row for row in rows if row["min_x"] is not None and is_current(row)]


def current_generated(store, dim: str | None = None) -> list:
    """Generator-authored objects with visually measured extents."""
    rows = (store.all("generated_features", "dim = ?", (dim,)) if dim
            else store.all("generated_features"))
    return [row for row in rows if row["min_x"] is not None]


def box_of(row) -> tuple[list[int], list[int]]:
    return ([row["min_x"], row["min_y"], row["min_z"]],
            [row["max_x"], row["max_y"], row["max_z"]])


def volume(lo, hi) -> int:
    return math.prod(max(0, hi[i] - lo[i] + 1) for i in range(3))


def intersection(a_lo, a_hi, b_lo, b_hi) -> int:
    size = [min(a_hi[i], b_hi[i]) - max(a_lo[i], b_lo[i]) + 1 for i in range(3)]
    return math.prod(size) if all(n > 0 for n in size) else 0


def _materials(value) -> collections.Counter:
    if hasattr(value, "keys") and "materials" in value.keys():
        value = value["materials"]
    if isinstance(value, str):
        try:
            value = json.loads(value or "{}")
        except (TypeError, ValueError):
            value = {}
    return collections.Counter(value or {})


def material_similarity(a, b) -> float:
    """Weighted Jaccard similarity; insensitive to namespace spelling."""
    a = collections.Counter({k.replace("minecraft:", ""): v
                             for k, v in _materials(a).items()})
    b = collections.Counter({k.replace("minecraft:", ""): v
                             for k, v in _materials(b).items()})
    keys = set(a) | set(b)
    total = sum(max(a[k], b[k]) for k in keys)
    return sum(min(a[k], b[k]) for k in keys) / total if total else 0.0


def match_score(row, observation: dict) -> float:
    """How likely a measured object is to be the prior version of ``row``.

    Intersection over the smaller box handles ordinary growth and shrinkage.  IoU prevents
    a tiny object inside a huge old box from stealing its identity.  Materials break ties
    between close neighbours.  No label participates: names may change and model language
    is not evidence of identity.
    """
    if row["dim"] != observation["dim"]:
        return 0.0
    old_lo, old_hi = box_of(row)
    new_lo, new_hi = observation["lo"], observation["hi"]
    shared = intersection(old_lo, old_hi, new_lo, new_hi)
    if not shared:
        return 0.0
    old_v, new_v = volume(old_lo, old_hi), volume(new_lo, new_hi)
    containment = shared / max(1, min(old_v, new_v))
    iou = shared / max(1, old_v + new_v - shared)
    materials = material_similarity(row, observation.get("materials", {}))
    return 0.55 * containment + 0.25 * iou + 0.20 * materials


def assign_identities(existing: list, observations: list,
                      floor: float = MATCH_FLOOR) -> dict[int, str]:
    """Globally rank candidate pairs and make a deterministic one-to-one assignment."""
    candidates = []
    for index, observation in enumerate(observations):
        for row in existing:
            score = match_score(row, observation)
            if score >= floor:
                candidates.append((score, row["id"], index))
    candidates.sort(key=lambda item: (-item[0], item[1], item[2]))
    assigned, used = {}, set()
    for _, sid, index in candidates:
        if index not in assigned and sid not in used:
            assigned[index] = sid
            used.add(sid)
    return assigned


def new_id(existing_ids=()) -> str:
    """A location-independent id: moving a boundary must never rename an object."""
    used = set(existing_ids)
    while True:
        sid = f"s_{uuid.uuid4().hex[:12]}"
        if sid not in used:
            return sid


def new_generated_id(existing_ids=()) -> str:
    used = set(existing_ids)
    while True:
        sid = f"g_{uuid.uuid4().hex[:12]}"
        if sid not in used:
            return sid


def observation_value(facts: dict, observed_at_ms: int, **extra) -> str:
    """Versioned JSON stored beside canonical geometry."""
    value = {
        "schema_version": SCHEMA_VERSION,
        "kind": KIND,
        "observed_at_ms": observed_at_ms,
        "blocks": facts["blocks"],
        "dims": facts["dims"],
        "fill": facts["fill"],
        "vertical": facts["vertical"],
        "significance": "work",
    }
    value.update(extra)
    return json.dumps(value)


def generated_observation_value(facts: dict, observed_at_ms: int, **extra) -> str:
    value = json.loads(observation_value(facts, observed_at_ms, **extra))
    value.update(kind=GENERATED_KIND, origin="world_generated")
    return json.dumps(value)


def blocks_inside(row, built: dict) -> set:
    lo, hi = box_of(row)
    return {p for p in built if all(lo[i] <= p[i] <= hi[i] for i in range(3))}


def explained_fraction(row, observations: list, built: dict) -> float:
    """Fraction of an old object's measured blocks covered by new object boundaries."""
    old = blocks_inside(row, built)
    if not old:
        return 0.0
    covered = {p for p in old if any(
        all(o["lo"][i] <= p[i] <= o["hi"][i] for i in range(3))
        for o in observations)}
    return len(covered) / len(old)


def _event_owner(store, dim: str, facts: dict) -> str | None:
    """Owner supported by the latest recorded mutation at each position in a box.

    A generated site's boundary is location context, not authorship.  Raw block history is
    the strongest authorship evidence we have: if a player placed most of the material that
    now defines an object, the object is player-built even when it stands inside a village.
    Replaying the latest mutation per coordinate also prevents a placed-and-later-broken
    block from claiming something that no longer stands.

    The event stream does not contain a full live voxel map, so material totals are matched
    conservatively against the measured current totals.  A few renovations inside a large
    vanilla building cannot cross the majority threshold.
    """
    lo, hi = facts["lo"], facts["hi"]
    rows = store.all(
        "event_history",
        "dim = ? AND kind IN ('block_place', 'block_break') "
        "AND x BETWEEN ? AND ? AND y BETWEEN ? AND ? AND z BETWEEN ? AND ? "
        "ORDER BY event_t, id",
        (dim, lo[0], hi[0], lo[1], hi[1], lo[2], hi[2]))
    latest = {}
    for row in rows:
        latest[(row["x"], row["y"], row["z"])] = row

    by_actor: dict[str, collections.Counter] = {}
    for row in latest.values():
        if row["kind"] != "block_place" or not row["actor"]:
            continue
        try:
            payload = json.loads(row["payload"] or "{}")
        except (TypeError, ValueError):
            payload = {}
        material = str(payload.get("after") or "").split("[")[0].replace(
            "minecraft:", "")
        if material and material != "air":
            by_actor.setdefault(row["actor"], collections.Counter())[material] += 1

    standing = _materials(facts.get("materials", {}))
    total = max(1, sum(standing.values()))
    scored = []
    for actor, placed in by_actor.items():
        supported = sum(min(standing[material], count)
                        for material, count in placed.items())
        scored.append((supported / total, supported, actor))
    if not scored:
        return None
    share, _, actor = max(scored)
    return actor if share >= PLAYER_ORIGIN_SHARE else None


def owner_from_work(store, dim: str, facts: dict) -> str | None:
    """Attribute a current object from exact events, then aggregated work history."""
    owner = _event_owner(store, dim, facts)
    if owner:
        return owner
    best = (0.0, "")
    standing = max(1, sum(_materials(facts.get("materials", {})).values()))
    for row in store.all("work_events", "dim = ? AND kind = 'placement'", (dim,)):
        score = match_score(row, {"dim": dim, "lo": facts["lo"], "hi": facts["hi"],
                                  "materials": facts.get("materials", {})})
        # Bounding-box containment alone cannot establish authorship: placing a door and a
        # short wall inside a 350-block vanilla house geometrically overlaps it perfectly.
        # Require the observed construction record to account for a substantive share of
        # what now stands before assigning the whole object to that player.
        constructed_share = sum(_materials(row).values()) / standing
        if constructed_share >= 0.25 and score > best[0] and row["actor"]:
            best = (score, row["actor"])
    return best[1] if best[0] >= MATCH_FLOOR else None


def migrate_player_built_generated(store) -> list[tuple[str, str]]:
    """Move generated-table rows contradicted by majority player-placement evidence.

    This repairs ontology, not labels.  In particular, proximity to a seed-backed village
    can never make a newly placed sculpture generator-authored.  Conversely, a few edits to
    a known vanilla building do not move that building: the current object must be mostly
    accounted for by player placements.
    """
    moved = []
    with store.transaction():
        for row in list(current_generated(store)):
            lo, hi = box_of(row)
            materials = _materials(row)
            facts = {
                "lo": lo, "hi": hi, "materials": materials,
                "blocks": sum(materials.values()),
                "dims": facts_of(row).get("dims") or
                        "x".join(str(hi[i] - lo[i] + 1) for i in range(3)),
                "fill": facts_of(row).get("fill"),
                "vertical": facts_of(row).get("vertical"),
            }
            # Moving an established generated identity is the highest-risk correction and
            # therefore requires exact majority evidence. Aggregated episode boxes remain a
            # useful fallback for attributing an ordinary structure, but are not precise
            # enough to reclassify a vanilla building after a substantial renovation.
            actor = _event_owner(store, row["dim"], facts)
            if not actor:
                continue

            target = new_id(r["id"] for r in store.all("structures"))
            value = facts_of(row)
            value.update(schema_version=SCHEMA_VERSION, kind=KIND,
                         origin="player_built", significance="work")
            value.pop("parent", None)
            value.pop("role_evidence", None)
            store.put("structures", {
                "id": target, "actor": actor, "dim": row["dim"], "name": None,
                "min_x": lo[0], "min_y": lo[1], "min_z": lo[2],
                "max_x": hi[0], "max_y": hi[1], "max_z": hi[2],
                "materials": row["materials"],
            }, Belief(provenance=Provenance[row["provenance"]],
                      confidence=row["confidence"],
                      first_seen_tick=row["first_seen_tick"],
                      verified_at_tick=row["verified_at_tick"],
                      verified_at_ms=row["verified_at_ms"],
                      source_event_ids=tuple(json.loads(row["source_event_ids"] or "[]")),
                      value=json.dumps(value)))

            for relation in store.all("relationships", "subject = ?", (row["id"],)):
                relation_id = relation["id"].replace(row["id"], target)
                store.db.execute(
                    "UPDATE relationships SET id = ?, subject = ? WHERE id = ?",
                    (relation_id, target, relation["id"]))
            store.db.execute("UPDATE activities SET place_id = ? WHERE place_id = ?",
                             (target, row["id"]))
            store.db.execute("UPDATE event_history SET place_id = ? WHERE place_id = ?",
                             (target, row["id"]))
            store.db.execute("DELETE FROM generated_features WHERE id = ?", (row["id"],))
            moved.append((row["id"], target))
    return moved


def generated_root(store, row):
    """Resolve any measured or semantic generated component to its broad site."""
    seen = set()
    while row and row["id"] not in seen:
        seen.add(row["id"])
        parent_id = facts_of(row).get("parent")
        parent = store.get("generated_features", parent_id) if parent_id else None
        if not parent:
            break
        row = parent
    return row


def generated_near(store, dim: str, facts: dict):
    """Best generator-authored site/component containing an observed object.

    A seed location is an anchor, not a building box. Broad sites therefore provide local
    context, while a component learned from a workstation has building-scale reach. Player
    placement history still wins before callers use this as an origin decision.
    """
    centre = [(facts["lo"][i] + facts["hi"][i]) / 2 for i in range(3)]
    candidates = []
    for row in store.all("generated_features", "dim = ?", (dim,)):
        if row["x"] is None or row["z"] is None:
            continue
        component = bool(facts_of(row).get("parent"))
        distance = math.hypot(row["x"] - centre[0], row["z"] - centre[2])
        if component and row["min_x"] is not None:
            shared = intersection(box_of(row)[0], box_of(row)[1], facts["lo"], facts["hi"])
            inside = shared > 0
        else:
            radius = GENERATED_COMPONENT_RADIUS if component else GENERATED_SITE_RADIUS
            inside = distance <= radius
        if inside:
            candidates.append((0 if component else 1, distance, row["id"], row))
    return min(candidates, key=lambda item: item[:3])[3] if candidates else None


def normalize_generated_hierarchy(store) -> int:
    """Flatten accidental component-of-component chains to one broad generated parent."""
    changed = 0
    for row in store.all("generated_features"):
        value = facts_of(row)
        if not value.get("parent"):
            continue
        root = generated_root(store, row)
        if root and value.get("parent") != root["id"]:
            value["parent"] = root["id"]
            store.db.execute("UPDATE generated_features SET value = ? WHERE id = ?",
                             (json.dumps(value), row["id"]))
            changed += 1
    store.commit()
    return changed


def migrate_unknown_generated(store) -> list[tuple[str, str]]:
    """Move unowned spatial rows supported by generated-site evidence to that ontology.

    This is a lossless table correction, not a new classification. It only acts when there
    is no observed owner, no placement history strong enough to establish one, and a known
    generated site/component spatially supports the object.
    """
    moved = []
    with store.transaction():
        for row in list(current(store)):
            value = facts_of(row)
            if row["actor"] or value.get("origin") == "player_built":
                continue
            lo, hi = box_of(row)
            facts = {"lo": lo, "hi": hi, "materials": _materials(row),
                     "blocks": sum(_materials(row).values()),
                     "dims": value.get("dims") or "x".join(
                         str(hi[i] - lo[i] + 1) for i in range(3)),
                     "fill": value.get("fill"), "vertical": value.get("vertical")}
            if owner_from_work(store, row["dim"], facts):
                continue
            parent = generated_near(store, row["dim"], facts)
            if not parent:
                continue
            target = (parent["id"] if facts_of(parent).get("parent")
                      else new_generated_id(r["id"] for r in
                                            store.all("generated_features")))
            prior = facts_of(parent) if target == parent["id"] else {}
            generated_value = dict(value)
            generated_value.update(schema_version=SCHEMA_VERSION, kind=GENERATED_KIND,
                                   origin="world_generated")
            if target != parent["id"]:
                generated_value["parent"] = parent["id"]
            else:
                generated_value.update({k: v for k, v in prior.items()
                                        if k in ("parent", "role_evidence")})
            kind = (parent["kind"] if target == parent["id"]
                    else "generated_building")
            store.put("generated_features", {
                "id": target, "dim": row["dim"], "kind": kind,
                "x": int((lo[0] + hi[0]) / 2), "y": int((lo[1] + hi[1]) / 2),
                "z": int((lo[2] + hi[2]) / 2),
                "discovered_from": "segmentation within known generated site",
                "min_x": lo[0], "min_y": lo[1], "min_z": lo[2],
                "max_x": hi[0], "max_y": hi[1], "max_z": hi[2],
                "materials": row["materials"],
            }, Belief(provenance=Provenance[row["provenance"]],
                      confidence=row["confidence"],
                      first_seen_tick=min((v for v in
                                           (row["first_seen_tick"],
                                            parent["first_seen_tick"])
                                           if v is not None), default=None),
                      verified_at_tick=row["verified_at_tick"],
                      verified_at_ms=row["verified_at_ms"],
                      source_event_ids=tuple(json.loads(row["source_event_ids"] or "[]")),
                      value=json.dumps(generated_value)))
            for relation in store.all("relationships", "subject = ?", (row["id"],)):
                relation_id = relation["id"].replace(row["id"], target)
                existing = store.get("relationships", relation_id)
                if existing:
                    store.put("relationships", {
                        "id": relation_id, "subject": target,
                        "predicate": relation["predicate"], "object": relation["object"],
                    }, Belief(provenance=Provenance[relation["provenance"]],
                              confidence=relation["confidence"],
                              first_seen_tick=relation["first_seen_tick"],
                              verified_at_tick=relation["verified_at_tick"],
                              verified_at_ms=relation["verified_at_ms"],
                              source_event_ids=tuple(json.loads(
                                  relation["source_event_ids"] or "[]")),
                              value=relation["value"]))
                    store.db.execute("DELETE FROM relationships WHERE id = ?",
                                     (relation["id"],))
                else:
                    store.db.execute(
                        "UPDATE relationships SET id = ?, subject = ? WHERE id = ?",
                        (relation_id, target, relation["id"]))
            store.db.execute("UPDATE activities SET place_id = ? WHERE place_id = ?",
                             (target, row["id"]))
            store.db.execute("UPDATE event_history SET place_id = ? WHERE place_id = ?",
                             (target, row["id"]))
            store.db.execute("DELETE FROM structures WHERE id = ?", (row["id"],))
            moved.append((row["id"], target))
    return moved


def attribute_origins(store) -> int:
    """Attach player provenance to canonical objects when work history supports it.

    Absence of placement history means ``unknown``, never ``world_generated``. Generated
    features can only receive that origin from the seed-backed environment API and live in
    their own table. This prevents an old or incomplete session log from inventing an owner.
    """
    changed = 0
    for row in current(store):
        lo, hi = box_of(row)
        facts = {"lo": lo, "hi": hi, "materials": _materials(row),
                 "blocks": sum(_materials(row).values()),
                 "dims": [hi[i] - lo[i] + 1 for i in range(3)],
                 "fill": facts_of(row).get("fill"),
                 "vertical": facts_of(row).get("vertical")}
        actor = owner_from_work(store, row["dim"], facts)
        value = facts_of(row)
        # Ownership is derived from the complete work record on every replay. Keeping an
        # old player_built label after its supporting attribution disappears makes a small
        # renovation permanently own a vanilla building even after the evidence is fixed.
        origin = "player_built" if actor else "unknown"
        if value.get("origin") == origin and row["actor"] == actor:
            continue
        value["origin"] = origin
        store.db.execute("UPDATE structures SET actor = ?, value = ? WHERE id = ?",
                         (actor, json.dumps(value), row["id"]))
        changed += 1
    store.commit()
    return changed


def audit(store) -> list[str]:
    """Mechanical invariants for the current-world materialized view."""
    issues = []
    rows = store.all("structures")
    canonical = current(store)
    for row in rows:
        if not is_current(row):
            issues.append(f"{row['id']}: historical/noncanonical row in structures")
            continue
        lo, hi = box_of(row)
        if any(lo[i] > hi[i] for i in range(3)):
            issues.append(f"{row['id']}: inside-out bounding box")
        facts = facts_of(row)
        if facts.get("schema_version") != SCHEMA_VERSION or facts.get("kind") != KIND:
            issues.append(f"{row['id']}: missing canonical schema marker")
        if not row["verified_at_ms"]:
            issues.append(f"{row['id']}: no wall-clock observation time")
        if facts.get("origin") == "world_generated":
            issues.append(f"{row['id']}: generated feature stored with player structures")
    for i, left in enumerate(canonical):
        for right in canonical[i + 1:]:
            rlo, rhi = box_of(right)
            score = match_score(left, {"dim": right["dim"], "lo": rlo, "hi": rhi,
                                       "materials": right["materials"]})
            if score >= 0.60:
                issues.append(f"{left['id']} and {right['id']}: likely duplicate ({score:.2f})")
    return issues
