#!/usr/bin/env python3
"""Deterministic grounding for claims about current places.

Names help resolve a player's words to canonical ids; they never establish identity or
topology. Spatial facts come from measured boxes. Direct physical connection is checked
from live voxels only when two boxes touch or overlap. Semantic membership ("part of") is
not geometry and remains unknown unless the store gains explicit evidence for it.
"""

from __future__ import annotations

import itertools
import json
import math
import re

from structures import (box_of, current, current_generated, facts_of, intersection,
                        volume)

STOPWORDS = {
    "a", "an", "and", "at", "by", "for", "in", "is", "made", "of", "on", "the",
    "to", "with", "wooden", "stone", "cobblestone", "birch", "timber", "oak",
}
SPATIAL_WORDS = {
    "above", "attached", "below", "beside", "between", "connect", "connected",
    "east", "inside", "join", "joined", "near", "north", "overlap", "part", "south",
    "touch", "touching", "west",
}
DEICTIC_WORDS = {"here", "nearby", "this", "these", "around"}
REFERENCE_STOPWORDS = SPATIAL_WORDS | DEICTIC_WORDS | {
    "about", "current", "does", "place", "stand", "still", "tell", "thing", "what",
    "where", "which",
}
SPATIAL_STEMS = ("attach", "connect", "join", "overlap", "part", "touch")

# Block families are world semantics, not object-specific patches. They let an answer know
# that panes and stained glass are both windows, or that lanterns and torches are both light.
FEATURE_MATERIALS = {
    "windows/glass": ("glass",),
    "lighting": ("torch", "lantern", "glowstone", "sea_lantern", "shroomlight",
                 "froglight", "candle", "redstone_lamp"),
    "entrances": ("door", "trapdoor", "fence_gate"),
    "foliage": ("leaves", "flower", "sapling", "azalea", "vine", "moss", "grass"),
    "storage": ("chest", "barrel", "shulker_box"),
    "rails": ("rail",),
    "fences": ("fence", "wall"),
    "stairs/slabs": ("stairs", "slab"),
}

ADVICE_TERMS = {
    "windows/glass": ("window", "windows", "glass"),
    "lighting": ("light", "lights", "lighting", "torch", "lantern"),
    "entrances": ("door", "doors", "entrance", "entry"),
    "foliage": ("plant", "plants", "flower", "flowers", "garden", "greenery"),
    "storage": ("storage", "chest", "barrel"),
    "rails": ("rail", "rails", "track", "tracks"),
    "fences": ("fence", "fences", "wall", "walls"),
    "stairs/slabs": ("stair", "stairs", "slab", "slabs"),
}


def _tokens(text: str) -> set[str]:
    return {word for word in re.findall(r"[a-z0-9]+", text.lower())
            if len(word) > 1 and word not in STOPWORDS}


def _is_spatial_word(word: str) -> bool:
    return word in SPATIAL_WORDS or word.startswith(SPATIAL_STEMS)


def _spatial_intent(text: str) -> bool:
    return any(_is_spatial_word(word) for word in _tokens(text))


def _name(store, row) -> tuple[str | None, float | None]:
    named = store.structure_name(row["id"])
    return ((named["object"], store.confidence_now("relationships", named))
            if named else (None, None))


def _reference_words(store, row) -> set[str]:
    """Words already used in the stored name/description; no generated synonym list."""
    named = store.structure_name(row["id"])
    if not named:
        return set()
    words = _tokens(named["object"] or "")
    try:
        value = json.loads(named["value"] or "{}")
    except (TypeError, ValueError):
        value = {}
    words |= _tokens((value.get("description") or "") + " " +
                     (value.get("rationale") or ""))
    return words


def _centre(row) -> tuple[float, float, float]:
    return tuple((row[f"min_{axis}"] + row[f"max_{axis}"]) / 2
                 for axis in "xyz")


def material_features(materials) -> list[str]:
    """Stable semantic features present in a material census."""
    if isinstance(materials, str):
        try:
            materials = json.loads(materials or "{}")
        except (TypeError, ValueError):
            materials = {}
    names = [str(name).replace("minecraft:", "").split("[")[0]
             for name in (materials or {})]
    return [feature for feature, needles in FEATURE_MATERIALS.items()
            if any(any(needle in name for needle in needles) for name in names)]


def advice_conflicts(speech: str, grounded: dict) -> list[str]:
    """Features a draft proposes as missing even though every subject already has them."""
    subjects = [item for item in grounded.get("structures", [])
                if item.get("role") == "referenced"]
    if not subjects:
        return []
    conflicts = []
    sentences = re.split(r"(?<=[.!?;])\s+", speech.lower())
    for sentence in sentences:
        if not re.search(
                r"\b(add|install|put|place|needs?|could use|should have|would benefit from)\b",
                sentence):
            continue
        for feature, terms in ADVICE_TERMS.items():
            terms_pattern = "|".join(re.escape(term) for term in terms)
            mentions = list(re.finditer(rf"\b(?:{terms_pattern})\b", sentence))
            if not mentions:
                continue
            # "More/larger windows" acknowledges the feature and can be legitimate
            # refinement. Scope that qualifier to this feature, not some other suggestion
            # elsewhere in the answer.
            prefixes = [sentence[max(0, match.start() - 28):match.start()]
                        for match in mentions]
            refined = any(re.search(
                r"\b(more|larger|wider|replace|rework|expand|different)(?:\s+\w+){0,2}\s*$",
                prefix) for prefix in prefixes)
            if not refined and all(feature in item.get("existing_features", [])
                                   for item in subjects):
                conflicts.append(feature)
    return list(dict.fromkeys(conflicts))


def distance_to_point(row, pos) -> float:
    """Euclidean distance from a point to the nearest point of a closed block box."""
    if not pos:
        return math.inf
    gaps = [max(row[f"min_{axis}"] - pos[i], 0, pos[i] - row[f"max_{axis}"])
            for i, axis in enumerate("xyz")]
    return math.sqrt(sum(gap * gap for gap in gaps))


def resolve_references(store, text: str, dim: str | None = None, pos=None,
                       limit: int = 8, strengths: dict | None = None) -> list:
    """Resolve names and explicit ids without choosing between ambiguous matches.

    Both ontologies are searched. A generated place is not the player's work, but it is a
    place they can name and stand in, and a vocabulary that omits it makes "the church" and
    "the village house" unresolvable — which in turn made the god refuse its own correct
    answer for mentioning one.
    """
    rows = current(store, dim) + current_generated(store, dim)
    query = text.lower()
    query_tokens = _tokens(text)
    primary, descriptive = [], []
    for row in rows:
        label, _ = _name(store, row)
        explicit = row["id"].lower() in query
        label_tokens = {word for word in _tokens(label or "")
                        if word not in REFERENCE_STOPWORDS and not _is_spatial_word(word)}
        description_tokens = {word for word in _reference_words(store, row) - label_tokens
                              if word not in REFERENCE_STOPWORDS
                              and not _is_spatial_word(word)}
        shared = query_tokens & label_tokens
        descriptive_shared = query_tokens & description_tokens
        # One meaningful noun is enough ("the cottage"); material/style adjectives are
        # excluded above. If "rail" names two objects, both survive resolution.
        lexical = len(shared)
        phrase = bool(label and label.lower() in query)
        if explicit or phrase or lexical:
            primary.append((3 if explicit else 2 if phrase else 1, lexical,
                            -distance_to_point(row, pos), row))
        elif descriptive_shared:
            descriptive.append((1, len(descriptive_shared),
                                -distance_to_point(row, pos), row))

    # Stored descriptions are useful aliases ("lookout" for a watchtower), but only when
    # the query did not directly match any current label. Otherwise incidental prose such
    # as "the rail is visible from the tower" turns every nearby building into "the rail".
    scored = primary or descriptive

    if not scored and query_tokens & DEICTIC_WORDS and pos:
        # Deictic language has no name to resolve. Preserve ambiguity among equally nearby
        # structures rather than silently deciding what "this" means.
        ordered = sorted(rows, key=lambda row: (distance_to_point(row, pos), row["id"]))
        if ordered:
            nearest = distance_to_point(ordered[0], pos)
            scored = [(0, 0, -distance_to_point(row, pos), row) for row in ordered
                      if distance_to_point(row, pos) <= min(12.0, nearest + 3.0)]

    scored.sort(key=lambda item: (-item[0], -item[1], -item[2], item[3]["id"]))
    chosen = scored[:limit]
    if strengths is not None:
        # How each one matched. A name is a reference; a word that merely appears in a
        # stored description is a coincidence, and treating the two alike made the god
        # refuse its own answer because "remains" occurs in a description somewhere.
        by_name = {id(item) for item in primary}
        for item in chosen:
            strengths[item[3]["id"]] = "name" if id(item) in by_name else "description"
    return [item[3] for item in chosen]


def box_relation(left, right) -> dict:
    """Exact, symmetric spatial measurements between two canonical boxes."""
    alo, ahi = box_of(left)
    blo, bhi = box_of(right)
    # Empty blocks separating the boxes. Zero means their occupied coordinates touch or
    # overlap on that axis.
    gaps = [max(blo[i] - ahi[i] - 1, alo[i] - bhi[i] - 1, 0) for i in range(3)]
    ac, bc = _centre(left), _centre(right)
    dx, dy, dz = bc[0] - ac[0], bc[1] - ac[1], bc[2] - ac[2]
    horizontal = math.hypot(gaps[0], gaps[2])
    vertical_overlap = min(ahi[1], bhi[1]) >= max(alo[1], blo[1])
    footprint_overlap = (min(ahi[0], bhi[0]) >= max(alo[0], blo[0])
                         and min(ahi[2], bhi[2]) >= max(alo[2], blo[2]))
    shared_volume = intersection(alo, ahi, blo, bhi)
    direction = []
    if abs(dx) >= abs(dz) and dx:
        direction.append("east" if dx > 0 else "west")
    if abs(dz) >= abs(dx) and dz:
        direction.append("south" if dz > 0 else "north")
    if abs(dy) > max(abs(dx), abs(dz)) / 2 and dy:
        direction.append("above" if dy > 0 else "below")
    contains = None
    if intersection(alo, ahi, blo, bhi) == volume(blo, bhi):
        contains = right["id"]
    elif intersection(alo, ahi, blo, bhi) == volume(alo, ahi):
        contains = left["id"]
    return {
        "left": left["id"], "right": right["id"],
        "relation_provenance": "DERIVED from current canonical geometry",
        "empty_gap_xyz": gaps,
        "horizontal_gap": round(horizontal, 2),
        "direction_from_left": direction or ["same centre"],
        "vertical_overlap": vertical_overlap,
        "footprint_overlap": footprint_overlap,
        "shared_box_volume": shared_volume,
        "beside": horizontal <= 2 and vertical_overlap,
        "beside_rule": "horizontal_gap <= 2 and vertical_overlap",
        "bbox_contains": contains,
        "semantic_part_of": "unknown",
    }


def _physically_connected(voxels: dict, left, right) -> bool | None:
    """Whether blocks in two boxes share one face-connected constructed component."""
    from segment import built_blocks

    built = set(built_blocks(voxels))
    alo, ahi = box_of(left)
    blo, bhi = box_of(right)
    a = {p for p in built if all(alo[i] <= p[i] <= ahi[i] for i in range(3))}
    b = {p for p in built if all(blo[i] <= p[i] <= bhi[i] for i in range(3))}
    if not a or not b:
        return None
    if a & b:
        return True
    neighbours = ((1, 0, 0), (-1, 0, 0), (0, 1, 0),
                  (0, -1, 0), (0, 0, 1), (0, 0, -1))
    seen = set(a)
    frontier = list(a)
    while frontier:
        x, y, z = frontier.pop()
        for ox, oy, oz in neighbours:
            point = (x + ox, y + oy, z + oz)
            if point in b:
                return True
            if point in built and point not in seen:
                seen.add(point)
                frontier.append(point)
    return False


async def _connection(store, left, right, url: str) -> tuple[bool | None, str]:
    from scan import request_voxels, to_voxels

    alo, ahi = box_of(left)
    blo, bhi = box_of(right)
    lo = [min(alo[i], blo[i]) - 1 for i in range(3)]
    hi = [max(ahi[i], bhi[i]) + 1 for i in range(3)]
    cells = math.prod(hi[i] - lo[i] + 1 for i in range(3))
    if cells > 250_000:
        return None, f"topology span is too large for a bounded read ({cells} cells)"
    try:
        result = await request_voxels(url, left["dim"], lo, hi)
    except Exception as error:
        return None, f"live topology unavailable ({type(error).__name__})"
    if not result.get("ok"):
        return None, "live topology unavailable"
    voxels = to_voxels(result)
    if not voxels:
        return None, "empty live topology read refused"
    return _physically_connected(voxels, left, right), "SCANNED live block topology"


def standing_in(store, dim: str, pos):
    """The innermost place on record that contains a point, and which table it lives in.

    Asked "what structure am I in right now" while standing in a village house, the god
    refused: it had rendered the place, described it correctly, and then its own guard
    rejected the answer because nothing had grounded it. Grounding only ever considered
    player-built structures, so a generated building could never be named — the god could
    see where it was and was not allowed to say.

    Where someone is standing is a fact, and a deterministic one. The smallest containing
    box wins, so a house inside a village is the house rather than the village.
    """
    if not pos:
        return None, None
    best = None
    for table, rows in (("structures", current(store, dim)),
                        ("generated_features", current_generated(store, dim))):
        for row in rows:
            lo, hi = box_of(row)
            if not all(lo[i] <= pos[i] <= hi[i] for i in range(3)):
                continue
            size = volume(lo, hi)
            if best is None or size < best[0]:
                best = (size, table, row)
    if best is None:
        return None, None
    return best[2], best[1]


def within_site(store, actor: str | None):
    """The broad generated place the player is currently in, from observed presence.

    Distinct from ``standing_in``, and deliberately so. That is containment in a measured
    box and is a fact about geometry. This is a village site located from the seed, which
    has an anchor and no bounds, so presence is a radius rather than a box. Reporting the
    two the same way is how "the closest village" once became the one underfoot; kept apart,
    "you are in the plains village, in its church" is exactly right.
    """
    if not actor:
        return None
    presence = store.get("actor_presence", actor)
    if presence is None or not presence["place_id"]:
        return None
    return store.get("generated_features", presence["place_id"])


async def ground_query(store, text: str, dim: str, pos, url: str,
                       actor: str | None = None) -> dict:
    """Resolve, verify, and relate the canonical objects a question can refer to."""
    from reconcile import confirm_before_speaking

    strengths: dict = {}
    matched = resolve_references(store, text, dim, pos, strengths=strengths)
    referenced = list(matched)
    roles = {row["id"]: ("referenced" if strengths.get(row["id"]) == "name"
                         else "loosely matched on a stored description")
             for row in matched}
    spatial = _spatial_intent(text)
    # A spatial question with no named subject gets a bounded local field, not the entire
    # world. These are candidates, explicitly not an interpretation of "this".
    if spatial and not referenced and pos:
        referenced = sorted(current(store, dim),
                            key=lambda row: (distance_to_point(row, pos), row["id"]))[:6]
        roles.update({row["id"]: "local candidate" for row in referenced})
    elif spatial and len(matched) == 1:
        # "What is beside the tower?" names the subject, not its answer. Bring a bounded
        # local field into the derived view and let exact gap measurements decide.
        subject = matched[0]
        candidates = []
        for row in current(store, dim):
            if row["id"] == subject["id"]:
                continue
            relation = box_relation(subject, row)
            distance = math.sqrt(sum(gap * gap for gap in relation["empty_gap_xyz"]))
            if distance <= 32:
                candidates.append((distance, row["id"], row))
        for _, _, row in sorted(candidates)[:6]:
            referenced.append(row)
            roles[row["id"]] = "local candidate"

    # Where the player is standing, whoever built it. Without this the god can render a
    # village house, describe it correctly, and then refuse its own answer for naming a
    # place nothing had grounded.
    here, here_table = standing_in(store, dim, pos)
    tables = {row["id"]: "structures" for row in referenced}
    if here is not None and here["id"] not in tables:
        referenced.append(here)
        roles[here["id"]] = "the player is standing in this"
        tables[here["id"]] = here_table
    elif here is not None:
        roles[here["id"]] = "the player is standing in this"

    structures = []
    for row in referenced:
        table = tables.get(row["id"], "structures")
        try:
            verification = (await confirm_before_speaking(store, row["id"], url)
                            if table == "structures" else None)
        except Exception as error:
            verification = {"ok": False,
                            "error": f"verification unavailable ({type(error).__name__})"}
        label, name_confidence = _name(store, row)
        item = {
            "id": row["id"], "name": label, "name_provenance": "INFERRED",
            "role": roles.get(row["id"], "local candidate"),
            "name_confidence_now": (round(name_confidence, 3)
                                    if name_confidence is not None else None),
            "bbox": [box_of(row)[0], box_of(row)[1]],
            "geometry_provenance": row["provenance"],
            "geometry_confidence_now": round(store.confidence_now(table, row), 3),
            "contradicted": store.contradicted(table, row["id"]),
            # A generated building is not one of the player's works, and saying so is the
            # difference between "your house" and "a village house you are standing in".
            "origin": ("world_generated" if table == "generated_features"
                       else facts_of(row).get("origin") or "unknown"),
            "verification": "not needed; canonical observation is fresh",
            "existing_features": material_features(row["materials"]),
            "representative_materials": [
                name.replace("minecraft:", "").split("[")[0]
                for name in list(json.loads(row["materials"] or "{}").keys())[:16]],
        }
        if verification is not None:
            item["verification"] = verification
        structures.append(item)

    relations = []
    if spatial:
        by_id = {row["id"]: row for row in referenced}
        pairs = (itertools.combinations(by_id, 2) if len(matched) != 1 else
                 ((matched[0]["id"], row["id"]) for row in referenced
                  if row["id"] != matched[0]["id"]))
        for left_id, right_id in pairs:
            left, right = by_id[left_id], by_id[right_id]
            relation = box_relation(left, right)
            if any(word.startswith(SPATIAL_STEMS) for word in _tokens(text)):
                connected, evidence = await _connection(store, left, right, url)
                relation["physically_connected"] = connected
                relation["connection_evidence"] = evidence
            relations.append(relation)

    site = within_site(store, actor)
    place = None
    if site is not None:
        label, _ = _name(store, site)
        place = {
            "id": site["id"], "kind": site["kind"], "name": label,
            "evidence": "observed presence within the site, not measured containment",
        }

    return {
        "rule": ("Only these canonical ids may support present-tense claims in this answer. "
                 "Names are INFERRED. `bbox_contains` is spatial containment, never proof "
                 "that one object is semantically part of another."),
        "within_place": place,
        "resolved_count": len(structures),
        "matched_count": len(matched),
        "ambiguous": len(matched) > 1,
        "structures": structures,
        "relations": relations,
    }


#: The biome variants a village generates in. A `village_<biome>` key names the whole
#: settlement and reads naturally reversed; a `village_<role>` key names one building in it
#: and reads naturally with the prefix dropped. Without the distinction a village_church
#: becomes a "church village", which is a different and non-existent place.
VILLAGE_BIOMES = ("plains", "desert", "savanna", "snowy", "taiga")


def _plain_place(kind: str) -> str:
    """A generator key as a person would say it: village_plains is a plains village."""
    bare = (kind or "").replace("minecraft:", "")
    for family in ("village", "ruined_portal", "ocean_ruin"):
        if not bare.startswith(family + "_"):
            continue
        rest = bare[len(family) + 1:]
        if family != "village" or rest in VILLAGE_BIOMES:
            return f"{rest} {family}".replace("_", " ")
        return rest.replace("_", " ")
    return bare.replace("_", " ")


def _render_place(grounded: dict) -> list[str]:
    place = grounded.get("within_place")
    if not place:
        return []
    what = place.get("name") or _plain_place(place.get("kind") or "")
    return [f"[OBSERVED] the player is currently within {what} "
            f"({place['evidence']})"]


def render_grounding(grounded: dict, immersive: bool = False) -> str:
    """Compact prompt text that keeps evidence levels attached to every claim."""
    lines = ["GROUNDING FOR THIS QUESTION:", grounded["rule"]]
    lines.extend(_render_place(grounded))
    if not grounded["structures"]:
        lines.append("No individual canonical structure was resolved. Do not invent one; "
                     "authoritative player or environment facts may still answer the question.")
        return "\n".join(lines)
    if grounded.get("ambiguous"):
        lines.append("The wording resolves to multiple objects; preserve that ambiguity.")
    for item in grounded["structures"]:
        verification = (json.dumps(item["verification"], sort_keys=True)
                        if isinstance(item["verification"], dict)
                        else item["verification"])
        bbox = "" if immersive else f" bbox={item['bbox']}"
        lines.append(
            f"[{item['id']}] name={item['name']!r} [INFERRED "
            f"{item['name_confidence_now']}] role={item['role']}{bbox} "
            f"[{item['geometry_provenance']}] features already present="
            f"{item.get('existing_features', [])}; representative materials="
            f"{item.get('representative_materials', [])}; verification={verification}")
    for relation in grounded["relations"]:
        lines.append("RELATION " + json.dumps(relation, sort_keys=True))
    return "\n".join(lines)
