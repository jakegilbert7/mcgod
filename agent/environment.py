#!/usr/bin/env python3
"""Seed-backed environment knowledge, kept separate from player-built structures.

Biome samples and vanilla structure locations describe the generated world. They are useful
world state, but they are not works and never enter ``structures`` or ``work_events``.
Coordinates remain internal evidence; prompts receive cardinal, landmark-style summaries.
"""

from __future__ import annotations

import collections
import hashlib
import json
import math
import re
import time

from scan import request_rpc
from store import Belief, Provenance

ENVIRONMENT_WORDS = {
    "area", "around", "beach", "biome", "coast", "desert", "direction", "east",
    "environment", "forest", "jungle", "lake", "land", "nearby", "north", "ocean",
    "sea", "shore", "south", "swamp", "terrain", "west", "world",
}
GENERATED_WORDS = {
    "city", "fortress", "generated", "igloo", "monument", "outpost", "pyramid", "ruin",
    "ruins", "shipwreck", "stronghold", "structure", "structures", "temple", "village",
    "witch", "mansion", "mineshaft",
}

# These are aliases for generator registry keys, not classifications of scanned builds.
FEATURE_ALIASES = {
    "village": ("village_plains", "village_desert", "village_savanna", "village_snowy",
                "village_taiga"),
    "shipwreck": ("shipwreck", "shipwreck_beached"),
    "monument": ("monument",),
    "stronghold": ("stronghold",),
    "mineshaft": ("mineshaft", "mineshaft_mesa"),
    "fortress": ("fortress",),
    "mansion": ("mansion",),
    "outpost": ("pillager_outpost",),
    "igloo": ("igloo",),
    "pyramid": ("desert_pyramid", "jungle_pyramid"),
    "temple": ("desert_pyramid", "jungle_pyramid"),
    "ruin": ("ocean_ruin_cold", "ocean_ruin_warm", "trail_ruins"),
    "ruins": ("ocean_ruin_cold", "ocean_ruin_warm", "trail_ruins"),
    "city": ("ancient_city", "end_city"),
    "witch": ("swamp_hut",),
}


def words(text: str) -> set[str]:
    return set(re.findall(r"[a-z]+", text.lower()))


def wants_environment(text: str) -> bool:
    found = words(text)
    return bool(found & (ENVIRONMENT_WORDS | GENERATED_WORDS))


def requested_features(text: str) -> list[str]:
    found = words(text)
    keys = []
    for word, aliases in FEATURE_ALIASES.items():
        if word in found:
            keys.extend(aliases)
    # Preserve registry order and enforce the same budget as the server.
    return list(dict.fromkeys(keys))[:8]


def _compass(dx: int, dz: int) -> str:
    """Eight-way in-world direction for a point relative to the player."""
    angle = math.atan2(dz, dx)
    names = ("east", "southeast", "south", "southwest",
             "west", "northwest", "north", "northeast")
    return names[round(angle / (math.pi / 4)) % 8]


def generated_feature_answer(result: dict, requested: list[str]) -> str | None:
    """A concise answer from an authoritative locate result, with no model improvisation."""
    if not result.get("ok") or not requested:
        return None
    features = result.get("generated_features") or []
    families = [word for word, aliases in FEATURE_ALIASES.items()
                if any(alias in requested for alias in aliases)]
    family = families[0] if families else "generated structure"
    if not features:
        return (f"I find no {family} within roughly "
                f"{int(result.get('radius') or 0)} blocks of here.")
    cx, cz = result["center"]
    feature = min(features, key=lambda item: math.hypot(
        item["pos"][0] - cx, item["pos"][2] - cz))
    dx, dz = feature["pos"][0] - cx, feature["pos"][2] - cz
    exact_distance = math.hypot(dx, dz)
    distance = int(round(exact_distance / 50.0) * 50)
    key = _plain(feature["kind"])
    # Variant suffixes describe generator internals; the player asked for the family.
    family = next((word for word, aliases in FEATURE_ALIASES.items()
                   if any(alias.replace("_", " ") == key for alias in aliases)), family)
    if distance == 0:
        return f"The nearest {family} is here."
    return f"The nearest {family} lies {_compass(dx, dz)}, roughly {distance} blocks away."


async def request_environment(url: str, dim: str, pos, radius: int = 256,
                              step: int = 16, features: list[str] = (),
                              timeout: float = 25.0) -> dict:
    return await request_rpc(url, {
        "rpc": "environment", "dim": dim,
        "x": int(pos[0]), "z": int(pos[2]),
        "radius": radius, "step": step, "features": list(features),
    }, "environment_result", timeout, max_size=16 * 1024 * 1024)


def _plain(biome: str) -> str:
    return biome.replace("minecraft:", "").replace("_", " ")


def _sector(dx: int, dz: int) -> str:
    if dx == dz == 0:
        return "here"
    if abs(dx) >= abs(dz):
        return "east" if dx > 0 else "west"
    return "south" if dz > 0 else "north"


def _describe_biomes(samples: list) -> dict[str, str]:
    sectors: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    for dx, dz, raw in samples:
        sector, biome = _sector(int(dx), int(dz)), _plain(str(raw))
        sectors[sector][biome] += 1

    descriptions = {}
    for sector, counts in sectors.items():
        total = sum(counts.values()) or 1
        ocean = sum(n for biome, n in counts.items() if "ocean" in biome)
        beach = sum(n for biome, n in counts.items() if "beach" in biome)
        notable = []
        if beach:
            notable.append("beach")
        if ocean:
            notable.append("ocean")
        dominant = [name for name, _ in counts.most_common(3)
                    if name not in notable and "ocean" not in name and "beach" not in name]
        parts = notable + dominant[:max(1, 3 - len(notable))]
        descriptions[sector] = ", then ".join(parts)
        if ocean / total >= .15 and beach:
            descriptions[sector] = "a beach giving way to ocean" + (
                f", with {dominant[0]} inland" if dominant else "")
    return descriptions


def persist_environment(store, result: dict) -> None:
    """Persist generator facts without ever touching the player structure tables."""
    if not result.get("ok"):
        return
    now = int(time.time() * 1000)
    dim = result["dim"]
    cx, cz = result["center"]
    raw = json.dumps(result.get("biomes") or [], separators=(",", ":"))
    digest = hashlib.sha1(f"{dim}:{cx}:{cz}:{result['radius']}:{result['step']}".encode()).hexdigest()[:16]
    # The seed itself, recorded once from the server that reported it. Everything the
    # seed map computes depends on this one number being right, and level.dat stopped
    # carrying it in 26.2, so the authoritative source is the running generator.
    if result.get("seed") is not None:
        store.put("relationships", {
            "id": "world:seed", "subject": "world", "predicate": "seed",
            "object": str(int(result["seed"])),
        }, Belief(provenance=Provenance.SCANNED, verified_at_ms=now,
                  value=json.dumps({"dim": dim})))
    store.put("terrain_regions", {
        "id": f"terrain:{digest}", "dim": dim, "center_x": cx, "center_z": cz,
        "radius": result["radius"], "step": result["step"], "biomes": raw,
    }, Belief(provenance=Provenance.SCANNED, verified_at_ms=now,
              value=json.dumps({"seed": result.get("seed")})))
    for feature in result.get("generated_features") or []:
        x, y, z = feature["pos"]
        kind = feature["kind"]
        identity = hashlib.sha1(f"{dim}:{kind}:{x}:{z}".encode()).hexdigest()[:16]
        store.put("generated_features", {
            "id": f"generated:{identity}", "dim": dim, "kind": kind,
            "x": x, "y": y, "z": z, "discovered_from": "world_generator",
        }, Belief(provenance=Provenance.SCANNED, verified_at_ms=now,
                  value=json.dumps({"origin": "world_generated"})))


def render_environment(result: dict) -> str:
    if not result.get("ok"):
        return "ENVIRONMENT: the seed-backed query was unavailable."
    lines = ["ENVIRONMENT [SCANNED from the world's generator]:"]
    descriptions = _describe_biomes(result.get("biomes") or [])
    for direction in ("here", "north", "east", "south", "west"):
        if direction in descriptions:
            lines.append(f"  {direction}: {descriptions[direction]}")
    for feature in result.get("generated_features") or []:
        x, _, z = feature["pos"]
        dx, dz = x - result["center"][0], z - result["center"][1]
        distance = round(math.hypot(dx, dz))
        direction = _sector(dx, dz)
        kind = _plain(feature["kind"])
        lines.append(f"  generated {kind}: {direction}, about {distance} blocks away")
    lines.append("Generated features above are world-authored, never player-built structures.")
    return "\n".join(lines)


def render_known_features(store, dim: str, pos, limit: int = 12) -> str:
    """Recall generator-authored features as directions, never player works or coordinates."""
    rows = store.all("generated_features", "dim = ?", (dim,))
    rows = sorted(rows, key=lambda row: math.hypot(
        (row["x"] or 0) - pos[0], (row["z"] or 0) - pos[2]))[:limit]
    if not rows:
        return ""
    lines = ["KNOWN GENERATED FEATURES [separate from player works]:"]
    for row in rows:
        dx, dz = (row["x"] or 0) - pos[0], (row["z"] or 0) - pos[2]
        named = store.structure_name(row["id"])
        label = named["object"] if named else _plain(row["kind"] or "unknown feature")
        lines.append(f"  {label}: {_sector(dx, dz)}, about {round(math.hypot(dx, dz))} "
                     "blocks away")
    return "\n".join(lines)


def summarize_voxels(voxels: dict, pos) -> str:
    """Fallback landmark summary from a bounded live terrain render."""
    columns: dict[tuple[int, int], tuple[int, str]] = {}
    for (x, y, z), material in voxels.items():
        key = (x, z)
        if key not in columns or y > columns[key][0]:
            columns[key] = (y, material.replace("minecraft:", "").split("[")[0])
    sectors: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    for (x, z), (_, material) in columns.items():
        sectors[_sector(x - pos[0], z - pos[2])][material] += 1
    lines = ["LIVE TERRAIN [SCANNED from the rendered area]:"]
    for direction in ("north", "east", "south", "west"):
        counts = sectors.get(direction, {})
        water = sum(v for k, v in counts.items() if "water" in k)
        shore = sum(v for k, v in counts.items() if any(s in k for s in ("sand", "gravel")))
        total = sum(counts.values()) or 1
        if water / total > .15:
            place = "shore and open water" if shore else "open water"
        else:
            common = sorted(counts.items(), key=lambda pair: -pair[1])[:2]
            place = ", ".join(k.replace("_", " ") for k, _ in common) or "unseen"
        lines.append(f"  {direction}: {place}")
    return "\n".join(lines)
