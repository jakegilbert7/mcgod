#!/usr/bin/env python3
"""The world as the generator would make it, without visiting it.

    python3 seedmap.py --biome 300 63 -300
    python3 seedmap.py --nearest village --at 300 -300
    python3 seedmap.py --count village --at 300 -300 --radius 10000
    python3 seedmap.py --check

Every other source of world knowledge in McGod requires someone to have been somewhere. The
event tap sees what a player did; a scan reads chunks the server has loaded; the plugin's
environment service asks the running generator, which is exact but needs the server up and
answers "where is the nearest one", never "how many are there".

Minecraft's generation is a pure function of the seed and the version, so all of it is
computable without visiting any of it. This binds cubiomes, which reimplements that
function in C, vendored under ``cubiomes/`` with a patch that carries it to 26.2.

Three rules keep this honest, and they are the whole reason it is a separate module:

* **This is the world as generated, not the world as it stands.** A village a player burned
  down is still here; a house they built is not. Anything present-tense needs a scan.
* **It is exact only for the version it was built for.** Generation changes between
  releases. ``verify.py`` compares it against the live server; if a future version drifts,
  that comparison fails rather than the god quietly inventing terrain.
* **It never writes to the store.** Results are evidence for one answer. Persisting a
  thousand computed village sites would drown the record of places anyone has actually
  been, which is what ``generated_features`` is for.
"""

from __future__ import annotations

import argparse
import ctypes
import functools
import json
import math
import os
import struct
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).parent
SOURCE_DIR = HERE / "cubiomes"
LIBRARY = SOURCE_DIR / "libmcgod_cubiomes.dylib"
SOURCES = ("biomenoise.c", "biomes.c", "finders.c", "generator.c", "layers.c",
           "noise.c", "quadbase.c", "util.c", "mcgod_shim.c")

DIMENSIONS = {"overworld": 0, "the_nether": -1, "nether": -1, "the_end": 1, "end": 1}

#: Which dimension each structure generates in. A caller asking for fortresses in the
#: overworld is asking a question with no answer, and should be told so rather than handed
#: an empty list that reads as "there are none".
STRUCTURE_DIMENSION = {
    "fortress": -1, "bastion": -1, "ruined_portal_n": -1,
    "end_city": 1, "end_gateway": 1,
}

#: Structures whose placement cubiomes exposes. Names match the game's, with the aliases a
#: person is likely to type.
STRUCTURES = (
    "village", "pillager_outpost", "desert_pyramid", "jungle_pyramid", "swamp_hut",
    "igloo", "ocean_ruin", "shipwreck", "monument", "mansion", "ruined_portal",
    "ancient_city", "buried_treasure", "mineshaft", "desert_well", "geode",
    "trail_ruins", "trial_chambers", "fortress", "bastion", "end_city", "end_gateway",
)

#: How well each structure's terrain check matched the live server, measured by taking the
#: seed map's own results and asking Paper to locate from each position (see the seed-map
#: tests). Placement is exact for every type; what varies is the check that decides whether
#: a candidate really becomes a structure, because the game consults surface height and
#: cubiomes cannot. A god that cannot see which of its answers are approximate cannot hedge,
#: so this travels with every result.
#:
#: `exact` matched on every sample. `approximate` carries a measured miss rate and should be
#: confirmed against the running server before a specific position is stated as fact.
#: `placement_only` means the server cannot locate that type at all, so the position is the
#: generation attempt and nothing has checked it.
ACCURACY = {
    "village": "exact", "pillager_outpost": "exact", "swamp_hut": "exact",
    "monument": "exact", "mansion": "exact", "trial_chambers": "exact",
    "igloo": "exact", "ruined_portal": "exact", "ocean_ruin": "exact",
    "buried_treasure": "exact",
    "ancient_city": "approximate", "shipwreck": "approximate",
    "jungle_pyramid": "approximate", "trail_ruins": "approximate",
    "desert_pyramid": "approximate",
    "desert_well": "placement_only", "geode": "placement_only",
    "mineshaft": "placement_only",
}
#: Roughly how often a reported position of an `approximate` type was not there, measured
#: on the test world. Reported so the model can weigh a count, not to correct one.
MISS_RATE = {"desert_pyramid": 0.53, "trail_ruins": 0.17, "shipwreck": 0.13,
             "jungle_pyramid": 0.13, "ancient_city": 0.07}

#: Refuse to sweep more ground than this in one call. A radius is quadratic in work, and a
#: model that asks for a million blocks should get a refusal it can read rather than a
#: process that stops answering.
MAX_RADIUS = 50_000
MAX_GRID_POINTS = 400_000


#: Said on every result, because a god that forgets it will describe a village the player
#: burned down as though it still stood.
AS_GENERATED = ("This is the world as the seed generates it. It does not know what anyone "
                "has since built, mined or destroyed.")


class SeedMapUnavailable(RuntimeError):
    """The seed map cannot answer. Never silently substituted with a guess."""


# --------------------------------------------------------------------------- the library

def build(force: bool = False) -> Path:
    """Compile the vendored cubiomes. Cheap, cached, and never done behind the user's back
    without saying so on failure."""
    sources = [SOURCE_DIR / name for name in SOURCES]
    missing = [s for s in sources if not s.exists()]
    if missing:
        raise SeedMapUnavailable(
            "vendored cubiomes sources are missing: "
            + ", ".join(s.name for s in missing))
    if not force and LIBRARY.exists():
        newest = max(s.stat().st_mtime for s in sources
                     + list((SOURCE_DIR / "tables").glob("*.h"))
                     + list(SOURCE_DIR.glob("*.h")))
        if LIBRARY.stat().st_mtime >= newest:
            return LIBRARY
    compiler = os.environ.get("CC", "cc")
    command = [compiler, "-O2", "-fPIC", "-shared", "-o", str(LIBRARY),
               *[str(s) for s in sources], "-lm"]
    try:
        done = subprocess.run(command, capture_output=True, text=True, cwd=SOURCE_DIR)
    except OSError as error:
        raise SeedMapUnavailable(f"cannot run {compiler}: {error}") from error
    if done.returncode != 0:
        raise SeedMapUnavailable(
            f"building cubiomes failed:\n{done.stderr.strip()[:2000]}")
    return LIBRARY


@functools.cache
def _library():
    lib = ctypes.CDLL(str(build()))
    lib.mcgod_open.restype = ctypes.c_void_p
    lib.mcgod_open.argtypes = [ctypes.c_int, ctypes.c_ulonglong, ctypes.c_int]
    lib.mcgod_close.argtypes = [ctypes.c_void_p]
    lib.mcgod_biome_name.restype = ctypes.c_char_p
    lib.mcgod_biome_name.argtypes = [ctypes.c_int, ctypes.c_int]
    lib.mcgod_biome_id.argtypes = [ctypes.c_char_p]
    lib.mcgod_version_name.restype = ctypes.c_char_p
    lib.mcgod_version_name.argtypes = [ctypes.c_int]
    lib.mcgod_version_from_string.argtypes = [ctypes.c_char_p]
    lib.mcgod_biome_at.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int,
                                   ctypes.c_int, ctypes.c_int,
                                   ctypes.POINTER(ctypes.c_ulonglong)]
    lib.mcgod_biome_grid.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int,
                                     ctypes.c_int, ctypes.c_int, ctypes.c_int,
                                     ctypes.c_int, ctypes.POINTER(ctypes.c_int)]
    lib.mcgod_structure_type.argtypes = [ctypes.c_char_p]
    lib.mcgod_region_size.argtypes = [ctypes.c_int, ctypes.c_int]
    lib.mcgod_structure_pos.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_ulonglong,
                                        ctypes.c_int, ctypes.c_int,
                                        ctypes.POINTER(ctypes.c_int),
                                        ctypes.POINTER(ctypes.c_int)]
    lib.mcgod_viable.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int]
    lib.mcgod_strongholds.argtypes = [ctypes.c_void_p, ctypes.c_int,
                                      ctypes.POINTER(ctypes.c_int)]
    lib.mcgod_spawn.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int)]
    lib.mcgod_nearest_biome.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int,
                                        ctypes.c_int, ctypes.c_int, ctypes.c_int,
                                        ctypes.c_int, ctypes.POINTER(ctypes.c_int),
                                        ctypes.POINTER(ctypes.c_int)]
    lib.mcgod_biome_histogram.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int,
                                          ctypes.c_int, ctypes.c_int, ctypes.c_int,
                                          ctypes.POINTER(ctypes.c_int)]
    return lib


def version_id(name: str | None = None) -> int:
    """The generation version to compute with, as cubiomes numbers them."""
    lib = _library()
    name = name or os.environ.get("MCGOD_SEED_MAP_VERSION")
    if not name:
        return lib.mcgod_newest_version()
    found = lib.mcgod_version_from_string(name.encode())
    if found < 0:
        raise SeedMapUnavailable(f"unknown Minecraft version for the seed map: {name}")
    return found


def version_name(mc: int | None = None) -> str:
    lib = _library()
    return lib.mcgod_version_name(lib.mcgod_newest_version() if mc is None
                                  else mc).decode()


# --------------------------------------------------------------------------- the seed

def seed_from_level_dat(path: Path | None = None) -> int | None:
    """Read the world seed out of level.dat.

    The NBT is gzipped and the field has moved between versions, so this looks for either
    spelling rather than parsing the whole tree: a wrong seed would make every answer here
    confidently wrong, so it is better to find nothing than to find the wrong long.
    """
    import gzip

    path = path or (HERE.parent / "server" / "world" / "level.dat")
    try:
        data = gzip.open(path, "rb").read()
    except (OSError, EOFError):
        return None
    for key in (b"RandomSeed", b"\x04\x00\x04seed"):
        index = data.find(key)
        if index >= 0:
            raw = data[index + len(key):index + len(key) + 8]
            if len(raw) == 8:
                return struct.unpack(">q", raw)[0]
    return None


def seed_from_store(store) -> int | None:
    """The seed the running server reported, as recorded by an environment query."""
    if store is None:
        return None
    try:
        row = store.get("relationships", "world:seed")
        if row is not None and row["object"]:
            return int(row["object"])
        rows = store.all("terrain_regions")
    except Exception:  # noqa: BLE001 - a store problem must not masquerade as no seed
        return None
    for row in rows:
        try:
            seed = (json.loads(row["value"] or "{}") or {}).get("seed")
        except (TypeError, ValueError):
            continue
        if seed is not None:
            return int(seed)
    return None


def world_seed(store=None, seed: int | None = None) -> int:
    """Where the seed comes from, in order of how much it can be trusted.

    An explicit argument, then the environment, then what the running server told us, then
    the world file. If none of those has it, this raises: computing a map for the wrong
    seed produces a completely plausible description of a world that does not exist, which
    is the single worst failure this module could have.
    """
    if seed is not None:
        return int(seed)
    from_env = os.environ.get("MCGOD_SEED")
    if from_env:
        return int(from_env)
    found = seed_from_store(store)
    if found is None:
        found = seed_from_level_dat()
    if found is None:
        raise SeedMapUnavailable(
            "no world seed is known: ask the server once (any environment query records "
            "it), set MCGOD_SEED, or make server/world/level.dat readable")
    return found


# --------------------------------------------------------------------------- the world

class World:
    """One seed, one version, one dimension. Cheap to make and safe to keep."""

    def __init__(self, seed: int, dim: str = "overworld", mc: int | None = None) -> None:
        if dim not in DIMENSIONS:
            raise SeedMapUnavailable(f"unknown dimension: {dim}")
        self.lib = _library()
        self.seed = int(seed)
        self.dim = dim
        self.mc = version_id() if mc is None else mc
        self.handle = self.lib.mcgod_open(self.mc, ctypes.c_ulonglong(self.seed),
                                          DIMENSIONS[dim])
        if not self.handle:
            raise SeedMapUnavailable("could not create a generator")

    def close(self) -> None:
        if getattr(self, "handle", None):
            self.lib.mcgod_close(ctypes.c_void_p(self.handle))
            self.handle = None

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def __del__(self):
        try:
            self.close()
        except Exception:  # noqa: BLE001 - interpreter shutdown
            pass

    def biome(self, x: int, y: int, z: int) -> str:
        """The biome at a block position, named as the game names it.

        Sampling happens on the quart grid the game actually stores biomes on, which is
        why the coordinates are shifted: asking at block resolution would interpolate a
        value the world does not have.
        """
        biome = self.lib.mcgod_biome_at(ctypes.c_void_p(self.handle), 4,
                                        x >> 2, y >> 2, z >> 2, None)
        return self.lib.mcgod_biome_name(self.mc, biome).decode()

    def biome_grid(self, lo_x: int, y: int, lo_z: int, width: int, height: int,
                   step: int = 4) -> list[list[str]]:
        """A rectangle of biomes, in one call into C.

        ``step`` is in blocks and is rounded to the quart grid, so 4 is every stored cell
        and larger values thin the sample out for a wide overview.
        """
        cells = max(1, step // 4)
        if width * height > MAX_GRID_POINTS:
            raise SeedMapUnavailable(
                f"{width * height} points is more than the {MAX_GRID_POINTS} allowed in "
                f"one grid; ask for a coarser step or a smaller area")
        buffer = (ctypes.c_int * (width * height))()
        # The C side walks a contiguous grid; a step wider than one cell is done here so
        # the fast path stays the common one.
        if cells == 1:
            self.lib.mcgod_biome_grid(ctypes.c_void_p(self.handle), 4, lo_x >> 2, y >> 2,
                                      lo_z >> 2, width, height, buffer)
            flat = list(buffer)
        else:
            flat = []
            for j in range(height):
                for i in range(width):
                    flat.append(self.lib.mcgod_biome_at(
                        ctypes.c_void_p(self.handle), 4,
                        (lo_x + i * step) >> 2, y >> 2, (lo_z + j * step) >> 2, None))
        names = {}
        for value in set(flat):
            names[value] = self.lib.mcgod_biome_name(self.mc, value).decode()
        return [[names[flat[j * width + i]] for i in range(width)]
                for j in range(height)]

    def biome_id(self, name: str) -> int:
        """The numeric id of a biome by name, or -1. Names are the game's own."""
        return self.lib.mcgod_biome_id(name.encode())

    def nearest_biome(self, name: str, x: int, z: int, y: int = 63,
                      max_radius: int = 10_000, step: int = 32) -> dict:
        """Where a named biome first appears, searching outward.

        ``step`` is how finely the search samples. A biome region smaller than the step can
        be stepped over entirely, so a miss is reported as a miss within that radius and at
        that resolution, never as "this biome does not exist".
        """
        biome = self.biome_id(name)
        if biome < 0:
            raise SeedMapUnavailable(f"unknown biome: {name}")
        if max_radius > MAX_RADIUS:
            raise SeedMapUnavailable(
                f"{max_radius} blocks is beyond the {MAX_RADIUS} this may sweep at once")
        out_x, out_z = ctypes.c_int(), ctypes.c_int()
        found = self.lib.mcgod_nearest_biome(
            ctypes.c_void_p(self.handle), biome, x, y, z, int(max_radius), int(step),
            ctypes.byref(out_x), ctypes.byref(out_z))
        result = {
            "ok": True, "biome": name, "dimension": self.dim,
            "from": [int(x), int(z)], "searched_radius": int(max_radius),
            "step": int(step), "version": version_name(self.mc),
            "as_generated": AS_GENERATED,
        }
        if not found:
            result.update(found=False, note=(
                f"no {name.replace('_', ' ')} within {max_radius} blocks of there, sampling "
                f"every {step} blocks; a smaller patch could have been stepped over, and "
                f"one may lie farther out"))
            return result
        distance = math.hypot(out_x.value - x, out_z.value - z)
        result.update(found=True, x=out_x.value, z=out_z.value,
                      distance=round(distance),
                      direction=compass(out_x.value - x, out_z.value - z))
        return result

    def biome_counts(self, x: int, z: int, radius: int, y: int = 63,
                     step: int = 32) -> dict:
        """How much of each biome lies within a radius, as sampled point counts."""
        if radius > MAX_RADIUS:
            raise SeedMapUnavailable(
                f"{radius} blocks is beyond the {MAX_RADIUS} this may sweep at once")
        counts = (ctypes.c_int * 256)()
        total = self.lib.mcgod_biome_histogram(
            ctypes.c_void_p(self.handle), int(x), int(y), int(z), int(radius), int(step),
            counts)
        by_name = {}
        for biome in range(256):
            if counts[biome]:
                name = self.lib.mcgod_biome_name(self.mc, biome)
                if name:
                    by_name[name.decode()] = counts[biome]
        ordered = dict(sorted(by_name.items(), key=lambda kv: -kv[1]))
        return {
            "ok": True, "dimension": self.dim, "centre": [int(x), int(z)],
            "radius": int(radius), "step": int(step), "samples": total,
            "version": version_name(self.mc),
            "biomes": {name: {"samples": n, "share": round(n / max(1, total), 4)}
                       for name, n in ordered.items()},
            "most_common": next(iter(ordered), None),
            "as_generated": AS_GENERATED,
        }

    def spawn(self):
        out = (ctypes.c_int * 2)()
        self.lib.mcgod_spawn(ctypes.c_void_p(self.handle), out)
        return [out[0], out[1]]

    def strongholds(self, count: int = 3) -> list[list[int]]:
        count = max(0, min(128, int(count)))
        out = (ctypes.c_int * (2 * max(1, count)))()
        written = self.lib.mcgod_strongholds(ctypes.c_void_p(self.handle), count, out)
        return [[out[2 * i], out[2 * i + 1]] for i in range(written)]

    def viable(self, structure: str, x: int, z: int) -> bool:
        kind = self.lib.mcgod_structure_type(structure.encode())
        return bool(self.lib.mcgod_viable(ctypes.c_void_p(self.handle), kind, x, z))


def structure_dimension(structure: str) -> str:
    value = STRUCTURE_DIMENSION.get(structure, 0)
    return {0: "overworld", -1: "the_nether", 1: "the_end"}[value]


def candidates(structure: str, seed: int, centre, radius: int, mc: int | None = None):
    """Every generation attempt for a structure within a radius, before any terrain check.

    This is the cheap half and it is pure arithmetic: the world is a grid of regions and
    each region's attempt comes from the seed, the region coordinates and the structure's
    own salt. Nothing is loaded and nothing is generated.
    """
    lib = _library()
    mc = version_id() if mc is None else mc
    kind = lib.mcgod_structure_type(structure.encode())
    if kind < 0:
        raise SeedMapUnavailable(f"unknown structure: {structure}")
    region = lib.mcgod_region_size(kind, mc)
    if region <= 0:
        raise SeedMapUnavailable(
            f"{structure} does not generate in Minecraft {version_name(mc)}")
    out_x, out_z = ctypes.c_int(), ctypes.c_int()
    found = []
    span = radius / 16.0 / region
    for rx in range(math.floor((centre[0] / 16.0 - radius / 16.0) / region),
                    math.ceil((centre[0] / 16.0 + radius / 16.0) / region) + 1):
        for rz in range(math.floor((centre[1] / 16.0 - radius / 16.0) / region),
                        math.ceil((centre[1] / 16.0 + radius / 16.0) / region) + 1):
            if not lib.mcgod_structure_pos(kind, mc, ctypes.c_ulonglong(seed), rx, rz,
                                           ctypes.byref(out_x), ctypes.byref(out_z)):
                continue
            x, z = out_x.value, out_z.value
            distance = math.hypot(x - centre[0], z - centre[1])
            if distance <= radius:
                found.append((distance, x, z))
    del span
    found.sort()
    return found


def find(structure: str, seed: int, centre, radius: int = 4096, limit: int = 0,
         check: bool = True, mc: int | None = None) -> dict:
    """Where a structure actually generates within a radius, nearest first.

    Placement gives candidates; the terrain check decides which of them become real. That
    check is the expensive half, so callers that only want the nearest few get exactly
    that, and a count over a wide radius pays for every candidate deliberately.

    ``truncated`` says whether the search stopped early, so a count is never reported as
    complete when it is not.
    """
    if radius > MAX_RADIUS:
        raise SeedMapUnavailable(
            f"{radius} blocks is beyond the {MAX_RADIUS} this may sweep at once")
    mc = version_id() if mc is None else mc
    dim = structure_dimension(structure)
    attempts = candidates(structure, seed, centre, radius, mc)
    found, checked = [], 0
    truncated = False
    with World(seed, dim, mc) as world:
        for distance, x, z in attempts:
            if not check:
                found.append({"x": x, "z": z, "distance": round(distance)})
            else:
                checked += 1
                if world.viable(structure, x, z):
                    found.append({"x": x, "z": z, "distance": round(distance),
                                  "biome": world.biome(x, 63, z)})
            if limit and len(found) >= limit:
                truncated = checked < len(attempts)
                break
    accuracy = ACCURACY.get(structure, "approximate")
    result = {
        "ok": True, "structure": structure, "dimension": dim,
        "centre": [int(centre[0]), int(centre[1])], "radius": int(radius),
        "version": version_name(mc), "terrain_checked": bool(check),
        "candidates": len(attempts), "checked": checked,
        "found": found, "count": len(found), "truncated": truncated,
        "accuracy": accuracy,
        "as_generated": AS_GENERATED,
    }
    if accuracy == "approximate":
        result["caution"] = (
            f"about {int(MISS_RATE.get(structure, 0.15) * 100)}% of reported "
            f"{structure.replace('_', ' ')} positions were not there when checked against "
            f"the running server; confirm a specific one before stating it as fact")
    elif accuracy == "placement_only":
        result["caution"] = (
            f"nothing has checked whether the terrain admits a {structure.replace('_', ' ')} "
            f"at these positions; they are generation attempts only")
    return result


def count(structure: str, seed: int, centre, radius: int, mc: int | None = None) -> dict:
    """How many of a structure generate within a radius. Every candidate is checked."""
    return find(structure, seed, centre, radius, limit=0, check=True, mc=mc)


def nearest(structure: str, seed: int, centre, radius: int = 4096,
            mc: int | None = None) -> dict:
    return find(structure, seed, centre, radius, limit=1, check=True, mc=mc)


def compass(dx: float, dz: float) -> str:
    names = ("east", "southeast", "south", "southwest",
             "west", "northwest", "north", "northeast")
    return names[round(math.atan2(dz, dx) / (math.pi / 4)) % 8]


def plural(noun: str, count: int) -> str:
    """English enough for a status line. The god's own speech comes from a model."""
    if count == 1:
        return noun
    if noun.endswith("y") and noun[-2:-1] not in "aeiou":
        return noun[:-1] + "ies"
    return noun + ("es" if noun.endswith(("s", "x", "ch", "sh")) else "s")


def describe(result: dict, centre) -> str:
    """A sentence a god could say, with directions instead of coordinates."""
    if not result.get("found"):
        return (f"No {result['structure'].replace('_', ' ')} generates within about "
                f"{result['radius']} blocks of there.")
    lines = []
    for item in result["found"][:5]:
        way = compass(item["x"] - centre[0], item["z"] - centre[1])
        distance = int(round(item["distance"] / 50.0) * 50)
        biome = f" in {item['biome'].replace('_', ' ')}" if item.get("biome") else ""
        lines.append(f"{way}, roughly {distance} blocks{biome}")
    more = ""
    if result["count"] > len(lines):
        more = f" and {result['count'] - len(lines)} more"
    return (f"{result['count']} "
            f"{plural(result['structure'].replace('_', ' '), result['count'])} within "
            f"{result['radius']} blocks: " + "; ".join(lines) + more + ".")


# --------------------------------------------------------------------------- self-check

def self_check(seed: int | None = None, store=None) -> list[str]:
    """Cheap invariants that catch a broken build or a drifted table.

    This is not the comparison against a live server, which lives in the regression suite
    and is what would actually catch a version change. This only proves the library loaded,
    the version is the one expected, and the tables are the size they were generated at.
    """
    issues = []
    try:
        lib = _library()
    except SeedMapUnavailable as error:
        return [str(error)]
    name = version_name()
    if name != "26.2":
        issues.append(f"newest supported version is {name}, expected 26.2")
    if lib.mcgod_biome_id(b"sulfur_caves") != 187:
        issues.append("sulfur_caves is missing or has the wrong id")
    if lib.mcgod_structure_type(b"village") < 0:
        issues.append("village is not a known structure")
    if lib.mcgod_region_size(lib.mcgod_structure_type(b"village"), version_id()) != 34:
        issues.append("village region size is not 34 chunks")
    try:
        with World(world_seed(store, seed), "overworld") as world:
            if not world.biome(0, 63, 0):
                issues.append("the overworld generator returned no biome at the origin")
    except SeedMapUnavailable as error:
        issues.append(str(error))
    return issues


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--dim", default="overworld", choices=sorted(DIMENSIONS))
    parser.add_argument("--at", nargs=2, type=int, metavar=("X", "Z"), default=[0, 0])
    parser.add_argument("--radius", type=int, default=4096)
    parser.add_argument("--biome", nargs=3, type=int, metavar=("X", "Y", "Z"))
    parser.add_argument("--nearest", metavar="STRUCTURE")
    parser.add_argument("--count", metavar="STRUCTURE")
    parser.add_argument("--all", metavar="STRUCTURE")
    parser.add_argument("--strongholds", type=int, metavar="N")
    parser.add_argument("--find-biome", metavar="BIOME")
    parser.add_argument("--biomes-within", action="store_true",
                        help="biome distribution within --radius of --at")
    parser.add_argument("--step", type=int, default=32)
    parser.add_argument("--spawn", action="store_true")
    parser.add_argument("--build", action="store_true", help="compile the library and exit")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)

    if args.build:
        print(f"built {build(force=True)}")
        return 0
    if args.check:
        from store import Store
        store = Store()
        try:
            issues = self_check(args.seed, store)
        finally:
            store.close()
        for issue in issues:
            print(f"ERROR  {issue}")
        print("ok" if not issues else f"{len(issues)} issue(s)")
        return 1 if issues else 0

    store = None
    if args.seed is None:
        from store import Store
        store = Store()
    try:
        seed = world_seed(store, args.seed)
    finally:
        if store is not None:
            store.close()
    print(f"seed {seed}, Minecraft {version_name()}", file=sys.stderr)

    if args.biome:
        with World(seed, args.dim) as world:
            print(world.biome(*args.biome))
    if args.find_biome:
        with World(seed, args.dim) as world:
            found = world.nearest_biome(args.find_biome, args.at[0], args.at[1],
                                        max_radius=args.radius, step=args.step)
        if found.get("found"):
            print(f"nearest {args.find_biome.replace('_', ' ')}: {found['direction']}, "
                  f"about {found['distance']} blocks away")
        else:
            print(found["note"])
    if args.biomes_within:
        with World(seed, args.dim) as world:
            spread = world.biome_counts(args.at[0], args.at[1], args.radius,
                                        step=args.step)
        print(f"{spread['samples']} samples within {args.radius} blocks:")
        for name, item in list(spread["biomes"].items())[:12]:
            print(f"  {name:28} {item['share'] * 100:5.1f}%")
    if args.spawn:
        with World(seed, "overworld") as world:
            print(json.dumps(world.spawn()))
    if args.strongholds:
        with World(seed, "overworld") as world:
            print(json.dumps(world.strongholds(args.strongholds)))
    for structure, limit in ((args.nearest, 1), (args.all, 0), (args.count, 0)):
        if not structure:
            continue
        result = find(structure, seed, args.at, args.radius, limit=limit)
        print(describe(result, args.at))
        if structure != args.count:
            print(json.dumps(result["found"][:20], indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
