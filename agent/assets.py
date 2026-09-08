#!/usr/bin/env python3
"""Reads block geometry and textures out of the Minecraft client jar.

    python3 assets.py oak_fence oak_log cobblestone_stairs

Blocks are not cubes. A fence is a post and two rails; stairs are two boxes; a pane is a
sheet. Guessing a texture from a block's name gets ~87% of the way and is wrong in exactly
the cases that matter visually — a fence rendered as a solid plank cube is a wall.

The game already ships the answer. `blockstates/<name>.json` picks a model, models inherit
through `parent`, and the resolved model carries `elements`: axis-aligned boxes with a
texture and UV rectangle per face. Reading that is a few hundred lines and is exact, which
is what "faithful to the models in texture and form" requires.

Nothing here talks to a model. Voxels and textures are agent-side data; only a rendered
image is ever put in a prompt.
"""

from __future__ import annotations

import functools
import io
import json
import os
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path

DEFAULT_JAR = Path(os.path.expanduser(
    "~/Library/Application Support/minecraft/versions/26.2/26.2.jar"))

#: Minecraft's own directional shading. Not lighting — a fixed multiplier per face
#: direction, which is why its renders read as solid without any light source.
FACE_SHADE = {"up": 1.0, "down": 0.5,
              "north": 0.8, "south": 0.8, "east": 0.6, "west": 0.6}

#: Blocks the client draws with hardcoded geometry instead of a block model.
#:
#: Their `models/block/*.json` carries only a `particle` texture — there is no shape in the
#: data at all — so without this a chest renders as a plain oak cube and a shulker box as a
#: blank one. The geometry below is the vanilla entity model transcribed into the same
#: element form everything else uses, so the existing rotation, culling and shading paths
#: apply unchanged. Rotation still comes from the blockstate, which does carry a variant per
#: facing even though the model is empty.
#:
#: UVs are given in the entity texture's own pixel space and scaled at build time; entity
#: atlases are 64x64 while block textures are 16x16.


def _box(frm, to, tex, faces_uv, atlas=64):
    """One element, with per-face UV given in atlas pixels."""
    k = 16.0 / atlas
    return {"from": list(frm), "to": list(to),
            "faces": {face: {"uv": [round(v * k, 4) for v in uv], "texture": tex}
                      for face, uv in faces_uv.items()}}


def _entity_box_uv(u, v, w, h, d):
    """Standard entity-model unwrap: the six faces around a box at atlas offset (u, v)."""
    return {
        "up":    [u + d, v, u + d + w, v + d],
        "down":  [u + d + w, v, u + d + w + w, v + d],
        "north": [u + d, v + d, u + d + w, v + d + h],
        "west":  [u, v + d, u + d, v + d + h],
        "east":  [u + d + w, v + d, u + d + w + d, v + d + h],
        "south": [u + d + w + d, v + d, u + d + w + d + w, v + d + h],
    }


def _chest(texture: str) -> tuple:
    """A chest: a base and a lid, as the client draws it."""
    return ([_box((1, 0, 1), (15, 10, 15), "#chest", _entity_box_uv(0, 19, 14, 10, 14)),
             _box((1, 10, 1), (15, 14, 15), "#chest", _entity_box_uv(0, 0, 14, 5, 14))],
            {"chest": texture})


def _shulker(texture: str) -> tuple:
    """A shulker box, drawn closed."""
    return ([_box((0, 0, 0), (16, 8, 16), "#shulker", _entity_box_uv(0, 28, 16, 8, 16)),
             _box((0, 8, 0), (16, 16, 16), "#shulker", _entity_box_uv(0, 0, 16, 8, 16))],
            {"shulker": texture})


def _banner(texture: str, wall: bool) -> tuple:
    """A banner: a post and a hanging cloth, or just the cloth when wall-mounted."""
    cloth = _box((0, 0, 0), (16, 40, 1), "#banner",
                 _entity_box_uv(0, 0, 20, 40, 1))
    if wall:
        return ([{**cloth, "from": [0, -6, 15], "to": [16, 34, 16]}],
                {"banner": texture})
    post = _box((7, 0, 7), (9, 42, 9), "#banner", _entity_box_uv(44, 0, 2, 42, 2))
    return ([post, {**cloth, "from": [0, 14, 7], "to": [16, 54, 8]}],
            {"banner": texture})


def _skull(texture: str) -> tuple:
    """A head, drawn as the cube the client draws."""
    return ([_box((4, 0, 4), (12, 8, 12), "#skull", _entity_box_uv(0, 0, 8, 8, 8),
                  atlas=64)], {"skull": texture})


def _pot(texture: str) -> tuple:
    return ([_box((1, 0, 1), (15, 16, 15), "#pot", _entity_box_uv(0, 0, 14, 16, 14))],
            {"pot": texture})


def _conduit(texture: str) -> tuple:
    return ([_box((5, 5, 5), (11, 11, 11), "#conduit",
                  _entity_box_uv(0, 0, 6, 6, 6), atlas=32)], {"conduit": texture})


#: Skull and banner textures are keyed by variety; these are the few that are not simply
#: the block name.
SKULL_TEXTURE = {
    "skeleton_skull": "entity/skeleton/skeleton",
    "skeleton_wall_skull": "entity/skeleton/skeleton",
    "wither_skeleton_skull": "entity/skeleton/wither_skeleton",
    "wither_skeleton_wall_skull": "entity/skeleton/wither_skeleton",
    "zombie_head": "entity/zombie/zombie",
    "zombie_wall_head": "entity/zombie/zombie",
    "creeper_head": "entity/creeper/creeper",
    "creeper_wall_head": "entity/creeper/creeper",
    "player_head": "entity/player/wide/steve",
    "player_wall_head": "entity/player/wide/steve",
    "dragon_head": "entity/enderdragon/dragon",
    "dragon_wall_head": "entity/enderdragon/dragon",
    "piglin_head": "entity/piglin/piglin",
    "piglin_wall_head": "entity/piglin/piglin",
}

ENTITY_BLOCKS = {
    "chest": lambda n: _chest("entity/chest/normal"),
    "trapped_chest": lambda n: _chest("entity/chest/trapped"),
    "ender_chest": lambda n: _chest("entity/chest/ender"),
    "decorated_pot": lambda n: _pot("entity/decorated_pot/decorated_pot_base"),
    "conduit": lambda n: _conduit("entity/conduit/base"),
}


#: A chest's blockstate is a single empty variant carrying no rotation — the client turns
#: it from the block entity's `facing` at render time. Anything hardcoded needs the same
#: treatment, or every chest points north however it was placed.
FACING_Y = {"north": 0, "east": 90, "south": 180, "west": 270}


def _entity_model(name: str):
    """Geometry for a block the client hardcodes, or None."""
    if name in ENTITY_BLOCKS:
        return ENTITY_BLOCKS[name](name)
    if name in SKULL_TEXTURE:
        return _skull(SKULL_TEXTURE[name])
    if name.endswith("_chest"):
        # Weathered copper chests invert the naming: the block is `exposed_copper_chest`
        # while the texture is `copper_exposed`.
        base = name[: -len("_chest")].replace("waxed_", "")
        if base.startswith(("exposed_", "weathered_", "oxidized_")):
            stage, _, metal = base.partition("_")
            base = f"{metal}_{stage}"
        return _chest(f"entity/chest/{base}")
    if name.endswith("shulker_box"):
        colour = name[: -len("_shulker_box")] if name != "shulker_box" else ""
        return _shulker(f"entity/shulker/shulker_{colour}" if colour
                        else "entity/shulker/shulker")
    if name.endswith("_banner"):
        # One shared base texture, dyed at render time; the colour is lost here, which is
        # a smaller error than a banner not appearing at all.
        return _banner("entity/banner/base", "_wall_banner" in name)
    return None


#: Blocks whose texture ships greyscale and is coloured at runtime by the fluid, not by a
#: colormap. Water is the one that matters; lava ships already coloured.
FLUID_TINT = {"water": (0.247, 0.463, 0.894)}


#: A full cube, used when a block has no model we can resolve.
FULL_CUBE = [{"from": [0, 0, 0], "to": [16, 16, 16],
              "faces": {d: {"uv": [0, 0, 16, 16], "texture": "#missing"}
                        for d in FACE_SHADE}}]


def parse_state(state: str) -> tuple:
    """Splits `minecraft:oak_stairs[facing=east,half=bottom]` into name and properties."""
    state = state.replace("minecraft:", "")
    if "[" not in state:
        return state, {}
    name, _, rest = state.partition("[")
    props = {}
    for pair in rest.rstrip("]").split(","):
        if "=" in pair:
            key, _, value = pair.partition("=")
            props[key.strip()] = value.strip()
    return name, props


@dataclass
class Part:
    """One model applied to a block, with the rotation the blockstate asks for.

    A block is not always one model. A fence is a post plus a side piece for each direction
    it connects, and each side is the same model rotated. Rendering only the first listed
    part is what made every fence a lone post and every stair face the same way.
    """

    elements: list
    textures: dict
    x: int = 0
    y: int = 0


@dataclass
class BlockModel:
    """A resolved block state: every part that makes it up."""

    name: str
    parts: list

    @property
    def elements(self) -> list:
        return [e for p in self.parts for e in p.elements]

    @property
    def textures(self) -> dict:
        merged = {}
        for p in self.parts:
            merged.update(p.textures)
        return merged

    @property
    def is_full_cube(self) -> bool:
        if len(self.parts) != 1 or len(self.parts[0].elements) != 1:
            return False
        e = self.parts[0].elements[0]
        return e["from"] == [0, 0, 0] and e["to"] == [16, 16, 16]


class Assets:
    def __init__(self, jar: Path = DEFAULT_JAR) -> None:
        if not jar.exists():
            raise SystemExit(f"no client jar at {jar}")
        self.zip = zipfile.ZipFile(jar)
        self._names = set(self.zip.namelist())
        self._tint_cache: dict = {}

    # --- raw access ---------------------------------------------------------------

    def _json(self, path: str):
        if path not in self._names:
            return None
        return json.loads(self.zip.read(path))

    @functools.lru_cache(maxsize=4096)
    def model(self, ref: str) -> dict | None:
        """A model by reference, with its parent chain already folded in.

        Children override the parent's textures and may omit elements to inherit them,
        which is how `oak_fence_post` gets its shape from `fence_post` while supplying
        `oak_planks` as the texture.
        """
        ref = ref.replace("minecraft:", "")
        raw = self._json(f"assets/minecraft/models/{ref}.json")
        if raw is None:
            return None
        merged = {"textures": {}, "elements": None}
        if raw.get("parent"):
            parent = self.model(raw["parent"])
            if parent:
                merged["textures"].update(parent["textures"])
                merged["elements"] = parent["elements"]
        merged["textures"].update(raw.get("textures", {}))
        if raw.get("elements") is not None:
            merged["elements"] = raw["elements"]
        return merged

    @staticmethod
    def _matches(when: dict, props: dict) -> bool:
        """Evaluates a multipart condition against a block's properties.

        Conditions are ANDs of property tests, values may be `a|b` alternatives, and `OR`
        holds a list of alternative conditions. This is what makes a fence grow an arm
        towards each neighbour it is actually joined to.
        """
        if "OR" in when:
            return any(Assets._matches(c, props) for c in when["OR"])
        if "AND" in when:
            return all(Assets._matches(c, props) for c in when["AND"])
        for key, expected in when.items():
            actual = props.get(key)
            if actual is None:
                return False
            if actual not in str(expected).split("|"):
                return False
        return True

    def _parts_for(self, block: str, props: dict) -> list:
        """Every (model ref, rotation) this block state actually applies."""
        state = self._json(f"assets/minecraft/blockstates/{block}.json")
        if state is None:
            return []
        found = []

        if "variants" in state:
            # Variant keys are property subsets: pick the one whose every term matches.
            best, best_score = None, -1
            for key, value in state["variants"].items():
                if key == "":
                    terms = {}
                else:
                    terms = dict(t.split("=", 1) for t in key.split(",") if "=" in t)
                if all(props.get(k) == v for k, v in terms.items()):
                    if len(terms) > best_score:
                        best, best_score = value, len(terms)
            if best is None:
                best = next(iter(state["variants"].values()))
            if isinstance(best, list):
                best = best[0]
            found.append(best)
        else:
            for part in state.get("multipart", []):
                when = part.get("when")
                if when and not self._matches(when, props):
                    continue
                apply = part.get("apply")
                if isinstance(apply, list):
                    apply = apply[0]
                if apply:
                    found.append(apply)
        return found

    # --- resolution ---------------------------------------------------------------

    @functools.lru_cache(maxsize=8192)
    def block(self, state: str) -> BlockModel:
        """Everything needed to draw one block state, rotations included."""
        name, props = parse_state(state)
        applies = self._parts_for(name, props)
        parts = []
        for apply in applies:
            model = self.model(apply.get("model", ""))
            if model is None or not model.get("elements"):
                continue
            parts.append(Part(model["elements"],
                              self._resolve_textures(model.get("textures", {})),
                              int(apply.get("x", 0)), int(apply.get("y", 0))))
        if parts:
            return BlockModel(name, parts)

        # No geometry in the data. Either the client hardcodes it, or we fall back to a
        # cube. The variant's rotation is read from `applies` rather than from `parts`,
        # which is empty here — reading it from parts silently left every chest facing
        # north however it had been placed.
        rotation = applies[0] if applies else {}
        entity = _entity_model(name)
        if entity:
            elements, textures = entity
            if "y" not in rotation and props.get("facing") in FACING_Y:
                rotation = dict(rotation, y=FACING_Y[props["facing"]])
        else:
            model = self.model(f"block/{name}") or {}
            elements = model.get("elements") or FULL_CUBE
            textures = self._resolve_textures(model.get("textures", {}))
        return BlockModel(name, [Part(elements, textures,
                                      int(rotation.get("x", 0)),
                                      int(rotation.get("y", 0)))])

    @staticmethod
    def _sprite(value):
        """Unwraps a texture value.

        As of 26.2 a texture slot may be either a plain name or an object carrying a
        `sprite` plus flags such as `force_translucent` — which is how glass declares
        itself see-through. Assuming a string crashed on the first pane encountered.
        """
        if isinstance(value, dict):
            return value.get("sprite"), bool(value.get("force_translucent"))
        return value, False

    def _resolve_textures(self, textures: dict) -> dict:
        """Follows `#name` indirection until each slot names a real texture."""
        out = {}
        for key in textures:
            value, seen = textures[key], set()
            while True:
                sprite, translucent = self._sprite(value)
                if not isinstance(sprite, str) or not sprite.startswith("#"):
                    break
                nxt = sprite[1:]
                if nxt in seen:
                    sprite = None
                    break
                seen.add(nxt)
                value = textures.get(nxt)
            if isinstance(sprite, str):
                out[key] = sprite.replace("minecraft:", "")
                if translucent:
                    out.setdefault("__translucent", []).append(key)
        return out

    @functools.lru_cache(maxsize=2048)
    def texture(self, ref: str):
        """A texture as RGBA pixels. Animated textures use their first frame."""
        from PIL import Image

        path = f"assets/minecraft/textures/{ref}.png"
        if path not in self._names:
            return None
        img = Image.open(io.BytesIO(self.zip.read(path))).convert("RGBA")
        if img.height > img.width:      # animated: a vertical strip of frames
            img = img.crop((0, 0, img.width, img.width))
        return img

    def face_texture(self, material: str, face: str):
        """The texture worn by one face of one block, ready to sample."""
        model = self.block(material)
        for element in model.elements:
            spec = element.get("faces", {}).get(face)
            if not spec:
                continue
            ref, _ = self._sprite(spec.get("texture", ""))
            if isinstance(ref, str) and ref.startswith("#"):
                ref = model.textures.get(ref[1:], "")
            if isinstance(ref, str) and ref:
                return self.texture(ref)
        for key in ("all", "side", "texture", "top", "end", "pane", "particle"):
            if key in model.textures and key != "__translucent":
                return self.texture(model.textures[key])
        return None

    def is_fluid(self, material: str) -> bool:
        return parse_state(material)[0] in FLUID_TINT

    #: Where the biome colour is sampled from. The colormaps are 256x256 indexed by
    #: temperature and downfall; this is roughly a plains biome, which is where this world
    #: is. A per-biome lookup would need the biome per block, which the voxel export does
    #: not carry and which would change almost nothing visually.
    _TINT_UV = (51, 173)

    @functools.lru_cache(maxsize=8)
    def _colormap(self, which: str):
        return self.texture(f"colormap/{which}")

    @functools.lru_cache(maxsize=1024)
    def tint(self, material: str) -> tuple:
        """The colour a tinted face is multiplied by.

        Grass and leaf textures ship greyscale and are coloured at runtime from a colormap —
        which is why every lawn rendered a flat grey until this existed. Faces that need it
        say so with `tintindex`; the block decides which map to read.
        """
        name = parse_state(material)[0]
        if name in FLUID_TINT:
            return FLUID_TINT[name]
        which = "foliage" if ("leaves" in name or "vine" in name) else "grass"
        cmap = self._colormap(which)
        if cmap is None:
            return (1.0, 1.0, 1.0)
        r, g, b = cmap.getpixel(self._TINT_UV)[:3]
        return (r / 255.0, g / 255.0, b / 255.0)

    def tinted(self, texture, material: str):
        """A tinted copy of a texture.

        Cached by identity rather than with lru_cache, because an Image is not hashable.
        Safe: `texture()` is itself cached, so the images passed here are long-lived and
        their identities stable.
        """
        factor = self.tint(material)
        if factor == (1.0, 1.0, 1.0):
            return texture
        key = (id(texture), factor)
        hit = self._tint_cache.get(key)
        if hit is not None:
            return hit
        px = texture.load()
        out = texture.copy()
        op = out.load()
        for y in range(texture.height):
            for x in range(texture.width):
                r, g, b, a = px[x, y]
                op[x, y] = (int(r * factor[0]), int(g * factor[1]),
                            int(b * factor[2]), a)
        self._tint_cache[key] = out
        return out

    def is_translucent(self, material: str) -> bool:
        return bool(self.block(material).textures.get("__translucent"))


def main(argv: list[str]) -> int:
    assets = Assets()
    for material in argv or ["oak_fence", "oak_log", "cobblestone", "glass_pane"]:
        model = assets.block(material)
        boxes = ", ".join(f"{e['from']}->{e['to']}" for e in model.elements[:3])
        print(f"  parts    {len(model.parts)} "
              + str([(p.x, p.y) for p in model.parts]))
        tex = assets.face_texture(material, "north")
        print(f"{material:22} {len(model.elements)} element(s)  "
              f"{'full cube' if model.is_full_cube else 'partial'}")
        print(f"  boxes    {boxes}")
        print(f"  textures {dict(list(model.textures.items())[:4])}")
        print(f"  north    {'ok ' + str(tex.size) if tex else 'MISSING'}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
