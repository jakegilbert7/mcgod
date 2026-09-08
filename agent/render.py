#!/usr/bin/env python3
"""A software renderer for Minecraft voxels.

    python3 render.py --demo out.png

Perspective, z-buffered, textured from the client jar, with per-face geometry read out of
the block models — a fence is drawn as a post, stairs as three boxes, a pane as a sheet.

Written rather than driven through deepslate for three reasons. It is deterministic, so the
same voxels always produce the same pixels and a content hash can key a cache. It needs no
browser in a pipeline that has to run unattended. And it stays in one language, which is
worth more over time than the smooth lighting we give up.

Lighting is ours: Minecraft's fixed per-face shade multipliers plus ambient occlusion
computed from neighbouring voxels. That is what makes a render read as solid rather than as
a flat sticker of textures.

Non-negotiable #3 still holds. Voxels and textures live here, agent-side. Only the finished
image is ever put in front of a model.
"""

from __future__ import annotations

import argparse
import functools
import math
import re
import sys
from pathlib import Path

import numpy as np
from PIL import Image

from assets import FACE_SHADE, Assets

# Face -> (normal, the four corners in bottom-left, bottom-right, top-right, top-left order
# of a unit box). Ordering is consistent so textures come out upright on vertical faces.
FACES = {
    "down":  ((0, -1, 0), lambda a, b: [(a[0], a[1], b[2]), (b[0], a[1], b[2]),
                                        (b[0], a[1], a[2]), (a[0], a[1], a[2])]),
    "up":    ((0, 1, 0),  lambda a, b: [(a[0], b[1], a[2]), (b[0], b[1], a[2]),
                                        (b[0], b[1], b[2]), (a[0], b[1], b[2])]),
    "north": ((0, 0, -1), lambda a, b: [(b[0], a[1], a[2]), (a[0], a[1], a[2]),
                                        (a[0], b[1], a[2]), (b[0], b[1], a[2])]),
    "south": ((0, 0, 1),  lambda a, b: [(a[0], a[1], b[2]), (b[0], a[1], b[2]),
                                        (b[0], b[1], b[2]), (a[0], b[1], b[2])]),
    "west":  ((-1, 0, 0), lambda a, b: [(a[0], a[1], a[2]), (a[0], a[1], b[2]),
                                        (a[0], b[1], b[2]), (a[0], b[1], a[2])]),
    "east":  ((1, 0, 0),  lambda a, b: [(b[0], a[1], b[2]), (b[0], a[1], a[2]),
                                        (b[0], b[1], a[2]), (b[0], b[1], b[2])]),
}
OPPOSITE = {"up": "down", "down": "up", "north": "south",
            "south": "north", "east": "west", "west": "east"}
OFFSET = {"up": (0, 1, 0), "down": (0, -1, 0), "north": (0, 0, -1),
          "south": (0, 0, 1), "east": (1, 0, 0), "west": (-1, 0, 0)}
NORMALS = {d: np.array(v, float) for d, v in OFFSET.items()}


def rotate(points, ax: int, ay: int):
    """Rotates model-space points about the block centre, as a blockstate variant asks.

    Blockstates do not ship a model per orientation — they ship one model and an angle.
    Ignoring it is why every stair faced the same way and every fence was a bare post.
    Rotation is about (8, 8, 8) in the model's own 16-unit space, x applied before y, which
    is the order the game uses.
    """
    p = np.asarray(points, float) - 8.0
    if ax % 360:
        a = math.radians(ax)
        c, s = math.cos(a), math.sin(a)
        y, z = p[:, 1].copy(), p[:, 2].copy()
        p[:, 1], p[:, 2] = y * c - z * s, y * s + z * c
    if ay % 360:
        # Clockwise seen from above, which is the direction the game rotates. Getting this
        # backwards is invisible on anything symmetric and obvious on a stair roof: every
        # step faced the wrong way.
        a = math.radians(ay)
        c, s = math.cos(a), math.sin(a)
        x, z = p[:, 0].copy(), p[:, 2].copy()
        p[:, 0], p[:, 2] = x * c - z * s, x * s + z * c
    return p + 8.0


def rotate_element(points, spec: dict):
    """Applies an element's own rotation.

    Separate from the variant rotation and easy to miss: a raised rail is a flat plane in
    the model with `rotation: {axis: x, angle: 45, rescale: true}` on it. Ignoring that left
    every ascending rail lying flat, floating where a ramp should be.

    `rescale` stretches the result back out so the tilted face still spans the block, which
    is what makes a sloped rail meet the ones above and below it.
    """
    if not spec:
        return points
    angle = float(spec.get("angle", 0))
    if not angle:
        return points
    axis = spec.get("axis", "y")
    origin = np.array(spec.get("origin", [8, 8, 8]), float)
    a = math.radians(angle)
    c, si = math.cos(a), math.sin(a)
    p = np.asarray(points, float) - origin
    i, j = {"x": (1, 2), "y": (2, 0), "z": (0, 1)}[axis]
    u, v = p[:, i].copy(), p[:, j].copy()
    p[:, i], p[:, j] = u * c - v * si, u * si + v * c
    if spec.get("rescale"):
        factor = 1.0 / math.cos(a)
        p[:, i] *= factor
        p[:, j] *= factor
    return p + origin


def nearest_face(normal) -> str:
    """The cardinal direction a rotated face now points, for shading."""
    best, score = "up", -2.0
    for name, vec in NORMALS.items():
        d = float(np.dot(normal, vec))
        if d > score:
            best, score = name, d
    return best


def look_at(eye, target, up=(0, 1, 0)) -> np.ndarray:
    f = np.array(target, float) - np.array(eye, float)
    f /= np.linalg.norm(f)
    s = np.cross(f, np.array(up, float))
    s /= np.linalg.norm(s)
    u = np.cross(s, f)
    m = np.eye(4)
    m[0, :3], m[1, :3], m[2, :3] = s, u, -f
    m[:3, 3] = -m[:3, :3] @ np.array(eye, float)
    return m


def perspective(fov_deg: float, aspect: float, near=0.1, far=500.0) -> np.ndarray:
    f = 1.0 / math.tan(math.radians(fov_deg) / 2)
    m = np.zeros((4, 4))
    m[0, 0], m[1, 1] = f / aspect, f
    m[2, 2], m[2, 3] = (far + near) / (near - far), (2 * far * near) / (near - far)
    m[3, 2] = -1.0
    return m


def orthographic(half_w: float, half_h: float, near=0.1, far=2000.0) -> np.ndarray:
    """A projection with no vanishing point, so a block is the same size wherever it is.

    Perspective is what a person sees, but it is the wrong tool for reading a boundary off a
    picture: two buildings at different depths overlap, and the same wall is wider at the
    near end than the far one. Under an orthographic plan view a pixel maps to a fixed patch
    of world, which is what makes a coordinate grid drawable over it and readable from it.
    """
    m = np.zeros((4, 4))
    m[0, 0], m[1, 1] = 1.0 / half_w, 1.0 / half_h
    m[2, 2], m[2, 3] = -2.0 / (far - near), -(far + near) / (far - near)
    m[3, 3] = 1.0
    return m


class Renderer:
    """Rasterises a voxel volume into an image."""

    #: Render at this multiple of the requested size, then average down. Voxel edges are
    #: hard diagonals on screen and alias badly; supersampling is the cheapest way to make
    #: a rail read as a rail rather than as a row of stair-stepped pixels.
    SUPERSAMPLE = 2

    def __init__(self, assets: Assets, size=(640, 480), background=(24, 26, 32),
                 supersample: int | None = None) -> None:
        self.assets = assets
        self.out_w, self.out_h = size
        self.ss = self.SUPERSAMPLE if supersample is None else max(1, supersample)
        self.w, self.h = self.out_w * self.ss, self.out_h * self.ss
        self.background = background

    # --- geometry ------------------------------------------------------------------

    @functools.lru_cache(maxsize=256)
    def entity_colour(self, kind: str):
        """A creature's rough colour, from the middle of its own texture."""
        ref = ENTITY_TEXTURE.get(kind)
        image = self.assets.texture(ref) if ref else None
        if image is None:
            return (0.75, 0.55, 0.55)
        pixels = [image.getpixel((x, y)) for x in range(2, image.width, 5)
                  for y in range(2, image.height // 2, 5)]
        solid = [p for p in pixels if len(p) < 4 or p[3] > 200]
        if not solid:
            return (0.75, 0.55, 0.55)
        return tuple(sum(p[i] for p in solid) / len(solid) / 255.0 for i in range(3))

    def _creature_quads(self, entities):
        """Living things, drawn with the game's own entity models.

        Hand-transcribed boxes were here first and the chickens came out looking like
        creepers — recognisably wrong, which is worse than an honest marker, because the
        model reading the render believes what it sees. These numbers come from the client's
        own `createBodyLayer` bytecode instead (see `entity_models.py`), so a chicken is the
        chicken the game draws: head, beak, wattle, a body pitched flat, two wings, two legs.
        """
        from entity_models import load, to_world

        models = _entity_models()
        out = []
        for entity in entities or ():
            raw = (entity.get("type") or "").lower()
            kind = raw.split(":")[-1]
            # Both forms: the list is written namespaced, and dropped items were rendering
            # as markers because the bare name never matched it.
            if not kind or raw in IGNORED_ENTITIES or kind in IGNORED_ENTITIES:
                continue
            model = models.get(kind)
            if not model:
                x, y, z = entity["pos"]
                colour = self.entity_colour(kind)
                lo, hi = (x - 0.28, y, z - 0.28), (x + 0.28, y + 0.65, z + 0.28)
                for face, (_, corners_of) in FACES.items():
                    out.append(([tuple(c) for c in corners_of(lo, hi)], None, None,
                                FACE_SHADE[face], False, False, colour))
                continue
            tex = self.assets.texture(model["texture"])
            if tex is None:
                continue
            tw, th = model["texture_size"]
            sx, sy = 16.0 / tw, 16.0 / th
            ex, ey, ez = entity["pos"]
            yaw = math.radians(180.0 - (entity.get("yaw") or 0.0))
            spin, cos = math.sin(yaw), math.cos(yaw)
            for cube in model["cubes"]:
                out += self._cube_quads(cube, tex, sx, sy, ex, ey, ez, cos, spin)
        return out

    def _cube_quads(self, cube, tex, sx, sy, ex, ey, ez, cos, spin):
        from entity_models import to_world

        """One box of an entity model, posed and placed in the world.

        A part rotates about its own pivot, not the entity's origin — a chicken's body is a
        vertical box laid flat by a 90 degree pitch, and a cow's legs hang off theirs. So the
        pivot is applied first, then the part's rotation, then the entity's facing.
        """
        px, py, pz = cube["pivot"]
        fx, fy, fz = cube["from"]
        w, h, d = cube["size"]
        rx, ry, rz = cube["rot"]
        cx, cy, cz = math.cos(rx), math.cos(ry), math.cos(rz)
        sxr, syr, szr = math.sin(rx), math.sin(ry), math.sin(rz)

        def place(mx, my, mz):
            # Model space, about the part pivot: z, then y, then x, as the game applies them.
            a, b = mx * cz - my * szr, mx * szr + my * cz
            a, c = a * cy + mz * syr, -a * syr + mz * cy
            b, c = b * cx - c * sxr, b * sxr + c * cx
            wx, wy, wz = to_world(px + a, py + b, pz + c)
            return (ex + wx * cos - wz * spin, ey + wy, ez + wx * spin + wz * cos)

        lo = (fx, fy, fz)
        hi = (fx + w, fy + h, fz + d)
        uw, uh, ud = cube["uv_size"]
        u, v = cube["tex"]
        out = []
        for face, (_, corners_of) in FACES.items():
            corners = [place(*c) for c in corners_of(lo, hi)]
            # Model space has Y pointing down, so what the game calls the top of a box
            # arrives here as its bottom. Swapping the names keeps the texture patches right.
            patch = _MODEL_FACE[face]
            if cube["mirror"]:
                patch = {"west": "east", "east": "west"}.get(patch, patch)
            uv = _box_uv(u, v, uw, uh, ud, patch)
            out.append((corners, tex, [uv[0] * sx, uv[1] * sy, uv[2] * sx, uv[3] * sy],
                        FACE_SHADE[face], False, face in ("up", "down"), None))
        return out

    def _quads(self, voxels: dict):
        """Every visible textured quad in the volume.

        A face is skipped when the neighbour in that direction is an opaque full cube,
        which removes the interior of any solid mass — usually most of the geometry.
        """
        solid_full = {p for p, m in voxels.items()
                      if self.assets.block(m).is_full_cube
                      and not self.assets.is_translucent(m)}
        out = []
        for (x, y, z), state in voxels.items():
            model = self.assets.block(state)
            fluid = self.assets.is_fluid(state)
            translucent = self.assets.is_translucent(state) or fluid
            for part in model.parts:
                for element in part.elements:
                    a, b = element["from"], element["to"]
                    for face, spec in (element.get("faces") or {}).items():
                        if face not in FACES:
                            continue
                        if model.is_full_cube:
                            dx, dy, dz = OFFSET[face]
                            if (x + dx, y + dy, z + dz) in solid_full:
                                continue
                        tex = self._face_texture(part, model, face)
                        if tex is None:
                            continue
                        # Water has no block model and no tintindex: it ships greyscale and
                        # is coloured by the fluid renderer, which is why it came out white.
                        if "tintindex" in spec or fluid:
                            tex = self.assets.tinted(tex, state)
                        # Build the quad in the model's own 16-unit space, rotate it as the
                        # variant asks, then place it in the world.
                        corners = np.array(FACES[face][1](a, b), float)
                        corners = rotate_element(corners, element.get("rotation"))
                        corners = rotate(corners, part.x, part.y)
                        world = corners / 16.0 + np.array([x, y, z], float)
                        normal = rotate_element(
                            np.array([NORMALS[face] * 4 + 8.0]), element.get("rotation"))
                        normal = rotate(normal, part.x, part.y)[0] - 8.0
                        shaded = nearest_face(normal)
                        ao = self._occlusion(voxels, (x, y, z), shaded)
                        out.append(([tuple(c) for c in world], tex,
                                    spec.get("uv", [0, 0, 16, 16]),
                                    FACE_SHADE[shaded] * ao, translucent,
                                    face in ("up", "down"), None))
        return out

    def _face_texture(self, part, model, face: str):
        """Texture for one face of one part, resolved through that part's own slots."""
        for element in part.elements:
            spec = (element.get("faces") or {}).get(face)
            if not spec:
                continue
            ref, _ = self.assets._sprite(spec.get("texture", ""))
            if isinstance(ref, str) and ref.startswith("#"):
                ref = part.textures.get(ref[1:]) or model.textures.get(ref[1:], "")
            if isinstance(ref, str) and ref:
                return self.assets.texture(ref)
        for key in ("all", "side", "texture", "top", "end", "pane", "particle"):
            if key in part.textures and key != "__translucent":
                return self.assets.texture(part.textures[key])
        return None

    @staticmethod
    def _occlusion(voxels: dict, pos, face) -> float:
        """Cheap ambient occlusion: how boxed-in the space in front of this face is.

        Not Minecraft's per-vertex smooth lighting, but it does the same job — it is what
        makes a doorway read as a recess rather than as a darker rectangle painted on.
        """
        dx, dy, dz = OFFSET[face]
        front = (pos[0] + dx, pos[1] + dy, pos[2] + dz)
        blocked = sum(
            1 for ox in (-1, 0, 1) for oy in (-1, 0, 1) for oz in (-1, 0, 1)
            if (ox, oy, oz) != (0, 0, 0)
            and (front[0] + ox, front[1] + oy, front[2] + oz) in voxels)
        return 1.0 - 0.45 * (blocked / 26.0)

    # --- rasterisation -------------------------------------------------------------

    orthographic = False

    def render(self, voxels: dict, azimuth=35.0, elevation=28.0, zoom=1.0,
               entities=None, ortho_extent=None) -> Image.Image:
        if not voxels:
            return Image.new("RGB", (self.out_w, self.out_h), self.background)

        pts = np.array(list(voxels.keys()), float)
        centre = (pts.min(0) + pts.max(0) + 1) / 2
        radius = float(np.linalg.norm(pts.max(0) - pts.min(0))) / 2 + 2.0
        dist = radius / math.tan(math.radians(25)) / max(zoom, 0.2)
        ay, ax = math.radians(azimuth), math.radians(elevation)
        eye = centre + dist * np.array([math.cos(ax) * math.sin(ay),
                                        math.sin(ax),
                                        math.cos(ax) * math.cos(ay)])
        self.orthographic = bool(ortho_extent)
        if ortho_extent:
            # A fixed world-space window rather than a field of view, so the caller knows
            # exactly which blocks land on which pixels and can draw a grid over them.
            half_w, half_z = ortho_extent
            eye = centre + np.array([0.0, max(dist, 256.0), 0.0])
            mvp = orthographic(half_w, half_z) @ look_at(eye, centre,
                                                         up=(0.0, 0.0, -1.0))
            self.last_mvp = mvp
        else:
            mvp = perspective(50.0, self.w / self.h) @ look_at(eye, centre)
            self.last_mvp = mvp

        colour = np.zeros((self.h, self.w, 3), np.float32)
        colour[:] = np.array(self.background, np.float32) / 255.0
        depth = np.full((self.h, self.w), np.inf, np.float32)

        quads = self._quads(voxels) + self._creature_quads(entities)
        # Translucent surfaces last, so they blend over what is already behind them.
        quads.sort(key=lambda q: q[4])
        for corners, tex, uv, shade, translucent, flip_v, flat in quads:
            self._quad(colour, depth, mvp, corners, tex, uv, shade, translucent, flip_v,
                       flat)
        image = Image.fromarray((np.clip(colour, 0, 1) * 255).astype(np.uint8))
        if self.ss > 1:
            image = image.resize((self.out_w, self.out_h), Image.LANCZOS)
        return image

    def _quad(self, colour, depth, mvp, corners, tex, uv, shade, translucent,
              flip_v: bool = False, flat=None) -> None:
        world = np.array([[c[0], c[1], c[2], 1.0] for c in corners]).T
        clip = mvp @ world
        w = clip[3]
        if np.any(w <= 1e-6):
            return
        ndc = clip[:3] / w
        sx = (ndc[0] * 0.5 + 0.5) * self.w
        sy = (1.0 - (ndc[1] * 0.5 + 0.5)) * self.h
        inv_w = 1.0 / w

        if flat is not None:
            # A solid marker: one colour, no texture to sample.
            pix = np.array([[[flat[0], flat[1], flat[2], 1.0]]], np.float32)
            uv = [0, 0, 16, 16]
        u0, v0, u1, v1 = [c / 16.0 for c in uv]
        # Corners are ordered bottom-left, bottom-right, top-right, top-left.
        uvs = np.array([[u0, v1], [u1, v1], [u1, v0], [u0, v0]])
        if flip_v:
            # Horizontal faces run the texture the other way down V. On a symmetric texture
            # this is invisible; on a corner rail it mirrors the curve, so every corner
            # appeared to turn the wrong way.
            uvs = np.array([[u0, v0], [u1, v0], [u1, v1], [u0, v1]])
        pix = (np.array([[[flat[0], flat[1], flat[2], 1.0]]], np.float32) if flat is not None
               else np.asarray(tex, np.float32) / 255.0)

        # Under orthographic projection the perspective divide carries no depth, so the
        # normalised z is handed over as the depth key instead.
        zs = ndc[2] if self.orthographic else None
        for tri in ((0, 1, 2), (0, 2, 3)):
            self._triangle(colour, depth, sx[list(tri)], sy[list(tri)],
                           inv_w[list(tri)], uvs[list(tri)], pix, shade, translucent,
                           zs=None if zs is None else zs[list(tri)])

    def _triangle(self, colour, depth, sx, sy, inv_w, uvs, pix, shade, translucent,
                  zs=None) -> None:
        x0, x1 = int(max(0, np.floor(sx.min()))), int(min(self.w - 1, np.ceil(sx.max())))
        y0, y1 = int(max(0, np.floor(sy.min()))), int(min(self.h - 1, np.ceil(sy.max())))
        if x1 < x0 or y1 < y0:
            return
        area = ((sx[1] - sx[0]) * (sy[2] - sy[0]) - (sx[2] - sx[0]) * (sy[1] - sy[0]))
        if abs(area) < 1e-9:
            return

        ys, xs = np.mgrid[y0:y1 + 1, x0:x1 + 1]
        px, py = xs + 0.5, ys + 0.5
        w0 = ((sx[1] - sx[0]) * (py - sy[0]) - (px - sx[0]) * (sy[1] - sy[0])) / area
        w1 = ((px - sx[0]) * (sy[2] - sy[0]) - (sx[2] - sx[0]) * (py - sy[0])) / area
        b1, b2 = w1, w0
        b0 = 1.0 - b1 - b2
        inside = (b0 >= -1e-6) & (b1 >= -1e-6) & (b2 >= -1e-6)
        if not inside.any():
            return

        # Perspective-correct interpolation: interpolate 1/w and u/w, then divide.
        iw = b0 * inv_w[0] + b1 * inv_w[1] + b2 * inv_w[2]
        z = 1.0 / np.maximum(iw, 1e-9)
        # Depth is its own quantity, not a by-product of the perspective divide. Under an
        # orthographic projection w is 1 everywhere, so reusing 1/w as the depth key made
        # every fragment equidistant and whichever quad drew last won — a plan view came
        # back as scattered underground blocks with the buildings missing.
        key = z if zs is None else (b0 * zs[0] + b1 * zs[1] + b2 * zs[2])
        closer = inside & (key < depth[y0:y1 + 1, x0:x1 + 1])
        if not closer.any():
            return

        u = (b0 * uvs[0, 0] * inv_w[0] + b1 * uvs[1, 0] * inv_w[1]
             + b2 * uvs[2, 0] * inv_w[2]) * z
        v = (b0 * uvs[0, 1] * inv_w[0] + b1 * uvs[1, 1] * inv_w[1]
             + b2 * uvs[2, 1] * inv_w[2]) * z
        th, tw = pix.shape[0], pix.shape[1]
        tx = np.clip((u * tw).astype(int), 0, tw - 1)
        ty = np.clip((v * th).astype(int), 0, th - 1)
        texel = pix[ty, tx]

        alpha = texel[..., 3] if pix.shape[2] == 4 else np.ones_like(u)
        visible = closer & (alpha > 0.15)
        if not visible.any():
            return
        rgb = texel[..., :3] * shade
        target = colour[y0:y1 + 1, x0:x1 + 1]
        if translucent:
            blend = np.where(visible[..., None], 0.55, 0.0)
            target[:] = target * (1 - blend) + rgb * blend
        else:
            target[visible] = rgb[visible]
            # Store the same quantity used by the comparison. Perspective uses camera
            # distance (`z`); orthographic uses projected z (`key`). Comparing projected z
            # and then storing constant 1/w made later-drawn lower blocks overwrite roofs,
            # fences, and every other upper surface in plan views.
            depth[y0:y1 + 1, x0:x1 + 1][visible] = key[visible]


def demo_voxels() -> dict:
    """A small cottage, to prove form and texture without needing a server."""
    v = {}
    for x in range(9):
        for z in range(7):
            v[(x, 0, z)] = "minecraft:cobblestone"
    for y in range(1, 5):
        for x in range(9):
            for z in range(7):
                edge = x in (0, 8) or z in (0, 6)
                if edge:
                    v[(x, y, z)] = "minecraft:oak_planks" if y > 1 else "minecraft:cobblestone"
    for y in (2, 3):
        v[(0, y, 3)] = "minecraft:glass_pane"
        v[(8, y, 3)] = "minecraft:glass_pane"
    for x in range(9):
        for z in range(7):
            v[(x, 5, z)] = "minecraft:cobblestone_stairs"
    for x in (2, 6):
        v[(x, 4, 0)] = "minecraft:torch"
    del v[(4, 1, 0)], v[(4, 2, 0)]
    for z in range(7, 13):
        v[(4, 0, z)] = "minecraft:oak_fence"
    return v


def main(argv: list[str]) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--demo", type=Path, default=Path("render_demo.png"))
    p.add_argument("--size", type=int, nargs=2, default=[640, 480])
    args = p.parse_args(argv)
    r = Renderer(Assets(), size=tuple(args.size))
    img = r.render(demo_voxels())
    img.save(args.demo)
    print(f"wrote {args.demo} ({img.size[0]}x{img.size[1]})")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))


#: Where review renders are written. Not part of the pipeline — nothing reads these back —
#: they exist so a person can see what the model saw.
REVIEW_DIR = Path(__file__).parent / "renders"


def save_for_review(picture: bytes, structure_id: str, label: str = "",
                    confidence: float | None = None) -> Path | None:
    """Writes the view a model was given, named so it can be found.

    Every time a structure's belief changes — it gets named, renamed, confirmed — the
    picture behind that decision lands here. Without it, judging whether the god is seeing
    what you think it is means re-deriving the render by hand and hoping it matches.

    One file per structure: the old one is removed when the name changes, so the folder
    shows what is believed now rather than a pile of every guess ever made.
    """
    if not picture:
        return None
    # The top level is a materialized view of current structures only.  Quest judgments
    # and historical work are useful evidence, but mixing them into that view made obsolete
    # and present objects visually indistinguishable.
    if structure_id.startswith("quest_"):
        folder = REVIEW_DIR / "quests"
    elif structure_id.startswith(("g_", "generated:", "generated_sub:")):
        folder = REVIEW_DIR / "generated"
    elif structure_id.startswith(("placement_", "removal_")):
        folder = REVIEW_DIR / "history"
    else:
        folder = REVIEW_DIR
    try:
        folder.mkdir(parents=True, exist_ok=True)
        for old in folder.glob(f"{structure_id}__*.png"):
            old.unlink()
        slug = re.sub(r"[^a-z0-9]+", "-", (label or "unnamed").lower()).strip("-")[:48]
        suffix = f"__{confidence:.2f}" if confidence is not None else ""
        path = folder / f"{structure_id}__{slug}{suffix}.png"
        path.write_bytes(picture)
        return path
    except OSError:
        # A review artefact failing to write must never break the thing it is reviewing.
        return None


#: Blocks that are the ground rather than something someone put there.
GROUND_MATERIALS = {
    "minecraft:grass_block", "minecraft:dirt", "minecraft:stone", "minecraft:short_grass",
    "minecraft:tall_grass", "minecraft:sand", "minecraft:gravel", "minecraft:water",
    "minecraft:andesite", "minecraft:diorite", "minecraft:granite", "minecraft:deepslate",
    "minecraft:coarse_dirt", "minecraft:rooted_dirt", "minecraft:clay", "minecraft:lava",
    "minecraft:sandstone", "minecraft:tuff", "minecraft:calcite", "minecraft:snow",
    "minecraft:podzol", "minecraft:mud", "minecraft:bedrock", "minecraft:seagrass",
    "minecraft:kelp", "minecraft:kelp_plant", "minecraft:dead_bush", "minecraft:fern",
}

#: Things the world grew or buried, which nobody placed. Leaves, ore and flowers came out
#: of the first spatial pass as "structures" — a birch tree read as a 58-block build — and
#: no amount of clustering fixes that, because they genuinely are connected masses of
#: non-ground blocks. They have to be named as natural.
NATURAL_SUFFIXES = ("_leaves", "_ore", "_sapling", "_mushroom", "_flower", "_tulip",
                    "_orchid", "_bush", "_fungus", "_roots", "_coral", "_coral_block",
                    "_coral_fan", "_sprouts", "_vine", "_lichen", "_bud", "_amethyst")
NATURAL_EXACT = {
    "minecraft:dandelion", "minecraft:poppy", "minecraft:oxeye_daisy",
    "minecraft:cornflower", "minecraft:allium", "minecraft:azure_bluet",
    "minecraft:lily_of_the_valley", "minecraft:sugar_cane", "minecraft:cactus",
    "minecraft:vine", "minecraft:glow_lichen", "minecraft:sculk", "minecraft:sculk_vein",
    "minecraft:moss_carpet", "minecraft:snow_block", "minecraft:ice", "minecraft:packed_ice",
    "minecraft:pointed_dripstone", "minecraft:dripstone_block", "minecraft:cobweb",
    "minecraft:amethyst_cluster", "minecraft:raw_iron_block", "minecraft:magma_block",
    "minecraft:obsidian", "minecraft:netherrack", "minecraft:soul_sand", "minecraft:tuff",
    # Found by the ledger of built things, which has no size floor to hide behind: every
    # one of these came out of a full-world read as an "unknown builder" mass. Vegetation
    # and geodes; a player who places one is still credited by exact block history.
    "minecraft:tall_seagrass", "minecraft:bush", "minecraft:sunflower", "minecraft:lilac",
    "minecraft:peony", "minecraft:rose_bush", "minecraft:torchflower",
    "minecraft:pitcher_plant", "minecraft:wildflowers", "minecraft:leaf_litter",
    "minecraft:short_dry_grass", "minecraft:tall_dry_grass", "minecraft:bubble_column",
    "minecraft:bee_nest", "minecraft:lily_pad", "minecraft:sea_pickle",
    "minecraft:spore_blossom", "minecraft:big_dripleaf", "minecraft:big_dripleaf_stem",
    "minecraft:small_dripleaf", "minecraft:cave_vines", "minecraft:cave_vines_plant",
    "minecraft:weeping_vines", "minecraft:weeping_vines_plant", "minecraft:twisting_vines",
    "minecraft:twisting_vines_plant", "minecraft:moss_block", "minecraft:pale_moss_block",
    "minecraft:pale_moss_carpet", "minecraft:pale_hanging_moss", "minecraft:bamboo",
    "minecraft:chorus_plant", "minecraft:amethyst_block", "minecraft:smooth_basalt",
    "minecraft:basalt", "minecraft:blackstone", "minecraft:end_stone", "minecraft:red_sand",
    "minecraft:red_sandstone", "minecraft:soul_soil", "minecraft:crimson_nylium",
    "minecraft:warped_nylium", "minecraft:blue_ice", "minecraft:powder_snow",
    "minecraft:infested_stone", "minecraft:infested_deepslate", "minecraft:suspicious_sand",
    "minecraft:suspicious_gravel", "minecraft:nether_wart_block",
    "minecraft:warped_wart_block", "minecraft:shroomlight",
}


def is_natural(material: str) -> bool:
    """Did the world put this here, rather than a player?

    Deliberately generous about ore and vegetation and deliberately silent about logs and
    planks, which are as often a wall as a tree. A tree with its leaves called natural
    collapses to a few logs and falls under the size floor; a log cabin does not.
    """
    name = material.split("[")[0]
    return (name in GROUND_MATERIALS or name in NATURAL_EXACT
            or name.endswith(NATURAL_SUFFIXES))


#: Colours for living things, sampled once from their own textures where one exists.
#: Entity models are bone-rigged and far outside what this renderer does, so an animal is
#: drawn as a small solid marker in roughly its own colour. That is honest — it says
#: "something alive stands here and this is what" — and it is what makes a pen full of
#: chickens or a hall full of villagers legible at all.
ENTITY_TEXTURE = {
    "minecraft:chicken": "entity/chicken", "minecraft:cow": "entity/cow/cow",
    "minecraft:pig": "entity/pig/pig", "minecraft:sheep": "entity/sheep/sheep",
    "minecraft:villager": "entity/villager/villager",
    "minecraft:iron_golem": "entity/iron_golem/iron_golem",
    "minecraft:zombie": "entity/zombie/zombie",
    "minecraft:skeleton": "entity/skeleton/skeleton",
    "minecraft:horse": "entity/horse/horse_brown",
    "minecraft:wolf": "entity/wolf/wolf", "minecraft:cat": "entity/cat/tabby",
    "minecraft:rabbit": "entity/rabbit/brown",
    "minecraft:bee": "entity/bee/bee", "minecraft:allay": "entity/allay/allay",
}

#: Entities that are not creatures and would only add noise to a picture of a building.
IGNORED_ENTITIES = {"minecraft:item", "minecraft:experience_orb", "minecraft:arrow",
                    "minecraft:item_frame", "minecraft:painting", "minecraft:text_display",
                    "minecraft:interaction", "minecraft:marker", "minecraft:armor_stand"}


def built_only(voxels: dict) -> dict:
    """Just the blocks someone put there, with the ground taken out.

    Distinct from `framed`, which crops the region to the build and keeps the terrain
    inside it — that is what a render wants. Measuring a structure wants the opposite:
    the blocks themselves and nothing else. Using the framed region to take a measurement
    turned a ten-block house into nine thousand blocks of hillside.

    Voxels carry a full block state, so the bare name has to be taken before comparing.
    """
    return {p: m for p, m in voxels.items() if not is_natural(m)}


def framed(voxels: dict, structure_only: bool = True) -> dict:
    """Trims a voxel volume to the built structure.

    A render of a generous box is mostly hillside, which is the same scoping mistake that
    made a scanned house read as 1760 dirt. Ground materials are dropped and the volume is
    cropped to what remains, so the model is shown the build rather than the field it
    stands in.
    """
    if not structure_only:
        return voxels
    GROUND = {"minecraft:grass_block", "minecraft:dirt", "minecraft:stone",
              "minecraft:short_grass", "minecraft:tall_grass", "minecraft:sand",
              "minecraft:gravel", "minecraft:water", "minecraft:andesite",
              "minecraft:diorite", "minecraft:granite", "minecraft:deepslate",
              "minecraft:coarse_dirt", "minecraft:rooted_dirt", "minecraft:clay"}
    # Voxels carry a full block state — `minecraft:grass_block[snowy=false]` — so the bare
    # name has to be taken before comparing. Without this the ground set matched nothing and
    # framing kept the entire region, which quietly turned a ten-block house into nine
    # thousand blocks of hillside the moment anything tried to re-measure it.
    built = built_only(voxels)
    if not built:
        return voxels
    xs = [p[0] for p in built]; ys = [p[1] for p in built]; zs = [p[2] for p in built]
    pad = 1
    lo = (min(xs) - pad, min(ys) - pad, min(zs) - pad)
    hi = (max(xs) + pad, max(ys) + pad, max(zs) + pad)
    return {p: m for p, m in voxels.items()
            if all(lo[i] <= p[i] <= hi[i] for i in range(3))}


#: The API resizes anything wider than this before the model sees it. Structure sheets use
#: two columns so every view keeps enough pixels for small block patterns and silhouettes.
MAX_SHEET_WIDTH = 1536


def surface(voxels: dict) -> dict:
    """Keeps only the topmost block in each column, plus anything standing on it.

    A 96-block-wide region is roughly a million voxels and almost all of it is stone nobody
    can see. Rendering only what is visible from above turns an impossible amount of
    geometry into a cheap one, and a landscape read from above is what siting a build
    actually needs.
    """
    top: dict = {}
    for (x, y, z), state in voxels.items():
        key = (x, z)
        if key not in top or y > top[key][0]:
            top[key] = (y, state)
    out = {(x, y, z): state for (x, z), (y, state) in top.items()}
    # Keep one layer under each surface block so cliffs and overhangs have a face.
    for (x, z), (y, _) in top.items():
        below = (x, y - 1, z)
        if below in voxels:
            out[below] = voxels[below]
    return out


def plan(voxels: dict, assets: Assets, lo, hi, size=(720, 720),
         step: int = 8, entities=None) -> Image.Image:
    """A straight-down orthographic plan with world coordinates ruled onto it.

    This is the view the segmentation task actually wants and did not have. A three-quarter
    perspective is what a person sees, but it is the wrong picture for deciding where a
    building ends: buildings at different depths overlap each other, and the same wall is
    wider at the near end. A plan view separates footprints — which is exactly the judgement
    being asked for — and because it is orthographic, a pixel maps to a fixed patch of world.

    So the grid can be ruled on and labelled, and the model can read a boundary straight off
    the image instead of cross-referencing a picture against a table of slices. Getting a
    coordinate out of a render was the weakest link in the whole pass.
    """
    from PIL import ImageDraw

    span_x = (hi[0] - lo[0]) / 2 + 1
    span_z = (hi[2] - lo[2]) / 2 + 1
    half = max(span_x, span_z)
    r = Renderer(assets, size=size, supersample=2)
    image = r.render(voxels, ortho_extent=(half, half), entities=entities).convert("RGB")

    centre = ((lo[0] + hi[0] + 1) / 2, (lo[2] + hi[2] + 1) / 2)
    scale = size[0] / (2 * half)

    def at(world_x: float, world_z: float):
        """Where a world column lands on the image. Derived from the same window the
        renderer used, so a label cannot drift from what it points at."""
        return (size[0] / 2 + (world_x - centre[0]) * scale,
                size[1] / 2 + (world_z - centre[1]) * scale)

    draw = ImageDraw.Draw(image, "RGBA")
    first_x = lo[0] - (lo[0] % step)
    first_z = lo[2] - (lo[2] % step)
    for x in range(first_x, hi[0] + step, step):
        px = at(x, 0)[0]
        if 0 <= px < size[0]:
            draw.line([(px, 0), (px, size[1])], fill=(255, 255, 255, 46))
            draw.text((px + 2, 2), f"x{x}", fill=(255, 255, 255, 190))
    for z in range(first_z, hi[2] + step, step):
        pz = at(0, z)[1]
        if 0 <= pz < size[1]:
            draw.line([(0, pz), (size[0], pz)], fill=(255, 255, 255, 46))
            draw.text((2, pz + 2), f"z{z}", fill=(255, 255, 255, 190))
    draw.text((size[0] - 118, size[1] - 14),
              f"north is up  {step}-block grid", fill=(255, 255, 255, 210))
    return image


def landscape(voxels: dict, assets: Assets, size=(760, 560)) -> Image.Image:
    """A wide view of the ground around a player, for siting a build.

    Two angles rather than three: from above at a steep angle, where terrain shape reads
    best, and from lower down, where relief and cliff faces do.
    """
    # No supersampling: a landscape is read for shape, not for edge quality, and it carries
    # tens of thousands of quads where a single build carries hundreds.
    r = Renderer(assets, size=size, supersample=1)
    visible = surface(voxels)
    tiles = [r.render(visible, azimuth=45, elevation=60),
             r.render(visible, azimuth=45, elevation=18)]
    canvas = Image.new("RGB", (size[0] * 2, size[1]), (24, 26, 32))
    for i, tile in enumerate(tiles):
        canvas.paste(tile, (i * size[0], 0))
    return canvas


#: Model space is Y-down and X-mirrored relative to the world, so a face that points up in
#: the world is the box's underside in the texture layout, and left and right are swapped.
_MODEL_FACE = {"up": "down", "down": "up", "north": "north", "south": "south",
               "east": "west", "west": "east"}


def _box_uv(u, v, w, h, d, face):
    """Where one face of a box sits on the sheet. Every entity texture is the same unwrapped
    net — the four sides in a row, with the top and bottom above them."""
    return {"down":  (u + d,             v,     u + d + w,         v + d),
            "up":    (u + d + w,         v,     u + d + w * 2,     v + d),
            "east":  (u,                 v + d, u + d,             v + d + h),
            "north": (u + d,             v + d, u + d + w,         v + d + h),
            "west":  (u + d + w,         v + d, u + d * 2 + w,     v + d + h),
            "south": (u + d * 2 + w,     v + d, u + d * 2 + w * 2, v + d + h)}[face]


@functools.lru_cache(maxsize=1)
def _entity_models() -> dict:
    """The extracted entity geometry, read once."""
    from entity_models import load
    return load()


def entities_within(entities, voxels: dict, pad: float = 0.5) -> list:
    """Only the creatures actually inside what is being drawn.

    A scan returns every entity in the box it read, which is always larger than the build
    being rendered — and after `framed()` crops the voxels, larger again. Passing the lot
    through drew chickens standing in mid-air outside the structure, which reads as part of
    the build to anything looking at the picture.
    """
    if not entities or not voxels:
        return []
    lo = [min(p[i] for p in voxels) - pad for i in range(3)]
    hi = [max(p[i] for p in voxels) + 1 + pad for i in range(3)]
    return [e for e in entities
            if e.get("pos") and all(lo[i] <= e["pos"][i] <= hi[i] for i in range(3))]


def focus_structure(voxels: dict, row, ground_pad: int = 2) -> dict:
    """Show exactly one stored boundary, plus enough ground to orient it.

    Fetches are padded for framing, but rendering that whole fetch made a correct rail box
    look like the workshop beside it.  The review sheet is an audit of the stored boundary:
    built and natural blocks inside that boundary are shown, with only low ground around its
    footprint added as context.  If a neighbour remains, the boundary genuinely swallowed
    it and the render should expose that.
    """
    lo = [row["min_x"], row["min_y"], row["min_z"]]
    hi = [row["max_x"], row["max_y"], row["max_z"]]
    subject = {p: material for p, material in voxels.items()
               if all(lo[i] <= p[i] <= hi[i] for i in range(3))}
    if not subject:
        return {}
    ground_top = min(p[1] for p in subject)
    context = {p: material for p, material in voxels.items()
               if lo[0] - ground_pad <= p[0] <= hi[0] + ground_pad
               and lo[2] - ground_pad <= p[2] <= hi[2] + ground_pad
               and ground_top - ground_pad <= p[1] <= ground_top
               and is_natural(material)}
    return subject | context


def forget_renders(structure_id: str) -> int:
    """Delete every review render of a structure that no longer exists.

    Two code paths remove a structure row — a fresh look retiring one, and the world model
    merging two — and only one of them cleaned up. The leftovers are indistinguishable from
    real structures in the renders folder, which is exactly where they get read.
    """
    gone = 0
    for path in REVIEW_DIR.glob(f"{structure_id}__*"):
        path.unlink()
        gone += 1
    for path in (REVIEW_DIR / "excavations").glob(f"{structure_id}__*"):
        path.unlink()
        gone += 1
    for path in (REVIEW_DIR / "generated").glob(f"{structure_id}__*"):
        path.unlink()
        gone += 1
    return gone


def move_review_render(old_id: str, new_id: str) -> int:
    """Move an existing review sheet when a database correction changes its ontology/id."""
    target_dir = (REVIEW_DIR / "generated"
                  if new_id.startswith(("g_", "generated:", "generated_sub:"))
                  else REVIEW_DIR)
    target_dir.mkdir(parents=True, exist_ok=True)
    moved = 0
    for folder in (REVIEW_DIR, REVIEW_DIR / "generated"):
        for path in folder.glob(f"{old_id}__*.png"):
            suffix = path.name[len(old_id):]
            target = target_dir / f"{new_id}{suffix}"
            path.replace(target)
            moved += 1
    return moved


def sheet(voxels: dict, assets: Assets, size=(512, 460)) -> Image.Image:
    """Four complementary views: perspective, plan, front elevation, and side elevation.

    Every structure gets the same views. The overhead plan makes flat builds and floor art
    legible; the two elevations expose upright planar work; the perspective preserves depth.
    A 2x2 canvas retains more detail per view than a very wide strip after API resizing.
    Entity geometry is deliberately excluded: occupancy is separate evidence, not part of
    the structure's shape.
    """
    width = min(size[0], MAX_SHEET_WIDTH // 2)
    size = (width, size[1])
    if not voxels:
        return Image.new("RGB", (size[0] * 2, size[1] * 2), (24, 26, 32))
    r = Renderer(assets, size=size)
    xs = [p[0] for p in voxels]
    zs = [p[2] for p in voxels]
    half = max(max(xs) - min(xs) + 1, max(zs) - min(zs) + 1) / 2 + 1
    tiles = [
        r.render(voxels, azimuth=35, elevation=24),
        r.render(voxels, ortho_extent=(half, half)),
        r.render(voxels, azimuth=0, elevation=0),
        r.render(voxels, azimuth=90, elevation=0),
    ]
    w, h = size
    canvas = Image.new("RGB", (w * 2, h * 2), (24, 26, 32))
    for i, tile in enumerate(tiles):
        canvas.paste(tile, ((i % 2) * w, (i // 2) * h))
    return canvas
