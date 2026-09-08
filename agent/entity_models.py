"""Entity geometry, read out of the game's own model classes.

Block models ship as JSON and can simply be read. Entity models do not: they are built in
Java, in `createBodyLayer()` methods that chain `CubeListBuilder.create().texOffs(u, v)
.addBox(x, y, z, w, h, d)` and hang the result on a part tree. There is no data file to
parse anywhere in the jar.

Transcribing the numbers by hand was tried first and the chickens came out looking like
creepers — close enough to be recognisably wrong, which is worse than a marker, because the
model reading the render believes it. So the numbers are taken from the game instead: this
walks the bytecode of those methods and pulls out what they actually build.

It is a small stack machine, not a decompiler. It knows how to push a constant, follow a
local variable, and recognise the handful of builder calls that define geometry. Anything
else it steps over. That is enough, because these methods are all shaped the same way.
"""
from __future__ import annotations

import json
import re
import subprocess
import zipfile
from pathlib import Path

CACHE = Path(__file__).parent / "entity_models.json"
CLASSES = Path("/tmp/mcgod-classes")

#: Minecraft renders a living entity with `scale(-1, -1, 1)` then `translate(0, -1.501, 0)`,
#: so model space is 16 units per block, Y pointing DOWN from a point 1.501 blocks above the
#: feet, and X mirrored. Getting this wrong puts every mob underground or upside down.
def to_world(mx: float, my: float, mz: float) -> tuple:
    return (-mx / 16.0, 1.501 - my / 16.0, mz / 16.0)


class Builder:
    """A `CubeListBuilder` as it accumulates: a texture corner, a mirror flag, and cubes."""

    def __init__(self):
        self.tex = (0, 0)
        self.mirror = False
        self.cubes: list = []

    def add(self, args: list) -> None:
        floats = [a for a in args if isinstance(a, (int, float))]
        if len(floats) < 6:
            return
        x, y, z, w, h, d = floats[:6]
        grow = floats[6] if len(floats) > 6 else 0.0
        self.cubes.append({"tex": list(self.tex), "mirror": self.mirror,
                           "from": [x - grow, y - grow, z - grow],
                           "size": [w + grow * 2, h + grow * 2, d + grow * 2],
                           "uv_size": [w, h, d]})


class Part:
    """One node of the part tree: a pose relative to its parent, and its cubes."""

    def __init__(self, name="root", pose=(0, 0, 0, 0, 0, 0), cubes=()):
        self.name = name
        self.pose = pose
        self.cubes = list(cubes)
        self.children: list = []

    def flat(self, ox=0.0, oy=0.0, oz=0.0, rot=(0.0, 0.0, 0.0)) -> list:
        """Every cube in the tree, with its pivot and rotation resolved against its parents.

        Rotation is kept per-cube rather than baked, because a rotated part rotates about
        its own pivot — a chicken's body is laid flat by a 90 degree pitch, and flattening
        that into corner coordinates would need the renderer to know nothing about pivots.
        """
        px, py, pz = ox + self.pose[0], oy + self.pose[1], oz + self.pose[2]
        turn = (rot[0] + self.pose[3], rot[1] + self.pose[4], rot[2] + self.pose[5])
        out = [{**cube, "pivot": [px, py, pz], "rot": list(turn)} for cube in self.cubes]
        for child in self.children:
            out += child.flat(px, py, pz, turn)
        return out


#: Bytecode constants this needs to recognise. Everything else is stepped over.
PUSH = {"iconst_m1": -1, "iconst_0": 0, "iconst_1": 1, "iconst_2": 2, "iconst_3": 3,
        "iconst_4": 4, "iconst_5": 5, "fconst_0": 0.0, "fconst_1": 1.0, "fconst_2": 2.0,
        "dconst_0": 0.0, "dconst_1": 1.0, "lconst_0": 0, "lconst_1": 1}
LDC = re.compile(r"//\s*(?:String (.*)|float ([-\d.E]+)f?|int (-?\d+)|long|double)")
CALL = re.compile(r"//\s*(?:Interface)?Method (?:([\w/$]+)\.)?\"?([\w<>$]+)\"?:\((.*?)\)(.*)")
FIELD = re.compile(r"//\s*Field ([\w/$]+)?\.?(\w+):(.*)")


def args_of(descriptor: str) -> int:
    """How many operands a JVM method descriptor consumes. Doubles and longs take one slot
    here because javap's constant pushes are one entry each in this machine."""
    n, i = 0, 0
    while i < len(descriptor):
        c = descriptor[i]
        if c == "L":
            i = descriptor.index(";", i)
        elif c == "[":
            i += 1
            continue
        n += 1
        i += 1
    return n


class Walker:
    """Walks the bytecode of the mesh-building methods and reconstructs the part tree."""

    def __init__(self):
        self.listings: dict = {}
        self.texture_size = (64, 32)

    def listing(self, klass: str) -> dict:
        """javap one class, split into methods. Cached — a superclass is asked for often."""
        if klass in self.listings:
            return self.listings[klass]
        try:
            text = subprocess.run(
                ["javap", "-c", "-p", "-classpath", str(CLASSES), klass.replace("/", ".")],
                capture_output=True, text=True, timeout=60).stdout
        except Exception:
            text = ""
        methods: dict = {}
        name, body = None, []
        for line in text.splitlines():
            head = re.match(r"\s{2}[\w.<> ]*?([\w<>$]+)\((.*?)\);?$", line)
            if head and not line.startswith("     "):
                if name:
                    methods.setdefault(name, []).append(body)
                name, body = head.group(1), []
            elif name is not None:
                body.append(line)
        if name:
            methods.setdefault(name, []).append(body)
        self.listings[klass] = methods
        return methods

    def run(self, klass: str, method: str, depth: int = 0):
        """Execute one method symbolically. Returns whatever it left on the stack."""
        if depth > 6:
            return None
        bodies = self.listing(klass).get(method) or []
        result = None
        for body in bodies:
            got = self._execute(klass, body, depth)
            if got is not None:
                result = got
        return result

    def _execute(self, klass: str, body: list, depth: int):
        stack: list = []
        locals_: dict = {}
        root = Part()
        produced = None

        for line in body:
            code = line.split(":", 1)[-1].strip()
            op = code.split()[0] if code else ""

            if op in PUSH:
                stack.append(PUSH[op])
            elif op in ("bipush", "sipush"):
                stack.append(int(code.split()[1]))
            elif op in ("ldc", "ldc_w", "ldc2_w"):
                found = LDC.search(line)
                if found:
                    text, number, whole = found.groups()
                    stack.append(text if text is not None else
                                 float(number) if number is not None else
                                 int(whole) if whole is not None else 0)
                else:
                    stack.append(0)
            elif op.startswith("aload") or op.startswith("iload") or op.startswith("fload"):
                slot = code.split()[-1] if "_" not in op else op.split("_")[1]
                stack.append(locals_.get(int(slot)))
            elif op.startswith("astore") or op.startswith("istore") or op.startswith("fstore"):
                slot = code.split()[-1] if "_" not in op else op.split("_")[1]
                locals_[int(slot)] = stack.pop() if stack else None
            elif op == "dup":
                stack.append(stack[-1] if stack else None)
            elif op == "pop":
                stack.pop() if stack else None
            elif op == "new":
                stack.append(None)
            elif op == "getstatic":
                found = FIELD.search(line)
                # PartPose.ZERO is how a part with no offset is written.
                stack.append((0.0,) * 6 if found and found.group(2) == "ZERO" else None)
            elif op == "putfield" or op == "putstatic":
                for _ in range(2 if op == "putfield" else 1):
                    stack.pop() if stack else None
            elif op == "getfield":
                stack.pop() if stack else None
                stack.append(None)
            elif op.startswith("invoke"):
                found = CALL.search(line)
                if not found:
                    continue
                owner, name, descriptor, _ = found.groups()
                count = args_of(descriptor)
                args = [stack.pop() if stack else None for _ in range(count)][::-1]
                receiver = None
                if op != "invokestatic":
                    receiver = stack.pop() if stack else None
                stack.append(self._call(klass, owner, name, args, receiver, root, depth))
                if name == "getRoot":
                    stack[-1] = root
            elif op in ("areturn", "ireturn", "freturn"):
                produced = stack[-1] if stack else None
        return produced if produced is not None else (root if root.children else None)

    def _call(self, klass, owner, name, args, receiver, root, depth):
        """The handful of calls that actually build geometry. Everything else returns
        whatever it was called on, so a chained builder survives an unknown call."""
        if name == "create" and owner and "CubeListBuilder" in owner:
            return Builder()
        if name == "texOffs" and isinstance(receiver, Builder):
            receiver.tex = (args[0] or 0, args[1] or 0)
            return receiver
        if name == "mirror" and isinstance(receiver, Builder):
            receiver.mirror = args[0] if args and isinstance(args[0], int) else True
            return receiver
        if name == "addBox" and isinstance(receiver, Builder):
            receiver.add(args)
            return receiver
        if name in ("offset", "offsetAndRotation", "rotation") and owner \
                and "PartPose" in owner:
            nums = [a if isinstance(a, (int, float)) else 0.0 for a in args]
            if name == "rotation":
                return (0.0, 0.0, 0.0, *nums[:3])
            return tuple(nums[:3]) + tuple(nums[3:6] if len(nums) > 3 else (0.0,) * 3)
        if name == "addOrReplaceChild":
            parent = receiver if isinstance(receiver, Part) else root
            label = next((a for a in args if isinstance(a, str)), "part")
            builder = next((a for a in args if isinstance(a, Builder)), None)
            pose = next((a for a in args if isinstance(a, tuple) and len(a) == 6),
                        (0.0,) * 6)
            child = Part(label, pose, builder.cubes if builder else [])
            parent.children.append(child)
            return child
        if name == "create" and owner and "LayerDefinition" in owner:
            sizes = [a for a in args if isinstance(a, int)]
            if len(sizes) >= 2:
                self.texture_size = (sizes[-2], sizes[-1])
            return next((a for a in args if isinstance(a, Part)), receiver)
        if name == "getRoot":
            return root
        # A mesh built by a helper or a superclass — follow it and graft what it returns.
        if owner and name not in ("<init>",) and (
                "Model" in owner or "MeshDefinition" in owner):
            got = self.run(owner, name, depth + 1)
            if isinstance(got, Part):
                # A helper that takes a part adds ITS parts to that one — `addCommonParts`
                # builds the hull of a boat onto the root it is handed. Recursing and
                # returning a detached tree threw all of that away and left a boat as one
                # quad of water.
                target = next((a for a in args if isinstance(a, Part)), None)
                if target is not None and got is not target:
                    target.children.extend(got.children)
                    return target
                return got
        if name in ("<init>",):
            return None
        return receiver


def find_class(zip_names: list, entity: str) -> list:
    """Every model class that might hold this entity's geometry, best guess first.

    One name is not enough. Animals were split into a base class holding the animation and
    an Adult/Baby pair holding the geometry, so the obvious `CatModel` does not exist and
    `AdultCatModel` carries only a constructor — the mesh is on a shared parent. Boats and
    minecarts hang several entity types off one class. So candidates are collected and each
    is tried in turn until one actually yields cubes.
    """
    stem = "".join(w.capitalize() for w in entity.split("_"))
    paths = [n[:-6] for n in zip_names
             if n.startswith("net/minecraft/client/model/") and n.endswith(".class")
             and "$" not in n]
    by_name = {p.rsplit("/", 1)[-1]: p for p in paths}
    out = []
    for want in (f"Adult{stem}Model", f"{stem}Model", f"{stem}EntityModel"):
        if want in by_name:
            out.append(by_name[want])
    # A boat, a raft or a chest minecart is a variant of one shared model; strip the wood
    # or cargo prefix and look for the base.
    for suffix in ("boat", "raft", "minecart", "chest_boat", "chest_raft"):
        if entity.endswith(suffix) and entity != suffix:
            base = "".join(w.capitalize() for w in suffix.split("_")) + "Model"
            if base in by_name and by_name[base] not in out:
                out.append(by_name[base])
    # Failing that, any class in the same package — that is where a shared parent lives.
    for candidate in out[:1]:
        package = candidate.rsplit("/", 1)[0]
        for path in paths:
            if path.rsplit("/", 1)[0] == package and path not in out \
                    and "Baby" not in path:
                out.append(path)
    return out


#: The method that builds the mesh. Most classes use the first; a few name it differently.
MESH_METHODS = ("createBodyLayer", "createMesh", "createBaseChickenModel", "createLayer",
                "createBodyMesh", "createAnimatedBodyLayer")


def geometry(entity: str, zip_names: list) -> dict | None:
    """The real part tree for one entity type, straight out of the game's own class."""
    for klass in find_class(zip_names, entity):
        walker = Walker()
        available = walker.listing(klass)
        methods = [m for m in MESH_METHODS if m in available]
        methods += [m for m in available
                    if m.startswith("create") and m not in methods
                    and "Baby" not in m and "Chest" not in m]
        # Take the richest mesh the class builds, not the first. A boat class also builds a
        # one-quad water patch, and that came back first and stood in for the whole boat.
        best = None
        for method in methods:
            try:
                part = walker.run(klass, method)
            except Exception:
                continue
            if isinstance(part, Part) and part.flat():
                cubes = part.flat()
                if best is None or len(cubes) > len(best["cubes"]):
                    best = {"class": klass, "texture_size": list(walker.texture_size),
                            "cubes": cubes}
        if best:
            return best
    return None


#: An entity whose own class holds no geometry usually borrows another's. These are the
#: cases where the name says so: a spruce boat is a boat, a husk is a zombie shape.
def base_of(entity: str) -> str | None:
    for suffix in ("chest_boat", "chest_raft", "boat", "raft", "minecart", "spider",
                   "horse", "skeleton", "zombie", "villager", "golem", "slime"):
        if entity.endswith("_" + suffix):
            return suffix
    for word, base in (("husk", "zombie"), ("stray", "skeleton"), ("wither_skeleton",
                       "skeleton"), ("mooshroom", "cow"), ("mule", "donkey"),
                       ("magma_cube", "slime"), ("glow_squid", "squid"),
                       ("pillager", "villager"), ("vindicator", "villager"),
                       ("illusioner", "villager"), ("giant", "zombie"),
                       ("snow_golem", "snow_golem"), ("trader_llama", "llama"),
                       ("skeleton_horse", "horse"), ("zombie_horse", "horse")):
        if entity == word:
            return base
    return None


def find_texture(entity: str, zip_names: list) -> str | None:
    """The texture sheet for an entity. Layout is inconsistent — some sit directly in
    `textures/entity/`, some in a folder of variants — so both are tried."""
    root = "assets/minecraft/textures/entity/"
    exact = [f"{root}{entity}.png", f"{root}{entity}/{entity}.png"]
    for path in exact:
        if path in zip_names:
            return path[len("assets/minecraft/textures/"):-4]
    folder = [n for n in zip_names if n.startswith(f"{root}{entity}/")
              and n.endswith(".png") and "baby" not in n]
    if folder:
        # Prefer the plain or temperate variant; a snowy fox is still a fox.
        best = sorted(folder, key=lambda n: (0 if "temperate" in n else
                                             1 if n.endswith(f"/{entity}.png") else 2,
                                             len(n)))[0]
        return best[len("assets/minecraft/textures/"):-4]
    loose = [n for n in zip_names if n.startswith(root) and n.endswith(f"/{entity}.png")]
    return loose[0][len("assets/minecraft/textures/"):-4] if loose else None


def build_cache(zip_names: list, entities: list) -> dict:
    """Extract every entity we can, once, and write it beside the code.

    This shells out to javap a few hundred times, which is far too slow to do per render —
    but the game's models do not change between runs, so it is done once and cached.
    """
    out: dict = {}
    for entity in entities:
        try:
            model = geometry(entity, zip_names)
        except Exception:
            model = None
        if not model:
            continue
        texture = find_texture(entity, zip_names)
        if not texture:
            continue
        out[entity] = {**model, "texture": texture}
    # Second pass: an entity with no class of its own borrows the shape it is a variant of.
    # Done after, because a fallback can only be resolved once its base has been extracted.
    for entity in entities:
        if entity in out:
            continue
        base = base_of(entity)
        if base and base in out:
            texture = find_texture(entity, zip_names) or out[base]["texture"]
            out[entity] = {**out[base], "texture": texture, "borrowed_from": base}
    return out


def load() -> dict:
    """The cached models. Empty if the cache has not been built — callers fall back."""
    try:
        return json.loads(CACHE.read_text())
    except Exception:
        return {}


def main() -> int:
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from assets import Assets

    assets = Assets()
    names = assets.zip.namelist()
    if not CLASSES.exists() or not any(CLASSES.rglob("*.class")):
        CLASSES.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(assets.zip.filename) as jar:
            for name in names:
                if name.startswith("net/minecraft/client/model/"):
                    jar.extract(name, CLASSES)
    lang = json.loads(assets.zip.read("assets/minecraft/lang/en_us.json"))
    entities = sorted({k.split(".")[-1] for k in lang
                       if k.startswith("entity.minecraft.") and k.count(".") == 2})
    print(f"{len(entities)} entity types in the game")
    cache = build_cache(names, entities)
    CACHE.write_text(json.dumps(cache, indent=0, sort_keys=True))
    print(f"extracted {len(cache)}, {sum(len(v['cubes']) for v in cache.values())} cubes")
    missing = [e for e in entities if e not in cache]
    print(f"no model for {len(missing)}: {', '.join(missing[:25])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
