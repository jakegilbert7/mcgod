# Vendored cubiomes, carried to Minecraft 26.2

Upstream is [Cubitect/cubiomes](https://github.com/Cubitect/cubiomes), MIT licensed; see
`LICENSE`. The commit this was taken from is in `UPSTREAM_COMMIT`. Upstream supports Java
releases through 1.21; everything below is what carries it to 26.2 "Chaos Cubed".

`seedmap.py` builds this on demand and binds it through ctypes. Nothing here is edited by
hand except the small, listed changes below.

## What was changed, and why each is safe

| file | change |
|---|---|
| `biomes.h` | adds `sulfur_caves = 187` and the `MC_26_2` version |
| `biomes.c` | `sulfur_caves` exists from 26.2 onward |
| `util.c` | the biome's name, its map colour, and the version's name |
| `biomenoise.c` | for 26.2, replaces the biome lookup with a port of the game's own search |
| `tables/rtree262.h` | generated; the game's parameter list, arranged as the game arranges it |
| `mcgod_shim.c` | McGod's own; a thin C surface for ctypes, no logic |

Everything except the shim is roughly seventy lines.

## Where the 26.2 data comes from

Nothing is guessed. Two artefacts, both produced by the game itself.

**The parameter list.** The server's data generator emits the exact multi-noise list it
uses to choose overworld biomes: 7,594 climate boxes over 55 biomes, sulfur caves included.

```bash
java -DbundlerMainClass=net.minecraft.data.Main -jar server/cache/mojang_26.2.jar \
     --reports --output /tmp/reports
```

**The search.** `tools/gen_rtree262.py` is a port of
`net.minecraft.world.level.biome.Climate$RTree`, read out of the 26.2 server jar with
`javap` (the jar ships unobfuscated, so this is the real control flow). It rebuilds the
tree the game builds — the same bucketing at `6^floor(log6(n - 0.01))`, the same rotating
seven-parameter sort, the same bounding-box cost, Java's truncating `(min+max)/2`, stable
sorts — and writes it out as a flat C table. `biomenoise.c` then walks that table exactly
as `Climate$RTree$SubTree.search` does, including the previous-result hint the game keeps
in a thread-local.

That hint is not a detail. It decides exact ties, and without it the port is 7 wrong in
24,323 samples; with it, zero. It is the difference between close and identical.

To regenerate after a Minecraft release, run the data generator again and:

```bash
python3 tools/gen_rtree262.py /tmp/reports/reports/biome_parameters/minecraft/overworld.json \
        biomes.h > tables/rtree262.h
```

New biomes must be added to the enum in `biomes.h` first, or the generator refuses rather
than silently dropping them.

## How exact it is

Measured against the running 26.2 server's own seed-backed sampler, at the quart grid the
game stores biomes on:

| set | points | mismatches |
|---|---:|---:|
| overworld, 25 centres across 32k blocks | 7,225 | 0 |
| overworld, corners at 80k and out to 120k | 2,023 | 0 |
| overworld, dense 4-block grid | 8,450 | 0 |
| overworld, three unseen centres | 3,267 | 0 |
| Nether | 12,675 | 0 |
| End | 12,675 | 0 |
| **total** | **46,315** | **0** |

Structure placement is exact: every position checked matched the server's own locate.

## What is not exact, and it is not the biomes

Placement is arithmetic and is exact. Whether a candidate becomes a real structure depends
on the terrain check, and the game consults surface height where cubiomes cannot. Measured
per type by asking the server about the seed map's own results:

- **exact on every sample**: village, pillager outpost, swamp hut, monument, mansion,
  trial chambers, igloo, ruined portal, ocean ruin, buried treasure
- **approximate**: ancient city and shipwreck and jungle temple and trail ruins miss
  occasionally; desert pyramid misses about half the time
- **placement only**: desert well, geode and mineshaft cannot be located by the server at
  all, so nothing has checked them

`seedmap.ACCURACY` carries this, and every result says which it is, so the god hedges
instead of asserting. Anything specific and important should still be confirmed against
the running server.

## The limit worth remembering

This describes the world **as generated**. It does not know what anyone has built, mined,
looted or destroyed. It is evidence about unvisited ground, never about the present.
