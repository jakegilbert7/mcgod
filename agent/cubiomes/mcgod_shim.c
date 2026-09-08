// The C surface McGod's seedmap.py binds through ctypes.
//
// cubiomes' own API is stateful in ways that are awkward across a foreign function
// interface: a Generator must be set up for a version, seeded for a dimension, and reused.
// This keeps one generator per handle so a caller can hold several worlds or dimensions at
// once, and returns plain arrays rather than structs, which ctypes handles without a
// declaration of cubiomes' layout that could silently drift from the C.
//
// Nothing here decides anything. Every function is a thin pass-through to cubiomes so that
// what McGod believes about the world stays traceable to the generator the game uses.

#include <stdlib.h>
#include <string.h>

#include "generator.h"
#include "finders.h"
#include "util.h"

typedef struct {
    Generator g;
    int mc;
    uint64_t seed;
    int dim;
} McgodWorld;

McgodWorld *mcgod_open(int mc, uint64_t seed, int dim)
{
    McgodWorld *w = (McgodWorld *) calloc(1, sizeof(McgodWorld));
    if (!w)
        return NULL;
    w->mc = mc;
    w->seed = seed;
    w->dim = dim;
    setupGenerator(&w->g, mc, 0);
    applySeed(&w->g, dim, seed);
    return w;
}

void mcgod_close(McgodWorld *w)
{
    free(w);
}

int mcgod_newest_version(void) { return MC_NEWEST; }

int mcgod_version_from_string(const char *name) { return str2mc(name); }

const char *mcgod_version_name(int mc) { return mc2str(mc); }

const char *mcgod_biome_name(int mc, int id) { return biome2str(mc, id); }

int mcgod_biome_id(const char *name)
{
    for (int id = 0; id < 256; id++)
    {
        const char *known = biome2str(MC_NEWEST, id);
        if (known && !strcmp(known, name))
            return id;
    }
    return -1;
}

// One biome. `scale` is 1 for block coordinates or 4 for the quart coordinates the game
// actually stores; `hint` carries the previous result forward, which is what the game's
// own thread-local last-result does and what makes exact ties resolve identically.
int mcgod_biome_at(McgodWorld *w, int scale, int x, int y, int z, uint64_t *hint)
{
    if (w->dim == DIM_OVERWORLD && scale == 4 && w->mc > MC_B1_7)
    {
        int64_t np[6];
        return sampleBiomeNoise(&w->g.bn, np, x, y, z, hint, 0);
    }
    return getBiomeAt(&w->g, scale, x, y, z);
}

// A grid of biomes in one call, since a map is thousands of points and one ctypes call per
// point costs more than the generation does. Coordinates are in `scale` units, and `out`
// must hold w*h ints.
int mcgod_biome_grid(McgodWorld *world, int scale, int x0, int y, int z0,
                     int w, int h, int *out)
{
    uint64_t hint = 0;
    for (int j = 0; j < h; j++)
        for (int i = 0; i < w; i++)
            out[j * w + i] = mcgod_biome_at(world, scale, x0 + i, y, z0 + j, &hint);
    return w * h;
}

int mcgod_structure_type(const char *name)
{
    static const struct { const char *name; int type; } known[] = {
        {"desert_pyramid", Desert_Pyramid}, {"jungle_pyramid", Jungle_Temple},
        {"jungle_temple", Jungle_Temple},   {"swamp_hut", Swamp_Hut},
        {"igloo", Igloo},                   {"village", Village},
        {"ocean_ruin", Ocean_Ruin},         {"shipwreck", Shipwreck},
        {"monument", Monument},             {"mansion", Mansion},
        {"outpost", Outpost},               {"pillager_outpost", Outpost},
        {"ruined_portal", Ruined_Portal},   {"ruined_portal_n", Ruined_Portal_N},
        {"ancient_city", Ancient_City},     {"treasure", Treasure},
        {"buried_treasure", Treasure},      {"mineshaft", Mineshaft},
        {"desert_well", Desert_Well},       {"geode", Geode},
        {"fortress", Fortress},             {"bastion", Bastion},
        {"end_city", End_City},             {"end_gateway", End_Gateway},
        {"trail_ruins", Trail_Ruins},       {"trial_chambers", Trial_Chambers},
        {"ocean_monument", Monument},       {"woodland_mansion", Mansion},
    };
    for (size_t i = 0; i < sizeof(known) / sizeof(known[0]); i++)
        if (!strcmp(known[i].name, name))
            return known[i].type;
    return -1;
}

// Region grid size in chunks for a structure type, or -1 if this version has no such
// structure. Callers need it to know which regions to ask about.
int mcgod_region_size(int type, int mc)
{
    StructureConfig sc;
    if (!getStructureConfig(type, mc, &sc))
        return -1;
    return sc.regionSize;
}

// The candidate position for one region, before any biome check. Returns 0 if this
// region has no attempt (rarity), 1 otherwise.
int mcgod_structure_pos(int type, int mc, uint64_t seed, int regionX, int regionZ,
                        int *outX, int *outZ)
{
    Pos p;
    if (!getStructurePos(type, mc, seed, regionX, regionZ, &p))
        return 0;
    *outX = p.x;
    *outZ = p.z;
    return 1;
}

// Whether the terrain at a candidate actually admits the structure. Expensive relative to
// placement, so callers filter by distance first.
int mcgod_viable(McgodWorld *w, int type, int x, int z)
{
    return isViableStructurePos(type, &w->g, x, z, 0);
}

// Strongholds are not in the structure-type table above: they do not use the region
// grid at all, but sit on concentric rings and are walked with their own iterator. `out` must hold
// 2*count ints; returns how many were written.
int mcgod_strongholds(McgodWorld *w, int count, int *out)
{
    StrongholdIter sh;
    Pos p = initFirstStronghold(&sh, w->mc, w->seed);
    int written = 0;
    if (count <= 0)
        return 0;
    out[0] = p.x;
    out[1] = p.z;
    written = 1;
    while (written < count)
    {
        if (nextStronghold(&sh, &w->g) <= 0)
            break;
        out[2 * written] = sh.pos.x;
        out[2 * written + 1] = sh.pos.z;
        written++;
    }
    return written;
}

int mcgod_spawn(McgodWorld *w, int *out)
{
    Pos p = getSpawn(&w->g);
    out[0] = p.x;
    out[1] = p.z;
    return 1;
}

// The nearest place a given biome generates, searched outward in square rings.
//
// cubiomes' own locateBiome picks a pseudo-random valid spot, which is what the game wants
// for spawn and strongholds and is not what a player means by "where is the closest one".
// This walks rings of increasing radius at `step` spacing and keeps searching one ring
// past the first hit, because the corner of an inner ring can be farther in a straight
// line than the middle of the next one out.
//
// `step` is a sampling interval in blocks: a biome region smaller than it can be stepped
// over, so callers trade thoroughness against work rather than this guessing for them.
// Returns 1 and writes block coordinates when found, 0 otherwise.
int mcgod_nearest_biome(McgodWorld *w, int biome, int x, int y, int z,
                        int maxRadius, int step, int *outX, int *outZ)
{
    if (step < 4)
        step = 4;
    long long best = -1;
    int found = 0, stopRing = -1;
    uint64_t hint = 0;
    for (int r = 0; r <= maxRadius; r += step)
    {
        if (stopRing >= 0 && r > stopRing)
            break;
        for (int dx = -r; dx <= r; dx += step)
        {
            for (int dz = -r; dz <= r; dz += step)
            {
                // Only the perimeter of this ring; the interior was covered already.
                if (r > 0 && dx > -r && dx < r && dz > -r && dz < r)
                    continue;
                int px = x + dx, pz = z + dz;
                if (mcgod_biome_at(w, 4, px >> 2, y >> 2, pz >> 2, &hint) != biome)
                    continue;
                long long d = (long long)dx * dx + (long long)dz * dz;
                if (!found || d < best)
                {
                    best = d;
                    *outX = px;
                    *outZ = pz;
                    found = 1;
                }
                if (stopRing < 0)
                    stopRing = r + step;
            }
        }
    }
    return found;
}

// How much of each biome lies in an area. `counts` must hold 256 ints and is written with
// the number of sampled points per biome id, so a caller can name the most common biome or
// describe the mix without shipping every sample across the boundary.
int mcgod_biome_histogram(McgodWorld *w, int x0, int y, int z0,
                          int radius, int step, int *counts)
{
    if (step < 4)
        step = 4;
    memset(counts, 0, 256 * sizeof(int));
    uint64_t hint = 0;
    int total = 0;
    for (int dz = -radius; dz <= radius; dz += step)
    {
        for (int dx = -radius; dx <= radius; dx += step)
        {
            if ((long long)dx * dx + (long long)dz * dz > (long long)radius * radius)
                continue;
            int id = mcgod_biome_at(w, 4, (x0 + dx) >> 2, y >> 2, (z0 + dz) >> 2, &hint);
            if (id >= 0 && id < 256)
            {
                counts[id]++;
                total++;
            }
        }
    }
    return total;
}
