package io.github.jakegilbert7.mcgod.capture;

import java.util.LinkedHashSet;
import java.util.List;
import java.util.Set;
import org.bukkit.Location;
import org.bukkit.NamespacedKey;
import org.bukkit.Registry;
import org.bukkit.World;
import org.bukkit.block.Biome;
import org.bukkit.generator.structure.Structure;
import org.bukkit.util.StructureSearchResult;

/** Seed-backed knowledge of terrain and vanilla-generated features.
 *
 * <p>This is intentionally not a block scan. Biomes and structure placements are properties
 * of the world's generator and can therefore answer questions beyond explored chunks. The
 * reply is kept on its own RPC channel so generated features can never accidentally enter
 * the player-built structure pipeline.
 */
public final class EnvironmentService {

    private static final int MAX_BIOME_RADIUS = 1024;
    private static final int MAX_FEATURE_RADIUS = 8192;
    private static final int MIN_STEP = 8;
    private static final int MAX_FEATURES = 8;

    public String query(String id, World world, int x, int z, int radius, int step,
                        List<String> featureKeys) {
        radius = Math.max(0, Math.min(MAX_FEATURE_RADIUS, radius));
        // A locate question is answered directly from the generator result; sweeping a
        // thousand-block biome grid first would add server work without informing it.
        int biomeRadius = featureKeys.isEmpty() ? Math.min(MAX_BIOME_RADIUS, radius) : 0;
        step = Math.max(MIN_STEP, step);
        // Bound the number of generator reads independently of caller input.
        while ((2 * biomeRadius / step + 1) * (2 * biomeRadius / step + 1) > 4225) {
            step *= 2;
        }

        StringBuilder out = new StringBuilder(16_384);
        out.append("{\"rpc\":\"environment_result\",\"ok\":true,\"id\":");
        Json.string(out, id);
        out.append(",\"dim\":");
        Json.string(out, Dims.of(world));
        out.append(",\"seed\":").append(world.getSeed());
        out.append(",\"center\":[").append(x).append(',').append(z).append(']');
        out.append(",\"radius\":").append(radius).append(",\"step\":").append(step);
        out.append(",\"biomes\":[");
        boolean first = true;
        for (int dz = -biomeRadius; dz <= biomeRadius; dz += step) {
            for (int dx = -biomeRadius; dx <= biomeRadius; dx += step) {
                Biome biome = world.getBiome(x + dx, world.getSeaLevel(), z + dz);
                if (!first) {
                    out.append(',');
                }
                first = false;
                out.append('[').append(dx).append(',').append(dz).append(',');
                Json.string(out, Registry.BIOME.getKeyOrThrow(biome).toString());
                out.append(']');
            }
        }
        out.append(']');

        // Locating structures is substantially more expensive than reading biomes, so it
        // is opt-in and bounded. Duplicate keys and duplicate locations are collapsed.
        out.append(",\"generated_features\":[");
        first = true;
        Set<String> requested = new LinkedHashSet<>(featureKeys);
        Set<String> seen = new LinkedHashSet<>();
        int searched = 0;
        int chunkRadius = Math.max(1, (radius + 15) / 16);
        Location origin = new Location(world, x, world.getSeaLevel(), z);
        for (String raw : requested) {
            if (searched++ >= MAX_FEATURES) {
                break;
            }
            String clean = raw.startsWith("minecraft:") ? raw.substring(10) : raw;
            Structure structure = Registry.STRUCTURE.get(NamespacedKey.minecraft(clean));
            if (structure == null) {
                continue;
            }
            StructureSearchResult result = world.locateNearestStructure(
                    origin, structure, chunkRadius, false);
            if (result == null) {
                continue;
            }
            Location at = result.getLocation();
            NamespacedKey structureKey = Registry.STRUCTURE.getKeyOrThrow(structure);
            String identity = structureKey + ":" + at.getBlockX() + ":" + at.getBlockZ();
            if (!seen.add(identity)) {
                continue;
            }
            if (!first) {
                out.append(',');
            }
            first = false;
            out.append("{\"kind\":");
            Json.string(out, structureKey.toString());
            out.append(",\"pos\":[").append(at.getBlockX()).append(',')
                    .append(at.getBlockY()).append(',').append(at.getBlockZ()).append("]}");
        }
        out.append("]}");
        return out.toString();
    }
}
