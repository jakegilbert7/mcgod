package io.github.jakegilbert7.mcgod.capture;

import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.Deque;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.TreeMap;
import java.util.function.Consumer;
import org.bukkit.Bukkit;
import org.bukkit.ChunkSnapshot;
import org.bukkit.Material;
import org.bukkit.World;
import org.bukkit.block.BlockState;
import org.bukkit.plugin.Plugin;
import org.bukkit.scheduler.BukkitRunnable;

/**
 * Reads a region of the world and summarises it.
 *
 * <p>The model never sees block arrays. A single chunk is roughly 98,000 block states; it
 * would not fit in a context window and the model would miscount it if it did. What comes
 * back is a material histogram, a bounding box, a block-entity census, and up to three
 * horizontal slices rendered as character grids with their own legend.
 *
 * <p>Two hard rules shape the implementation. Chunks are loaded with
 * {@code getChunkAtAsync} and immediately turned into a {@link ChunkSnapshot} on the main
 * thread; the snapshot is then safe to read from anywhere, so the aggregation runs off the
 * tick. And loading is rate limited to a few chunks per tick, because a large region would
 * otherwise stall the server for as long as the disk takes.
 */
public final class ScanService {

    /** Characters assigned to materials in a slice, commonest first. */
    private static final String PALETTE = "#*o+x=%&$@ABCDEFGHJKLMNPQRSTUVWXYZ";
    private static final char AIR = '.';
    private static final char OVERFLOW = '?';

    private static final int MAX_SPAN = 96;      // blocks per horizontal axis
    private static final int MAX_CHUNKS = 144;
    /**
     * Voxel exports are per-block, so they are capped tighter than histogram scans.
     *
     * <p>96 horizontally is enough to see the ground a build would sit in; the vertical
     * cap stays tight because terrain context needs the surface, not the whole stone column
     * beneath it.
     */
    private static final int MAX_VOXEL_SPAN = 96;
    private static final int MAX_VOXEL_HEIGHT = 48;
    private static final int MAX_HISTOGRAM = 40;
    private static final int MAX_GRID = 48;

    private final Plugin plugin;
    private final int chunksPerTick;

    public ScanService(Plugin plugin, int chunksPerTick) {
        this.plugin = plugin;
        this.chunksPerTick = Math.max(1, chunksPerTick);
    }

    /**
     * Exports a region as voxels: a material palette plus one byte per block.
     *
     * <p>This is the one place block-level data leaves the server, and it exists for the
     * renderer, which is agent-side. Non-negotiable #3 is about what reaches the MODEL: a
     * model is given the finished image, never this. A 48-cube packs to about 110KB before
     * base64, which is nothing over loopback and would be absurd in a prompt.
     */
    public void voxels(String id, World world, int[] min, int[] max, Consumer<String> reply) {
        int minX = Math.min(min[0], max[0]);
        int maxX = Math.max(min[0], max[0]);
        int minZ = Math.min(min[2], max[2]);
        int maxZ = Math.max(min[2], max[2]);
        int minY = Math.max(world.getMinHeight(), Math.min(min[1], max[1]));
        int maxY = Math.min(world.getMaxHeight() - 1, Math.max(min[1], max[1]));

        // Errors must answer on the channel the caller is listening to. Replying to a
        // voxel request on the scan channel does not fail loudly — the caller waits for a
        // reply that will never come and hangs until it times out, which looks exactly
        // like the server being slow rather than the region being too big.
        if (maxX - minX + 1 > MAX_VOXEL_SPAN || maxZ - minZ + 1 > MAX_VOXEL_SPAN) {
            reply.accept(voxelError(id, "voxel region exceeds " + MAX_VOXEL_SPAN
                    + " per horizontal axis"));
            return;
        }
        if (maxY - minY + 1 > MAX_VOXEL_HEIGHT) {
            reply.accept(voxelError(id, "voxel region exceeds " + MAX_VOXEL_HEIGHT
                    + " in height"));
            return;
        }
        collect(world, minX, minY, minZ, maxX, maxY, maxZ, (snapshots, living) ->
                reply.accept(packVoxels(id, world, snapshots,
                        "[" + String.join(",", living) + "]",
                        minX, minY, minZ, maxX, maxY, maxZ)));
    }

    /**
     * Living things in a region, with where they are.
     *
     * <p>Blocks alone cannot show what a farm is. A pen holding twelve chickens, a hall
     * with villagers in it and an empty room of the same shape are the same voxels; the
     * animals are the point. Must run on the main thread.
     */
    /** Loads the chunks a region covers, rate limited, then hands over their snapshots. */
    private void collect(World world, int minX, int minY, int minZ,
                         int maxX, int maxY, int maxZ,
                         java.util.function.BiConsumer<List<ChunkSnapshot>,
                                 List<String>> done) {
        int cx0 = minX >> 4, cx1 = maxX >> 4, cz0 = minZ >> 4, cz1 = maxZ >> 4;
        Deque<long[]> pending = new ArrayDeque<>();
        for (int cx = cx0; cx <= cx1; cx++) {
            for (int cz = cz0; cz <= cz1; cz++) {
                pending.add(new long[]{cx, cz});
            }
        }
        List<ChunkSnapshot> snapshots = new ArrayList<>();
        List<String> living = java.util.Collections.synchronizedList(new ArrayList<>());
        int[] outstanding = {pending.size()};
        new BukkitRunnable() {
            @Override
            public void run() {
                for (int i = 0; i < chunksPerTick && !pending.isEmpty(); i++) {
                    long[] c = pending.poll();
                    world.getChunkAtAsync((int) c[0], (int) c[1]).thenAccept(chunk -> {
                        snapshots.add(chunk.getChunkSnapshot(false, false, false));
                        // Read the living things while the chunk is in hand. Doing it in a
                        // second pass afterwards found nothing: the chunk had been let go
                        // again by then, and an unloaded chunk has no entities to give.
                        for (org.bukkit.entity.Entity e : chunk.getEntities()) {
                            org.bukkit.Location at = e.getLocation();
                            if (at.getX() < minX || at.getX() > maxX + 1
                                    || at.getY() < minY || at.getY() > maxY + 1
                                    || at.getZ() < minZ || at.getZ() > maxZ + 1) {
                                continue;
                            }
                            StringBuilder one = new StringBuilder("{\"type\":");
                            Json.string(one, e.getType().getKey().toString());
                            one.append(",\"pos\":[").append(at.getX()).append(',')
                                    .append(at.getY()).append(',').append(at.getZ())
                                    .append("]}");
                            living.add(one.toString());
                        }
                        if (--outstanding[0] == 0) {
                            Bukkit.getScheduler().runTaskAsynchronously(plugin,
                                    () -> done.accept(snapshots, living));
                        }
                    });
                }
                if (pending.isEmpty()) {
                    cancel();
                }
            }
        }.runTaskTimer(plugin, 0L, 1L);
    }

    private static String packVoxels(String id, World world, List<ChunkSnapshot> snapshots,
                                     String living,
                                     int minX, int minY, int minZ,
                                     int maxX, int maxY, int maxZ) {
        Map<ChunkKey, ChunkSnapshot> byChunk = new LinkedHashMap<>();
        for (ChunkSnapshot s : snapshots) {
            byChunk.put(new ChunkKey(s.getX(), s.getZ()), s);
        }
        int sx = maxX - minX + 1, sy = maxY - minY + 1, sz = maxZ - minZ + 1;
        Map<String, Integer> palette = new LinkedHashMap<>();
        palette.put("minecraft:air", 0);
        // Keyed by the BlockData object, not its string. A wide region holds hundreds of
        // thousands of blocks and only a few dozen distinct states, and getAsString()
        // builds a fresh string every call — doing that per block took minutes for a
        // landscape and seconds for a house, which is why it went unnoticed until the
        // region grew.
        Map<org.bukkit.block.data.BlockData, Integer> seen = new java.util.HashMap<>();
        byte[] data = new byte[sx * sy * sz];
        int i = 0;
        for (int y = minY; y <= maxY; y++) {
            for (int z = minZ; z <= maxZ; z++) {
                for (int x = minX; x <= maxX; x++) {
                    ChunkSnapshot chunk = byChunk.get(new ChunkKey(x >> 4, z >> 4));
                    int index = 0;
                    if (chunk != null) {
                        org.bukkit.block.data.BlockData bd =
                                chunk.getBlockData(x & 15, y, z & 15);
                        if (bd != null && !bd.getMaterial().isAir()) {
                            Integer known = seen.get(bd);
                            if (known == null) {
                                // The full block state, not just the material: facing,
                                // half and connection shape all live in the properties,
                                // and the client's blockstate JSON is keyed by this string.
                                known = palette.size();
                                if (known > 255) {
                                    known = 0;
                                } else {
                                    palette.put(bd.getAsString(), known);
                                }
                                seen.put(bd, known);
                            }
                            index = known;
                        }
                    }
                    data[i++] = (byte) index;
                }
            }
        }
        StringBuilder out = new StringBuilder(data.length * 2);
        out.append("{\"rpc\":\"voxel_result\",\"ok\":true,\"id\":");
        Json.string(out, id);
        out.append(",\"dim\":");
        Json.string(out, Dims.of(world));
        out.append(",\"origin\":[").append(minX).append(',').append(minY).append(',')
                .append(minZ).append(']');
        out.append(",\"size\":[").append(sx).append(',').append(sy).append(',')
                .append(sz).append(']');
        out.append(",\"palette\":[");
        boolean first = true;
        for (String key : palette.keySet()) {
            if (!first) {
                out.append(',');
            }
            first = false;
            Json.string(out, key);
        }
        out.append("],\"entities\":").append(living);
        out.append(",\"data\":");
        Json.string(out, java.util.Base64.getEncoder().encodeToString(data));
        out.append('}');
        return out.toString();
    }

    /** Resolves a schema dimension name to a loaded world. */
    public static World world(String dim) {
        for (World world : Bukkit.getWorlds()) {
            if (Dims.of(world).equals(dim) || world.getName().equals(dim)) {
                return world;
            }
        }
        return null;
    }

    /**
     * Scans a region and hands the finished JSON to {@code reply}.
     *
     * <p>Must be called on the main thread. {@code reply} is invoked off it.
     */
    public void scan(String id, World world, int[] min, int[] max, Consumer<String> reply) {
        int minX = Math.min(min[0], max[0]);
        int maxX = Math.max(min[0], max[0]);
        int minZ = Math.min(min[2], max[2]);
        int maxZ = Math.max(min[2], max[2]);
        int minY = Math.max(world.getMinHeight(), Math.min(min[1], max[1]));
        int maxY = Math.min(world.getMaxHeight() - 1, Math.max(min[1], max[1]));

        if (maxX - minX + 1 > MAX_SPAN || maxZ - minZ + 1 > MAX_SPAN) {
            reply.accept(error(id, "region exceeds " + MAX_SPAN + " blocks per horizontal axis"));
            return;
        }
        if (maxY < minY) {
            reply.accept(error(id, "empty vertical range"));
            return;
        }

        int cx0 = minX >> 4, cx1 = maxX >> 4, cz0 = minZ >> 4, cz1 = maxZ >> 4;
        int chunks = (cx1 - cx0 + 1) * (cz1 - cz0 + 1);
        if (chunks > MAX_CHUNKS) {
            reply.accept(error(id, "region spans " + chunks + " chunks, limit " + MAX_CHUNKS));
            return;
        }

        Deque<long[]> pending = new ArrayDeque<>();
        for (int cx = cx0; cx <= cx1; cx++) {
            for (int cz = cz0; cz <= cz1; cz++) {
                pending.add(new long[]{cx, cz});
            }
        }

        final int fMinX = minX, fMaxX = maxX, fMinY = minY, fMaxY = maxY, fMinZ = minZ, fMaxZ = maxZ;
        List<ChunkSnapshot> snapshots = new ArrayList<>();
        Map<String, Integer> blockEntities = new TreeMap<>();
        Map<String, Integer> entities = new TreeMap<>();
        int[] outstanding = {chunks};
        long startedTick = Bukkit.getCurrentTick();

        new BukkitRunnable() {
            @Override
            public void run() {
                for (int i = 0; i < chunksPerTick && !pending.isEmpty(); i++) {
                    long[] c = pending.poll();
                    world.getChunkAtAsync((int) c[0], (int) c[1]).thenAccept(chunk -> {
                        // Paper completes this on the main thread, so the snapshot and the
                        // block-entity read below are both safe here.
                        snapshots.add(chunk.getChunkSnapshot(false, false, false));
                        for (BlockState state : chunk.getTileEntities(false)) {
                            if (state.getX() >= fMinX && state.getX() <= fMaxX
                                    && state.getY() >= fMinY && state.getY() <= fMaxY
                                    && state.getZ() >= fMinZ && state.getZ() <= fMaxZ) {
                                blockEntities.merge(
                                        state.getType().getKey().toString(), 1, Integer::sum);
                            }
                        }
                        // What is actually alive in there. A pen holding twelve chickens
                        // and an empty pen are the same blocks; only the animals tell them
                        // apart, and the event stream cannot — it sees a breeding, never a
                        // flock. Entities must be read on the main thread, which is where
                        // Paper completes this.
                        for (org.bukkit.entity.Entity e : chunk.getEntities()) {
                            org.bukkit.Location at = e.getLocation();
                            if (at.getBlockX() >= fMinX && at.getBlockX() <= fMaxX
                                    && at.getBlockY() >= fMinY && at.getBlockY() <= fMaxY
                                    && at.getBlockZ() >= fMinZ && at.getBlockZ() <= fMaxZ) {
                                entities.merge(e.getType().getKey().toString(), 1,
                                        Integer::sum);
                            }
                        }
                        if (--outstanding[0] == 0) {
                            Bukkit.getScheduler().runTaskAsynchronously(plugin, () ->
                                    reply.accept(summarise(id, world, snapshots, blockEntities,
                                            entities, fMinX, fMinY, fMinZ, fMaxX, fMaxY,
                                            fMaxZ, startedTick)));
                        }
                    });
                }
                if (pending.isEmpty()) {
                    cancel();
                }
            }
        }.runTaskTimer(plugin, 0L, 1L);
    }

    /** Runs off the tick, over snapshots only. */
    private static String summarise(String id, World world, List<ChunkSnapshot> snapshots,
                                    Map<String, Integer> blockEntities,
                                    Map<String, Integer> entities,
                                    int minX, int minY, int minZ,
                                    int maxX, int maxY, int maxZ, long tick) {
        Map<ChunkKey, ChunkSnapshot> byChunk = new LinkedHashMap<>();
        for (ChunkSnapshot s : snapshots) {
            byChunk.put(new ChunkKey(s.getX(), s.getZ()), s);
        }

        Map<String, Integer> histogram = new LinkedHashMap<>();
        long air = 0;
        long total = 0;
        for (int y = minY; y <= maxY; y++) {
            for (int x = minX; x <= maxX; x++) {
                for (int z = minZ; z <= maxZ; z++) {
                    Material m = materialAt(byChunk, x, y, z);
                    total++;
                    if (m == null || m.isAir()) {
                        air++;
                    } else {
                        histogram.merge(m.getKey().toString(), 1, Integer::sum);
                    }
                }
            }
        }

        StringBuilder out = new StringBuilder(4096);
        out.append("{\"rpc\":\"scan_result\",\"ok\":true,\"id\":");
        Json.string(out, id);
        out.append(",\"tick\":").append(tick);
        out.append(",\"dim\":");
        Json.string(out, Dims.of(world));
        out.append(",\"bbox\":{\"min\":[").append(minX).append(',').append(minY).append(',')
                .append(minZ).append("],\"max\":[").append(maxX).append(',').append(maxY)
                .append(',').append(maxZ).append("]}");
        out.append(",\"volume\":").append(total);
        out.append(",\"air\":").append(air);
        out.append(",\"solid\":").append(total - air);

        List<Map.Entry<String, Integer>> ranked = new ArrayList<>(histogram.entrySet());
        ranked.sort(Comparator.<Map.Entry<String, Integer>>comparingInt(Map.Entry::getValue).reversed());
        out.append(",\"materials\":{");
        int shown = 0;
        long other = 0;
        for (Map.Entry<String, Integer> e : ranked) {
            if (shown < MAX_HISTOGRAM) {
                if (shown > 0) {
                    out.append(',');
                }
                Json.string(out, e.getKey());
                out.append(':').append(e.getValue());
                shown++;
            } else {
                other += e.getValue();
            }
        }
        out.append('}');
        out.append(",\"materials_truncated\":").append(other);
        out.append(",\"distinct_materials\":").append(histogram.size());

        out.append(",\"block_entities\":{");
        boolean firstBe = true;
        for (Map.Entry<String, Integer> e : blockEntities.entrySet()) {
            if (!firstBe) {
                out.append(',');
            }
            firstBe = false;
            Json.string(out, e.getKey());
            out.append(':').append(e.getValue());
        }
        out.append('}');

        out.append(",\"entities\":{");
        boolean firstEnt = true;
        for (Map.Entry<String, Integer> e : entities.entrySet()) {
            if (!firstEnt) {
                out.append(',');
            }
            firstEnt = false;
            Json.string(out, e.getKey());
            out.append(':').append(e.getValue());
        }
        out.append('}');

        out.append(",\"slices\":[");
        int[] levels = sliceLevels(minY, maxY);
        for (int i = 0; i < levels.length; i++) {
            if (i > 0) {
                out.append(',');
            }
            slice(out, byChunk, levels[i], minX, minZ, maxX, maxZ);
        }
        out.append("]}");
        return out.toString();
    }

    /** Bottom, middle and top of the requested range, deduplicated. */
    private static int[] sliceLevels(int minY, int maxY) {
        if (maxY == minY) {
            return new int[]{minY};
        }
        int mid = minY + (maxY - minY) / 2;
        return mid == minY || mid == maxY
                ? new int[]{minY, maxY}
                : new int[]{minY, mid, maxY};
    }

    private static void slice(StringBuilder out, Map<ChunkKey, ChunkSnapshot> byChunk, int y,
                              int minX, int minZ, int maxX, int maxZ) {
        int stepX = Math.max(1, (maxX - minX + 1 + MAX_GRID - 1) / MAX_GRID);
        int stepZ = Math.max(1, (maxZ - minZ + 1 + MAX_GRID - 1) / MAX_GRID);

        Map<String, Integer> counts = new LinkedHashMap<>();
        for (int x = minX; x <= maxX; x += stepX) {
            for (int z = minZ; z <= maxZ; z += stepZ) {
                Material m = materialAt(byChunk, x, y, z);
                if (m != null && !m.isAir()) {
                    counts.merge(m.getKey().toString(), 1, Integer::sum);
                }
            }
        }
        List<Map.Entry<String, Integer>> ranked = new ArrayList<>(counts.entrySet());
        ranked.sort(Comparator.<Map.Entry<String, Integer>>comparingInt(Map.Entry::getValue).reversed());
        Map<String, Character> legend = new LinkedHashMap<>();
        for (Map.Entry<String, Integer> e : ranked) {
            if (legend.size() >= PALETTE.length()) {
                break;
            }
            legend.put(e.getKey(), PALETTE.charAt(legend.size()));
        }

        out.append("{\"y\":").append(y);
        out.append(",\"step\":[").append(stepX).append(',').append(stepZ).append(']');
        out.append(",\"origin\":[").append(minX).append(',').append(minZ).append(']');
        out.append(",\"legend\":{");
        boolean first = true;
        for (Map.Entry<String, Character> e : legend.entrySet()) {
            if (!first) {
                out.append(',');
            }
            first = false;
            out.append('"').append(e.getValue()).append("\":");
            Json.string(out, e.getKey());
        }
        out.append("},\"rows\":[");
        boolean firstRow = true;
        StringBuilder row = new StringBuilder();
        for (int z = minZ; z <= maxZ; z += stepZ) {
            row.setLength(0);
            for (int x = minX; x <= maxX; x += stepX) {
                Material m = materialAt(byChunk, x, y, z);
                if (m == null || m.isAir()) {
                    row.append(AIR);
                } else {
                    row.append(legend.getOrDefault(m.getKey().toString(), OVERFLOW));
                }
            }
            if (!firstRow) {
                out.append(',');
            }
            firstRow = false;
            Json.string(out, row.toString());
        }
        out.append("]}");
    }

    private static Material materialAt(Map<ChunkKey, ChunkSnapshot> byChunk, int x, int y, int z) {
        ChunkSnapshot s = byChunk.get(new ChunkKey(x >> 4, z >> 4));
        return s == null ? null : s.getBlockType(x & 15, y, z & 15);
    }

    /** Full block state string, e.g. {@code minecraft:oak_stairs[facing=east,...]}. */
    private static String stateAt(Map<ChunkKey, ChunkSnapshot> byChunk, int x, int y, int z) {
        ChunkSnapshot s = byChunk.get(new ChunkKey(x >> 4, z >> 4));
        if (s == null) {
            return null;
        }
        org.bukkit.block.data.BlockData data = s.getBlockData(x & 15, y, z & 15);
        if (data == null || data.getMaterial().isAir()) {
            return null;
        }
        return data.getAsString();
    }

    private static String voxelError(String id, String message) {
        return failure("voxel_result", id, message);
    }

    private static String error(String id, String message) {
        return failure("scan_result", id, message);
    }

    private static String failure(String rpc, String id, String message) {
        StringBuilder out = new StringBuilder();
        out.append("{\"rpc\":");
        Json.string(out, rpc);
        out.append(",\"ok\":false,\"id\":");
        Json.string(out, id);
        out.append(",\"error\":");
        Json.string(out, message);
        out.append('}');
        return out.toString();
    }

    private record ChunkKey(int x, int z) {
    }
}
