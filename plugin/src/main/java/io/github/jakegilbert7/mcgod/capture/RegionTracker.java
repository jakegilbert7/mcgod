package io.github.jakegilbert7.mcgod.capture;

import java.util.HashMap;
import java.util.Map;
import java.util.UUID;

/**
 * Decides when a player has meaningfully changed region.
 *
 * <p>The world is cut into a fixed XZ grid, but a grid alone is not enough: a player working
 * near a cell edge crosses it constantly, and a naive implementation reports an entry on every
 * crossing. That turns a stationary build into what looks downstream like frantic travel.
 *
 * <p>So the recorded cell only changes once the player is {@code hysteresis} blocks clear of
 * the cell they were last recorded in. Stepping over a boundary and back reports nothing;
 * genuinely leaving reports once. Deliberately not updating the recorded cell while the player
 * is inside the widened box is what makes crossing-and-returning free.
 *
 * <p>This separates jitter from movement. It does not separate movement from building — a
 * build large enough to span cells still reports real entries as the player works across it,
 * and that is correct here. Deciding those entries belong to one continuous build rather than
 * a journey is L1's job (P4), where episodes cluster on spatial-temporal density.
 *
 * <p>Deliberately free of any Bukkit dependency so it can be tested directly against recorded
 * sessions.
 */
public final class RegionTracker {

    private record Cell(UUID world, int cx, int cz) {
    }

    private final int regionSize;
    private final int hysteresis;
    private final Map<UUID, Cell> lastCell = new HashMap<>();

    public RegionTracker(int regionSize, int hysteresis) {
        this.regionSize = regionSize;
        this.hysteresis = hysteresis;
    }

    /**
     * Offers one observed position.
     *
     * @return true if this position counts as entering a new region.
     */
    public boolean enter(UUID player, UUID world, int x, int z) {
        Cell candidate = new Cell(world, Math.floorDiv(x, regionSize), Math.floorDiv(z, regionSize));
        Cell previous = lastCell.get(player);

        if (candidate.equals(previous)) {
            return false;
        }
        boolean changedWorld = previous == null || !previous.world().equals(candidate.world());
        if (!changedWorld && !isClearOf(previous, x, z)) {
            return false;
        }
        lastCell.put(player, candidate);
        return true;
    }

    /** True once the player is more than {@code hysteresis} blocks outside {@code cell}. */
    private boolean isClearOf(Cell cell, int x, int z) {
        int minX = cell.cx() * regionSize - hysteresis;
        int maxX = cell.cx() * regionSize + regionSize - 1 + hysteresis;
        int minZ = cell.cz() * regionSize - hysteresis;
        int maxZ = cell.cz() * regionSize + regionSize - 1 + hysteresis;
        return x < minX || x > maxX || z < minZ || z > maxZ;
    }

    public void forget(UUID player) {
        lastCell.remove(player);
    }
}
