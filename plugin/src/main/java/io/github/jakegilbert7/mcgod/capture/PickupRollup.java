package io.github.jakegilbert7.mcgod.capture;

import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.UUID;
import org.bukkit.Bukkit;
import org.bukkit.Location;

/**
 * Item pickup downsampling, done at the tap.
 *
 * <p>Picking up a stack of cobblestone can fire dozens of events a second. Instead of one line
 * per pickup, counts accumulate per (actor, item) and flush on a fixed period as one
 * {@code item_pickup} event carrying a {@code count}. The position reported is where the actor
 * made their most recent pickup in the window.
 *
 * <p>Main thread only: both {@link #record} (from the listener) and {@link #flush} (from a sync
 * task) run on the tick, so no synchronization is needed.
 */
public final class PickupRollup {

    private static final class Pending {
        private final Map<String, Integer> counts = new LinkedHashMap<>();
        private String dim;
        private int[] pos;
    }

    private final EventQueue queue;
    private final Map<UUID, Pending> pending = new HashMap<>();

    public PickupRollup(EventQueue queue) {
        this.queue = queue;
    }

    public void record(UUID actor, Location where, String item, int amount) {
        Pending p = pending.computeIfAbsent(actor, key -> new Pending());
        p.dim = Dims.of(where.getWorld());
        p.pos = Dims.pos(where);
        p.counts.merge(item, amount, Integer::sum);
    }

    /** Emits one event per (actor, item) accumulated since the last flush, and resets. */
    public void flush() {
        if (pending.isEmpty()) {
            return;
        }
        long tick = Bukkit.getCurrentTick();
        for (Map.Entry<UUID, Pending> entry : pending.entrySet()) {
            String actor = entry.getKey().toString();
            Pending p = entry.getValue();
            for (Map.Entry<String, Integer> item : p.counts.entrySet()) {
                queue.offer(GameEvent.itemPickup(tick, actor, p.dim, p.pos, item.getKey(), item.getValue()));
            }
        }
        pending.clear();
    }
}
