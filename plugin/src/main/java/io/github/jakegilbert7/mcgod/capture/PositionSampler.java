package io.github.jakegilbert7.mcgod.capture;

import java.util.UUID;
import org.bukkit.Bukkit;
import org.bukkit.Location;
import org.bukkit.entity.Player;
import org.bukkit.scheduler.BukkitRunnable;

/**
 * Movement downsampling, done at the tap.
 *
 * <p>Instead of subscribing to PlayerMoveEvent (which fires up to 20x/sec/player and would be
 * discarded almost entirely), a repeating main-thread task samples every online player's block
 * position on a fixed period. Region entry is derived from the same sample; the decision of
 * what counts as entering a new region lives in {@link RegionTracker}.
 */
public final class PositionSampler extends BukkitRunnable {

    private final EventQueue queue;
    private final RegionTracker regions;

    public PositionSampler(EventQueue queue, int regionSize, int hysteresis) {
        this.queue = queue;
        this.regions = new RegionTracker(regionSize, hysteresis);
    }

    @Override
    public void run() {
        for (Player player : Bukkit.getOnlinePlayers()) {
            observe(player, player.getLocation());
        }
    }

    /** Records one position sample and any region change. Main thread only. */
    public void observe(Player player, Location location) {
        long tick = Bukkit.getCurrentTick();
        String actor = player.getUniqueId().toString();
        String dim = Dims.of(location.getWorld());
        int[] pos = Dims.pos(location);

        queue.offer(GameEvent.move(tick, actor, dim, pos));

        if (regions.enter(player.getUniqueId(), location.getWorld().getUID(), pos[0], pos[2])) {
            queue.offer(GameEvent.regionEnter(tick, actor, dim, pos));
        }
    }

    public void forget(UUID player) {
        regions.forget(player);
    }
}
