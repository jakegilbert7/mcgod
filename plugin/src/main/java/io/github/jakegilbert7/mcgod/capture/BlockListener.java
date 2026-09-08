package io.github.jakegilbert7.mcgod.capture;

import org.bukkit.Bukkit;
import org.bukkit.block.Block;
import org.bukkit.block.BlockState;
import org.bukkit.entity.Player;
import org.bukkit.event.EventHandler;
import org.bukkit.event.EventPriority;
import org.bukkit.event.Listener;
import org.bukkit.event.block.BlockBreakEvent;
import org.bukkit.event.block.BlockPlaceEvent;
import org.bukkit.event.world.PortalCreateEvent;
import org.bukkit.event.player.PlayerBucketEmptyEvent;
import org.bukkit.event.player.PlayerBucketFillEvent;

/**
 * Changes a player makes to the world's blocks.
 *
 * <p>MONITOR + ignoreCancelled: we record what actually happened, after every other plugin
 * has had its say. Each handler builds a record and hands it off. No I/O, no world reads.
 */
public final class BlockListener implements Listener {

    private final EventQueue queue;

    public BlockListener(EventQueue queue) {
        this.queue = queue;
    }

    @EventHandler(priority = EventPriority.MONITOR, ignoreCancelled = true)
    public void onPlace(BlockPlaceEvent event) {
        Player player = event.getPlayer();
        Block placed = event.getBlockPlaced();
        BlockState replaced = event.getBlockReplacedState();
        queue.offer(GameEvent.blockPlace(
                Bukkit.getCurrentTick(),
                player.getUniqueId().toString(),
                Dims.of(placed.getWorld()),
                new int[]{placed.getX(), placed.getY(), placed.getZ()},
                replaced.getType().getKey().toString(),
                placed.getType().getKey().toString(),
                Dims.heldTool(player)));
    }

    @EventHandler(priority = EventPriority.MONITOR, ignoreCancelled = true)
    public void onBreak(BlockBreakEvent event) {
        Player player = event.getPlayer();
        Block broken = event.getBlock();
        queue.offer(GameEvent.blockBreak(
                Bukkit.getCurrentTick(),
                player.getUniqueId().toString(),
                Dims.of(broken.getWorld()),
                new int[]{broken.getX(), broken.getY(), broken.getZ()},
                broken.getType().getKey().toString(),
                Dims.heldTool(player)));
    }

    @EventHandler(priority = EventPriority.MONITOR, ignoreCancelled = true)
    public void onBucketFill(PlayerBucketFillEvent event) {
        bucket(event.getPlayer(), event.getBlock(), event.getBucket().getKey().toString(), "fill");
    }

    @EventHandler(priority = EventPriority.MONITOR, ignoreCancelled = true)
    public void onBucketEmpty(PlayerBucketEmptyEvent event) {
        bucket(event.getPlayer(), event.getBlock(), event.getBucket().getKey().toString(), "empty");
    }

    private void bucket(Player player, Block block, String item, String action) {
        queue.offer(GameEvent.of(Bukkit.getCurrentTick(), "bucket_use",
                        player.getUniqueId().toString(), Dims.of(block.getWorld()),
                        new int[]{block.getX(), block.getY(), block.getZ()})
                .with("item", item)
                .with("action", action));
    }

    /** Lighting a nether portal, or an end portal forming. Only recorded when a player caused it. */
    @EventHandler(priority = EventPriority.MONITOR, ignoreCancelled = true)
    public void onPortalCreate(PortalCreateEvent event) {
        if (!(event.getEntity() instanceof Player player)) {
            return;
        }
        queue.offer(GameEvent.of(Bukkit.getCurrentTick(), "portal_create",
                        player.getUniqueId().toString(), Dims.of(event.getWorld()),
                        Dims.pos(player.getLocation()))
                .with("reason", Dims.name(event.getReason()))
                .with("blocks", event.getBlocks().size()));
    }
}
