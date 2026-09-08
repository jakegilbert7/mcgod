package io.github.jakegilbert7.mcgod.capture;

import java.util.EnumSet;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Set;
import java.util.UUID;
import org.bukkit.Bukkit;
import org.bukkit.Location;
import org.bukkit.entity.Player;
import org.bukkit.event.EventHandler;
import org.bukkit.event.EventPriority;
import org.bukkit.event.Listener;
import org.bukkit.event.inventory.InventoryCloseEvent;
import org.bukkit.event.inventory.InventoryOpenEvent;
import org.bukkit.event.inventory.InventoryType;
import org.bukkit.event.player.PlayerQuitEvent;
import org.bukkit.inventory.Inventory;
import org.bukkit.inventory.ItemStack;
import org.bukkit.loot.Lootable;

/**
 * What a player put into and took out of storage.
 *
 * <p>Implemented as a diff between opening and closing rather than by watching clicks. Clicks
 * are a nightmare of special cases — shift-click, drag, hotbar swap, number-key swap, double
 * click to gather — and getting any one of them wrong silently corrupts the count. Comparing
 * the container's contents before and against after is immune to all of it, and yields the
 * net movement rather than the gross fumbling, which is the number that actually matters.
 *
 * <p>Only true storage is tracked. Furnaces and brewing stands change their own contents while
 * open, so a diff would credit the player with "depositing" an iron ingot the furnace smelted;
 * those have {@code smelt} instead. Crafting grids are not storage at all.
 *
 * <p>Known limit: a hopper feeding a chest while a player stands in it is attributed to the
 * player. Rare, and cheaper to accept than to reconcile against every automation event.
 */
public final class ContainerTracker implements Listener {

    private static final Set<InventoryType> STORAGE = EnumSet.of(
            InventoryType.CHEST, InventoryType.ENDER_CHEST, InventoryType.BARREL,
            InventoryType.SHULKER_BOX, InventoryType.HOPPER,
            InventoryType.DISPENSER, InventoryType.DROPPER);

    private record Snapshot(String container, String dim, int[] pos, String lootTable,
                            Map<String, Integer> counts) {
    }

    private final EventQueue queue;
    private final Map<UUID, Snapshot> open = new HashMap<>();

    public ContainerTracker(EventQueue queue) {
        this.queue = queue;
    }

    @EventHandler(priority = EventPriority.MONITOR)
    public void onOpen(InventoryOpenEvent event) {
        if (!(event.getPlayer() instanceof Player player)) {
            return;
        }
        Inventory inventory = event.getInventory();
        if (!STORAGE.contains(inventory.getType())) {
            return;
        }
        Location where = inventory.getLocation() != null
                ? inventory.getLocation() : player.getLocation();
        String lootTable = null;
        if (inventory.getHolder(false) instanceof Lootable lootable
                && lootable.getLootTable() != null) {
            // Read before inventory contents: resolving a generated chest may clear its
            // pending loot table, and this key is the authoritative clue that distinguishes
            // a village weaponsmith chest from an arbitrary chest nearby.
            lootTable = lootable.getLootTable().key().asString();
        }
        open.put(player.getUniqueId(), new Snapshot(
                Dims.name(inventory.getType()),
                Dims.of(where.getWorld()),
                Dims.pos(where),
                lootTable,
                count(inventory)));
    }

    @EventHandler(priority = EventPriority.MONITOR)
    public void onClose(InventoryCloseEvent event) {
        if (!(event.getPlayer() instanceof Player player)) {
            return;
        }
        Snapshot before = open.remove(player.getUniqueId());
        if (before == null) {
            return;
        }
        Map<String, Integer> after = count(event.getInventory());
        long tick = Bukkit.getCurrentTick();
        String actor = player.getUniqueId().toString();

        Map<String, Integer> keys = new LinkedHashMap<>(before.counts());
        after.forEach((k, v) -> keys.putIfAbsent(k, 0));
        for (String item : keys.keySet()) {
            int delta = after.getOrDefault(item, 0) - before.counts().getOrDefault(item, 0);
            if (delta == 0) {
                continue;
            }
            queue.offer(GameEvent.of(tick, delta > 0 ? "container_put" : "container_take",
                            actor, before.dim(), before.pos())
                    .with("item", item)
                    .with("count", Math.abs(delta))
                    .with("container", before.container())
                    .with("loot_table", before.lootTable()));
        }
    }

    /** Nothing closes the inventory if the player vanishes mid-transaction. */
    @EventHandler(priority = EventPriority.MONITOR)
    public void onQuit(PlayerQuitEvent event) {
        open.remove(event.getPlayer().getUniqueId());
    }

    private static Map<String, Integer> count(Inventory inventory) {
        Map<String, Integer> counts = new LinkedHashMap<>();
        for (ItemStack stack : inventory.getContents()) {
            if (stack != null && !stack.getType().isAir()) {
                counts.merge(stack.getType().getKey().toString(), stack.getAmount(), Integer::sum);
            }
        }
        return counts;
    }
}
