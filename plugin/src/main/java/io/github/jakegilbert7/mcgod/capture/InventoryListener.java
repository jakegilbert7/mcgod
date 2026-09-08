package io.github.jakegilbert7.mcgod.capture;

import java.util.EnumSet;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Set;
import org.bukkit.Bukkit;
import org.bukkit.Keyed;
import org.bukkit.Location;
import org.bukkit.entity.Player;
import org.bukkit.event.EventHandler;
import org.bukkit.event.EventPriority;
import org.bukkit.event.Listener;
import org.bukkit.event.enchantment.EnchantItemEvent;
import org.bukkit.event.entity.EntityPickupItemEvent;
import org.bukkit.event.inventory.CraftItemEvent;
import org.bukkit.event.inventory.FurnaceExtractEvent;
import org.bukkit.event.inventory.InventoryOpenEvent;
import org.bukkit.event.inventory.InventoryType;
import org.bukkit.event.player.PlayerDropItemEvent;
import org.bukkit.inventory.ItemStack;

/**
 * Where a player's items come from and go.
 *
 * <p>Pickups are rolled up per minute by {@link PickupRollup}; drops are recorded individually
 * because they are rare and because a drop is the counterweight that makes pickup counts mean
 * anything. Without drops, a player who gathers 300 cobblestone and throws away 200 reads as
 * having kept 300, and net accounting silently overstates every haul.
 */
public final class InventoryListener implements Listener {

    /** Openings that are not storage: the player's own inventory, a crafting grid, creative. */
    private static final Set<InventoryType> IGNORED_CONTAINERS = EnumSet.of(
            InventoryType.PLAYER, InventoryType.CRAFTING, InventoryType.CREATIVE);

    private final EventQueue queue;
    private final PickupRollup pickups;

    public InventoryListener(EventQueue queue, PickupRollup pickups) {
        this.queue = queue;
        this.pickups = pickups;
    }

    @EventHandler(priority = EventPriority.MONITOR, ignoreCancelled = true)
    public void onPickup(EntityPickupItemEvent event) {
        if (!(event.getEntity() instanceof Player player)) {
            return;
        }
        ItemStack stack = event.getItem().getItemStack();
        pickups.record(player.getUniqueId(), player.getLocation(),
                stack.getType().getKey().toString(), stack.getAmount());
    }

    @EventHandler(priority = EventPriority.MONITOR, ignoreCancelled = true)
    public void onDrop(PlayerDropItemEvent event) {
        ItemStack stack = event.getItemDrop().getItemStack();
        queue.offer(event(event.getPlayer(), "item_drop")
                .with("item", Dims.item(stack))
                .with("count", stack.getAmount()));
    }

    /**
     * Crafting, with the real quantity for bulk crafts.
     *
     * <p>Shift-clicking a recipe crafts as many as the ingredients allow but fires the event
     * exactly once, so reporting the recipe's result size undercounts badly — a shift-click
     * that made 64 stairs looked like 4. The repeat count is the smallest ingredient stack in
     * the grid, which is what limits how many times the recipe can run.
     *
     * <p>It is an upper bound: it does not model running out of inventory space mid-craft.
     * {@code bulk} marks the events where that caveat applies, so downstream can tell an exact
     * number from an estimated one rather than trusting both equally.
     */
    @EventHandler(priority = EventPriority.MONITOR, ignoreCancelled = true)
    public void onCraft(CraftItemEvent event) {
        if (!(event.getWhoClicked() instanceof Player player)) {
            return;
        }
        ItemStack result = event.getRecipe().getResult();
        int perCraft = result.getAmount();
        int crafts = 1;
        if (event.isShiftClick()) {
            int limit = Integer.MAX_VALUE;
            for (ItemStack ingredient : event.getInventory().getMatrix()) {
                if (ingredient != null && !ingredient.getType().isAir()) {
                    limit = Math.min(limit, ingredient.getAmount());
                }
            }
            if (limit != Integer.MAX_VALUE) {
                crafts = limit;
            }
        }
        Map<String, Integer> inputs = new LinkedHashMap<>();
        for (ItemStack ingredient : event.getInventory().getMatrix()) {
            if (ingredient != null && !ingredient.getType().isAir()) {
                // Vanilla crafting consumes one item from each occupied matrix slot per
                // recipe run. Aggregate equal materials so downstream can reconstruct a
                // transformation without knowing the shape of every recipe.
                inputs.merge(Dims.item(ingredient), crafts, Integer::sum);
            }
        }
        String recipe = event.getRecipe() instanceof Keyed keyed
                ? keyed.getKey().toString() : null;
        queue.offer(event(player, "craft")
                .with("item", Dims.item(result))
                .with("count", perCraft * crafts)
                .with("per_craft", perCraft)
                .with("bulk", event.isShiftClick())
                .with("recipe", recipe)
                .with("inputs", inputs));
    }

    @EventHandler(priority = EventPriority.MONITOR, ignoreCancelled = true)
    public void onSmelt(FurnaceExtractEvent event) {
        queue.offer(GameEvent.of(Bukkit.getCurrentTick(), "smelt",
                        event.getPlayer().getUniqueId().toString(),
                        Dims.of(event.getBlock().getWorld()),
                        new int[]{event.getBlock().getX(), event.getBlock().getY(),
                                event.getBlock().getZ()})
                .with("item", event.getItemType().getKey().toString())
                .with("count", event.getItemAmount()));
    }

    @EventHandler(priority = EventPriority.MONITOR, ignoreCancelled = true)
    public void onEnchant(EnchantItemEvent event) {
        queue.offer(event(event.getEnchanter(), "enchant")
                .with("item", Dims.item(event.getItem()))
                .with("cost", event.getExpLevelCost()));
    }

    /**
     * Opening a container.
     *
     * <p>{@code pos} is the container's own location, not the player's, so it agrees with the
     * {@code container_put} / {@code container_take} events that follow. Knowing which chest
     * matters more than knowing where someone stood while reaching into it. Falls back to the
     * player for inventories with no block behind them, such as an ender chest.
     */
    @EventHandler(priority = EventPriority.MONITOR)
    public void onOpen(InventoryOpenEvent event) {
        InventoryType type = event.getInventory().getType();
        if (IGNORED_CONTAINERS.contains(type) || !(event.getPlayer() instanceof Player player)) {
            return;
        }
        Location where = event.getInventory().getLocation() != null
                ? event.getInventory().getLocation() : player.getLocation();
        queue.offer(GameEvent.of(Bukkit.getCurrentTick(), "container_open",
                        player.getUniqueId().toString(), Dims.of(where.getWorld()),
                        Dims.pos(where))
                .with("container", Dims.name(type)));
    }

    private GameEvent event(Player player, String type) {
        return GameEvent.of(Bukkit.getCurrentTick(), type, player.getUniqueId().toString(),
                Dims.of(player.getWorld()), Dims.pos(player.getLocation()));
    }
}
