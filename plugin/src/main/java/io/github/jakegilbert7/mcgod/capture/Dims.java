package io.github.jakegilbert7.mcgod.capture;

import org.bukkit.Location;
import org.bukkit.World;
import org.bukkit.entity.Entity;
import org.bukkit.entity.Player;
import org.bukkit.inventory.ItemStack;

/** Small main-thread helpers that turn Bukkit objects into schema values. */
public final class Dims {

    private Dims() {
    }

    /** Vanilla dimension key, or the world name for a custom world. */
    public static String of(World world) {
        return switch (world.getEnvironment()) {
            case NORMAL -> "overworld";
            case NETHER -> "the_nether";
            case THE_END -> "the_end";
            default -> world.getName();
        };
    }

    public static int[] pos(Location location) {
        return new int[]{location.getBlockX(), location.getBlockY(), location.getBlockZ()};
    }

    /** Namespaced key of the item in the main hand, or null if the hand is empty. */
    public static String heldTool(Player player) {
        return item(player.getInventory().getItemInMainHand());
    }

    /** Namespaced key of a stack's material, or null for null/air. */
    public static String item(ItemStack stack) {
        return stack == null || stack.getType().isAir() ? null : stack.getType().getKey().toString();
    }

    /** Namespaced key of an entity's type, e.g. minecraft:sheep. */
    public static String entity(Entity entity) {
        return entity == null ? null : entity.getType().getKey().toString();
    }

    /** Namespaced biome key at a location, e.g. minecraft:sunflower_plains. */
    public static String biome(World world, Location location) {
        try {
            return world.getBiome(location.getBlockX(), location.getBlockY(),
                    location.getBlockZ()).getKey().toString();
        } catch (RuntimeException e) {
            // Custom/datapack biomes may not expose a key; context is optional, never fatal.
            return null;
        }
    }

    /** Lowercased enum name, for causes and states that have no namespaced key. */
    public static String name(Enum<?> value) {
        return value == null ? null : value.name().toLowerCase(java.util.Locale.ROOT);
    }
}
