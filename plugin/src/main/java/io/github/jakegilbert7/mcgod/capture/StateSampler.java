package io.github.jakegilbert7.mcgod.capture;

import org.bukkit.Bukkit;
import org.bukkit.Statistic;
import org.bukkit.World;
import org.bukkit.block.Block;
import org.bukkit.entity.Entity;
import org.bukkit.entity.Player;
import org.bukkit.inventory.PlayerInventory;
import org.bukkit.scheduler.BukkitRunnable;
import java.util.LinkedHashMap;
import java.util.Map;

/**
 * Periodic snapshot of a player's situation: the world context that is state rather than event.
 *
 * <p>Health, hunger, XP, biome, light, time of day and weather are never "emitted" by the game
 * — nothing fires when it gets dark or when you walk into a swamp. Without sampling them the
 * god can see that a player died to a skeleton but not that it happened at night, underground,
 * on two hearts, in the rain.
 *
 * <p>Kept separate from {@code move} deliberately. Movement is the highest-volume event we
 * record and stays lean; this is an order of magnitude rarer and carries the heavy context.
 */
public final class StateSampler extends BukkitRunnable {

    private final EventQueue queue;

    public StateSampler(EventQueue queue) {
        this.queue = queue;
    }

    @Override
    public void run() {
        for (Player player : Bukkit.getOnlinePlayers()) {
            sample(player);
        }
    }

    public void sample(Player player) {
        World world = player.getWorld();
        Block block = player.getLocation().getBlock();
        PlayerInventory inventory = player.getInventory();
        Map<String, Integer> travel = new LinkedHashMap<>();
        travel.put("minecart", player.getStatistic(Statistic.MINECART_ONE_CM));
        travel.put("boat", player.getStatistic(Statistic.BOAT_ONE_CM));
        travel.put("horse", player.getStatistic(Statistic.HORSE_ONE_CM));
        travel.put("pig", player.getStatistic(Statistic.PIG_ONE_CM));
        travel.put("strider", player.getStatistic(Statistic.STRIDER_ONE_CM));

        Entity vehicle = player.getVehicle();
        Map<String, String> passengers = new LinkedHashMap<>();
        if (vehicle != null) {
            for (Entity occupant : vehicle.getPassengers()) {
                passengers.put(occupant.getUniqueId().toString(), Dims.entity(occupant));
            }
        }

        GameEvent state = GameEvent.of(Bukkit.getCurrentTick(), "player_state",
                        player.getUniqueId().toString(), Dims.of(world),
                        Dims.pos(player.getLocation()))
                .with("health", player.getHealth())
                .with("food", player.getFoodLevel())
                .with("saturation", (double) player.getSaturation())
                .with("level", player.getLevel())
                .with("total_xp", player.getTotalExperience())
                .with("biome", Dims.biome(world, player.getLocation()))
                .with("light", (int) block.getLightLevel())
                .with("sky_light", (int) block.getLightFromSky())
                // 0-24000; 13000-23000 is night. Kept raw so downstream can bucket it.
                .with("time", world.getTime())
                .with("weather", world.isThundering() ? "thunder"
                        : world.hasStorm() ? "rain" : "clear")
                .with("gamemode", Dims.name(player.getGameMode()))
                .with("held", Dims.item(inventory.getItemInMainHand()))
                .with("helmet", Dims.item(inventory.getHelmet()))
                .with("chestplate", Dims.item(inventory.getChestplate()))
                .with("leggings", Dims.item(inventory.getLeggings()))
                .with("boots", Dims.item(inventory.getBoots()))
                // Vanilla keeps these counters in the player save. Sampling them recovers
                // journeys that predate this plugin version as well as rides whose enter or
                // exit event was missed during a restart.
                .with("travel_cm", travel)
                .with("on_ground", player.isOnGround())
                .with("sneaking", player.isSneaking());
        if (vehicle != null) {
            state.with("vehicle", Dims.name(vehicle.getType()))
                    .with("vehicle_id", vehicle.getUniqueId().toString())
                    .with("vehicle_passengers", passengers)
                    .with("vehicle_in_water", vehicle.isInWater())
                    .with("vehicle_surface", vehicle.getLocation().clone().subtract(0, 0.2, 0)
                            .getBlock().getType().getKey().toString());
        }
        queue.offer(state);
    }
}
