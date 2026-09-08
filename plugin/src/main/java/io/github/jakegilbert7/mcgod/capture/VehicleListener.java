package io.github.jakegilbert7.mcgod.capture;

import org.bukkit.Bukkit;
import org.bukkit.entity.Entity;
import org.bukkit.entity.Player;
import org.bukkit.entity.Vehicle;
import org.bukkit.event.EventHandler;
import org.bukkit.event.EventPriority;
import org.bukkit.event.Listener;
import org.bukkit.event.vehicle.VehicleEnterEvent;
import org.bukkit.event.vehicle.VehicleExitEvent;
import java.util.LinkedHashMap;
import java.util.Map;

/** Records deliberate use of transport as activity, independently of any structure. */
public final class VehicleListener implements Listener {

    private final EventQueue queue;

    public VehicleListener(EventQueue queue) {
        this.queue = queue;
    }

    @EventHandler(priority = EventPriority.MONITOR, ignoreCancelled = true)
    public void onEnter(VehicleEnterEvent event) {
        Entity entered = event.getEntered();
        offer(entered instanceof Player ? "vehicle_enter" : "vehicle_passenger_enter",
                entered, event.getVehicle());
    }

    @EventHandler(priority = EventPriority.MONITOR, ignoreCancelled = true)
    public void onExit(VehicleExitEvent event) {
        Entity exited = event.getExited();
        offer(exited instanceof Player ? "vehicle_exit" : "vehicle_passenger_exit",
                exited, event.getVehicle());
    }

    private void offer(String type, Entity passenger, Vehicle vehicle) {
        String actor = actorFor(passenger, vehicle);
        Map<String, String> passengers = new LinkedHashMap<>();
        for (Entity occupant : vehicle.getPassengers()) {
            passengers.put(occupant.getUniqueId().toString(), Dims.entity(occupant));
        }
        queue.offer(GameEvent.of(Bukkit.getCurrentTick(), type,
                        actor, Dims.of(vehicle.getWorld()), Dims.pos(vehicle.getLocation()))
                .with("vehicle", Dims.name(vehicle.getType()))
                .with("vehicle_id", vehicle.getUniqueId().toString())
                .with("passenger_entity", Dims.entity(passenger))
                .with("passenger_id", passenger.getUniqueId().toString())
                .with("passengers", passengers)
                .with("vehicle_in_water", vehicle.isInWater())
                .with("vehicle_surface", vehicle.getLocation().clone().subtract(0, 0.2, 0)
                        .getBlock().getType().getKey().toString()));
    }

    /**
     * Attribute a mob transition to a player already sharing the vehicle when possible.
     * Otherwise retain it as world evidence; a later player transition can still join it by
     * vehicle_id without pretending an unrelated nearby player caused it.
     */
    private String actorFor(Entity passenger, Vehicle vehicle) {
        if (passenger instanceof Player player) {
            return player.getUniqueId().toString();
        }
        for (Entity occupant : vehicle.getPassengers()) {
            if (occupant instanceof Player player) {
                return player.getUniqueId().toString();
            }
        }
        return "world";
    }
}
