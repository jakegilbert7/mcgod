package io.github.jakegilbert7.mcgod.capture;

import org.bukkit.Bukkit;
import org.bukkit.event.EventHandler;
import org.bukkit.event.EventPriority;
import org.bukkit.event.Listener;
import org.bukkit.event.weather.ThunderChangeEvent;
import org.bukkit.event.weather.WeatherChangeEvent;

/**
 * World-level context with no actor. These carry {@code actor: "world"} rather than a UUID,
 * because something genuinely happened that no player caused and the god should know about it.
 */
public final class WorldListener implements Listener {

    private final EventQueue queue;

    public WorldListener(EventQueue queue) {
        this.queue = queue;
    }

    @EventHandler(priority = EventPriority.MONITOR, ignoreCancelled = true)
    public void onWeather(WeatherChangeEvent event) {
        queue.offer(GameEvent.of(Bukkit.getCurrentTick(), "weather_change", "world",
                        Dims.of(event.getWorld()), null)
                .with("weather", event.toWeatherState() ? "rain" : "clear"));
    }

    @EventHandler(priority = EventPriority.MONITOR, ignoreCancelled = true)
    public void onThunder(ThunderChangeEvent event) {
        queue.offer(GameEvent.of(Bukkit.getCurrentTick(), "thunder_change", "world",
                        Dims.of(event.getWorld()), null)
                .with("weather", event.toThunderState() ? "thunder" : "clear"));
    }
}
