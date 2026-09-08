package io.github.jakegilbert7.mcgod;

import io.github.jakegilbert7.mcgod.capture.BlockListener;
import io.github.jakegilbert7.mcgod.capture.EventDispatcher;
import io.github.jakegilbert7.mcgod.capture.CommandRunner;
import io.github.jakegilbert7.mcgod.capture.PowerService;
import io.github.jakegilbert7.mcgod.capture.EventQueue;
import io.github.jakegilbert7.mcgod.capture.ChatListener;
import io.github.jakegilbert7.mcgod.capture.ContainerTracker;
import io.github.jakegilbert7.mcgod.capture.EntityListener;
import io.github.jakegilbert7.mcgod.capture.EventSink;
import io.github.jakegilbert7.mcgod.capture.InventoryListener;
import io.github.jakegilbert7.mcgod.capture.JsonlSink;
import io.github.jakegilbert7.mcgod.capture.PickupRollup;
import io.github.jakegilbert7.mcgod.capture.PlayerListener;
import io.github.jakegilbert7.mcgod.capture.PositionSampler;
import io.github.jakegilbert7.mcgod.capture.ScanCommandHandler;
import io.github.jakegilbert7.mcgod.capture.ScanService;
import io.github.jakegilbert7.mcgod.capture.StateSampler;
import io.github.jakegilbert7.mcgod.capture.StatsCommand;
import io.github.jakegilbert7.mcgod.capture.TickClock;
import io.github.jakegilbert7.mcgod.capture.TradeListener;
import io.github.jakegilbert7.mcgod.capture.VehicleListener;
import io.github.jakegilbert7.mcgod.capture.WorldListener;
import io.github.jakegilbert7.mcgod.capture.WebSocketSink;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.logging.Level;
import org.bukkit.configuration.file.FileConfiguration;
import org.bukkit.plugin.java.JavaPlugin;
import org.bukkit.scheduler.BukkitTask;

/**
 * L0 capture and bridge: listeners feed a bounded queue, a dispatcher thread serializes each
 * event once and fans it out to the session file and the WebSocket bridge. Nothing in the
 * event path does I/O or blocks the tick.
 */
public final class McGodPlugin extends JavaPlugin {

    private EventQueue queue;
    private CommandRunner runner;
    private PowerService powers;
    private EventDispatcher dispatcher;
    private JsonlSink jsonl;
    private WebSocketSink bridge;
    private PickupRollup pickups;
    private PositionSampler sampler;
    private StateSampler state;
    private BukkitTask rollupTask;

    @Override
    public void onEnable() {
        getLogger().info("=== MCGOD ONLINE ===");

        saveDefaultConfig();
        FileConfiguration config = getConfig();
        int capacity = config.getInt("capture.queue-capacity", 65536);
        long flushMillis = config.getLong("capture.flush-interval-ms", 1000);
        long sampleTicks = config.getLong("capture.position-sample-ticks", 40);
        long rollupTicks = config.getLong("capture.pickup-rollup-ticks", 1200);
        int regionSize = config.getInt("capture.region-size", 64);
        int hysteresis = config.getInt("capture.region-hysteresis", 16);
        long stateTicks = config.getLong("capture.state-sample-ticks", 200);

        queue = new EventQueue(capacity);

        List<EventSink> sinks = new ArrayList<>();
        try {
            jsonl = new JsonlSink(getDataFolder().toPath().resolve("events"), getLogger());
            sinks.add(jsonl);
        } catch (IOException e) {
            getLogger().log(Level.SEVERE, "Could not open capture file; disabling McGod", e);
            getServer().getPluginManager().disablePlugin(this);
            return;
        }

        // The bridge is best-effort. A port conflict must not cost us the session file.
        if (config.getBoolean("bridge.enabled", true)) {
            try {
                bridge = new WebSocketSink(
                        config.getString("bridge.host", "127.0.0.1"),
                        config.getInt("bridge.port", 8765),
                        getLogger());
                sinks.add(bridge);
            } catch (RuntimeException e) {
                getLogger().log(Level.SEVERE, "Bridge failed to start; capturing to file only", e);
                bridge = null;
            }
        }

        dispatcher = new EventDispatcher(queue, sinks, flushMillis, getLogger());
        runner = new CommandRunner(this, queue, getLogger());
        runner.start();
        powers = new PowerService(this, queue, runner, getLogger());
        getServer().getPluginManager().registerEvents(powers, this);
        dispatcher.start();

        // Region scans share the socket but not the event stream: replies carry an "rpc" key
        // and go only to the caller, so the JSONL stays a pure event log.
        if (bridge != null) {
            ScanService scans = new ScanService(this, config.getInt("scan.chunks-per-tick", 4));
            ScanCommandHandler commands = new ScanCommandHandler(this, scans, queue, runner, powers, getLogger());
            bridge.onCommand(commands::handle);
        }

        pickups = new PickupRollup(queue);
        sampler = new PositionSampler(queue, regionSize, hysteresis);
        state = new StateSampler(queue);

        // Refreshed every tick so asynchronous events (chat) can stamp a tick safely.
        TickClock clock = new TickClock();
        getServer().getScheduler().runTaskTimer(this, clock, 1L, 1L);

        getServer().getPluginManager().registerEvents(new BlockListener(queue), this);
        getServer().getPluginManager().registerEvents(new PlayerListener(queue, sampler), this);
        getServer().getPluginManager().registerEvents(new InventoryListener(queue, pickups), this);
        getServer().getPluginManager().registerEvents(new EntityListener(queue), this);
        getServer().getPluginManager().registerEvents(new TradeListener(queue), this);
        getServer().getPluginManager().registerEvents(new VehicleListener(queue), this);
        getServer().getPluginManager().registerEvents(new WorldListener(queue), this);
        getServer().getPluginManager().registerEvents(new ContainerTracker(queue), this);
        getServer().getPluginManager().registerEvents(new ChatListener(queue, clock, this), this);

        sampler.runTaskTimer(this, sampleTicks, sampleTicks);
        state.runTaskTimer(this, stateTicks, stateTicks);
        rollupTask = getServer().getScheduler().runTaskTimer(this, pickups::flush, rollupTicks, rollupTicks);

        getCommand("mcgod").setExecutor(new StatsCommand(queue, dispatcher, jsonl, bridge, runner, powers));
    }

    @Override
    public void onDisable() {
        if (sampler != null) {
            sampler.cancel();
        }
        if (state != null) {
            state.cancel();
        }
        if (rollupTask != null) {
            rollupTask.cancel();
        }
        if (pickups != null) {
            pickups.flush();
        }
        // Repeating effects must not outlive the plugin: a spell whose owner is gone would
        // keep running with nothing able to stop it.
        if (runner != null) {
            runner.shutdown();
        }
        if (powers != null) {
            powers.revokeAll();
        }
        if (dispatcher != null) {
            dispatcher.stop();
        }
        getLogger().info("=== MCGOD OFFLINE ===");
    }
}
