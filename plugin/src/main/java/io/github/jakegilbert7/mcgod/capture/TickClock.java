package io.github.jakegilbert7.mcgod.capture;

import org.bukkit.Bukkit;

/**
 * The current tick, readable from any thread.
 *
 * <p>{@code Bukkit.getCurrentTick()} reads server state that is only safely visible from the
 * main thread. Chat arrives asynchronously, so it needs a value published across threads: this
 * is refreshed by a main-thread task every tick and read through a volatile field.
 */
public final class TickClock implements Runnable {

    private volatile long tick;

    public TickClock() {
        this.tick = Bukkit.getCurrentTick();
    }

    @Override
    public void run() {
        tick = Bukkit.getCurrentTick();
    }

    public long tick() {
        return tick;
    }
}
