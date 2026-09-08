package io.github.jakegilbert7.mcgod.capture;

import java.util.List;
import java.util.concurrent.atomic.AtomicLong;
import java.util.logging.Level;
import java.util.logging.Logger;

/**
 * Drains {@link EventQueue} on a single daemon thread, serializes each event once, and hands
 * the resulting line to every {@link EventSink}.
 *
 * <p>Serializing once and fanning out is the whole point: the JSONL file and the WebSocket
 * bridge must see byte-identical lines, or the replay harness (P3) stops being a faithful
 * stand-in for the live bridge.
 *
 * <p>Nothing here touches the Bukkit API, so it is safe off the main thread.
 */
public final class EventDispatcher {

    private static final long POLL_MILLIS = 200;

    private final EventQueue queue;
    private final List<EventSink> sinks;
    private final long flushIntervalMillis;
    private final Logger log;

    private final AtomicLong dispatched = new AtomicLong();
    private volatile boolean running;
    private Thread thread;

    public EventDispatcher(EventQueue queue, List<EventSink> sinks, long flushIntervalMillis, Logger log) {
        this.queue = queue;
        this.sinks = List.copyOf(sinks);
        this.flushIntervalMillis = flushIntervalMillis;
        this.log = log;
    }

    public void start() {
        running = true;
        thread = new Thread(this::run, "mcgod-dispatcher");
        thread.setDaemon(true);
        thread.start();
    }

    /** Stops accepting new work, drains what is queued, then closes every sink. */
    public void stop() {
        if (thread == null) {
            return;
        }
        running = false;
        try {
            thread.join(5000);
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
        }
        for (EventSink sink : sinks) {
            try {
                sink.close();
            } catch (RuntimeException e) {
                log.log(Level.WARNING, "Sink " + sink.name() + " failed to close", e);
            }
        }
        log.info("Dispatched " + dispatched.get() + " events");
    }

    private void run() {
        StringBuilder buffer = new StringBuilder(256);
        long lastFlush = System.currentTimeMillis();
        try {
            while (running || queue.depth() > 0) {
                GameEvent event = running ? queue.poll(POLL_MILLIS) : queue.pollNow();
                if (event != null) {
                    buffer.setLength(0);
                    event.writeJson(buffer);
                    String line = buffer.toString();
                    for (EventSink sink : sinks) {
                        try {
                            sink.accept(line);
                        } catch (RuntimeException e) {
                            log.log(Level.WARNING, "Sink " + sink.name() + " threw; continuing", e);
                        }
                    }
                    dispatched.incrementAndGet();
                }
                long now = System.currentTimeMillis();
                if (now - lastFlush >= flushIntervalMillis) {
                    lastFlush = now;
                    sinks.forEach(EventSink::flush);
                }
            }
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
        }
        sinks.forEach(EventSink::flush);
    }

    public long dispatched() {
        return dispatched.get();
    }

    public List<EventSink> sinks() {
        return sinks;
    }
}
