package io.github.jakegilbert7.mcgod.capture;

import java.util.concurrent.ArrayBlockingQueue;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicLong;

/**
 * Hand-off between the main thread and the writer thread.
 *
 * <p>{@link #offer} never blocks and never allocates beyond the queue node: if the queue is
 * full the event is dropped and counted. Dropping events is always preferable to stalling
 * the server tick.
 */
public final class EventQueue {

    private final ArrayBlockingQueue<GameEvent> queue;
    private final AtomicLong enqueued = new AtomicLong();
    private final AtomicLong dropped = new AtomicLong();

    public EventQueue(int capacity) {
        this.queue = new ArrayBlockingQueue<>(capacity);
    }

    /** Main-thread entry point. Non-blocking. */
    public void offer(GameEvent event) {
        if (queue.offer(event)) {
            enqueued.incrementAndGet();
        } else {
            dropped.incrementAndGet();
        }
    }

    GameEvent poll(long timeoutMillis) throws InterruptedException {
        return queue.poll(timeoutMillis, TimeUnit.MILLISECONDS);
    }

    GameEvent pollNow() {
        return queue.poll();
    }

    public int depth() {
        return queue.size();
    }

    public long enqueued() {
        return enqueued.get();
    }

    public long dropped() {
        return dropped.get();
    }
}
