package io.github.jakegilbert7.mcgod.capture;

/**
 * A destination for serialized events. Sinks are driven exclusively by the
 * {@link EventDispatcher} thread, never by the main thread, so implementations may do I/O.
 *
 * <p>A sink must not throw. Transport problems are the sink's business to absorb and count:
 * one failing destination must never stall the others or the drain loop.
 */
public interface EventSink {

    /** Short name, used in log lines and {@code /mcgod stats}. */
    String name();

    /** Consumes one serialized event. Called for every event, in order. */
    void accept(String json);

    /** Periodic hint that buffered data should be pushed out. */
    default void flush() {
    }

    /** Final flush and release. Called once, after the queue has drained. */
    void close();

    /** Events this sink failed to deliver. */
    long failed();
}
