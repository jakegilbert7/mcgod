package io.github.jakegilbert7.mcgod.capture;

import java.net.InetSocketAddress;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicLong;
import java.util.logging.Level;
import java.util.logging.Logger;
import org.java_websocket.WebSocket;
import org.java_websocket.handshake.ClientHandshake;
import org.java_websocket.server.WebSocketServer;

/**
 * The L0 bridge: a WebSocket server that pushes serialized events to attached consumers.
 *
 * <p>The event stream is pure and one-way. Every event frame is exactly one event,
 * byte-identical to the corresponding line in the session JSONL. That equivalence is what lets
 * the P3 replay harness feed the same consumer interface as the live bridge, so nothing may
 * ever be added to it that is not an event.
 *
 * <p>Region scans therefore travel on a separate logical channel over the same socket. A
 * request carries an {@code "rpc"} key and the reply goes only to the connection that asked;
 * neither is broadcast and neither reaches the JSONL. Consumers route any frame with an
 * {@code "rpc"} key away from their event pipeline.
 *
 * <p>Java-WebSocket runs its own threads; nothing here touches the Bukkit API or the main
 * thread. A consumer that cannot keep up is disconnected rather than allowed to grow an
 * unbounded outbound buffer — the same drop-don't-stall rule the capture queue follows.
 */
public final class EventBridgeServer extends WebSocketServer {

    /** Consecutive sends with data still buffered before a consumer is considered stuck. */
    private static final int LAG_LIMIT = 500;

    private final Logger log;
    private final Map<WebSocket, Integer> lag = new ConcurrentHashMap<>();
    private final AtomicLong sent = new AtomicLong();
    private final AtomicLong failed = new AtomicLong();
    private volatile boolean up;
    private volatile java.util.function.BiConsumer<WebSocket, String> handler;

    public EventBridgeServer(InetSocketAddress address, Logger log) {
        super(address);
        this.log = log;
        setReuseAddr(true);
    }

    @Override
    public void onStart() {
        up = true;
        log.info("Bridge listening on ws://" + getAddress().getHostString() + ":" + getPort());
    }

    @Override
    public void onOpen(WebSocket conn, ClientHandshake handshake) {
        lag.put(conn, 0);
        log.info("Bridge consumer connected: " + conn.getRemoteSocketAddress());
    }

    @Override
    public void onClose(WebSocket conn, int code, String reason, boolean remote) {
        lag.remove(conn);
        log.info("Bridge consumer disconnected: " + conn.getRemoteSocketAddress()
                + " (code " + code + (reason == null || reason.isEmpty() ? "" : ", " + reason) + ")");
    }

    /** Inbound control frames. Never events, never broadcast, never written to the session. */
    @Override
    public void onMessage(WebSocket conn, String message) {
        if (handler != null) {
            handler.accept(conn, message);
        }
    }

    /** Installed by the plugin so this class stays free of any world knowledge. */
    public void onCommand(java.util.function.BiConsumer<WebSocket, String> handler) {
        this.handler = handler;
    }

    @Override
    public void onError(WebSocket conn, Exception e) {
        if (conn == null) {
            // Server-level failure, most often the port already being in use.
            up = false;
            log.log(Level.SEVERE, "Bridge server error; capture to file is unaffected", e);
        } else {
            failed.incrementAndGet();
            lag.remove(conn);
        }
    }

    /** Fans one serialized event out to every attached consumer. Never throws. */
    public void publish(String json) {
        for (WebSocket conn : getConnections()) {
            if (!conn.isOpen()) {
                continue;
            }
            try {
                conn.send(json);
                sent.incrementAndGet();
                lag.put(conn, conn.hasBufferedData() ? lag.getOrDefault(conn, 0) + 1 : 0);
                if (lag.getOrDefault(conn, 0) > LAG_LIMIT) {
                    log.warning("Bridge consumer " + conn.getRemoteSocketAddress()
                            + " is not keeping up; disconnecting");
                    lag.remove(conn);
                    conn.close();
                }
            } catch (RuntimeException e) {
                failed.incrementAndGet();
            }
        }
    }

    public boolean isUp() {
        return up;
    }

    public int consumers() {
        return getConnections().size();
    }

    public long sent() {
        return sent.get();
    }

    public long failed() {
        return failed.get();
    }
}
