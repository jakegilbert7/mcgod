package io.github.jakegilbert7.mcgod.capture;

import java.net.InetSocketAddress;
import java.util.logging.Level;
import java.util.logging.Logger;

/** Adapts {@link EventBridgeServer} to the sink interface the dispatcher drives. */
public final class WebSocketSink implements EventSink {

    private final EventBridgeServer server;
    private final Logger log;

    public WebSocketSink(String host, int port, Logger log) {
        this.log = log;
        this.server = new EventBridgeServer(new InetSocketAddress(host, port), log);
        this.server.start();
    }

    @Override
    public String name() {
        return "websocket";
    }

    @Override
    public void accept(String json) {
        server.publish(json);
    }

    @Override
    public void close() {
        try {
            server.stop(1000);
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
        } catch (RuntimeException e) {
            log.log(Level.WARNING, "Bridge did not stop cleanly", e);
        }
    }

    @Override
    public long failed() {
        return server.failed();
    }

    public EventBridgeServer server() {
        return server;
    }

    /** Routes inbound control frames; see {@link EventBridgeServer#onCommand}. */
    public void onCommand(java.util.function.BiConsumer<org.java_websocket.WebSocket, String> h) {
        server.onCommand(h);
    }
}
