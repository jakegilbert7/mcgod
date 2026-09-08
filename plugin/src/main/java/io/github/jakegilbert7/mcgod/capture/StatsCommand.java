package io.github.jakegilbert7.mcgod.capture;

import java.util.List;
import net.kyori.adventure.text.Component;
import net.kyori.adventure.text.format.NamedTextColor;
import org.bukkit.Bukkit;
import org.bukkit.command.Command;
import org.bukkit.command.CommandExecutor;
import org.bukkit.command.CommandSender;
import org.bukkit.command.TabCompleter;
import org.jetbrains.annotations.NotNull;

/**
 * {@code /mcgod stats} — the readout the P1 and P2 acceptance tests are checked against: how
 * many events were seen, how many reached disk and the bridge, how many were lost, and the
 * tick rate.
 */
public final class StatsCommand implements CommandExecutor, TabCompleter {

    private final EventQueue queue;
    private final EventDispatcher dispatcher;
    private final JsonlSink jsonl;
    private final WebSocketSink bridge;
    private final CommandRunner runner;
    private final PowerService powers;

    public StatsCommand(EventQueue queue, EventDispatcher dispatcher, JsonlSink jsonl,
                        WebSocketSink bridge, CommandRunner runner,
                        PowerService powers) {
        this.queue = queue;
        this.dispatcher = dispatcher;
        this.jsonl = jsonl;
        this.bridge = bridge;
        this.runner = runner;
        this.powers = powers;
    }

    @Override
    public boolean onCommand(@NotNull CommandSender sender, @NotNull Command command,
                             @NotNull String label, String @NotNull [] args) {
        // The switch. A god that can do anything needs one that does not depend on the
        // agent being reachable, or on anyone knowing which effect to name.
        if (args.length == 1 && args[0].equalsIgnoreCase("stop")) {
            int stopped = runner.stopAll() + powers.revokeAll();
            sender.sendMessage(Component.text(
                    stopped == 0 ? "Nothing of the god's was running."
                                 : "Stopped " + stopped + " of the god's doings.",
                    NamedTextColor.GOLD));
            return true;
        }
        if (args.length != 1 || !args[0].equalsIgnoreCase("stats")) {
            sender.sendMessage(Component.text("Usage: /" + label + " stats|stop",
                    NamedTextColor.RED));
            return true;
        }

        sender.sendMessage(Component.text("=== McGod capture ===", NamedTextColor.GOLD));
        line(sender, "file", jsonl.file().getFileName().toString());
        line(sender, "enqueued", Long.toString(queue.enqueued()));
        line(sender, "dispatched", Long.toString(dispatcher.dispatched()));
        line(sender, "written", Long.toString(jsonl.written()));
        line(sender, "dropped", Long.toString(queue.dropped()));
        line(sender, "commands queued", Integer.toString(runner.queued()));
        line(sender, "effects running", String.join(", ", runner.active()));
        line(sender, "queue depth", Integer.toString(queue.depth()));
        line(sender, "write errors", Long.toString(jsonl.failed()));

        sender.sendMessage(Component.text("=== bridge ===", NamedTextColor.GOLD));
        if (bridge == null) {
            line(sender, "state", "disabled");
        } else {
            EventBridgeServer server = bridge.server();
            line(sender, "state", server.isUp() ? "listening on port " + server.getPort() : "down");
            line(sender, "consumers", Integer.toString(server.consumers()));
            line(sender, "frames sent", Long.toString(server.sent()));
            line(sender, "send failures", Long.toString(server.failed()));
        }

        line(sender, "tps (1m)", String.format("%.2f", Bukkit.getTPS()[0]));
        return true;
    }

    private void line(CommandSender sender, String key, String value) {
        sender.sendMessage(Component.text(key + ": ", NamedTextColor.GRAY)
                .append(Component.text(value, NamedTextColor.WHITE)));
    }

    @Override
    public List<String> onTabComplete(@NotNull CommandSender sender, @NotNull Command command,
                                      @NotNull String label, String @NotNull [] args) {
        return args.length == 1 ? List.of("stats", "stop") : List.of();
    }
}
