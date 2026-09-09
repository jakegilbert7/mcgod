package io.github.jakegilbert7.mcgod.capture;

import com.google.gson.JsonArray;
import com.google.gson.JsonObject;
import com.google.gson.JsonParser;
import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Deque;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.function.Consumer;
import java.util.logging.Level;
import java.util.logging.Logger;
import net.kyori.adventure.text.Component;
import net.kyori.adventure.text.serializer.plain.PlainTextComponentSerializer;
import net.kyori.adventure.text.format.NamedTextColor;
import org.bukkit.Bukkit;
import org.bukkit.World;
import org.bukkit.command.CommandSender;
import org.bukkit.entity.Player;
import org.bukkit.plugin.Plugin;
import org.java_websocket.WebSocket;

/**
 * Parses inbound control frames and dispatches them.
 *
 * <p>Runs on a Java-WebSocket thread, so it touches no Bukkit state directly: work is hopped
 * onto the main thread, and replies come back asynchronously to the one connection that asked.
 *
 * <p>Actions are whitelisted here and budgeted here. `speak` is the only one implemented, and
 * deliberately: it is the lowest-severity action there is, changing nothing in the world. The
 * budget is enforced server-side rather than in the agent because the agent is the thing most
 * likely to be wrong, and a loop that spams chat is the failure a player actually notices.
 *
 * <p>Commands are the second action, and the same principle decides where their gate lives:
 * {@link CommandService} holds the allowlist, the limits and the budget, here in the server,
 * because the agent drafting the command is a language model. The agent may ask for anything;
 * what is permitted is not its decision.
 */
public final class ScanCommandHandler {

    /**
     * Whitelisted actions, with the cost each one charges against the hourly budget.
     *
     * <p>Answering a direct question is budgeted separately and far more generously than
     * speaking unbidden. The thing worth rationing is a god that interrupts; a god that
     * will not answer when spoken to is just broken. Both are still capped, because the
     * agent is the component most likely to be wrong and a loop that floods chat is the
     * failure a player actually notices.
     */
    private static final Map<String, Integer> ACTIONS = Map.of("speak", 1);
    private static final int BUDGET_PER_HOUR = 20;
    private static final int REPLY_BUDGET_PER_HOUR = 120;
    private static final long COOLDOWN_MS = 3000;
    private static final long REPLY_COOLDOWN_MS = 400;
    private static final int MAX_SPEECH = 900;

    private final Plugin plugin;
    private final ScanService scans;
    private final EnvironmentService environments;
    private final Logger log;
    private static final PlainTextComponentSerializer PLAIN =
            PlainTextComponentSerializer.plainText();
    private final CommandRunner runner;
    private final PowerService powers;
    private final EventQueue queue;
    private final Deque<Long> spent = new ArrayDeque<>();
    private final Deque<Long> replies = new ArrayDeque<>();
    private long lastAction;
    private long lastReply;

    public ScanCommandHandler(Plugin plugin, ScanService scans, EventQueue queue,
                              CommandRunner runner, PowerService powers, Logger log) {
        this.plugin = plugin;
        this.scans = scans;
        this.queue = queue;
        this.runner = runner;
        this.powers = powers;
        this.environments = new EnvironmentService();
        this.log = log;
    }

    public void handle(WebSocket conn, String message) {
        String id = "?";
        try {
            JsonObject request = JsonParser.parseString(message).getAsJsonObject();
            id = request.has("id") ? request.get("id").getAsString() : "?";
            String rpc = request.has("rpc") ? request.get("rpc").getAsString() : "";
            if ("speak".equals(rpc)) {
                speak(conn, id, request);
                return;
            }
            if ("run_command".equals(rpc)) {
                runCommand(conn, id, request);
                return;
            }
            if ("stop_commands".equals(rpc)) {
                stopCommands(conn, id, request);
                return;
            }
            if ("grant_power".equals(rpc) || "revoke_power".equals(rpc)) {
                power(conn, id, request, "grant_power".equals(rpc));
                return;
            }
            if ("held_powers".equals(rpc)) {
                heldPowers(conn, id, request);
                return;
            }
            if ("environment".equals(rpc)) {
                environment(conn, id, request);
                return;
            }
            boolean wantVoxels = "voxels".equals(rpc);
            if (!wantVoxels && !"scan_region".equals(rpc)) {
                send(conn, id, "unknown rpc: " + rpc);
                return;
            }
            String dim = request.get("dim").getAsString();
            int[] min = triple(request.getAsJsonArray("min"));
            int[] max = triple(request.getAsJsonArray("max"));

            final String replyId = id;
            Bukkit.getScheduler().runTask(plugin, () -> {
                World world = ScanService.world(dim);
                if (world == null) {
                    send(conn, replyId, "no such dimension: " + dim);
                    return;
                }
                Consumer<String> reply = payload -> {
                    if (conn.isOpen()) {
                        conn.send(payload);
                    }
                };
                if (wantVoxels) {
                    scans.voxels(replyId, world, min, max, reply);
                } else {
                    scans.scan(replyId, world, min, max, reply);
                }
            });
        } catch (RuntimeException e) {
            log.log(Level.WARNING, "Bad scan request", e);
            send(conn, id, "malformed request: " + e.getClass().getSimpleName());
        }
    }

    private void environment(WebSocket conn, String id, JsonObject request) {
        String dim = request.get("dim").getAsString();
        int x = request.get("x").getAsInt();
        int z = request.get("z").getAsInt();
        int radius = request.has("radius") ? request.get("radius").getAsInt() : 256;
        int step = request.has("step") ? request.get("step").getAsInt() : 16;
        java.util.List<String> features = new java.util.ArrayList<>();
        if (request.has("features")) {
            for (var value : request.getAsJsonArray("features")) {
                features.add(value.getAsString());
            }
        }
        final String replyId = id;
        Bukkit.getScheduler().runTask(plugin, () -> {
            World world = ScanService.world(dim);
            if (world == null) {
                send(conn, replyId, "environment_result", "no such dimension: " + dim);
                return;
            }
            String payload = environments.query(replyId, world, x, z, radius, step, features);
            if (conn.isOpen()) {
                conn.send(payload);
            }
        });
    }

    /** Says something in chat, if the budget allows it. */
    private void speak(WebSocket conn, String id, JsonObject request) {
        String text = request.has("text") ? request.get("text").getAsString() : "";
        if (text.isBlank()) {
            send(conn, id, "speak_result", "speak requires text");
            return;
        }
        if (text.length() > MAX_SPEECH) {
            text = text.substring(0, MAX_SPEECH) + "…";
        }
        boolean reply = request.has("reply") && request.get("reply").getAsBoolean();
        int remaining;
        synchronized (spent) {
            long now = System.currentTimeMillis();
            Deque<Long> ledger = reply ? replies : spent;
            int budget = reply ? REPLY_BUDGET_PER_HOUR : BUDGET_PER_HOUR;
            long cooldown = reply ? REPLY_COOLDOWN_MS : COOLDOWN_MS;
            long last = reply ? lastReply : lastAction;
            ledger.removeIf(t -> now - t > 3_600_000L);
            if (ledger.size() >= budget) {
                send(conn, id, "speak_result",
                        (reply ? "reply" : "speech") + " budget spent for this hour");
                return;
            }
            if (now - last < cooldown) {
                send(conn, id, "speak_result",
                        "on cooldown, " + (cooldown - (now - last)) + "ms left");
                return;
            }
            ledger.add(now);
            if (reply) {
                lastReply = now;
            } else {
                lastAction = now;
            }
            remaining = budget - ledger.size();
        }

        final String message = text;
        final String target = request.has("target") ? request.get("target").getAsString() : null;
        Bukkit.getScheduler().runTask(plugin, () -> {
            Component line = Component.text("\u271e ", NamedTextColor.GOLD)
                    .append(Component.text(message, NamedTextColor.WHITE));
            if (target == null) {
                Bukkit.getServer().sendMessage(line);
            } else {
                Player player = Bukkit.getPlayerExact(target);
                if (player != null) {
                    player.sendMessage(line);
                }
            }
            log.info("god: " + message);
        });
        StringBuilder out = new StringBuilder("{\"rpc\":\"speak_result\",\"ok\":true,\"id\":");
        Json.string(out, id);
        out.append(",\"remaining\":").append(remaining).append('}');
        if (conn.isOpen()) {
            conn.send(out.toString());
        }
    }

    /**
     * Runs what the god asked for: one command, a thousand, or a repeating effect.
     *
     * <p>Nothing here caps how much is asked for. Every command is checked against
     * {@link CommandService}, and whatever passes is handed to {@link CommandRunner},
     * which paces it so the tick survives. The reply says what was accepted rather than
     * what it did, because a bridge takes longer to build than a sentence should wait; each
     * command reports itself through the event stream as it runs.
     */
    private void runCommand(WebSocket conn, String id, JsonObject request) {
        List<String> drafted = new ArrayList<>();
        if (request.has("commands") && request.get("commands").isJsonArray()) {
            for (var value : request.getAsJsonArray("commands")) {
                drafted.add(value.getAsString());
            }
        } else if (request.has("command")) {
            drafted.add(request.get("command").getAsString());
        }
        String anchor = request.has("anchor") && !request.get("anchor").isJsonNull()
                ? request.get("anchor").getAsString() : null;
        boolean anchored = anchor != null && !anchor.isBlank();

        List<String> accepted = new ArrayList<>();
        List<String> refused = new ArrayList<>();
        for (String raw : drafted) {
            CommandService.Verdict verdict = CommandService.inspect(raw, anchored);
            if (verdict.allowed()) {
                accepted.add(verdict.command());
            } else {
                refused.add(verdict.refusal());
            }
        }
        if (accepted.isEmpty() && !refused.isEmpty()) {
            send(conn, id, "command_result", refused.get(0));
            return;
        }
        if (accepted.isEmpty()) {
            send(conn, id, "command_result", "no command given");
            return;
        }

        int every = request.has("every") ? request.get("every").getAsInt() : 0;
        long duration = request.has("duration") ? request.get("duration").getAsLong() : 0;
        String spell = null;
        if (every > 0 && duration > 0) {
            spell = request.has("spell") ? request.get("spell").getAsString()
                    : "spell-" + System.currentTimeMillis();
            runner.cast(spell, accepted, anchor, every, duration);
            log.info("god casts " + spell + ": " + accepted.size() + " command(s) every "
                    + every + " ticks for " + duration + " ticks");
            // A spell runs for minutes; nobody waits for it to finish before replying.
            reply(conn, id, accepted.size(), refused, spell, List.of());
            return;
        }
        // A one-shot batch answers when it has actually run. The runner drains two hundred
        // a tick, so even a large build finishes in under a second, and a god that says "I
        // am giving you a sword" for a command that then fails on guessed syntax is worse
        // than one that waits.
        log.info("god runs " + accepted.size() + " command(s)"
                + (anchored ? " at " + anchor : ""));
        runner.submit(accepted, anchor,
                outcomes -> reply(conn, id, accepted.size(), refused, null, outcomes));
    }

    /** One reply, carrying whatever is known: acceptance for a spell, outcomes for a batch. */
    private void reply(WebSocket conn, String id, int accepted, List<String> refused,
                       String spell, List<CommandRunner.Outcome> outcomes) {
        StringBuilder out = new StringBuilder("{\"rpc\":\"command_result\",\"ok\":true,\"id\":");
        Json.string(out, id);
        out.append(",\"accepted\":").append(accepted);
        if (spell != null) {
            out.append(",\"spell\":");
            Json.string(out, spell);
        }
        out.append(",\"refused\":[");
        for (int i = 0; i < refused.size(); i++) {
            if (i > 0) {
                out.append(',');
            }
            Json.string(out, refused.get(i));
        }
        out.append("],\"ran\":[");
        for (int i = 0; i < outcomes.size(); i++) {
            CommandRunner.Outcome outcome = outcomes.get(i);
            if (i > 0) {
                out.append(',');
            }
            out.append("{\"command\":");
            Json.string(out, outcome.command());
            out.append(",\"ok\":").append(outcome.ok()).append(",\"output\":");
            Json.string(out, outcome.output() == null ? "" : outcome.output());
            out.append('}');
        }
        out.append("]}");
        if (conn.isOpen()) {
            conn.send(out.toString());
        }
    }

    /**
     * Gives or takes an ability the god has invented.
     *
     * <p>An ability is a trigger, some commands, and a few switches, rather than a name from
     * a list. The commands pass the same gate a direct command passes, and always as though
     * anchored, because a power runs from wherever its owner is standing. Binding a command
     * to a gesture is not a way around what may be run.
     *
     * <p>Runs on the main thread because it touches players directly.
     */
    private void power(WebSocket conn, String id, JsonObject request, boolean give) {
        String player = request.has("player") && !request.get("player").isJsonNull()
                ? request.get("player").getAsString() : null;
        String name = request.has("name") && !request.get("name").isJsonNull()
                ? request.get("name").getAsString() : null;

        if (!give) {
            Bukkit.getScheduler().runTask(plugin, () -> {
                List<String> removed = powers.revoke(player, name);
                StringBuilder out = new StringBuilder("{\"rpc\":\"power_result\",\"id\":");
                Json.string(out, id);
                if (removed == null) {
                    out.append(",\"ok\":false,\"error\":");
                    Json.string(out, "no player called " + player + " is here");
                } else {
                    out.append(",\"ok\":true,\"revoked\":");
                    strings(out, removed);
                    out.append(",\"holding\":");
                    strings(out, powers.current(player));
                }
                out.append('}');
                if (conn.isOpen()) {
                    conn.send(out.toString());
                }
            });
            return;
        }

        List<String> switches = new ArrayList<>();
        if (request.has("switches") && request.get("switches").isJsonArray()) {
            for (var value : request.getAsJsonArray("switches")) {
                switches.add(value.getAsString());
            }
        }
        Map<String, List<String>> scripts = new LinkedHashMap<>();
        List<String> refused = new ArrayList<>();
        if (request.has("scripts") && request.get("scripts").isJsonObject()) {
            for (var entry : request.getAsJsonObject("scripts").entrySet()) {
                if (!entry.getValue().isJsonArray()) {
                    continue;
                }
                List<String> accepted = new ArrayList<>();
                for (var value : entry.getValue().getAsJsonArray()) {
                    CommandService.Verdict verdict =
                            CommandService.inspect(value.getAsString(), true);
                    if (verdict.allowed()) {
                        accepted.add(verdict.command());
                    } else {
                        refused.add(verdict.refusal());
                    }
                }
                if (!accepted.isEmpty()) {
                    scripts.put(entry.getKey(), accepted);
                }
            }
        }
        List<String> on = new ArrayList<>();
        if (request.has("on") && request.get("on").isJsonArray()) {
            for (var value : request.getAsJsonArray("on")) {
                on.add(value.getAsString());
            }
        }
        PowerService.Spec spec = new PowerService.Spec(
                name, switches, scripts, on,
                text(request, "projectile"), number(request, "speed"),
                text(request, "impulse"), number(request, "power"),
                text(request, "beam"), (int) number(request, "range"),
                number(request, "damage"),
                (int) number(request, "every"), (long) number(request, "duration"),
                request.has("cooldown_ms") ? request.get("cooldown_ms").getAsLong() : 200);

        Bukkit.getScheduler().runTask(plugin, () -> {
            PowerService.Granted granted = powers.grant(player, spec);
            StringBuilder out = new StringBuilder("{\"rpc\":\"power_result\",\"id\":");
            Json.string(out, id);
            if (granted.error() != null) {
                out.append(",\"ok\":false,\"error\":");
                Json.string(out, granted.error());
                out.append('}');
            } else {
                out.append(",\"ok\":true,\"granted\":");
                Json.string(out, granted.name());
                out.append(",\"switches\":");
                strings(out, granted.switches());
                out.append(",\"triggers\":");
                strings(out, granted.triggers());
                out.append(",\"projectile\":");
                if (granted.projectile() == null) {
                    out.append("null");
                } else {
                    Json.string(out, granted.projectile());
                }
                out.append(",\"replaced\":").append(granted.replaced());
                out.append(",\"holding\":");
                strings(out, granted.holding());
                out.append(",\"unknown\":");
                strings(out, granted.unknown());
                out.append(",\"refused\":");
                strings(out, refused);
                out.append(",\"triggers_available\":");
                strings(out, PowerService.TRIGGERS);
                out.append(",\"switches_available\":");
                strings(out, PowerService.SWITCHES);
                out.append('}');
            }
            if (conn.isOpen()) {
                conn.send(out.toString());
            }
        });
    }

    /**
     * What a player is holding right now.
     *
     * <p>Asked before every answer, because a god that cannot see what it has already given
     * cannot change it or take it back. Told "make my arrow power automatic" while blind to
     * the name of that power, the model invented a second one beside it, and the first went
     * on firing.
     */
    private void heldPowers(WebSocket conn, String id, JsonObject request) {
        String player = request.has("player") && !request.get("player").isJsonNull()
                ? request.get("player").getAsString() : null;
        Bukkit.getScheduler().runTask(plugin, () -> {
            StringBuilder out = new StringBuilder("{\"rpc\":\"power_result\",\"ok\":true,\"id\":");
            Json.string(out, id);
            out.append(",\"holding\":");
            strings(out, powers.current(player));
            out.append('}');
            if (conn.isOpen()) {
                conn.send(out.toString());
            }
        });
    }

    private static String text(JsonObject request, String key) {
        return request.has(key) && !request.get(key).isJsonNull()
                ? request.get(key).getAsString() : null;
    }

    private static double number(JsonObject request, String key) {
        return request.has(key) && !request.get(key).isJsonNull()
                ? request.get(key).getAsDouble() : 0;
    }

    /** A JSON array of strings, which the power reply needs six times over. */
    private static void strings(StringBuilder out, List<String> values) {
        out.append('[');
        for (int i = 0; i < values.size(); i++) {
            if (i > 0) {
                out.append(',');
            }
            Json.string(out, values.get(i));
        }
        out.append(']');
    }

    /** Stops a repeating effect, or all of them. */
    private void stopCommands(WebSocket conn, String id, JsonObject request) {
        int stopped;
        if (request.has("spell") && !request.get("spell").isJsonNull()) {
            stopped = runner.cancel(request.get("spell").getAsString()) ? 1 : 0;
        } else {
            stopped = runner.stopAll();
        }
        StringBuilder out = new StringBuilder("{\"rpc\":\"stop_result\",\"ok\":true,\"id\":");
        Json.string(out, id);
        out.append(",\"stopped\":").append(stopped).append('}');
        if (conn.isOpen()) {
            conn.send(out.toString());
        }
    }

    private static int[] triple(JsonArray a) {
        return new int[]{a.get(0).getAsInt(), a.get(1).getAsInt(), a.get(2).getAsInt()};
    }

    /**
     * Replies with a failure on the channel the caller is listening to.
     *
     * <p>The reply type must echo the request type. A `speak` that was refused for cooldown
     * once answered on the `scan_result` channel, so the caller — correctly waiting for
     * `speak_result` — hung until it timed out. A refusal nobody can hear is worse than no
     * limit at all: it looks exactly like the server being down.
     */
    private void send(WebSocket conn, String id, String error) {
        send(conn, id, "scan_result", error);
    }

    private void send(WebSocket conn, String id, String replyType, String error) {
        if (!conn.isOpen()) {
            return;
        }
        StringBuilder out = new StringBuilder("{\"rpc\":");
        Json.string(out, replyType);
        out.append(",\"ok\":false,\"id\":");
        Json.string(out, id);
        out.append(",\"error\":");
        Json.string(out, error);
        out.append('}');
        conn.send(out.toString());
    }
}
