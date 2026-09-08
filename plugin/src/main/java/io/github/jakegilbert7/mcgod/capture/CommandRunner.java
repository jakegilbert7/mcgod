package io.github.jakegilbert7.mcgod.capture;

import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Deque;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.function.Consumer;
import java.util.logging.Logger;

import org.bukkit.Bukkit;
import org.bukkit.Location;
import org.bukkit.command.CommandSender;
import org.bukkit.entity.Player;
import org.bukkit.plugin.Plugin;
import org.bukkit.scheduler.BukkitTask;

import net.kyori.adventure.text.serializer.plain.PlainTextComponentSerializer;

/**
 * Runs whatever the god asks, however much of it, without stopping the game.
 *
 * <p>There is no cap on how many commands one request may contain and no gap enforced
 * between them. A bridge is thousands of blocks; rain of fire for a minute is a command
 * every few ticks for twelve hundred ticks. Refusing those would be refusing the point.
 *
 * <p>What cannot be given up is the tick. Bukkit runs commands on the main thread, and a
 * thousand of them in one tick is a visible freeze — the game stuttering is always worse
 * than the thing being slow. So work is queued and drained a bounded number per tick. The
 * request is unbounded; the pace is not, and the two are different questions.
 *
 * <p>Repetition is a first-class thing rather than the agent looping. A spell that runs for
 * five minutes must survive the agent being busy, restarting, or losing its socket, and it
 * must stop cleanly when the plugin unloads. It also has to be cancellable by a person: the
 * price of an omnipotent god is a switch that turns it off, which is {@code /mcgod stop}.
 */
public final class CommandRunner {

    /**
     * How many queued commands run per tick.
     *
     * <p>High enough that a large build finishes in a second or two, low enough that the
     * server does not stutter. Twenty ticks a second, so 200 here is four thousand commands
     * a second.
     */
    private static final int PER_TICK = 200;

    /** What one command did: the text, whether it worked, and the server's own words. */
    public record Outcome(String command, boolean ok, String output) { }

    /**
     * A batch of commands submitted together, and who to tell when they have all run.
     *
     * <p>The reply used to go out as soon as work was queued, and the god said "I am giving
     * you a sword" for a command that then failed on a syntax the model had guessed. Saying
     * what was accepted is not saying what happened.
     */
    private static final class Batch {
        final List<Outcome> outcomes = new ArrayList<>();
        final Consumer<List<Outcome>> done;
        int outstanding;

        Batch(int outstanding, Consumer<List<Outcome>> done) {
            this.outstanding = outstanding;
            this.done = done;
        }
    }

    /**
     * One command waiting its turn, where to run it from, and the batch it belongs to.
     *
     * <p>{@code at} is a place rather than a person, for the things that happen somewhere
     * nobody is standing: where a thrown pig came down, where an arrow struck. Only one of
     * the two is ever set.
     */
    private record Pending(String command, String anchor, Location at, Batch batch) { }

    /** A repeating effect: the same commands, every so often, until it has run its course. */
    private static final class Spell {
        final String id;
        final List<String> commands;
        final String anchor;
        final int every;
        long remaining;
        BukkitTask task;

        Spell(String id, List<String> commands, String anchor, int every, long remaining) {
            this.id = id;
            this.commands = commands;
            this.anchor = anchor;
            this.every = every;
            this.remaining = remaining;
        }
    }

    private static final PlainTextComponentSerializer PLAIN =
            PlainTextComponentSerializer.plainText();

    private final Plugin plugin;
    private final EventQueue queue;
    private final Logger log;
    private final Deque<Pending> pending = new ArrayDeque<>();
    private final Map<String, Spell> spells = new LinkedHashMap<>();
    private BukkitTask drain;

    public CommandRunner(Plugin plugin, EventQueue queue, Logger log) {
        this.plugin = plugin;
        this.queue = queue;
        this.log = log;
    }

    /** Starts the drain loop. Idempotent. */
    public synchronized void start() {
        if (drain == null) {
            drain = Bukkit.getScheduler().runTaskTimer(plugin, this::tick, 1L, 1L);
        }
    }

    /**
     * Queues commands to run as soon as the pace allows.
     *
     * <p>Returns immediately. The caller is told what was accepted, not what it did, because
     * a large build takes longer than a reply should wait; each command reports itself as it
     * runs, through the event stream.
     */
    public synchronized int submit(List<String> commands, String anchor,
                                   Consumer<List<Outcome>> done) {
        Batch batch = done == null ? null : new Batch(commands.size(), done);
        for (String command : commands) {
            pending.add(new Pending(command, anchor, null, batch));
        }
        start();
        return pending.size();
    }

    public synchronized int submit(List<String> commands, String anchor) {
        return submit(commands, anchor, null);
    }

    /**
     * Queues commands to run at a place instead of from a player.
     *
     * <p>What an impact needs. The thing that landed is not a player and may be gone by the
     * time anyone reads about it, so the position is copied now and the commands run
     * against it, which is what makes {@code ~ ~ ~} mean "here, where this happened".
     */
    public synchronized int submitAt(List<String> commands, Location where) {
        for (String command : commands) {
            pending.add(new Pending(command, null, where.clone(), null));
        }
        start();
        return pending.size();
    }

    /**
     * Starts a repeating effect and returns its id.
     *
     * @param every     ticks between runs; at least one
     * @param duration  total ticks to keep going, or 0 to run once
     */
    public synchronized String cast(String id, List<String> commands, String anchor,
                                    int every, long duration) {
        cancel(id);
        Spell spell = new Spell(id, List.copyOf(commands), anchor, Math.max(1, every),
                Math.max(0, duration));
        spells.put(id, spell);
        spell.task = Bukkit.getScheduler().runTaskTimer(plugin, () -> {
            synchronized (this) {
                for (String command : spell.commands) {
                    pending.add(new Pending(command, spell.anchor, null, null));
                }
                spell.remaining -= spell.every;
                if (spell.remaining <= 0) {
                    cancel(spell.id);
                }
            }
        }, 0L, spell.every);
        start();
        return id;
    }

    /** Stops one repeating effect. Returns whether there was one. */
    public synchronized boolean cancel(String id) {
        Spell spell = spells.remove(id);
        if (spell == null) {
            return false;
        }
        if (spell.task != null) {
            spell.task.cancel();
        }
        return true;
    }

    /**
     * Stops everything: every repeating effect and every command still waiting.
     *
     * <p>The switch. A god that can do anything needs one, and it must not depend on the
     * agent being reachable, so it is here and reachable from {@code /mcgod stop}.
     */
    public synchronized int stopAll() {
        int stopped = spells.size() + pending.size();
        for (Spell spell : List.copyOf(spells.values())) {
            cancel(spell.id);
        }
        pending.clear();
        return stopped;
    }

    public synchronized int queued() {
        return pending.size();
    }

    public synchronized List<String> active() {
        return List.copyOf(spells.keySet());
    }

    /** Cancels the drain loop and every spell. Called when the plugin unloads. */
    public synchronized void shutdown() {
        stopAll();
        if (drain != null) {
            drain.cancel();
            drain = null;
        }
    }

    private void tick() {
        List<Pending> batch = new ArrayList<>(PER_TICK);
        synchronized (this) {
            for (int i = 0; i < PER_TICK && !pending.isEmpty(); i++) {
                batch.add(pending.poll());
            }
        }
        for (Pending item : batch) {
            Outcome outcome = run(item.command(), item.anchor(), item.at());
            Batch owner = item.batch();
            if (owner == null) {
                continue;
            }
            Consumer<List<Outcome>> finished = null;
            List<Outcome> results = null;
            synchronized (this) {
                owner.outcomes.add(outcome);
                if (--owner.outstanding <= 0) {
                    finished = owner.done;
                    results = List.copyOf(owner.outcomes);
                }
            }
            if (finished != null) {
                finished.accept(results);
            }
        }
    }

    /**
     * Runs one command, from the anchor player when there is one.
     *
     * <p>The wrapping in {@code execute at} is composed here and never comes from the agent,
     * whose command was already checked. That is what makes {@code ~} usable without making
     * {@code execute} available: the model may say "ten blocks above me", it may not say
     * "as the server, run anything".
     */
    private Outcome run(String command, String anchor, Location at) {
        StringBuilder feedback = new StringBuilder();
        boolean ok;
        String dispatched = command;
        try {
            if (at != null) {
                dispatched = "execute in " + at.getWorld().getKey() + " positioned "
                        + at.getX() + " " + at.getY() + " " + at.getZ() + " run " + command;
            } else if (anchor != null && !anchor.isBlank()) {
                Player player = Bukkit.getPlayerExact(anchor);
                if (player == null) {
                    String missing = "no player called " + anchor + " is here";
                    record(command, anchor, at, false, missing);
                    return new Outcome(command, false, missing);
                }
                dispatched = "execute as " + anchor + " at " + anchor + " run " + command;
            }
            CommandSender sender = Bukkit.createCommandSender(line -> {
                if (feedback.length() > 0) {
                    feedback.append(' ');
                }
                feedback.append(PLAIN.serialize(line));
            });
            ok = Bukkit.dispatchCommand(sender, dispatched);
        } catch (Exception failure) {
            ok = false;
            feedback.setLength(0);
            feedback.append(failure.getClass().getSimpleName()).append(": ")
                    .append(String.valueOf(failure.getMessage()));
        }
        record(command, anchor, at, ok, feedback.toString());
        return new Outcome(command, ok, feedback.toString());
    }

    /**
     * Every act the god performs goes on the record it can read back.
     *
     * <p>With the place it happened, when there is one. "Break the ice I just made" is
     * unanswerable if the god's own log says only {@code setblock ~ ~-1 ~ ice} a hundred
     * times: the command text is the same every time and the ice is wherever the player
     * was standing. Recording the anchor's position turns the god's acts into something it
     * can find again.
     */
    private void record(String command, String anchor, Location at, boolean ok,
                        String output) {
        if (!ok) {
            log.info("god failed /" + command + (output.isBlank() ? "" : "  " + output));
        }
        int[] where = {0, 0, 0};
        String dim = Dims.of(Bukkit.getWorlds().get(0));
        if (at != null) {
            where = Dims.pos(at);
            dim = Dims.of(at.getWorld());
        } else if (anchor != null && !anchor.isBlank()) {
            Player player = Bukkit.getPlayerExact(anchor);
            if (player != null) {
                where = Dims.pos(player.getLocation());
                dim = Dims.of(player.getWorld());
            }
        }
        queue.offer(GameEvent.of(Bukkit.getCurrentTick(), "god_command", "god", dim, where)
                .with("command", command)
                .with("anchor", anchor == null || anchor.isBlank() ? null : anchor)
                .with("ok", ok)
                .with("output", output.isEmpty() ? null : output));
    }
}
