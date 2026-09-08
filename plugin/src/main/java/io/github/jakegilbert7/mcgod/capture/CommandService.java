package io.github.jakegilbert7.mcgod.capture;

import java.util.List;
import java.util.Locale;
import java.util.Set;
import java.util.regex.Pattern;

/**
 * What the god is allowed to do to the world, and how often.
 *
 * <p>Until now the only action was speech, which changes nothing. Running commands is a
 * different order of thing, so the decision about what may run lives here, in the server,
 * and not in the agent. The agent is the component most likely to be wrong: it is a model
 * drafting text. This class assumes every command arriving is a mistake until it matches
 * something safe, and it is the only place that opinion is expressed.
 *
 * <p>Three refusals, in order of how badly they would end:
 *
 * <ul>
 *   <li><b>Not on the list.</b> An allowlist, never a denylist. A denylist is a bet that you
 *       thought of everything, and every Minecraft release adds commands.
 *   <li><b>Anything that grants power or ends the server.</b> op, stop, reload, whitelist,
 *       ban. These are listed explicitly as well as absent from the allowlist, so that
 *       widening the allowlist carelessly still cannot reach them.
 *   <li><b>Anything that speaks.</b> say, tell, tellraw, me. Speech is budgeted separately
 *       and capped, and a command that prints to chat is a way around that budget.
 * </ul>
 *
 * <p>Two structural rules beyond the list. {@code execute} is refused outright because it
 * wraps any other command and would make the allowlist decorative. And coordinates must be
 * absolute: a console command's {@code ~} is relative to the world origin rather than to
 * the player, so a relative build lands somewhere nobody asked for.
 */
public final class CommandService {

    /**
     * Commands the god may run.
     *
     * <p>Chosen so it can give things, move things, make weather and sound and light, and
     * build; and so that nothing on it can grant permission, remove a player, alter the
     * server's own state, or speak.
     */
    private static final Set<String> ALLOWED = Set.of(
            "give", "summon", "effect", "particle", "playsound", "stopsound", "title",
            "time", "weather", "xp", "experience", "enchant", "setblock", "fill",
            "clone", "tp", "teleport", "spawnpoint", "gamemode", "clear", "item",
            "ride", "damage", "attribute", "loot", "place", "fillbiome", "spreadplayers");

    /**
     * Never, whatever else changes. Redundant with the allowlist by design: a future hand
     * widening one should still not be able to reach these.
     */
    private static final Set<String> FORBIDDEN = Set.of(
            "op", "deop", "ban", "ban-ip", "banlist", "pardon", "pardon-ip", "kick",
            "whitelist", "stop", "restart", "reload", "rl", "save-all", "save-off",
            "save-on", "execute", "function", "datapack", "debug", "jfr", "perf",
            "plugins", "pl", "version", "ver", "seed", "say", "tell", "msg", "w",
            "tellraw", "me", "trigger", "gamerule", "difficulty", "setidletimeout",
            "setworldspawn", "worldborder", "kill", "publish", "transfer");

    /**
     * How many blocks one command may write: the game's own limit, not a smaller one.
     *
     * <p>There was a tighter cap here and it was wrong. A bridge across a chasm is a big
     * fill, and a god that cannot build one is not the thing being built.
     */
    private static final int MAX_BLOCKS = 32768;
    private static final int MAX_LENGTH = 512;

    private static final Pattern RELATIVE = Pattern.compile("(?<![\\w.])[~^]");
    private static final Pattern NUMBER = Pattern.compile("-?\\d+");

    /** Why a command was refused, or null if it may run. */
    public record Verdict(String refusal, String root, String command) {
        public boolean allowed() {
            return refusal == null;
        }
    }

    /**
     * Whether a command is one the god may run.
     *
     * <p>Pure, so callers can ask freely and hand the same answer back to the agent as an
     * explanation. There is no budget and no cooldown: how much the god does is a question
     * about what it was asked for, not something to ration. What is still refused is what
     * would end the server or hand out its keys.
     *
     * <p>{@code anchored} says whether the caller supplied a player to run from. Relative
     * coordinates are meaningless from the console — {@code ~} is the world origin — but
     * perfectly meaningful when the command runs at somebody, so they are allowed exactly
     * then. Ice under your feet and fire above your head are relative by nature.
     */
    public static Verdict inspect(String raw, boolean anchored) {
        String command = raw == null ? "" : raw.trim();
        while (command.startsWith("/")) {
            command = command.substring(1).trim();
        }
        if (command.isEmpty()) {
            return new Verdict("no command given", "", command);
        }
        if (command.length() > MAX_LENGTH) {
            return new Verdict("command longer than " + MAX_LENGTH + " characters", "",
                    command);
        }
        if (command.indexOf('\n') >= 0 || command.indexOf('\r') >= 0) {
            return new Verdict("a command is one line", "", command);
        }
        String root = command.split("\\s+")[0].toLowerCase(Locale.ROOT);
        if (root.startsWith("minecraft:")) {
            root = root.substring("minecraft:".length());
        }
        if (root.indexOf(':') >= 0) {
            return new Verdict("plugin commands are not available to the god", root, command);
        }
        if (FORBIDDEN.contains(root)) {
            return new Verdict(root + " is never available", root, command);
        }
        if (!ALLOWED.contains(root)) {
            return new Verdict(root + " is not one of the commands the god may run", root,
                    command);
        }
        if (!anchored && RELATIVE.matcher(command).find()) {
            return new Verdict("~ and ^ need somebody to be relative to: name an anchor "
                    + "player, or use absolute coordinates", root, command);
        }
        String area = volumeRefusal(root, command);
        if (area != null) {
            return new Verdict(area, root, command);
        }
        return new Verdict(null, root, command);
    }

    /**
     * Refuses a fill or clone larger than the cap.
     *
     * <p>The server enforces its own limit and would refuse a huge fill anyway, but it does
     * so after the fact and with a message nobody sees. The point of measuring here is that
     * an accidental extra digit is caught before the world changes at all.
     */
    /** The old signature, for callers with nobody to anchor to. */
    public static Verdict inspect(String raw) {
        return inspect(raw, false);
    }

    private static String volumeRefusal(String root, String command) {
        if (!root.equals("fill") && !root.equals("clone") && !root.equals("fillbiome")) {
            return null;
        }
        if (RELATIVE.matcher(command).find()) {
            // Anchored and relative: the corners are wherever the player is, and the game
            // enforces its own limit on the result.
            return null;
        }
        var found = NUMBER.matcher(command);
        long[] values = new long[6];
        int seen = 0;
        while (found.find() && seen < 6) {
            values[seen++] = Long.parseLong(found.group());
        }
        if (seen < 6) {
            return "expected two absolute corners";
        }
        long volume = 1;
        for (int axis = 0; axis < 3; axis++) {
            volume *= Math.abs(values[axis + 3] - values[axis]) + 1;
            if (volume > MAX_BLOCKS) {
                return "that would change more than " + MAX_BLOCKS + " blocks at once";
            }
        }
        return null;
    }

    /** The commands the god may run, for anything that wants to explain itself. */
    public static List<String> allowed() {
        return ALLOWED.stream().sorted().toList();
    }
}
