package io.github.jakegilbert7.mcgod.capture;

import java.util.LinkedHashMap;
import java.util.Map;

/**
 * One captured event, in the schema from CLAUDE.md.
 *
 * <p>Core fields (tick, t, type, actor, dim, pos) are on every event and are serialized in a
 * fixed order. Everything else is type-specific and lives in an ordered extras map, written
 * after the core fields in insertion order and omitted entirely when absent.
 *
 * <p>The extras map exists because there are ~18 event types and a positional record with a
 * field for every type-specific value stopped being readable. Adding an event type is
 * additive: a new {@code type} value and its own keys. No existing field changes meaning,
 * which is what keeps the schema compatible with sessions recorded earlier.
 *
 * <p>Instances are built and populated on the main thread and then handed to the queue. The
 * queue's happens-before edge publishes them safely to the dispatcher thread, which only
 * reads. Do not mutate an event after offering it.
 */
public final class GameEvent {

    private final long tick;
    private final long t;
    private final String type;
    private final String actor;
    private final String dim;
    private final int[] pos;
    private final Map<String, Object> extra = new LinkedHashMap<>(4);

    private GameEvent(long tick, String type, String actor, String dim, int[] pos) {
        this.tick = tick;
        this.t = System.currentTimeMillis();
        this.type = type;
        this.actor = actor;
        this.dim = dim;
        this.pos = pos;
    }

    public static GameEvent of(long tick, String type, String actor, String dim, int[] pos) {
        return new GameEvent(tick, type, actor, dim, pos);
    }

    /** Adds a type-specific field. Null values are dropped, so callers need not pre-check. */
    public GameEvent with(String key, Object value) {
        if (value != null) {
            extra.put(key, value);
        }
        return this;
    }

    public String type() {
        return type;
    }

    // --- Named factories for the types whose field order is pinned by the schema example ---

    public static GameEvent blockPlace(long tick, String actor, String dim, int[] pos,
                                       String before, String after, String tool) {
        return of(tick, "block_place", actor, dim, pos)
                .with("before", before).with("after", after).with("tool", tool);
    }

    public static GameEvent blockBreak(long tick, String actor, String dim, int[] pos,
                                       String before, String tool) {
        return of(tick, "block_break", actor, dim, pos)
                .with("before", before).with("after", "minecraft:air").with("tool", tool);
    }

    public static GameEvent move(long tick, String actor, String dim, int[] pos) {
        return of(tick, "move", actor, dim, pos);
    }

    public static GameEvent regionEnter(long tick, String actor, String dim, int[] pos) {
        return of(tick, "region_enter", actor, dim, pos);
    }

    /** Per-minute rollup of one item type for one actor. */
    public static GameEvent itemPickup(long tick, String actor, String dim, int[] pos,
                                       String item, int count) {
        return of(tick, "item_pickup", actor, dim, pos).with("item", item).with("count", count);
    }

    // --- Serialization -------------------------------------------------------------------

    /** Appends this event as one JSON object. Core field order matches CLAUDE.md. */
    public void writeJson(StringBuilder out) {
        out.append('{');
        out.append("\"tick\":").append(tick);
        out.append(",\"t\":").append(t);
        appendKey(out, "type");
        appendString(out, type);
        appendKey(out, "actor");
        appendString(out, actor);
        appendKey(out, "dim");
        appendString(out, dim);
        if (pos != null) {
            out.append(",\"pos\":[").append(pos[0]).append(',')
                    .append(pos[1]).append(',').append(pos[2]).append(']');
        }
        for (Map.Entry<String, Object> entry : extra.entrySet()) {
            appendKey(out, entry.getKey());
            appendValue(out, entry.getValue());
        }
        out.append('}');
    }

    private static void appendKey(StringBuilder out, String key) {
        out.append(",\"").append(key).append("\":");
    }

    private static void appendValue(StringBuilder out, Object value) {
        switch (value) {
            case java.util.Map<?, ?> m -> appendMap(out, m);
            case String s -> appendString(out, s);
            case Integer i -> out.append(i.intValue());
            case Long l -> out.append(l.longValue());
            case Boolean b -> out.append(b.booleanValue());
            case Double d -> out.append(Double.isFinite(d) ? trim(d) : "0");
            default -> appendString(out, String.valueOf(value));
        }
    }

    /**
     * A histogram, never a block array.
     *
     * <p>The rule is that the model never receives per-block data, and a count keyed by
     * material is a census rather than an array: bounded by the number of distinct materials
     * involved, not by the number of blocks. An explosion that removes 34 blocks of three
     * kinds costs three entries.
     */
    private static void appendMap(StringBuilder out, java.util.Map<?, ?> map) {
        out.append('{');
        boolean first = true;
        for (java.util.Map.Entry<?, ?> e : map.entrySet()) {
            if (!first) {
                out.append(',');
            }
            first = false;
            appendString(out, String.valueOf(e.getKey()));
            out.append(':');
            appendValue(out, e.getValue());
        }
        out.append('}');
    }

    /** Keeps decimals short and stable; damage values are the only doubles we emit. */
    private static String trim(double d) {
        if (d == Math.rint(d)) {
            return Long.toString((long) d);
        }
        return String.valueOf(Math.round(d * 100.0) / 100.0);
    }

    private static void appendString(StringBuilder out, String value) {
        out.append('"');
        for (int i = 0; i < value.length(); i++) {
            char c = value.charAt(i);
            switch (c) {
                case '"' -> out.append("\\\"");
                case '\\' -> out.append("\\\\");
                case '\n' -> out.append("\\n");
                case '\r' -> out.append("\\r");
                case '\t' -> out.append("\\t");
                default -> {
                    if (c < 0x20) {
                        out.append(String.format("\\u%04x", (int) c));
                    } else {
                        out.append(c);
                    }
                }
            }
        }
        out.append('"');
    }
}
