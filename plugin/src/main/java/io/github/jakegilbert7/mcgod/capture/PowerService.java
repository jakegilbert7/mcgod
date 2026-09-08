package io.github.jakegilbert7.mcgod.capture;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.UUID;
import java.util.logging.Logger;

import org.bukkit.Bukkit;
import org.bukkit.GameMode;
import org.bukkit.entity.Entity;
import org.bukkit.entity.EntityType;
import org.bukkit.entity.Fireball;
import org.bukkit.entity.Player;
import org.bukkit.entity.Projectile;
import org.bukkit.event.EventHandler;
import org.bukkit.event.EventPriority;
import org.bukkit.event.Listener;
import org.bukkit.event.block.Action;
import org.bukkit.event.entity.EntityDamageByEntityEvent;
import org.bukkit.event.entity.EntityDamageEvent;
import org.bukkit.event.player.PlayerAnimationEvent;
import org.bukkit.event.player.PlayerAnimationType;
import org.bukkit.event.player.PlayerInteractEvent;
import org.bukkit.event.player.PlayerMoveEvent;
import org.bukkit.event.player.PlayerQuitEvent;
import org.bukkit.event.player.PlayerToggleSneakEvent;
import org.bukkit.plugin.Plugin;
import org.bukkit.util.Vector;

/**
 * Abilities the god invents, rather than abilities it picks from a list.
 *
 * <p>This began as an enum: flight, fireball, fire immunity, a few trails. That was the wrong
 * shape. A fixed list means the god can only grant what somebody thought of in advance, and
 * the interesting requests are never on it — web shooters, a wake of ice, lightning where you
 * look, a shockwave when you land.
 *
 * <p>So an ability here is a small script the model writes: some things that happen when the
 * player does something, plus a few switches that no command can express. Everything else is
 * commands, which the model already knows how to write and which already pass through
 * {@link CommandService}. The vocabulary is small enough to hold in mind and open enough that
 * "give me a power that..." usually has an answer.
 *
 * <dl>
 *   <dt>Triggers</dt><dd>{@code on_use} (either mouse button), {@code on_sneak},
 *       {@code on_move}, {@code on_attack}, {@code on_damaged}, {@code every}</dd>
 *   <dt>Switches</dt><dd>{@code fly}, {@code no_fall}, {@code glow},
 *       {@code immune:<cause>}, {@code walk_speed:<n>}</dd>
 *   <dt>Aiming</dt><dd>{@code projectile:<entity>[:speed]}, launched along the line of
 *       sight, because a summoned entity's motion is written in world axes and cannot
 *       follow where someone is looking</dd>
 * </dl>
 *
 * <p>Scripted commands run anchored to the player, so {@code ~} is where they stand and
 * {@code ^} is where they face, and they go through the same gate and the same paced runner
 * as everything else the god does.
 */
public final class PowerService implements Listener {

    /** One ability a player currently has. */
    private static final class Ability {
        String name = "power";
        final List<String> switches = new ArrayList<>();
        final Map<String, List<String>> scripts = new HashMap<>();
        String projectile;
        double speed = 1.6;
        int every;
        long expiresAtTick = Long.MAX_VALUE;
        long cooldownMs = 200;
    }

    /** What a player holds, and what was true before it was given. */
    private static final class Held {
        final List<Ability> abilities = new ArrayList<>();
        boolean couldFlyBefore;
        float walkSpeedBefore = 0.2f;
    }

    /** The triggers a script may hang from. Anything else is refused by name. */
    public static final List<String> TRIGGERS = List.of(
            "on_use", "on_sneak", "on_move", "on_attack", "on_damaged", "every");

    /** The switches, which are the things no command expresses. */
    public static final List<String> SWITCHES = List.of(
            "fly", "no_fall", "glow", "immune:<cause>", "walk_speed:<n>");

    private final Plugin plugin;
    private final EventQueue queue;
    private final CommandRunner runner;
    private final Logger log;
    private final Map<UUID, Held> held = new HashMap<>();
    private final Map<String, Long> lastFired = new HashMap<>();

    public PowerService(Plugin plugin, EventQueue queue, CommandRunner runner, Logger log) {
        this.plugin = plugin;
        this.queue = queue;
        this.runner = runner;
        this.log = log;
        Bukkit.getScheduler().runTaskTimer(plugin, this::tick, 20L, 1L);
    }

    /**
     * What a grant did, in the shape the reply needs.
     *
     * <p>{@code unknown} is not an error: it is the vocabulary the model reached for and did
     * not find, handed back with the real vocabulary beside it, because a model told nothing
     * will invent the same word again.
     */
    public record Granted(String error, String name, List<String> switches,
                          List<String> triggers, String projectile, List<String> unknown) { }

    /**
     * Gives a player an ability described by the model.
     *
     * <p>Unknown switches and triggers are reported rather than ignored, so a model that
     * invents vocabulary is told what exists instead of quietly getting nothing.
     */
    public synchronized Granted grant(String playerName, String name,
                                      List<String> switches,
                                      Map<String, List<String>> scripts,
                                      String projectile, double speed,
                                      int every, long durationTicks,
                                      long cooldownMs) {
        Player player = Bukkit.getPlayerExact(playerName);
        if (player == null) {
            return new Granted("no player called " + playerName + " is here",
                    null, List.of(), List.of(), null, List.of());
        }
        List<String> unknown = new ArrayList<>();
        Ability ability = new Ability();
        ability.name = name == null || name.isBlank() ? "power" : name;
        ability.projectile = projectile == null || projectile.isBlank() ? null : projectile;
        ability.speed = speed > 0 ? speed : 1.6;
        ability.every = Math.max(0, every);
        ability.cooldownMs = Math.max(0, cooldownMs);
        ability.expiresAtTick = durationTicks <= 0 ? Long.MAX_VALUE
                : Bukkit.getCurrentTick() + durationTicks;

        for (String raw : switches == null ? List.<String>of() : switches) {
            String flag = raw.toLowerCase(Locale.ROOT).trim();
            if (flag.equals("fly") || flag.equals("no_fall") || flag.equals("glow")
                    || flag.startsWith("immune:") || flag.startsWith("walk_speed:")) {
                ability.switches.add(flag);
            } else {
                unknown.add(flag);
            }
        }
        for (Map.Entry<String, List<String>> entry
                : (scripts == null ? Map.<String, List<String>>of() : scripts).entrySet()) {
            String trigger = entry.getKey().toLowerCase(Locale.ROOT).trim();
            if (!TRIGGERS.contains(trigger)) {
                unknown.add(trigger);
                continue;
            }
            ability.scripts.put(trigger, List.copyOf(entry.getValue()));
        }
        if (ability.projectile != null && entityType(ability.projectile) == null) {
            unknown.add("projectile:" + ability.projectile);
            ability.projectile = null;
        }

        Held holder = held.computeIfAbsent(player.getUniqueId(), id -> {
            Held fresh = new Held();
            fresh.couldFlyBefore = player.getAllowFlight();
            fresh.walkSpeedBefore = player.getWalkSpeed();
            return fresh;
        });
        holder.abilities.add(ability);
        applySwitches(player);

        record(playerName, ability.name, true, durationTicks);
        log.info("god grants " + playerName + " '" + ability.name + "' switches="
                + ability.switches + " triggers=" + ability.scripts.keySet()
                + (ability.projectile == null ? "" : " projectile=" + ability.projectile));
        return new Granted(null, ability.name, List.copyOf(ability.switches),
                List.copyOf(ability.scripts.keySet()), ability.projectile, unknown);
    }

    /** Re-asserts every switch a player's abilities imply. */
    private void applySwitches(Player player) {
        Held holder = held.get(player.getUniqueId());
        if (holder == null) {
            return;
        }
        boolean fly = false;
        boolean glow = false;
        Float walk = null;
        for (Ability ability : holder.abilities) {
            for (String flag : ability.switches) {
                if (flag.equals("fly")) {
                    fly = true;
                } else if (flag.equals("glow")) {
                    glow = true;
                } else if (flag.startsWith("walk_speed:")) {
                    try {
                        walk = Float.parseFloat(flag.substring("walk_speed:".length()));
                    } catch (NumberFormatException ignored) {
                        // a malformed number is not worth refusing the whole ability over
                    }
                }
            }
        }
        if (fly) {
            player.setAllowFlight(true);
        }
        player.setGlowing(glow);
        if (walk != null) {
            player.setWalkSpeed(Math.max(-1f, Math.min(1f, walk)));
        }
    }

    /** Takes an ability back by name, or everything when no name is given. */
    public synchronized List<String> revoke(String playerName, String name) {
        Player player = Bukkit.getPlayerExact(playerName);
        if (player == null) {
            return null;
        }
        Held holder = held.get(player.getUniqueId());
        if (holder == null) {
            return List.of();
        }
        List<String> removed = new ArrayList<>();
        holder.abilities.removeIf(ability -> {
            if (name != null && !name.isBlank() && !ability.name.equalsIgnoreCase(name)) {
                return false;
            }
            removed.add(ability.name);
            return true;
        });
        restore(player, holder);
        if (holder.abilities.isEmpty()) {
            held.remove(player.getUniqueId());
        } else {
            applySwitches(player);
        }
        if (!removed.isEmpty()) {
            record(playerName, String.join(", ", removed), false, 0);
        }
        return removed;
    }

    /**
     * Puts back what was there before.
     *
     * <p>Never takes flight from someone who had it anyway: a granted minute running out must
     * not strand a builder in creative mode mid-air.
     */
    private void restore(Player player, Held holder) {
        if (!holder.couldFlyBefore && player.getGameMode() != GameMode.CREATIVE
                && player.getGameMode() != GameMode.SPECTATOR) {
            player.setAllowFlight(false);
            player.setFlying(false);
        }
        player.setGlowing(false);
        player.setWalkSpeed(holder.walkSpeedBefore);
    }

    private void tick() {
        long now = Bukkit.getCurrentTick();
        List<Runnable> due = new ArrayList<>();
        synchronized (this) {
            for (UUID id : List.copyOf(held.keySet())) {
                Held holder = held.get(id);
                Player player = Bukkit.getPlayer(id);
                if (holder == null) {
                    continue;
                }
                if (player == null) {
                    held.remove(id);
                    continue;
                }
                boolean expired = holder.abilities.removeIf(a -> now >= a.expiresAtTick);
                if (holder.abilities.isEmpty()) {
                    restore(player, holder);
                    held.remove(id);
                    continue;
                }
                if (expired) {
                    restore(player, holder);
                    applySwitches(player);
                }
                for (Ability ability : holder.abilities) {
                    if (ability.every > 0 && now % ability.every == 0) {
                        List<String> script = ability.scripts.get("every");
                        if (script != null && !script.isEmpty()) {
                            due.add(() -> runner.submit(script, player.getName()));
                        }
                    }
                    // A switch a plugin or the game may have cleared between ticks.
                    if (ability.switches.contains("fly") && !player.getAllowFlight()) {
                        player.setAllowFlight(true);
                    }
                }
            }
        }
        due.forEach(Runnable::run);
    }

    /**
     * Runs whatever hangs off one trigger, and throws whatever is aimed.
     *
     * <p>Each ability is throttled on its own clock, per trigger. A single clock per player
     * was wrong in a way that only shows up once someone holds two powers at once: a trail
     * hung from {@code on_move} fires every few ticks and would swallow the cooldown their
     * fireball was waiting on, so the interesting power silently stopped working whenever
     * the dull one was running.
     */
    private void fire(Player player, String trigger) {
        List<Ability> firing = new ArrayList<>();
        long now = System.currentTimeMillis();
        synchronized (this) {
            Held holder = held.get(player.getUniqueId());
            if (holder == null) {
                return;
            }
            for (Ability ability : holder.abilities) {
                boolean aims = trigger.equals("on_use") && ability.projectile != null;
                if (!aims && !ability.scripts.containsKey(trigger)) {
                    continue;
                }
                String clock = player.getUniqueId() + "\0" + ability.name + "\0" + trigger;
                Long last = lastFired.get(clock);
                if (last != null && now - last < ability.cooldownMs) {
                    continue;
                }
                lastFired.put(clock, now);
                firing.add(ability);
            }
        }
        for (Ability ability : firing) {
            if (trigger.equals("on_use") && ability.projectile != null) {
                launch(player, ability);
            }
            List<String> script = ability.scripts.get(trigger);
            if (script != null && !script.isEmpty()) {
                runner.submit(script, player.getName());
            }
        }
    }

    /**
     * Throws something along the line of sight.
     *
     * <p>The one thing commands genuinely cannot do. A summoned entity's motion is written in
     * world axes, so a fireball from {@code /summon} flies the same way whichever way you are
     * facing; aiming has to happen where the player's direction is known.
     */
    private void launch(Player player, Ability ability) {
        EntityType type = entityType(ability.projectile);
        if (type == null) {
            return;
        }
        Vector aim = player.getEyeLocation().getDirection().normalize();
        Entity thrown;
        try {
            thrown = player.getWorld().spawnEntity(
                    player.getEyeLocation().add(aim.clone().multiply(1.4)), type);
        } catch (IllegalArgumentException notSpawnable) {
            // A real entity name is not the same as a spawnable one. Refusing quietly beats
            // throwing on every click for as long as the power is held.
            log.info("cannot throw " + ability.projectile + ": " + notSpawnable.getMessage());
            ability.projectile = null;
            return;
        }
        if (thrown instanceof Fireball fireball) {
            fireball.setShooter(player);
            fireball.setDirection(aim);
        } else if (thrown instanceof Projectile projectile) {
            projectile.setShooter(player);
            projectile.setVelocity(aim.multiply(ability.speed));
        } else {
            thrown.setVelocity(aim.multiply(ability.speed));
        }
    }

    private static EntityType entityType(String name) {
        if (name == null) {
            return null;
        }
        String clean = name.toLowerCase(Locale.ROOT).trim()
                .replace("minecraft:", "").replace(' ', '_');
        try {
            return EntityType.valueOf(clean.toUpperCase(Locale.ROOT));
        } catch (IllegalArgumentException missing) {
            return null;
        }
    }

    /**
     * Either mouse button counts as using a power.
     *
     * <p>Right-clicking air with an empty hand sends the server nothing at all — the client
     * only reports a use when there is an item or a block within reach — so a power bound to
     * it appeared to work only when facing something. The arm swing always arrives, so both
     * are treated as the same intent.
     */
    @EventHandler(priority = EventPriority.MONITOR, ignoreCancelled = true)
    public void onInteract(PlayerInteractEvent event) {
        Action action = event.getAction();
        if (action == Action.RIGHT_CLICK_AIR || action == Action.RIGHT_CLICK_BLOCK) {
            fire(event.getPlayer(), "on_use");
        }
    }

    @EventHandler(priority = EventPriority.MONITOR)
    public void onSwing(PlayerAnimationEvent event) {
        if (event.getAnimationType() == PlayerAnimationType.ARM_SWING) {
            fire(event.getPlayer(), "on_use");
        }
    }

    @EventHandler(priority = EventPriority.MONITOR, ignoreCancelled = true)
    public void onSneak(PlayerToggleSneakEvent event) {
        if (event.isSneaking()) {
            fire(event.getPlayer(), "on_sneak");
        }
    }

    /**
     * Moving, which is not the same as looking around.
     *
     * <p>A teleport is a move event too, and one across worlds cannot be measured against
     * where it came from — asking for the distance throws, on the main thread, inside an
     * event handler. Different world means they have certainly moved.
     */
    @EventHandler(priority = EventPriority.MONITOR, ignoreCancelled = true)
    public void onMove(PlayerMoveEvent event) {
        if (event.getFrom().getWorld() == event.getTo().getWorld()
                && event.getTo().distanceSquared(event.getFrom()) < 0.02) {
            return;
        }
        fire(event.getPlayer(), "on_move");
    }

    @EventHandler(priority = EventPriority.MONITOR, ignoreCancelled = true)
    public void onAttack(EntityDamageByEntityEvent event) {
        if (event.getDamager() instanceof Player player) {
            fire(player, "on_attack");
        }
    }

    /** Immunity, and anything hung from being hurt. */
    @EventHandler(priority = EventPriority.HIGHEST, ignoreCancelled = true)
    public void onDamage(EntityDamageEvent event) {
        if (!(event.getEntity() instanceof Player player)) {
            return;
        }
        String cause = event.getCause().name().toLowerCase(Locale.ROOT);
        boolean immune;
        synchronized (this) {
            Held holder = held.get(player.getUniqueId());
            if (holder == null) {
                return;
            }
            immune = holder.abilities.stream().anyMatch(ability ->
                    ability.switches.contains("immune:" + cause)
                            || ability.switches.contains("immune:all")
                            || (ability.switches.contains("no_fall") && cause.equals("fall"))
                            || (ability.switches.contains("immune:fire")
                                && (cause.equals("fire") || cause.equals("fire_tick")
                                    || cause.equals("lava") || cause.equals("hot_floor"))));
        }
        if (immune) {
            event.setCancelled(true);
            if (cause.startsWith("fire") || cause.equals("lava")) {
                player.setFireTicks(0);
            }
            return;
        }
        fire(player, "on_damaged");
    }

    @EventHandler
    public void onQuit(PlayerQuitEvent event) {
        synchronized (this) {
            held.remove(event.getPlayer().getUniqueId());
            lastFired.keySet().removeIf(clock ->
                    clock.startsWith(event.getPlayer().getUniqueId().toString()));
        }
    }

    /** What a player holds, for status and for the god to read back. */
    public synchronized List<String> current(String playerName) {
        Player player = Bukkit.getPlayerExact(playerName);
        if (player == null) {
            return List.of();
        }
        Held holder = held.get(player.getUniqueId());
        if (holder == null) {
            return List.of();
        }
        List<String> names = new ArrayList<>();
        for (Ability ability : holder.abilities) {
            names.add(ability.name);
        }
        return names;
    }

    /** Takes everything from everyone. Part of the stop switch. */
    public synchronized int revokeAll() {
        int count = 0;
        for (UUID id : List.copyOf(held.keySet())) {
            Player player = Bukkit.getPlayer(id);
            if (player != null) {
                count += revoke(player.getName(), null).size();
            } else {
                held.remove(id);
            }
        }
        return count;
    }

    private void record(String player, String name, boolean given, long durationTicks) {
        queue.offer(GameEvent.of(Bukkit.getCurrentTick(),
                        given ? "god_granted_power" : "god_revoked_power", "god",
                        Dims.of(Bukkit.getWorlds().get(0)), new int[]{0, 0, 0})
                .with("player", player)
                .with("power", name)
                .with("duration_ticks", durationTicks <= 0 ? null : durationTicks));
    }
}
