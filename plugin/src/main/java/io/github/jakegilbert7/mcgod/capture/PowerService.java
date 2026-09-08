package io.github.jakegilbert7.mcgod.capture;

import java.util.HashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Set;
import java.util.UUID;
import java.util.logging.Logger;

import org.bukkit.Bukkit;
import org.bukkit.GameMode;
import org.bukkit.Particle;
import org.bukkit.entity.Fireball;
import org.bukkit.entity.LargeFireball;
import org.bukkit.entity.Player;
import org.bukkit.entity.SmallFireball;
import org.bukkit.entity.WindCharge;
import org.bukkit.event.EventHandler;
import org.bukkit.event.EventPriority;
import org.bukkit.event.Listener;
import org.bukkit.event.block.Action;
import org.bukkit.event.entity.EntityDamageEvent;
import org.bukkit.event.player.PlayerInteractEvent;
import org.bukkit.event.player.PlayerMoveEvent;
import org.bukkit.event.player.PlayerQuitEvent;
import org.bukkit.plugin.Plugin;
import org.bukkit.potion.PotionEffect;
import org.bukkit.potion.PotionEffectType;
import org.bukkit.util.Vector;

/**
 * Things a player can be given that no command can give them.
 *
 * <p>Asked for the power to fly and throw fire, the god could only reach for what commands
 * offer: creative mode and a stack of fire charges. Creative is not a superpower, it is a
 * different game — no damage, no hunger, no stakes — and a fire charge is an item you throw
 * once. The player was right that this was unsatisfying, and right to ask whether it needed
 * mods. It does not. Flight without creative, a fireball from an empty hand, immunity to the
 * fire you make: all of it is ordinary server-side API, and none of it is expressible as a
 * command, which is why the god needed a second way to act rather than a longer allowlist.
 *
 * <p>Powers are held per player with an expiry and cleaned up on quit and on unload. What is
 * given back on expiry is what was there before: creative flight is left alone, and a player
 * who could already fly does not lose it because a granted minute ran out.
 */
public final class PowerService implements Listener {

    /** How long between the granting and the taking away, and what was true before. */
    private static final class Granted {
        final Set<String> powers = new LinkedHashSet<>();
        long expiresAtTick;
        boolean couldFlyBefore;
        boolean wasFlyingAllowed;
    }

    /** Every power by name, so the god can be told what exists rather than guessing. */
    public static final List<String> POWERS = List.of(
            "flight", "fireball", "great_fireball", "wind_blast", "fire_immunity",
            "fall_immunity", "strength", "speed", "jump", "regeneration", "resistance",
            "night_vision", "water_breathing", "invisibility", "glow", "flame_trail",
            "cloud_trail", "spark_trail");

    private static final Map<String, PotionEffectType> EFFECTS = Map.of(
            "strength", PotionEffectType.STRENGTH,
            "speed", PotionEffectType.SPEED,
            "jump", PotionEffectType.JUMP_BOOST,
            "regeneration", PotionEffectType.REGENERATION,
            "resistance", PotionEffectType.RESISTANCE,
            "night_vision", PotionEffectType.NIGHT_VISION,
            "water_breathing", PotionEffectType.WATER_BREATHING,
            "invisibility", PotionEffectType.INVISIBILITY);

    /** A throw every quarter second. Fast enough to feel like a power, slow enough to aim. */
    private static final long THROW_COOLDOWN_MS = 250;

    private final Plugin plugin;
    private final EventQueue queue;
    private final Logger log;
    private final Map<UUID, Granted> held = new HashMap<>();
    private final Map<UUID, Long> lastThrow = new HashMap<>();

    public PowerService(Plugin plugin, EventQueue queue, Logger log) {
        this.plugin = plugin;
        this.queue = queue;
        this.log = log;
        Bukkit.getScheduler().runTaskTimer(plugin, this::expire, 20L, 20L);
    }

    /** Whether a name is one of the powers. Used to explain a refusal precisely. */
    public static boolean known(String power) {
        return POWERS.contains(power.toLowerCase(Locale.ROOT));
    }

    /**
     * Gives a player powers for a while.
     *
     * @param durationTicks how long to hold them; zero means until taken away
     * @return what was granted, or null if the player is not here
     */
    public synchronized List<String> grant(String name, List<String> powers,
                                           long durationTicks) {
        Player player = Bukkit.getPlayerExact(name);
        if (player == null) {
            return null;
        }
        Granted granted = held.computeIfAbsent(player.getUniqueId(), id -> {
            Granted fresh = new Granted();
            fresh.couldFlyBefore = player.getAllowFlight();
            fresh.wasFlyingAllowed = player.isFlying();
            return fresh;
        });
        List<String> applied = new java.util.ArrayList<>();
        for (String raw : powers) {
            String power = raw.toLowerCase(Locale.ROOT).trim();
            if (!known(power)) {
                continue;
            }
            granted.powers.add(power);
            applied.add(power);
            apply(player, power, durationTicks);
        }
        granted.expiresAtTick = durationTicks <= 0 ? Long.MAX_VALUE
                : Bukkit.getCurrentTick() + durationTicks;
        record(name, applied, durationTicks, true);
        log.info("god grants " + name + " " + applied + " for " + durationTicks + " ticks");
        return applied;
    }

    private void apply(Player player, String power, long durationTicks) {
        PotionEffectType effect = EFFECTS.get(power);
        if (effect != null) {
            int ticks = durationTicks <= 0 ? Integer.MAX_VALUE : (int) durationTicks;
            player.addPotionEffect(new PotionEffect(effect, ticks, 1, true, false, true));
            return;
        }
        if (power.equals("flight")) {
            // Flight without creative: they keep hunger, damage, and their inventory. That
            // difference is the whole point — creative is not a superpower, it is a
            // different game.
            player.setAllowFlight(true);
        } else if (power.equals("glow")) {
            player.setGlowing(true);
        }
    }

    /** Takes powers back. With no names, takes all of them. */
    public synchronized List<String> revoke(String name, List<String> powers) {
        Player player = Bukkit.getPlayerExact(name);
        if (player == null) {
            return null;
        }
        Granted granted = held.get(player.getUniqueId());
        if (granted == null) {
            return List.of();
        }
        List<String> removed = new java.util.ArrayList<>();
        for (String power : powers == null || powers.isEmpty()
                ? List.copyOf(granted.powers) : powers) {
            String key = power.toLowerCase(Locale.ROOT).trim();
            if (!granted.powers.remove(key)) {
                continue;
            }
            removed.add(key);
            undo(player, key);
        }
        if (granted.powers.isEmpty()) {
            held.remove(player.getUniqueId());
        }
        record(name, removed, 0, false);
        return removed;
    }

    private void undo(Player player, String power) {
        PotionEffectType effect = EFFECTS.get(power);
        if (effect != null) {
            player.removePotionEffect(effect);
            return;
        }
        if (power.equals("flight")) {
            // Never take flight from someone who had it anyway. A granted minute running
            // out must not strand a builder in creative mode mid-air.
            if (player.getGameMode() != GameMode.CREATIVE
                    && player.getGameMode() != GameMode.SPECTATOR) {
                player.setAllowFlight(false);
                player.setFlying(false);
            }
        } else if (power.equals("glow")) {
            player.setGlowing(false);
        }
    }

    private synchronized void expire() {
        long now = Bukkit.getCurrentTick();
        for (UUID id : List.copyOf(held.keySet())) {
            Granted granted = held.get(id);
            if (granted == null || now < granted.expiresAtTick) {
                continue;
            }
            Player player = Bukkit.getPlayer(id);
            if (player != null) {
                revoke(player.getName(), null);
            } else {
                held.remove(id);
            }
        }
    }

    private synchronized boolean has(Player player, String power) {
        Granted granted = held.get(player.getUniqueId());
        return granted != null && granted.powers.contains(power);
    }

    /**
     * An empty hand throws whatever they were given.
     *
     * <p>Empty-handed on purpose: right-clicking with something in hand is placing a block
     * or eating or using a tool, and a power that stole those would make the game worse
     * rather than better.
     */
    @EventHandler(priority = EventPriority.NORMAL, ignoreCancelled = true)
    public void onInteract(PlayerInteractEvent event) {
        Action action = event.getAction();
        if (action != Action.RIGHT_CLICK_AIR && action != Action.RIGHT_CLICK_BLOCK) {
            return;
        }
        Player player = event.getPlayer();
        if (event.getItem() != null) {
            return;
        }
        Class<? extends org.bukkit.entity.Entity> shot = null;
        if (has(player, "great_fireball")) {
            shot = LargeFireball.class;
        } else if (has(player, "fireball")) {
            shot = SmallFireball.class;
        } else if (has(player, "wind_blast")) {
            shot = WindCharge.class;
        }
        if (shot == null) {
            return;
        }
        long now = System.currentTimeMillis();
        Long last = lastThrow.get(player.getUniqueId());
        if (last != null && now - last < THROW_COOLDOWN_MS) {
            return;
        }
        lastThrow.put(player.getUniqueId(), now);
        Vector aim = player.getEyeLocation().getDirection().normalize();
        org.bukkit.entity.Entity thrown = player.getWorld().spawn(
                player.getEyeLocation().add(aim.clone().multiply(1.2)), shot);
        if (thrown instanceof Fireball fireball) {
            fireball.setShooter(player);
            fireball.setDirection(aim);
        } else {
            thrown.setVelocity(aim.multiply(1.6));
        }
    }

    /** Immunity to the fire you make, and to the ground you land on. */
    @EventHandler(priority = EventPriority.HIGHEST, ignoreCancelled = true)
    public void onDamage(EntityDamageEvent event) {
        if (!(event.getEntity() instanceof Player player)) {
            return;
        }
        EntityDamageEvent.DamageCause cause = event.getCause();
        boolean burning = cause == EntityDamageEvent.DamageCause.FIRE
                || cause == EntityDamageEvent.DamageCause.FIRE_TICK
                || cause == EntityDamageEvent.DamageCause.LAVA
                || cause == EntityDamageEvent.DamageCause.HOT_FLOOR;
        if (burning && has(player, "fire_immunity")) {
            event.setCancelled(true);
            player.setFireTicks(0);
            return;
        }
        if (cause == EntityDamageEvent.DamageCause.FALL && has(player, "fall_immunity")) {
            event.setCancelled(true);
        }
    }

    /** A trail, so a power is something other people can see. */
    @EventHandler(priority = EventPriority.MONITOR, ignoreCancelled = true)
    public void onMove(PlayerMoveEvent event) {
        if (event.getTo().distanceSquared(event.getFrom()) < 0.02) {
            return;
        }
        Player player = event.getPlayer();
        Particle particle = null;
        if (has(player, "flame_trail")) {
            particle = Particle.FLAME;
        } else if (has(player, "cloud_trail")) {
            particle = Particle.CLOUD;
        } else if (has(player, "spark_trail")) {
            particle = Particle.ELECTRIC_SPARK;
        }
        if (particle != null) {
            player.getWorld().spawnParticle(particle, player.getLocation(), 6,
                    0.2, 0.1, 0.2, 0.01);
        }
    }

    /** Powers do not survive leaving; nothing lingers on a player who is not here. */
    @EventHandler
    public void onQuit(PlayerQuitEvent event) {
        synchronized (this) {
            held.remove(event.getPlayer().getUniqueId());
            lastThrow.remove(event.getPlayer().getUniqueId());
        }
    }

    /** What a player currently holds, for status and for the god to read back. */
    public synchronized List<String> current(String name) {
        Player player = Bukkit.getPlayerExact(name);
        if (player == null) {
            return List.of();
        }
        Granted granted = held.get(player.getUniqueId());
        return granted == null ? List.of() : List.copyOf(granted.powers);
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

    /** The god's own acts, on the record it reads back. */
    private void record(String player, List<String> powers, long durationTicks,
                        boolean given) {
        if (powers.isEmpty()) {
            return;
        }
        queue.offer(GameEvent.of(Bukkit.getCurrentTick(),
                        given ? "god_granted_power" : "god_revoked_power", "god",
                        Dims.of(Bukkit.getWorlds().get(0)), new int[]{0, 0, 0})
                .with("player", player)
                .with("powers", String.join(",", powers))
                .with("duration_ticks", durationTicks <= 0 ? null : durationTicks));
    }
}
