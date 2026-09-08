package io.github.jakegilbert7.mcgod.capture;

import net.kyori.adventure.text.serializer.plain.PlainTextComponentSerializer;

import org.bukkit.Bukkit;
import org.bukkit.entity.Entity;
import org.bukkit.entity.LivingEntity;
import org.bukkit.entity.Monster;
import org.bukkit.entity.Player;
import org.bukkit.entity.Projectile;
import org.bukkit.entity.Tameable;
import org.bukkit.event.EventHandler;
import org.bukkit.event.EventPriority;
import org.bukkit.event.Listener;
import org.bukkit.event.entity.EntityBreedEvent;
import org.bukkit.event.entity.EntityDamageByEntityEvent;
import org.bukkit.event.entity.EntityDamageEvent;
import org.bukkit.event.entity.EntityPotionEffectEvent;
import org.bukkit.event.entity.EntityDeathEvent;
import org.bukkit.event.entity.EntityTameEvent;
import org.bukkit.event.entity.ProjectileHitEvent;
import org.bukkit.event.entity.EntityExplodeEvent;
import org.bukkit.event.player.PlayerFishEvent;
import org.bukkit.event.player.PlayerShearEntityEvent;
import org.bukkit.projectiles.ProjectileSource;

/**
 * What a player does to, and suffers from, the living world.
 *
 * <p>These are the events behind L2's risk profile. Both directions are recorded: what hurt
 * the player and what the player hurt, each with its cause and — where one exists — the
 * entity responsible. "Died to a skeleton" and "died to fall damage while fleeing a skeleton"
 * are different stories, and only the cause distinguishes them.
 */
public final class EntityListener implements Listener {

    /** A named pet is worth naming; the name is a component, so it needs flattening. */
    private static final PlainTextComponentSerializer PLAIN =
            PlainTextComponentSerializer.plainText();

    private final EventQueue queue;

    public EntityListener(EventQueue queue) {
        this.queue = queue;
    }

    /**
     * A death. A player's kill stays {@code mob_kill}; everything else worth keeping
     * becomes {@code mob_death}.
     *
     * <p>A creeper that blows up beside a player also kills the sheep standing there, and
     * nothing recorded that: the death of one mob at the hands of another was invisible, so
     * "what just happened to me" could describe the blast and not what it cost.
     *
     * <p>Not every death is worth an event. Hostile mobs burn by the dozen at every sunrise
     * and drown and fall constantly, which is weather, not news. So this keeps a death when
     * something else killed it — that is the case being fixed, and it is bounded by how
     * often mobs actually fight — and otherwise only when the victim was not a monster,
     * because a sheep that burned or a wolf that fell is a thing that happened to the
     * player's world while a zombie catching fire is merely morning.
     */
    @EventHandler(priority = EventPriority.MONITOR)
    public void onDeath(EntityDeathEvent event) {
        LivingEntity victim = event.getEntity();
        Player killer = victim.getKiller();
        if (killer != null) {
            queue.offer(GameEvent.of(Bukkit.getCurrentTick(), "mob_kill",
                            killer.getUniqueId().toString(), Dims.of(victim.getWorld()),
                            Dims.pos(victim.getLocation()))
                    .with("entity", Dims.entity(victim))
                    .with("tool", Dims.heldTool(killer)));
            return;
        }
        Entity slayer = culprit(victim.getLastDamageCause());
        if (slayer == null && victim instanceof Monster) {
            return;
        }
        EntityDamageEvent last = victim.getLastDamageCause();
        queue.offer(GameEvent.of(Bukkit.getCurrentTick(), "mob_death", "world",
                        Dims.of(victim.getWorld()), Dims.pos(victim.getLocation()))
                .with("entity", Dims.entity(victim))
                .with("name", victim.customName() == null
                        ? null : PLAIN.serialize(victim.customName()))
                .with("cause", last == null ? null : Dims.name(last.getCause()))
                .with("killer", slayer == null ? null : Dims.entity(slayer))
                .with("tame", victim instanceof Tameable tameable && tameable.isTamed()));
    }

    /**
     * The entity behind a killing blow, following a projectile back to whatever fired it.
     *
     * <p>An arrow did not kill the sheep; a skeleton did. Reporting the projectile would be
     * true and useless, and would make "a skeleton shot your wolf" unanswerable.
     */
    private static Entity culprit(EntityDamageEvent cause) {
        if (!(cause instanceof EntityDamageByEntityEvent byEntity)) {
            return null;
        }
        Entity damager = byEntity.getDamager();
        if (damager instanceof Projectile projectile
                && projectile.getShooter() instanceof Entity shooter) {
            return shooter;
        }
        return damager;
    }

    /**
     * Damage a player took.
     *
     * <p>At MONITOR the damage has not been applied yet, so {@code player.getHealth()} is still
     * the pre-hit value. {@code health} is therefore the health this hit *leaves* them on,
     * which is the number that says whether they nearly died. {@code source} names the entity
     * responsible when there is one, so "poison" and "skeleton" are distinguishable.
     */
    @EventHandler(priority = EventPriority.MONITOR, ignoreCancelled = true)
    public void onDamage(EntityDamageEvent event) {
        if (!(event.getEntity() instanceof Player player)) {
            return;
        }
        double remaining = Math.max(0.0, player.getHealth() - event.getFinalDamage());
        GameEvent record = GameEvent.of(Bukkit.getCurrentTick(), "damage",
                        player.getUniqueId().toString(), Dims.of(player.getWorld()),
                        Dims.pos(player.getLocation()))
                .with("cause", Dims.name(event.getCause()))
                .with("amount", event.getFinalDamage())
                .with("health", remaining)
                .with("health_before", player.getHealth());
        if (event instanceof EntityDamageByEntityEvent byEntity) {
            record.with("source", Dims.entity(byEntity.getDamager()));
        }
        queue.offer(record);
    }

    /** Damage a player dealt. Fires alongside {@code damage} when a player hits a player. */
    @EventHandler(priority = EventPriority.MONITOR, ignoreCancelled = true)
    public void onDamageDealt(EntityDamageByEntityEvent event) {
        if (!(event.getDamager() instanceof Player player)) {
            return;
        }
        queue.offer(GameEvent.of(Bukkit.getCurrentTick(), "damage_dealt",
                        player.getUniqueId().toString(), Dims.of(player.getWorld()),
                        Dims.pos(event.getEntity().getLocation()))
                .with("entity", Dims.entity(event.getEntity()))
                .with("cause", Dims.name(event.getCause()))
                .with("amount", event.getFinalDamage())
                .with("tool", Dims.heldTool(player)));
    }

    /** Potions, beacons, milk, and every other status change on a player. */
    @EventHandler(priority = EventPriority.MONITOR, ignoreCancelled = true)
    public void onPotionEffect(EntityPotionEffectEvent event) {
        if (!(event.getEntity() instanceof Player player)) {
            return;
        }
        queue.offer(GameEvent.of(Bukkit.getCurrentTick(), "potion_effect",
                        player.getUniqueId().toString(), Dims.of(player.getWorld()),
                        Dims.pos(player.getLocation()))
                .with("effect", event.getModifiedType() == null
                        ? null : event.getModifiedType().getKey().toString())
                .with("action", Dims.name(event.getAction()))
                .with("reason", Dims.name(event.getCause()))
                .with("duration", event.getNewEffect() == null
                        ? null : event.getNewEffect().getDuration())
                .with("amplifier", event.getNewEffect() == null
                        ? null : event.getNewEffect().getAmplifier()));
    }

    @EventHandler(priority = EventPriority.MONITOR, ignoreCancelled = true)
    public void onShear(PlayerShearEntityEvent event) {
        Player player = event.getPlayer();
        queue.offer(GameEvent.of(Bukkit.getCurrentTick(), "shear",
                        player.getUniqueId().toString(), Dims.of(player.getWorld()),
                        Dims.pos(event.getEntity().getLocation()))
                .with("entity", Dims.entity(event.getEntity())));
    }

    /**
     * Creepers, TNT, beds in the nether. Recorded with no actor; the god infers blame.
     *
     * <p>Records WHAT was destroyed, not merely how much. A count is useless for accounting:
     * measured on one recorded session, eight creeper explosions removed 114 blocks and the
     * belief store went on believing every one of them was still standing, because nothing
     * in the event stream ever said otherwise. The destroyed materials close that hole
     * without a scan.
     */
    @EventHandler(priority = EventPriority.MONITOR, ignoreCancelled = true)
    public void onExplode(EntityExplodeEvent event) {
        queue.offer(GameEvent.of(Bukkit.getCurrentTick(), "explosion", "world",
                        Dims.of(event.getLocation().getWorld()),
                        Dims.pos(event.getLocation()))
                .with("entity", Dims.entity(event.getEntity()))
                .with("blocks", event.blockList().size())
                .with("destroyed", census(event.blockList())));
    }

    /** TNT and beds detonating with no entity behind them. */
    @EventHandler(priority = EventPriority.MONITOR, ignoreCancelled = true)
    public void onBlockExplode(org.bukkit.event.block.BlockExplodeEvent event) {
        queue.offer(GameEvent.of(Bukkit.getCurrentTick(), "explosion", "world",
                        Dims.of(event.getBlock().getWorld()),
                        new int[]{event.getBlock().getX(), event.getBlock().getY(),
                                event.getBlock().getZ()})
                .with("blocks", event.blockList().size())
                .with("destroyed", census(event.blockList())));
    }

    /** Material histogram of a block list. Bounded by distinct materials, not block count. */
    private static java.util.Map<String, Integer> census(java.util.List<org.bukkit.block.Block> blocks) {
        java.util.Map<String, Integer> counts = new java.util.LinkedHashMap<>();
        for (org.bukkit.block.Block b : blocks) {
            if (!b.getType().isAir()) {
                counts.merge(b.getType().getKey().toString(), 1, Integer::sum);
            }
        }
        return counts;
    }

    @EventHandler(priority = EventPriority.MONITOR, ignoreCancelled = true)
    public void onBreed(EntityBreedEvent event) {
        if (!(event.getBreeder() instanceof Player player)) {
            return;
        }
        queue.offer(GameEvent.of(Bukkit.getCurrentTick(), "breed",
                        player.getUniqueId().toString(), Dims.of(event.getEntity().getWorld()),
                        Dims.pos(event.getEntity().getLocation()))
                .with("entity", Dims.entity(event.getEntity())));
    }

    @EventHandler(priority = EventPriority.MONITOR, ignoreCancelled = true)
    public void onTame(EntityTameEvent event) {
        if (!(event.getOwner() instanceof Player player)) {
            return;
        }
        queue.offer(GameEvent.of(Bukkit.getCurrentTick(), "tame",
                        player.getUniqueId().toString(), Dims.of(event.getEntity().getWorld()),
                        Dims.pos(event.getEntity().getLocation()))
                .with("entity", Dims.entity(event.getEntity())));
    }

    /** Only successful catches; the FISHING/BITE states fire constantly while waiting. */
    @EventHandler(priority = EventPriority.MONITOR, ignoreCancelled = true)
    public void onFish(PlayerFishEvent event) {
        if (event.getState() != PlayerFishEvent.State.CAUGHT_FISH
                && event.getState() != PlayerFishEvent.State.CAUGHT_ENTITY) {
            return;
        }
        Player player = event.getPlayer();
        queue.offer(GameEvent.of(Bukkit.getCurrentTick(), "fish",
                        player.getUniqueId().toString(), Dims.of(player.getWorld()),
                        Dims.pos(player.getLocation()))
                .with("state", Dims.name(event.getState()))
                .with("entity", Dims.entity(event.getCaught())));
    }

    /** Projectiles a player fired, and what they struck. */
    @EventHandler(priority = EventPriority.MONITOR)
    public void onProjectileHit(ProjectileHitEvent event) {
        Projectile projectile = event.getEntity();
        ProjectileSource source = projectile.getShooter();
        if (!(source instanceof Player player)) {
            return;
        }
        queue.offer(GameEvent.of(Bukkit.getCurrentTick(), "projectile_hit",
                        player.getUniqueId().toString(), Dims.of(projectile.getWorld()),
                        Dims.pos(projectile.getLocation()))
                .with("projectile", Dims.entity(projectile))
                .with("entity", Dims.entity(event.getHitEntity()))
                .with("block", event.getHitBlock() == null
                        ? null : event.getHitBlock().getType().getKey().toString()));
    }
}
