package io.github.jakegilbert7.mcgod.capture;

import org.bukkit.Bukkit;
import org.bukkit.entity.Player;
import org.bukkit.event.EventHandler;
import org.bukkit.event.EventPriority;
import org.bukkit.event.Listener;
import org.bukkit.event.entity.PlayerDeathEvent;
import org.bukkit.event.player.PlayerAdvancementDoneEvent;
import org.bukkit.event.player.PlayerBedEnterEvent;
import org.bukkit.event.player.PlayerChangedWorldEvent;
import org.bukkit.event.player.PlayerCommandPreprocessEvent;
import org.bukkit.event.player.PlayerExpChangeEvent;
import org.bukkit.event.player.PlayerGameModeChangeEvent;
import org.bukkit.event.player.PlayerItemBreakEvent;
import org.bukkit.event.player.PlayerLevelChangeEvent;
import org.bukkit.event.player.PlayerRespawnEvent;
import org.bukkit.event.player.PlayerItemConsumeEvent;
import org.bukkit.event.player.PlayerJoinEvent;
import org.bukkit.event.player.PlayerQuitEvent;
import org.bukkit.event.player.PlayerTeleportEvent;

/** Player lifecycle, and the things a player does to themselves rather than to the world. */
public final class PlayerListener implements Listener {

    /** Recipe unlocks fire constantly and say nothing about intent. */
    private static final String RECIPE_PREFIX = "minecraft:recipes/";

    private final EventQueue queue;
    private final PositionSampler sampler;

    public PlayerListener(EventQueue queue, PositionSampler sampler) {
        this.queue = queue;
        this.sampler = sampler;
    }

    @EventHandler(priority = EventPriority.MONITOR)
    public void onJoin(PlayerJoinEvent event) {
        Player player = event.getPlayer();
        // The name rides along on join and quit only. Actor identity stays the UUID —
        // names can be changed and reused, so they are a label, never a key — but without
        // carrying it at all the god can only ever address someone as "59cca301".
        queue.offer(event(player, "player_join").with("name", player.getName()));
        // Seeds the region cell so the first sample does not report a spurious entry.
        sampler.observe(player, player.getLocation());
    }

    @EventHandler(priority = EventPriority.MONITOR)
    public void onQuit(PlayerQuitEvent event) {
        queue.offer(event(event.getPlayer(), "player_quit")
                .with("name", event.getPlayer().getName()));
        sampler.forget(event.getPlayer().getUniqueId());
    }

    @EventHandler(priority = EventPriority.MONITOR)
    public void onDeath(PlayerDeathEvent event) {
        Player player = event.getEntity();
        String cause = player.getLastDamageCause() == null
                ? null : Dims.name(player.getLastDamageCause().getCause());
        queue.offer(event(player, "player_death").with("cause", cause));
    }

    /** A teleport can cross many region cells at once; observe it rather than wait for a sample. */
    @EventHandler(priority = EventPriority.MONITOR, ignoreCancelled = true)
    public void onTeleport(PlayerTeleportEvent event) {
        sampler.observe(event.getPlayer(), event.getTo());
    }

    @EventHandler(priority = EventPriority.MONITOR)
    public void onChangedWorld(PlayerChangedWorldEvent event) {
        Player player = event.getPlayer();
        queue.offer(event(player, "dimension_change")
                .with("from", Dims.of(event.getFrom()))
                .with("to", Dims.of(player.getWorld())));
        // The destination is a different world entirely, so the old cell means nothing.
        sampler.forget(player.getUniqueId());
        sampler.observe(player, player.getLocation());
    }

    @EventHandler(priority = EventPriority.MONITOR, ignoreCancelled = true)
    public void onConsume(PlayerItemConsumeEvent event) {
        queue.offer(event(event.getPlayer(), "eat").with("item", Dims.item(event.getItem())));
    }

    /**
     * Sleeping, including failed attempts.
     *
     * <p>The result is recorded rather than filtered on. Trying to sleep at noon, or being
     * blocked because monsters are near, says something about what the player wanted; a tap
     * that only records successes would show nothing at all for a player repeatedly trying
     * to skip a night they cannot survive.
     */
    @EventHandler(priority = EventPriority.MONITOR, ignoreCancelled = true)
    public void onBedEnter(PlayerBedEnterEvent event) {
        queue.offer(GameEvent.of(Bukkit.getCurrentTick(), "sleep",
                        event.getPlayer().getUniqueId().toString(),
                        Dims.of(event.getBed().getWorld()),
                        Dims.pos(event.getBed().getLocation()))
                .with("result", Dims.name(event.getBedEnterResult())));
    }

    /**
     * Minecraft's own record of firsts. Recipe unlocks are filtered out: they fire on every
     * new item picked up and would drown the real milestones.
     */
    @EventHandler(priority = EventPriority.MONITOR)
    public void onAdvancement(PlayerAdvancementDoneEvent event) {
        String key = event.getAdvancement().getKey().toString();
        if (key.startsWith(RECIPE_PREFIX)) {
            return;
        }
        queue.offer(event(event.getPlayer(), "advancement").with("advancement", key));
    }

    /**
     * Experience gained. XP is the clearest available proxy for effort spent on mining,
     * smelting, breeding and combat, and nothing else in the tap captures it.
     */
    @EventHandler(priority = EventPriority.MONITOR)
    public void onExpChange(PlayerExpChangeEvent event) {
        if (event.getAmount() == 0) {
            return;
        }
        queue.offer(event(event.getPlayer(), "xp_change")
                .with("amount", event.getAmount())
                .with("total_xp", event.getPlayer().getTotalExperience()));
    }

    @EventHandler(priority = EventPriority.MONITOR)
    public void onLevelChange(PlayerLevelChangeEvent event) {
        queue.offer(event(event.getPlayer(), "level_change")
                .with("from", event.getOldLevel())
                .with("to", event.getNewLevel()));
    }

    @EventHandler(priority = EventPriority.MONITOR)
    public void onRespawn(PlayerRespawnEvent event) {
        queue.offer(GameEvent.of(Bukkit.getCurrentTick(), "respawn",
                        event.getPlayer().getUniqueId().toString(),
                        Dims.of(event.getRespawnLocation().getWorld()),
                        Dims.pos(event.getRespawnLocation()))
                .with("bed", event.isBedSpawn())
                .with("anchor", event.isAnchorSpawn()));
        sampler.forget(event.getPlayer().getUniqueId());
    }

    @EventHandler(priority = EventPriority.MONITOR, ignoreCancelled = true)
    public void onGameModeChange(PlayerGameModeChangeEvent event) {
        queue.offer(event(event.getPlayer(), "gamemode_change")
                .with("from", Dims.name(event.getPlayer().getGameMode()))
                .with("to", Dims.name(event.getNewGameMode())));
    }

    /** A tool wearing out is a real milestone in a survival session. */
    @EventHandler(priority = EventPriority.MONITOR)
    public void onItemBreak(PlayerItemBreakEvent event) {
        queue.offer(event(event.getPlayer(), "item_break")
                .with("item", Dims.item(event.getBrokenItem())));
    }

    /** What the player typed. The command itself, not its effects. */
    @EventHandler(priority = EventPriority.MONITOR, ignoreCancelled = true)
    public void onCommand(PlayerCommandPreprocessEvent event) {
        queue.offer(event(event.getPlayer(), "command")
                .with("text", event.getMessage()));
    }

    private GameEvent event(Player player, String type) {
        return GameEvent.of(Bukkit.getCurrentTick(), type, player.getUniqueId().toString(),
                Dims.of(player.getWorld()), Dims.pos(player.getLocation()));
    }
}
