package io.github.jakegilbert7.mcgod.capture;

import io.papermc.paper.event.player.PlayerTradeEvent;
import org.bukkit.Bukkit;
import org.bukkit.entity.Player;
import org.bukkit.event.EventHandler;
import org.bukkit.event.EventPriority;
import org.bukkit.event.Listener;

/** Villager trading. Paper-only event, kept apart so a Paper API change is easy to find. */
public final class TradeListener implements Listener {

    private final EventQueue queue;

    public TradeListener(EventQueue queue) {
        this.queue = queue;
    }

    @EventHandler(priority = EventPriority.MONITOR, ignoreCancelled = true)
    public void onTrade(PlayerTradeEvent event) {
        Player player = event.getPlayer();
        queue.offer(GameEvent.of(Bukkit.getCurrentTick(), "villager_trade",
                        player.getUniqueId().toString(), Dims.of(player.getWorld()),
                        Dims.pos(player.getLocation()))
                .with("item", Dims.item(event.getTrade().getResult()))
                .with("count", event.getTrade().getResult().getAmount())
                .with("entity", Dims.entity(event.getVillager())));
    }
}
