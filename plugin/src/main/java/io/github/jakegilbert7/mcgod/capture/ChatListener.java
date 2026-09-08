package io.github.jakegilbert7.mcgod.capture;

import io.papermc.paper.event.player.AsyncChatEvent;
import net.kyori.adventure.text.serializer.plain.PlainTextComponentSerializer;
import org.bukkit.entity.Player;
import org.bukkit.event.EventHandler;
import org.bukkit.event.EventPriority;
import org.bukkit.event.Listener;
import org.bukkit.event.block.SignChangeEvent;
import org.bukkit.plugin.Plugin;

/**
 * Text players produce: chat, and what they write on signs.
 *
 * <p>This is the only place a player says anything in their own words, which makes it the
 * richest context the tap can capture for a system whose whole purpose is talking back.
 *
 * <p>Chat fires asynchronously, hence {@link TickClock}. Everything else here is main-thread.
 */
public final class ChatListener implements Listener {

    private static final PlainTextComponentSerializer PLAIN = PlainTextComponentSerializer.plainText();

    private final EventQueue queue;
    private final TickClock clock;
    private final Plugin plugin;

    public ChatListener(EventQueue queue, TickClock clock, Plugin plugin) {
        this.queue = queue;
        this.clock = clock;
        this.plugin = plugin;
    }

    @EventHandler(priority = EventPriority.MONITOR, ignoreCancelled = true)
    public void onChat(AsyncChatEvent event) {
        Player player = event.getPlayer();
        String actor = player.getUniqueId().toString();
        String text = PLAIN.serialize(event.message());
        // AsyncChatEvent is not on the server thread. World and location are Bukkit state,
        // so snapshot them one tick later on the scheduler instead of reading them here.
        plugin.getServer().getScheduler().runTask(plugin, () -> {
            if (!player.isOnline()) {
                return;
            }
            queue.offer(GameEvent.of(clock.tick(), "chat", actor,
                            Dims.of(player.getWorld()), Dims.pos(player.getLocation()))
                    .with("text", text));
        });
    }

    @EventHandler(priority = EventPriority.MONITOR, ignoreCancelled = true)
    public void onSign(SignChangeEvent event) {
        StringBuilder text = new StringBuilder();
        for (net.kyori.adventure.text.Component line : event.lines()) {
            String plain = PLAIN.serialize(line);
            if (!plain.isBlank()) {
                if (!text.isEmpty()) {
                    text.append(" / ");
                }
                text.append(plain);
            }
        }
        if (text.isEmpty()) {
            return;
        }
        queue.offer(GameEvent.of(clock.tick(), "sign_change",
                        event.getPlayer().getUniqueId().toString(),
                        Dims.of(event.getBlock().getWorld()),
                        new int[]{event.getBlock().getX(), event.getBlock().getY(),
                                event.getBlock().getZ()})
                .with("text", text.toString()));
    }
}
