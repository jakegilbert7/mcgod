package io.github.jakegilbert7.mcgod.capture;

import java.io.BufferedWriter;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.concurrent.atomic.AtomicLong;
import java.util.logging.Level;
import java.util.logging.Logger;

/**
 * Appends one JSON object per line to a session file. One file per server run, so a session
 * file is a self-contained replay input for P3.
 */
public final class JsonlSink implements EventSink {

    private static final DateTimeFormatter STAMP = DateTimeFormatter.ofPattern("yyyyMMdd-HHmmss");

    private final Path file;
    private final Logger log;
    private final AtomicLong written = new AtomicLong();
    private final AtomicLong failed = new AtomicLong();
    private BufferedWriter out;

    public JsonlSink(Path directory, Logger log) throws IOException {
        this.file = directory.resolve("session-" + STAMP.format(LocalDateTime.now()) + ".jsonl");
        this.log = log;
        Files.createDirectories(directory);
        this.out = Files.newBufferedWriter(file, StandardCharsets.UTF_8,
                StandardOpenOption.CREATE, StandardOpenOption.APPEND);
        log.info("Capturing events to " + file);
    }

    @Override
    public String name() {
        return "jsonl";
    }

    @Override
    public void accept(String json) {
        if (out == null) {
            failed.incrementAndGet();
            return;
        }
        try {
            out.write(json);
            out.write('\n');
            written.incrementAndGet();
        } catch (IOException e) {
            record(e);
        }
    }

    @Override
    public void flush() {
        if (out == null) {
            return;
        }
        try {
            out.flush();
        } catch (IOException e) {
            record(e);
        }
    }

    @Override
    public void close() {
        if (out == null) {
            return;
        }
        try {
            out.flush();
            out.close();
        } catch (IOException e) {
            record(e);
        }
        out = null;
        log.info("Wrote " + written.get() + " events to " + file);
    }

    private void record(IOException e) {
        if (failed.getAndIncrement() == 0) {
            log.log(Level.SEVERE, "Event write failed; further errors will only be counted", e);
        }
    }

    public Path file() {
        return file;
    }

    public long written() {
        return written.get();
    }

    @Override
    public long failed() {
        return failed.get();
    }
}
