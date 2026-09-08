#!/usr/bin/env python3
"""L3: the belief store.

A single local SQLite file. No server, no account, no network.

Every row carries where it came from. Provenance is an ordered enum and it is the
anti-hallucination backbone of the whole system:

    SCANNED   direct region read        assertable as fact
    OBSERVED  event stream              assertable as fact
    DERIVED   deterministic computation assertable as fact
    INFERRED  model classification      only above a confidence threshold
    ASSERTED  something the god said    never

ASSERTED can never be promoted. That rule is load-bearing: without it the god reads back its
own speculation as evidence, and one confident guess becomes permanent canon. It is enforced
by a database trigger rather than by convention, so no future caller can forget it.
"""

from __future__ import annotations

import json
import sqlite3
import time
from contextlib import contextmanager
from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path

from config import INFERRED_THRESHOLD

DEFAULT_PATH = Path(__file__).parent / "mcgod.db"


class Provenance(IntEnum):
    """Lower is more trustworthy. Order matters; do not renumber casually."""

    SCANNED = 0
    OBSERVED = 1
    DERIVED = 2
    INFERRED = 3
    ASSERTED = 4

    @property
    def assertable(self) -> bool:
        """Whether the god may state this as fact without hedging."""
        return self <= Provenance.DERIVED


#: How long a belief keeps half its confidence, in milliseconds, by what it is about.
#:
#: Different facts rot at different speeds and treating them alike is how a store starts
#: lying. Blocks do not move on their own, so a structure's geometry stays good for a long
#: while. Animals wander and despawn within minutes, so a count of what is inside a pen is
#: nearly worthless an hour later. What a place IS barely changes at all — a coop is still
#: a coop next month — so a name decays slowest of the three.
#:
#: Decay never rewrites the stored confidence. It is applied when the belief is read, so
#: the record of what was measured stays exactly as it was measured.
HALF_LIFE_MS = {
    "holds": 20 * 60_000,             # a live entity count
    "survives": 24 * 3_600_000,       # whether a build still stands
    "players": 60 * 60_000,           # position, health, what they were carrying
    "named": 30 * 24 * 3_600_000,     # what a place is
    "structures": 14 * 24 * 3_600_000,
    "masses": 14 * 24 * 3_600_000,      # measured geometry, same as a structure's
    "role": 30 * 24 * 3_600_000,        # what a mass is for, like a name
    "regions": 6 * 3_600_000,
}
DEFAULT_HALF_LIFE_MS = 7 * 24 * 3_600_000

#: Actor values that are not players. `world` is the game itself — weather, explosions, a
#: mob killing another mob. `god` is this program acting through a command. Neither has a
#: profile, an inventory or a place, so neither may create a row in `players`, an activity,
#: or a region cell; the god's command in particular carries no meaningful position.
NON_PLAYER_ACTORS = frozenset({"world", "god"})

#: What a belief's confidence drops to the moment something contradicts it. Not zero: the
#: measurement still happened and is still the best guess until someone looks again.
CONTRADICTED_CONFIDENCE = 0.05

PROVENANCE_VALUES = ", ".join(f"'{p.name}'" for p in Provenance)

# Columns every belief row carries, whatever it is a belief about.
BELIEF_COLUMNS = f"""
    value             TEXT,
    confidence        REAL    NOT NULL DEFAULT 1.0
                              CHECK (confidence >= 0.0 AND confidence <= 1.0),
    provenance        TEXT    NOT NULL CHECK (provenance IN ({PROVENANCE_VALUES})),
    first_seen_tick   INTEGER,
    verified_at_tick  INTEGER,
    verified_at_ms    INTEGER,
    source_event_ids  TEXT
"""

TABLES = {
    "structures": """
        id TEXT PRIMARY KEY, actor TEXT, dim TEXT, name TEXT,
        min_x INTEGER, min_y INTEGER, min_z INTEGER,
        max_x INTEGER, max_y INTEGER, max_z INTEGER,
        materials TEXT
    """,
    # Generator-authored world knowledge is deliberately disjoint from player works. A
    # village may be relevant to the player, but it must never become one of "their builds"
    # merely because a scan saw its blocks. If they edit one, the edit remains work history.
    "generated_features": """
        id TEXT PRIMARY KEY, dim TEXT, kind TEXT,
        x INTEGER, y INTEGER, z INTEGER, discovered_from TEXT,
        min_x INTEGER, min_y INTEGER, min_z INTEGER,
        max_x INTEGER, max_y INTEGER, max_z INTEGER,
        materials TEXT
    """,
    # Seed-backed biome observations. Coordinates are internal evidence and are never
    # rendered into the god's speech; chat receives cardinal descriptions instead.
    "terrain_regions": """
        id TEXT PRIMARY KEY, dim TEXT, center_x INTEGER, center_z INTEGER,
        radius INTEGER, step INTEGER, biomes TEXT
    """,
    # Historical evidence that somebody placed or removed blocks.  These used to share
    # the structures table with present-tense objects.  That made a demolished wall, the
    # hole beneath a house, and the house itself three competing versions of one fact.
    # Work is immutable history; structures are what a fresh world read says stands now.
    "work_events": """
        id TEXT PRIMARY KEY, kind TEXT NOT NULL, actor TEXT, dim TEXT,
        min_x INTEGER, min_y INTEGER, min_z INTEGER,
        max_x INTEGER, max_y INTEGER, max_z INTEGER,
        materials TEXT
    """,
    # The complete present-tense ledger of built things: one row per connected mass of
    # non-natural blocks in every patch of ground that has been read, whatever its size and
    # whoever put it there. Deterministic and model-free, so nothing a model declines to
    # name can vanish from the record. A landmark in ``structures`` is a named grouping of
    # these; ``place_id`` points at the structure or generated feature a mass belongs to.
    "masses": """
        id TEXT PRIMARY KEY, dim TEXT, actor TEXT, place_id TEXT,
        min_x INTEGER, min_y INTEGER, min_z INTEGER,
        max_x INTEGER, max_y INTEGER, max_z INTEGER,
        materials TEXT
    """,
    "farms": """
        id TEXT PRIMARY KEY, actor TEXT, dim TEXT, kind TEXT,
        min_x INTEGER, min_y INTEGER, min_z INTEGER,
        max_x INTEGER, max_y INTEGER, max_z INTEGER,
        yield_profile TEXT
    """,
    "players": """
        id TEXT PRIMARY KEY, name TEXT, dim TEXT,
        last_seen_tick INTEGER, profile TEXT
    """,
    "quests": """
        id TEXT PRIMARY KEY, actor TEXT, spec TEXT, state TEXT,
        issued_tick INTEGER, deadline_tick INTEGER, resolved_tick INTEGER
    """,
    "watchers": """
        id TEXT PRIMARY KEY, quest_id TEXT, predicate TEXT, state TEXT,
        registered_tick INTEGER, tripped_tick INTEGER
    """,
    "episodes": """
        id TEXT PRIMARY KEY, actor TEXT, dim TEXT,
        tick_start INTEGER, tick_end INTEGER, t_start INTEGER, t_end INTEGER,
        min_x INTEGER, min_y INTEGER, min_z INTEGER,
        max_x INTEGER, max_y INTEGER, max_z INTEGER,
        gross_placed INTEGER, gross_broken INTEGER, net_total INTEGER,
        net_by_material TEXT, tools TEXT, event_counts TEXT, path_blocks REAL
    """,
    "utterances": """
        id TEXT PRIMARY KEY, actor TEXT, tick INTEGER, speech TEXT, trigger TEXT
    """,
    # Chronological observed deeds. Episodes summarize sustained work; this ledger keeps
    # the exact recent actions needed for questions such as "what did I just do?".
    "activities": """
        id TEXT PRIMARY KEY, actor TEXT, dim TEXT, tick INTEGER, event_t INTEGER,
        kind TEXT, place_id TEXT, x INTEGER, y INTEGER, z INTEGER, detail TEXT
    """,
    # Lossless searchable mirror of the JSONL event source. Activities are interpretations
    # of deeds; this is the low-level evidence they can always be rebuilt from. Keeping the
    # two separate prevents a new summarizer from depending on what an older one considered
    # salient enough to retain.
    "event_history": """
        id TEXT PRIMARY KEY, actor TEXT, dim TEXT, tick INTEGER, event_t INTEGER,
        kind TEXT, place_id TEXT, x INTEGER, y INTEGER, z INTEGER, payload TEXT
    """,
    # Current membership in a generator-authored place.  This is deliberately separate
    # from activities: an arrival is history, while presence is replaceable state.  Keeping
    # it makes "found a village" mean an actual outside -> inside transition rather than
    # whichever movement sample happened to fall inside a distance threshold.
    "actor_presence": """
        id TEXT PRIMARY KEY, actor TEXT, dim TEXT, place_id TEXT, event_t INTEGER
    """,
    "relationships": """
        id TEXT PRIMARY KEY, subject TEXT, predicate TEXT, object TEXT
    """,
    # One row per region cell a player has actually occupied. This is the scan work queue,
    # and it is deliberately not a map of the world: nothing changes where nobody is,
    # because Minecraft does not tick unloaded chunks. Scanning exists to correct beliefs
    # about places someone has been, never to discover places nobody has.
    "regions": """
        id TEXT PRIMARY KEY, dim TEXT, cx INTEGER, cz INTEGER,
        last_activity_t INTEGER, last_scanned_t INTEGER,
        last_activity_ms INTEGER, last_scanned_ms INTEGER,
        dirty INTEGER DEFAULT 0, dirty_reason TEXT, actors TEXT
    """,
}


@dataclass(frozen=True)
class Belief:
    """The provenance envelope around any stored fact."""

    provenance: Provenance
    confidence: float = 1.0
    first_seen_tick: int | None = None
    verified_at_tick: int | None = None
    verified_at_ms: int | None = None
    source_event_ids: tuple[str, ...] = ()
    value: str | None = None

    @property
    def assertable(self) -> bool:
        """Whether the god may state this without hedging."""
        if self.provenance == Provenance.ASSERTED:
            return False
        if self.provenance == Provenance.INFERRED:
            return self.confidence >= INFERRED_THRESHOLD
        return True


class Store:
    def __init__(self, path: Path | str = DEFAULT_PATH) -> None:
        self.path = Path(path)
        self.db = sqlite3.connect(self.path)
        self.db.row_factory = sqlite3.Row
        self._transaction_depth = 0
        self._savepoint_serial = 0
        self.db.execute("PRAGMA foreign_keys = ON")
        self.db.execute("PRAGMA journal_mode = WAL")
        self._create()

    def _create(self) -> None:
        for table, columns in TABLES.items():
            self.db.execute(
                f"CREATE TABLE IF NOT EXISTS {table} ({columns}, {BELIEF_COLUMNS})")
            # The rule that makes the whole provenance scheme mean anything. Enforced here
            # rather than in Python so that no caller, present or future, can route around it.
            self.db.execute(f"""
                CREATE TRIGGER IF NOT EXISTS {table}_asserted_is_final
                BEFORE UPDATE ON {table}
                FOR EACH ROW WHEN OLD.provenance = 'ASSERTED'
                             AND NEW.provenance <> 'ASSERTED'
                BEGIN
                    SELECT RAISE(ABORT,
                        'ASSERTED provenance can never be promoted');
                END
            """)
            self.db.execute(f"""
                CREATE TRIGGER IF NOT EXISTS {table}_asserted_cannot_be_replaced
                BEFORE INSERT ON {table}
                FOR EACH ROW WHEN NEW.provenance <> 'ASSERTED'
                             AND EXISTS (
                                 SELECT 1 FROM {table}
                                 WHERE id = NEW.id AND provenance = 'ASSERTED')
                BEGIN
                    SELECT RAISE(ABORT,
                        'ASSERTED provenance can never be promoted');
                END
            """)
            # SQLite's CREATE IF NOT EXISTS does not add columns to an existing database.
            # Keep the migration mechanical and additive so old worlds open safely.
            columns_now = {r["name"] for r in self.db.execute(
                f"PRAGMA table_info({table})")}
            if "verified_at_ms" not in columns_now:
                self.db.execute(f"ALTER TABLE {table} ADD COLUMN verified_at_ms INTEGER")
            if table == "regions":
                for name in ("last_activity_ms", "last_scanned_ms"):
                    if name not in columns_now:
                        self.db.execute(f"ALTER TABLE regions ADD COLUMN {name} INTEGER")
            if table == "generated_features":
                additions = {
                    "min_x": "INTEGER", "min_y": "INTEGER", "min_z": "INTEGER",
                    "max_x": "INTEGER", "max_y": "INTEGER", "max_z": "INTEGER",
                    "materials": "TEXT",
                }
                for name, kind in additions.items():
                    if name not in columns_now:
                        self.db.execute(
                            f"ALTER TABLE generated_features ADD COLUMN {name} {kind}")
        self.db.execute(
            "CREATE INDEX IF NOT EXISTS episodes_actor ON episodes(actor, tick_start)")
        self.db.execute(
            "CREATE INDEX IF NOT EXISTS structures_dim ON structures(dim, min_x, min_z)")
        self.db.execute(
            "CREATE INDEX IF NOT EXISTS work_events_dim ON work_events(dim, min_x, min_z)")
        self.db.execute(
            "CREATE INDEX IF NOT EXISTS masses_dim ON masses(dim, min_x, min_z)")
        self.db.execute(
            "CREATE INDEX IF NOT EXISTS masses_place ON masses(place_id)")
        self.db.execute(
            "CREATE INDEX IF NOT EXISTS generated_features_dim "
            "ON generated_features(dim, x, z)")
        self.db.execute(
            "CREATE INDEX IF NOT EXISTS terrain_regions_dim "
            "ON terrain_regions(dim, center_x, center_z)")
        self.db.execute(
            "CREATE INDEX IF NOT EXISTS activities_actor_time "
            "ON activities(actor, event_t)")
        self.db.execute(
            "CREATE INDEX IF NOT EXISTS event_history_actor_time "
            "ON event_history(actor, event_t)")
        self.db.execute(
            "CREATE INDEX IF NOT EXISTS event_history_kind_time "
            "ON event_history(kind, event_t)")
        self.db.execute(
            "CREATE INDEX IF NOT EXISTS event_history_place_time "
            "ON event_history(place_id, event_t)")
        self.db.execute(
            "CREATE INDEX IF NOT EXISTS event_history_space_time "
            "ON event_history(dim, x, z, event_t)")
        self._migrate_historical_structures()
        # Legacy spatial rows were labelled DERIVED even though a model chose their
        # boundaries.  Correct the provenance once; fresh rows carry the schema marker.
        self.db.execute("""
            UPDATE structures SET provenance = 'INFERRED'
            WHERE id LIKE 's_%'
              AND (value IS NULL OR value NOT LIKE '%\"schema_version\"%')
        """)
        self._normalize_current_structures()
        self.db.commit()

    def commit(self) -> None:
        """Commit unless an enclosing unit of work owns the transaction."""
        if self._transaction_depth == 0:
            self.db.commit()

    @contextmanager
    def transaction(self):
        """Make a group of store mutations one recoverable state transition.

        Store methods normally commit immediately, which is convenient for individual
        observations but unsafe for a segmentation pass: five objects written and the
        sixth failing would leave a world representation assembled from two different
        observations. Nested callers use savepoints so this remains safe as larger units
        of work are introduced.
        """
        outer = self._transaction_depth == 0
        savepoint = None
        if outer:
            self.db.execute("BEGIN IMMEDIATE")
        else:
            self._savepoint_serial += 1
            savepoint = f"mcgod_{self._savepoint_serial}"
            self.db.execute(f"SAVEPOINT {savepoint}")
        self._transaction_depth += 1
        try:
            yield
        except BaseException:
            self._transaction_depth -= 1
            if outer:
                self.db.rollback()
            else:
                self.db.execute(f"ROLLBACK TO {savepoint}")
                self.db.execute(f"RELEASE {savepoint}")
            raise
        else:
            self._transaction_depth -= 1
            if outer:
                self.db.commit()
            else:
                self.db.execute(f"RELEASE {savepoint}")

    def _migrate_historical_structures(self) -> int:
        """Move legacy placement/removal rows out of present-tense world state.

        This migration is deliberately lossless: every belief column and source id is
        copied before the old row is removed.  Relationships may continue to point at a
        work id (for example a quest answered by that work), but context only enumerates
        current structures.
        """
        rows = self.db.execute(
            "SELECT * FROM structures WHERE id LIKE 'placement_%' OR id LIKE 'removal_%'"
        ).fetchall()
        if not rows:
            return 0
        target = [r["name"] for r in self.db.execute("PRAGMA table_info(work_events)")]
        marks = ", ".join("?" for _ in target)
        names = ", ".join(target)
        for row in rows:
            kind = "placement" if row["id"].startswith("placement_") else "removal"
            values = []
            for name in target:
                values.append(kind if name == "kind" else row[name])
            self.db.execute(
                f"INSERT OR IGNORE INTO work_events ({names}) VALUES ({marks})", values)
        self.db.execute(
            "DELETE FROM structures WHERE id LIKE 'placement_%' OR id LIKE 'removal_%'")
        return len(rows)

    def _normalize_current_structures(self) -> int:
        """Bring pre-v2 spatial rows under the canonical current-object contract."""
        changed = 0
        for row in self.db.execute(
                "SELECT * FROM structures WHERE id LIKE 's_%'").fetchall():
            try:
                value = json.loads(row["value"] or "{}")
            except (TypeError, ValueError):
                value = {}
            if value.get("schema_version") == 2 and value.get("kind") == "current_structure":
                continue
            observed = row["verified_at_ms"] or value.get("observed_at_ms") \
                or value.get("when") or int(time.time() * 1000)
            value.update(schema_version=2, kind="current_structure",
                         observed_at_ms=observed)
            self.db.execute(
                "UPDATE structures SET value = ?, verified_at_ms = ?, provenance = 'INFERRED' "
                "WHERE id = ?", (json.dumps(value), observed, row["id"]))
            self.db.execute(
                "UPDATE relationships SET verified_at_ms = COALESCE(verified_at_ms, ?) "
                "WHERE subject = ?", (observed, row["id"]))
            changed += 1
        return changed

    def put(self, table: str, row: dict, belief: Belief) -> None:
        """Inserts or replaces one row with its provenance envelope.

        Refuses to overwrite a row with one of strictly worse provenance. A scan should not
        be clobbered by a guess.
        """
        if table not in TABLES:
            raise ValueError(f"unknown table: {table}")

        existing = self.db.execute(
            f"SELECT provenance FROM {table} WHERE id = ?", (row["id"],)).fetchone()
        if existing is not None:
            current = Provenance[existing["provenance"]]
            if belief.provenance > current:
                return  # keep the better-sourced row
            if current == Provenance.ASSERTED and belief.provenance != Provenance.ASSERTED:
                raise sqlite3.IntegrityError(
                    "ASSERTED provenance can never be promoted")

        payload = dict(row)
        payload.update(
            value=belief.value,
            confidence=belief.confidence,
            provenance=belief.provenance.name,
            first_seen_tick=belief.first_seen_tick,
            verified_at_tick=belief.verified_at_tick,
            verified_at_ms=belief.verified_at_ms,
            source_event_ids=json.dumps(list(belief.source_event_ids)),
        )
        columns = ", ".join(payload)
        marks = ", ".join("?" for _ in payload)
        # REPLACE is a delete followed by an insert in SQLite.  Besides disturbing row
        # identity, that can route around BEFORE UPDATE triggers such as ASSERTED-is-final.
        # A real UPSERT preserves both the row and every database-level invariant.
        updates = ", ".join(
            f"{column} = excluded.{column}" for column in payload if column != "id")
        self.db.execute(
            f"INSERT INTO {table} ({columns}) VALUES ({marks}) "
            f"ON CONFLICT(id) DO UPDATE SET {updates}", tuple(payload.values()))
        self.commit()

    def get(self, table: str, row_id: str) -> sqlite3.Row | None:
        return self.db.execute(
            f"SELECT * FROM {table} WHERE id = ?", (row_id,)).fetchone()

    def all(self, table: str, where: str = "", params: tuple = ()) -> list[sqlite3.Row]:
        clause = f" WHERE {where}" if where else ""
        return self.db.execute(f"SELECT * FROM {table}{clause}", params).fetchall()

    def name_structure(self, structure_id: str, label: str, confidence: float,
                       rationale: str, tick: int | None, key: str,
                       category: str = "other", saw: bool = False,
                       description: str = "") -> None:
        """Records what a model thinks a structure is.

        A name is a separate belief from the geometry it describes, and they do not share a
        provenance. The bounding box and material census are DERIVED — deterministic
        computation over observed events. The name is INFERRED — a model's guess. Writing
        the name onto the structure row would have meant one provenance column standing for
        two beliefs of different quality, and `put()` correctly refused the write, silently
        discarding every classification. Names live in `relationships` so each belief carries
        its own provenance, decays on its own schedule, and a later scan can sharpen the
        geometry without disturbing the name or vice versa.
        """
        self.put("relationships", {
            "id": f"named:{structure_id}",
            "subject": structure_id,
            "predicate": "named",
            "object": label,
        }, Belief(provenance=Provenance.INFERRED, confidence=confidence,
                  verified_at_tick=tick, verified_at_ms=int(time.time() * 1000),
                  source_event_ids=(key,),
                  value=json.dumps({"category": category, "rationale": rationale,
                                    # A sentence about what this place is, kept and
                                    # rewritten as the thing changes. A name says what it
                                    # is; the description says what is true of it, and
                                    # carrying it forward is what lets the god speak about
                                    # a building it has known for weeks rather than one it
                                    # re-derives from a block list every time.
                                    "description": description,
                                    # Whether this name came from looking at the thing or
                                    # from reasoning about its measurements. Both are
                                    # INFERRED; one is much better evidence than the other.
                                    "saw_it": saw})))

    def record_utterance(self, actor: str, tick: int, speech: str, trigger: str) -> None:
        """Writes down something the god said.

        Always ASSERTED, which can never be promoted. This is the one table whose whole
        purpose is to hold things that are NOT evidence: it lets the god see what it told
        someone a minute ago without that becoming a fact it can cite back.
        """
        self.put("utterances", {
            "id": f"{actor}:{tick}:{abs(hash(speech)) % 10 ** 8}",
            "actor": actor, "tick": tick, "speech": speech, "trigger": trigger,
        }, Belief(provenance=Provenance.ASSERTED, first_seen_tick=tick,
                  verified_at_tick=tick))

    def recent_utterances(self, actor: str, limit: int = 6) -> list:
        return list(self.db.execute(
            "SELECT * FROM utterances WHERE actor = ? ORDER BY tick DESC LIMIT ?",
            (actor, limit)))

    def all_names(self) -> dict:
        """Every stored name, keyed by subject. One query instead of N."""
        return {r["subject"]: r for r in
                self.db.execute("SELECT * FROM relationships WHERE predicate = 'named'")}

    def structure_description(self, structure_id: str) -> str:
        row = self.get("relationships", f"named:{structure_id}")
        if row is None:
            return ""
        try:
            return (json.loads(row["value"] or "{}") or {}).get("description") or ""
        except (TypeError, ValueError):
            return ""

    def structure_name(self, structure_id: str):
        """The stored name for a structure, or None."""
        return self.get("relationships", f"named:{structure_id}")

    @staticmethod
    def half_life(table: str, predicate: str | None = None) -> int:
        return HALF_LIFE_MS.get(predicate or "", HALF_LIFE_MS.get(table,
                                                                  DEFAULT_HALF_LIFE_MS))

    def confidence_now(self, table: str, row, now_ms: int | None = None) -> float:
        """Stored confidence, decayed by how long it has been since anyone checked.

        Applied on read, never written back: the record of what was measured must stay what
        was measured. A belief nobody has confirmed for a long time is not wrong, it is
        merely old, and the difference matters — old is recoverable by looking again.
        """
        if row is None:
            return 0.0
        now_ms = int(time.time() * 1000) if now_ms is None else now_ms
        seen = self._seen_at(row)
        if not seen:
            return float(row["confidence"])
        predicate = row["predicate"] if "predicate" in row.keys() else None
        age = max(0, now_ms - seen)
        return float(row["confidence"]) * (0.5 ** (age / self.half_life(table, predicate)))

    @staticmethod
    def _seen_at(row) -> int:
        """When this belief was last confirmed, in wall-clock milliseconds."""
        for key in ("verified_at_ms", "verified_at_t"):
            if key in row.keys() and row[key]:
                return int(row[key])
        try:
            value = json.loads(row["value"] or "{}")
        except (TypeError, ValueError):
            return 0
        if isinstance(value, dict):
            for key in ("when", "t", "seen_at"):
                if value.get(key):
                    return int(value[key])
        return 0

    def assertable(self, table: str, row_id: str, now_ms: int | None = None) -> bool:
        """Whether the god may state this row as fact. The gate before it speaks."""
        row = self.get(table, row_id)
        if row is None:
            return False
        provenance = Provenance[row["provenance"]]
        if provenance == Provenance.ASSERTED:
            return False
        if provenance == Provenance.INFERRED:
            return self.confidence_now(table, row, now_ms) >= INFERRED_THRESHOLD
        return True

    def contradict(self, table: str, row_id: str, reason: str, tick: int = 0) -> bool:
        """Something happened that this belief cannot survive. Mark it, do not rewrite it.

        Decay handles beliefs that merely went quiet. This handles the other case: an event
        that directly contradicts one. Someone killed the animals we counted, broke the
        blocks we measured, emptied the chest we listed — the belief is wrong NOW, and
        waiting for a half-life to expire would leave the god confidently reciting a flock
        that has been eaten.

        Confidence is driven to the floor rather than the belief being edited, for the same
        reason decay never writes back: what was measured stays what was measured, and only
        looking again can replace it. The row keeps its provenance, so a scan that has been
        contradicted is still a scan — an old one, known to be out of date.
        """
        row = self.get(table, row_id)
        if row is None:
            return False
        try:
            value = json.loads(row["value"] or "{}")
        except (TypeError, ValueError):
            value = {}
        if not isinstance(value, dict):
            value = {"value": value}
        if "contradicted" not in value:
            value["confidence_before"] = float(row["confidence"])
        value["contradicted"] = reason
        value["contradicted_tick"] = tick
        self.db.execute(
            f"UPDATE {table} SET confidence = ?, value = ? WHERE id = ?",
            (CONTRADICTED_CONFIDENCE, json.dumps(value), row_id))
        self.commit()
        return True

    def reaffirm(self, table: str, row_id: str, verified_at_ms: int,
                 tick: int | None = None) -> bool:
        """A contradicted belief has been looked at again and still holds.

        The mirror of ``contradict``: the confidence it took away comes back, the mark is
        cleared, and the verification time moves to now. Only a fresh measurement may call
        this; nothing else can un-contradict a belief.
        """
        row = self.get(table, row_id)
        if row is None:
            return False
        try:
            value = json.loads(row["value"] or "{}")
        except (TypeError, ValueError):
            value = {}
        if not isinstance(value, dict) or "contradicted" not in value:
            return False
        confidence = float(value.pop("confidence_before", row["confidence"]))
        value.pop("contradicted", None)
        value.pop("contradicted_tick", None)
        self.db.execute(
            f"UPDATE {table} SET confidence = ?, value = ?, verified_at_ms = ?, "
            f"verified_at_tick = COALESCE(?, verified_at_tick) WHERE id = ?",
            (confidence, json.dumps(value), verified_at_ms, tick, row_id))
        self.commit()
        return True

    def contradicted(self, table: str, row_id: str) -> str | None:
        row = self.get(table, row_id)
        if row is None:
            return None
        try:
            return (json.loads(row["value"] or "{}") or {}).get("contradicted")
        except (TypeError, ValueError):
            return None

    def stale(self, table: str, row_id: str, floor: float = 0.5,
              now_ms: int | None = None) -> bool:
        """Has a belief decayed far enough that it should be looked at again?"""
        row = self.get(table, row_id)
        return row is not None and self.confidence_now(table, row, now_ms) < floor

    def record_episodes(self, episodes, source: str) -> int:
        """Persists L1 episodes. Deterministic computation, hence DERIVED."""
        for ep in episodes:
            bbox = ep.bbox
            (lo, hi) = bbox if bbox else ((None,) * 3, (None,) * 3)
            self.put("episodes", {
                "id": ep.id, "actor": ep.actor, "dim": ep.dim,
                "tick_start": ep.tick_start, "tick_end": ep.tick_end,
                "t_start": ep.t_start, "t_end": ep.t_end,
                "min_x": lo[0], "min_y": lo[1], "min_z": lo[2],
                "max_x": hi[0], "max_y": hi[1], "max_z": hi[2],
                "gross_placed": ep.gross_placed, "gross_broken": ep.gross_broken,
                "net_total": ep.net_total,
                "net_by_material": json.dumps(dict(ep.net)),
                "tools": json.dumps(dict(ep.tools)),
                "event_counts": json.dumps(dict(ep.kinds)),
                "path_blocks": ep.path,
            }, Belief(
                provenance=Provenance.DERIVED,
                first_seen_tick=ep.tick_start,
                verified_at_tick=ep.tick_end,
                source_event_ids=(source,),
            ))
        return len(episodes)

    def close(self) -> None:
        self.db.close()
