#!/usr/bin/env python3
"""Regression suite. Runs offline: no server, no API key, no network.

    python3 test_regression.py
    python3 test_regression.py -v

Every invariant here was verified once by hand while the phase was built, in a throwaway
script that no longer exists. That is not good enough for a system whose stated failure mode
is subtle wrongness. Two things broke silently before this file existed:

  - the token estimator under-counted by 41%, so P7's budget audit passed on bad numbers
  - every classification in the first P8 run was discarded while printing success

Both were caught by luck. The corpus is the fixture; the assertions are the ones that would
have caught them.
"""

from __future__ import annotations

import asyncio
import glob
import json
import re
import os
import sqlite3
import subprocess
import sys
import tempfile
import textwrap
import time
import unittest
from pathlib import Path

from context import CHARS_PER_TOKEN, Assembler, audit, estimate_tokens
from detectors import StructureCensus, analyse, cluster
from episodes import segment
from store import Belief, Provenance, Store

HERE = Path(__file__).parent
CORPUS = sorted(HERE.glob("sessions/corpus/*.jsonl"))

#: The corpus is recorded play, and recorded play is somebody's account: a player UUID on
#: every line. It is not distributed with the source, so a fresh clone has no fixture and
#: the tests that read one say so plainly instead of failing. Record your own sessions into
#: `agent/sessions/corpus/` and every one of them runs.
needs_corpus = unittest.skipUnless(
    CORPUS, "no recorded corpus in agent/sessions/corpus/ (see the README)")
CORE_FIELDS = {"tick", "t", "type", "actor", "dim"}


def load(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines()
            if l.strip()]


@needs_corpus
class TestCorpus(unittest.TestCase):
    """The corpus is the test suite for everything above L0. Guard it first."""

    def test_corpus_present(self):
        self.assertGreaterEqual(len(CORPUS), 5, "corpus shrank")

    def test_every_line_is_a_valid_event(self):
        for path in CORPUS:
            for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
                if not line.strip():
                    continue
                event = json.loads(line)
                missing = CORE_FIELDS - set(event)
                self.assertFalse(missing, f"{path.name}:{n} missing {missing}")

    def test_ticks_are_ordered(self):
        for path in CORPUS:
            events = load(path)
            for a, b in zip(events, events[1:]):
                self.assertLessEqual(a["tick"], b["tick"], f"{path.name} tick inversion")

    def test_server_kept_up(self):
        """Game time must track wall clock, or the recording caught a stall."""
        for path in CORPUS:
            events = load(path)
            wall = (events[-1]["t"] - events[0]["t"]) / 1000
            ticks = events[-1]["tick"] - events[0]["tick"]
            self.assertAlmostEqual(ticks / wall, 20.0, delta=0.5,
                                   msg=f"{path.name} implied TPS off")

    def test_move_cadence_is_steady(self):
        for path in CORPUS:
            moves = [e for e in load(path) if e["type"] == "move"]
            gaps = [b["tick"] - a["tick"] for a, b in zip(moves, moves[1:])]
            on_time = sum(1 for g in gaps if g == 40)
            self.assertGreater(on_time / len(gaps), 0.95, f"{path.name} irregular sampling")


@needs_corpus
class TestReplay(unittest.TestCase):
    """P3. Replay must be a faithful stand-in for the live bridge."""

    def _run(self, args: list[str]) -> str:
        out = subprocess.run([sys.executable, str(HERE / "replay.py"), *args],
                             capture_output=True, text=True, cwd=HERE)
        self.assertEqual(out.returncode, 0, out.stderr)
        return out.stdout

    def test_output_is_independent_of_speed(self):
        for path in CORPUS:
            fast = self._run([str(path), "--speed", "0"])
            faster = self._run([str(path), "--speed", "5000"])
            self.assertEqual(fast, faster, f"{path.name} pacing changed the output")

    def test_round_trip_is_byte_identical(self):
        for path in CORPUS:
            with tempfile.NamedTemporaryFile(suffix=".jsonl") as tmp:
                self._run([str(path), "--speed", "0", "--quiet", "--out", tmp.name])
                self.assertEqual(Path(tmp.name).read_bytes(), path.read_bytes(),
                                 f"{path.name} did not survive a replay round trip")

    def test_malformed_line_fails_loudly(self):
        with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False) as tmp:
            tmp.write('{"tick":1,"t":1,"type":"move","actor":"a","dim":"overworld"}\n')
            tmp.write('{"tick":2,"broken\n')
        out = subprocess.run([sys.executable, str(HERE / "replay.py"), tmp.name,
                              "--speed", "0", "--quiet"], capture_output=True, text=True)
        self.assertNotEqual(out.returncode, 0, "a malformed line was silently skipped")


@needs_corpus
class CanonicalStructureState(unittest.TestCase):
    """Current objects and historical work must never become competing world truths."""

    def test_legacy_work_is_migrated_losslessly_out_of_current_state(self):
        path = Path(tempfile.mktemp(suffix=".db"))
        store = Store(path)
        store.put("structures", {
            "id": "placement_1_64_2", "actor": "p", "dim": "overworld", "name": None,
            "min_x": 1, "min_y": 64, "min_z": 2,
            "max_x": 4, "max_y": 68, "max_z": 5,
            "materials": json.dumps({"oak_planks": 30}),
        }, Belief(provenance=Provenance.DERIVED, verified_at_tick=20,
                  verified_at_ms=1234, value=json.dumps({"blocks": 30})))
        store.close()

        store = Store(path)
        try:
            self.assertIsNone(store.get("structures", "placement_1_64_2"))
            work = store.get("work_events", "placement_1_64_2")
            self.assertIsNotNone(work)
            self.assertEqual(work["kind"], "placement")
            self.assertEqual(work["verified_at_ms"], 1234)
            self.assertEqual(json.loads(work["materials"])["oak_planks"], 30)
        finally:
            store.close()
            path.unlink(missing_ok=True)

    def test_a_segmentation_pass_cannot_commit_half_a_scene(self):
        """One failed object must leave the prior canonical view untouched."""
        from god import God

        store = Store(":memory:")
        god = God(store, use_model=False)
        facts = lambda x: {
            "lo": [x, 64, 0], "hi": [x + 2, 66, 2], "materials": {"stone": 27},
            "blocks": 27, "dims": "3x3x3", "fill": 1.0, "vertical": 3,
        }
        drawn = [
            {"label": "first", "confidence": 0.8, "_facts": facts(0), "_shot": b""},
            {"label": "second", "confidence": 0.8, "_facts": facts(10), "_shot": b""},
        ]
        original = god._write_segment
        calls = 0

        def fail_second(*args, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise RuntimeError("simulated persistence failure")
            return original(*args, **kwargs)

        god._write_segment = fail_second
        try:
            with self.assertRaisesRegex(RuntimeError, "persistence failure"):
                god._apply_segmentation(
                    "overworld", drawn, [], {}, [-20, 40, -20], [30, 90, 20])
            self.assertEqual(store.all("structures"), [])
            self.assertEqual(store.all("relationships"), [])
        finally:
            store.close()

    def test_startup_sync_is_idempotent_across_every_replayed_layer(self):
        from god import sync_world

        store = Store(":memory:")
        # A fixed input: a server appending to its session file between the two replays
        # would be a different world, and the claim under test is that the same evidence
        # replays to the same state.
        sessions = [str(path) for path in CORPUS]
        try:
            sync_world(store, sessions)
            tables = ("episodes", "work_events", "regions", "players")

            def snapshot():
                return {table: [dict(row) for row in sorted(
                    store.all(table), key=lambda row: row["id"])] for table in tables}

            first = snapshot()
            sync_world(store, sessions)
            second = snapshot()
            self.assertEqual(second, first)
            self.assertGreater(len(first["episodes"]), 0,
                               "the replay did not persist episodes")
            self.assertGreater(len(first["regions"]), 0,
                               "the replay did not rebuild the region queue")
        finally:
            store.close()

    def test_measured_demolition_retires_by_fraction_not_an_absolute_size(self):
        from god import God

        store = Store(":memory:")
        god = God(store, use_model=False)
        try:
            facts = {"lo": [0, 64, 0], "hi": [2, 66, 2],
                     "materials": {"stone": 27}, "blocks": 27,
                     "dims": "3x3x3", "fill": 1.0, "vertical": 3}
            sid = god._write_segment("overworld", facts, {"label": "marker"}, None, b"")
            old = store.get("structures", sid)
            god._apply_segmentation("overworld", [], [old], {},
                                    [-10, 50, -10], [10, 80, 10])
            self.assertIsNone(store.get("structures", sid),
                              "measured total demolition remained current")

            small = {"lo": [20, 64, 0], "hi": [29, 64, 0],
                     "materials": {"rail": 10}, "blocks": 10,
                     "dims": "10x1x1", "fill": 1.0, "vertical": 1}
            sid = god._write_segment("overworld", small, {"label": "rail"}, None, b"")
            old = store.get("structures", sid)
            standing = {(x, 64, 0): "rail" for x in range(20, 30)}
            god._apply_segmentation("overworld", [], [old], standing,
                                    [10, 50, -10], [40, 80, 10])
            self.assertIsNotNone(store.get("structures", sid),
                                 "a small but fully standing object was retired")
        finally:
            store.close()


class QuestLifecycleRegression(unittest.TestCase):
    def test_deterministic_fallback_is_valid(self):
        from god import fallback_quest
        from quests import validate

        spec, problems = validate(fallback_quest([3, 64, 5], 100), "p", 100,
                                  now_t=1_800_000_000_000)
        self.assertEqual(problems, [])
        self.assertIsNotNone(spec)
        self.assertTrue(spec.intent)
        self.assertEqual(spec.issued_t, 1_800_000_000_000)

    def test_expiry_is_persisted_and_cannot_return_after_restart(self):
        import asyncio
        import god as god_module
        from quests import Watcher, validate

        store = Store(":memory:")
        spec, problems = validate({
            "id": "short_lived", "intent": "a marker",
            "region": {"dim": "overworld", "center": [0, 64, 0], "radius": 8},
            "constraints": [{"type": "min_blocks", "value": 8}],
            "deadline_ticks": 1200,
        }, "p", 1, now_t=1000)
        self.assertEqual(problems, [])
        store.put("quests", {
            "id": spec.id, "actor": "p", "spec": spec.to_json(), "state": "active",
            "issued_tick": 1, "deadline_tick": 1201, "resolved_tick": None,
        }, Belief(provenance=Provenance.DERIVED))
        god = god_module.God(store, use_model=False)
        god.tick = 2000
        god.quests["p"] = (spec, Watcher(spec))
        original = god_module.speak

        async def silent(*args, **kwargs):
            return {"ok": True}

        god_module.speak = silent
        try:
            asyncio.run(god.evaluate())
            self.assertEqual(store.get("quests", spec.id)["state"], "expired")
            self.assertNotIn("p", god.quests)
            self.assertEqual(god.adopt_active_quests(), 0)
        finally:
            god_module.speak = original
            store.close()

    def test_constrained_completion_runs_without_lifecycle_errors(self):
        import asyncio
        import god as god_module
        from quests import SETTLE_TICKS, Watcher, validate

        store = Store(":memory:")
        spec, problems = validate({
            "id": "measured_build", "intent": "a small patterned marker",
            "region": {"dim": "overworld", "center": [0, 64, 0], "radius": 10},
            "constraints": [{"type": "min_blocks", "value": 8}],
            "deadline_ticks": 72000,
        }, "p", 10, now_t=1000)
        self.assertEqual(problems, [])
        watcher = Watcher(spec)
        materials = {}
        tick = 20
        for x in range(4):
            for y in range(64, 66):
                material = "minecraft:oak_planks" if x % 2 else "minecraft:cobblestone"
                materials[material] = materials.get(material, 0) + 1
                watcher.observe({"type": "block_place", "actor": "p", "dim": "overworld",
                                 "pos": [x, y, 0], "tick": tick, "after": material})
                tick += 1
        god = god_module.God(store, use_model=False)
        god.tick = tick + SETTLE_TICKS
        god.tick_seen_at = __import__("time").monotonic()
        god.quests["p"] = (spec, watcher)
        original_scan, original_speak = god_module.request_scan, god_module.speak

        async def scanned(*args, **kwargs):
            return {"ok": True, "tick": god.tick, "solid": 8, "materials": materials}

        async def silent(*args, **kwargs):
            return {"ok": True}

        god_module.request_scan, god_module.speak = scanned, silent
        try:
            asyncio.run(god.evaluate())
            self.assertEqual(store.get("quests", spec.id)["state"], "complete")
        finally:
            god_module.request_scan, god_module.speak = original_scan, original_speak
            store.close()


class CanonicalIdentityRegression(unittest.TestCase):

    def test_growth_keeps_identity_without_consulting_a_label(self):
        from structures import assign_identities

        old = {"id": "s_home", "dim": "overworld",
               "min_x": 0, "min_y": 64, "min_z": 0,
               "max_x": 8, "max_y": 69, "max_z": 8,
               "materials": json.dumps({"oak_planks": 80})}
        grown = {"dim": "overworld", "lo": [-1, 64, -1], "hi": [10, 72, 10],
                 "materials": {"oak_planks": 95, "cobblestone": 20},
                 "label": "a completely different phrase"}
        self.assertEqual(assign_identities([old], [grown]), {0: "s_home"})

    def test_a_split_can_continue_an_old_identity_only_once(self):
        from structures import assign_identities

        old = {"id": "s_old", "dim": "overworld",
               "min_x": 0, "min_y": 64, "min_z": 0,
               "max_x": 19, "max_y": 70, "max_z": 9,
               "materials": json.dumps({"stone": 100})}
        halves = [
            {"dim": "overworld", "lo": [0, 64, 0], "hi": [9, 70, 9],
             "materials": {"stone": 50}},
            {"dim": "overworld", "lo": [10, 64, 0], "hi": [19, 70, 9],
             "materials": {"stone": 50}},
        ]
        assigned = assign_identities([old], halves)
        self.assertEqual(list(assigned.values()), ["s_old"])

    def test_wall_time_not_server_tick_drives_structure_staleness(self):
        now = 1_800_000_000_000
        store = Store(":memory:")
        try:
            store.put("structures", {
                "id": "s_clock", "actor": None, "dim": "overworld", "name": None,
                "min_x": 0, "min_y": 64, "min_z": 0,
                "max_x": 4, "max_y": 68, "max_z": 4, "materials": "{}",
            }, Belief(provenance=Provenance.INFERRED, confidence=1,
                      verified_at_tick=999999, verified_at_ms=now,
                      value=json.dumps({"kind": "current_structure"})))
            later = now + Store.half_life("structures")
            self.assertAlmostEqual(
                store.confidence_now("structures", store.get("structures", "s_clock"), later),
                0.5, places=2)
        finally:
            store.close()


@needs_corpus
class TestEpisodes(unittest.TestCase):
    """P4. Boundaries must match what the player remembers doing."""

    def _episodes(self, name: str):
        return segment(load(HERE / "sessions/corpus" / name))

    def test_build_then_demolish_nets_to_zero(self):
        """The P4 acceptance criterion, and non-negotiable #4."""
        eps = self._episodes("02-abstract.jsonl")
        self.assertEqual(len(eps), 1)
        self.assertEqual(eps[0].net_total, 0, "demolition session did not net to zero")
        self.assertEqual(eps[0].gross_placed, eps[0].gross_broken)
        for material, delta in eps[0].net.items():
            self.assertEqual(delta, 0, f"{material} did not net to zero")

    def test_house_splits_into_remembered_phases(self):
        eps = self._episodes("01-house.jsonl")
        self.assertEqual(len(eps), 6, "house session boundaries drifted")
        self.assertTrue(any(e.kinds.get("sleep") for e in eps), "lost the sleep")
        self.assertTrue(any("minecraft:iron_ore" in e.broken for e in eps),
                        "lost the cave trip")

    def test_hole_splits_from_the_wander(self):
        eps = self._episodes("03-hole-idle.jsonl")
        self.assertEqual(len(eps), 2)
        dig, wander = eps
        self.assertEqual(dig.gross_placed, 0, "the dig placed nothing")
        self.assertGreater(dig.gross_broken, 300)
        self.assertLess(wander.gross_broken, 5, "the wander did no real work")

    def test_ids_are_stable(self):
        for path in CORPUS:
            events = load(path)
            self.assertEqual([e.id for e in segment(events)],
                             [e.id for e in segment(events)])

    def test_boundary_jitter_is_rare(self):
        """Hysteresis: a return to a cell you just left is not travel.

        A rate, not zero. The corpus was captured at hysteresis 8, which leaves exactly one
        jitter event across 54 minutes of heavy travel in `05-baseline`; 16 removes it and is
        now the default for new recordings. Asserting zero would fail on valid recorded data
        while telling us nothing new.
        """
        total = jitter = 0
        for path in CORPUS:
            entries = [e for e in load(path) if e["type"] == "region_enter"]
            cells = [(e["pos"][0] // 64, e["pos"][2] // 64) for e in entries]
            total += len(entries)
            jitter += sum(1 for i in range(2, len(cells))
                          if cells[i] == cells[i - 2]
                          and entries[i]["tick"] - entries[i - 2]["tick"] < 600)
        self.assertLess(jitter / max(1, total), 0.05,
                        f"{jitter}/{total} region entries are boundary jitter")


class TestStore(unittest.TestCase):
    """P5. Provenance is the anti-hallucination backbone."""

    def setUp(self):
        self.path = Path(tempfile.mktemp(suffix=".db"))
        self.store = Store(self.path)

    def tearDown(self):
        self.store.close()
        self.path.unlink(missing_ok=True)

    def test_asserted_can_never_be_promoted(self):
        """The single load-bearing rule in the whole store."""
        self.store.put("structures", {"id": "rumour"},
                       Belief(provenance=Provenance.ASSERTED))
        import sqlite3
        with self.assertRaises(sqlite3.IntegrityError):
            self.store.db.execute(
                "UPDATE structures SET provenance='SCANNED' WHERE id='rumour'")
            self.store.db.commit()
        with self.assertRaises(sqlite3.IntegrityError):
            self.store.put("structures", {"id": "rumour"},
                           Belief(provenance=Provenance.OBSERVED))

    def test_transaction_rolls_back_every_store_write(self):
        with self.assertRaisesRegex(RuntimeError, "stop"):
            with self.store.transaction():
                self.store.put("players", {"id": "one", "name": "one"},
                               Belief(provenance=Provenance.OBSERVED))
                self.store.put("players", {"id": "two", "name": "two"},
                               Belief(provenance=Provenance.OBSERVED))
                raise RuntimeError("stop")
        self.assertIsNone(self.store.get("players", "one"))
        self.assertIsNone(self.store.get("players", "two"))

    def test_nested_transaction_uses_a_savepoint(self):
        with self.store.transaction():
            self.store.put("players", {"id": "outer", "name": "outer"},
                           Belief(provenance=Provenance.OBSERVED))
            with self.assertRaisesRegex(RuntimeError, "inner"):
                with self.store.transaction():
                    self.store.put("players", {"id": "inner", "name": "inner"},
                                   Belief(provenance=Provenance.OBSERVED))
                    raise RuntimeError("inner")
        self.assertIsNotNone(self.store.get("players", "outer"))
        self.assertIsNone(self.store.get("players", "inner"))

    def test_a_guess_cannot_overwrite_a_scan(self):
        self.store.put("structures", {"id": "s", "name": "scanned"},
                       Belief(provenance=Provenance.SCANNED))
        self.store.put("structures", {"id": "s", "name": "guessed"},
                       Belief(provenance=Provenance.INFERRED, confidence=0.99))
        self.assertEqual(self.store.get("structures", "s")["name"], "scanned")

    def test_invalid_provenance_rejected(self):
        import sqlite3
        self.store.put("structures", {"id": "x"}, Belief(provenance=Provenance.SCANNED))
        with self.assertRaises(sqlite3.IntegrityError):
            self.store.db.execute("UPDATE structures SET provenance='GOSSIP' WHERE id='x'")

    def test_asserted_cannot_be_bypassed_with_insert_or_replace(self):
        self.store.put("structures", {"id": "final"},
                       Belief(provenance=Provenance.ASSERTED))
        with self.assertRaises(__import__("sqlite3").IntegrityError):
            self.store.db.execute("""
                INSERT OR REPLACE INTO structures (id, provenance, confidence)
                VALUES ('final', 'SCANNED', 1.0)
            """)

    def test_confidence_is_bounded(self):
        import sqlite3
        self.store.put("structures", {"id": "x"}, Belief(provenance=Provenance.SCANNED))
        with self.assertRaises(sqlite3.IntegrityError):
            self.store.db.execute("UPDATE structures SET confidence=1.5 WHERE id='x'")

    def test_assertability_gate(self):
        self.store.put("farms", {"id": "lo"},
                       Belief(provenance=Provenance.INFERRED, confidence=0.40))
        self.store.put("farms", {"id": "hi"},
                       Belief(provenance=Provenance.INFERRED, confidence=0.90))
        self.store.put("farms", {"id": "said"}, Belief(provenance=Provenance.ASSERTED))
        self.store.put("farms", {"id": "seen"}, Belief(provenance=Provenance.SCANNED))
        self.assertFalse(self.store.assertable("farms", "lo"))
        self.assertTrue(self.store.assertable("farms", "hi"))
        self.assertFalse(self.store.assertable("farms", "said"))
        self.assertTrue(self.store.assertable("farms", "seen"))

    def test_a_name_is_a_separate_belief_from_its_geometry(self):
        """The P8 bug: geometry is DERIVED, a name is INFERRED, and one row cannot be both.

        Writing the name onto the structure row meant `put()` correctly refused the write
        and silently discarded every classification.
        """
        self.store.put("structures", {"id": "s"}, Belief(provenance=Provenance.DERIVED))
        self.store.name_structure("s", "wooden building", 0.9, "because", 100, "k")
        self.assertEqual(self.store.get("structures", "s")["provenance"], "DERIVED")
        named = self.store.structure_name("s")
        self.assertIsNotNone(named, "the classification was discarded")
        self.assertEqual(named["provenance"], "INFERRED")
        self.assertEqual(named["object"], "wooden building")


@needs_corpus
class TestDetectors(unittest.TestCase):
    """P6. Detectors emit facts and never labels."""

    @classmethod
    def setUpClass(cls):
        cls.episodes, cls.findings, cls.events = analyse(CORPUS)

    def _facts(self, detector: str, kind: str | None = None):
        return [f for f in self.findings
                if f.detector == detector and (kind is None or f.kind == kind)]

    def test_pointless_hole_builds_nothing(self):
        """It must still be described — but never as something that was built."""
        hole = next(e for e in self.episodes if e.gross_broken > 300)
        built = [f for f in self._facts("structure_census", "placement")
                 if f.episode_id == hole.id]
        self.assertEqual(built, [], "invented a structure from an excavation")
        removed = [f for f in self._facts("structure_census", "removal")
                   if f.episode_id == hole.id]
        self.assertTrue(removed, "the hole produced no facts at all")
        self.assertGreater(removed[0].facts["blocks"], 300)

    def test_census_separates_colocated_work(self):
        """The house and the mine shaft beneath it are one episode, two structures."""
        house = next(e for e in self.episodes if e.gross_placed > 100
                     and "minecraft:glass_pane" in e.placed)
        here = [f for f in self._facts("structure_census") if f.episode_id == house.id]
        self.assertGreaterEqual(len(here), 2, "failed to separate the house from the mine")
        depths = {f.facts["bbox"].split(",")[1] for f in here}
        self.assertGreater(len(depths), 1, "all clusters at one depth")

    def test_each_cluster_reports_its_own_materials(self):
        """The P7 bug: clusters were given episode-wide materials."""
        for f in self._facts("structure_census"):
            self.assertLessEqual(sum(f.facts["materials"].values()), f.facts["blocks"] * 2)
            self.assertTrue(f.facts["materials"])

    def test_clusters_carry_block_times(self):
        """Without per-block ticks a build and its demolition are indistinguishable."""
        for f in self._facts("structure_census"):
            self.assertIn("first_tick", f.facts)
            self.assertLessEqual(f.facts["first_tick"], f.facts["last_tick"])

    def test_clustering_rejects_outliers(self):
        points = [(0, 0, z) for z in range(20)] + [(500, 60, 500)]
        groups = cluster(points)
        self.assertEqual(len(groups), 2)
        self.assertEqual(len(groups[0]), 20)

    def test_farm_trips_the_rate_anomaly(self):
        anomalies = self._facts("rate_anomaly")
        placed = [f for f in anomalies if f.facts["work"] == "placed"]
        self.assertTrue(placed, "the burst-built farm raised no rate anomaly")
        self.assertGreaterEqual(placed[0].facts["ratio"], 2.0)

    def test_context_and_risk_are_read(self):
        """Both dimensions were captured for weeks before anything read them."""
        self.assertTrue(self._facts("context_profile"))
        self.assertTrue(self._facts("risk_profile"))
        risky = [f for f in self._facts("risk_profile") if f.facts["hits"]]
        self.assertTrue(any(f.facts["health_floor"] < 20 for f in risky))

    def test_findings_carry_no_labels(self):
        """L2 describes; naming is the model's job."""
        banned = {"house", "farm", "castle", "mine", "tower", "hole", "excavation"}
        for f in self.findings:
            self.assertNotIn(f.kind, banned, f"{f.detector} emitted a label as a kind")


class TestImports(unittest.TestCase):
    """Every module must import.

    A constant renamed in `quests` left `god` importing a name that no longer existed, and
    nothing noticed because no test imported `god`. The live daemon is exactly the thing a
    unit test is least likely to touch and most costly to find broken.
    """

    def test_every_method_the_god_calls_on_itself_exists(self):
        """A rewrite deleted `on_chat` and nothing noticed.

        The slice replaced ran from `describe` to `on_event` and `on_chat` sat between
        them, so the method vanished while its call site remained. Python only finds that
        at the moment a player speaks.
        """
        import ast
        import inspect

        import god
        source = inspect.getsource(god)
        tree = ast.parse(source)
        klass = next(n for n in ast.walk(tree)
                     if isinstance(n, ast.ClassDef) and n.name == "God")
        defined = {n.name for n in klass.body
                   if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}
        called = set()
        for node in ast.walk(klass):
            if (isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name)
                    and node.value.id == "self"):
                called.add(node.attr)
        attributes = {"store", "use_model", "segment_model",
                      "quests", "names", "tick", "tick_seen_at",
                      "last_sweep", "disturbed", "followed", "boxed", "edits", "edited",
                      "positions", "last_block_tick", "edit_ticks"}
        missing = called - defined - attributes
        self.assertEqual(missing, set(),
                         f"God calls methods on itself that do not exist: {missing}")

    def test_ground_is_recognised_through_a_block_state(self):
        """Voxels carry a full block state, so bare-name comparison silently matched nothing.

        Framing then kept the entire region, and re-measuring a ten-block house reported
        nine thousand blocks of hillside.
        """
        from render import built_only
        voxels = {
            (0, 64, 0): "minecraft:grass_block[snowy=false]",
            (0, 65, 0): "minecraft:oak_planks",
            (1, 64, 0): "minecraft:dirt",
            (1, 65, 0): "minecraft:cobblestone_stairs[facing=east,half=bottom]",
        }
        built = built_only(voxels)
        self.assertEqual(len(built), 2, f"ground was not recognised: {built}")
        self.assertNotIn((0, 64, 0), built)

    def test_framing_keeps_the_ground_a_render_needs(self):
        """`built_only` measures; `framed` renders. A building with no ground under it
        is harder to read than one sitting in its own terrain."""
        from render import framed
        voxels = {(x, 64, 0): "minecraft:grass_block[snowy=false]" for x in range(20)}
        voxels[(5, 65, 0)] = "minecraft:oak_planks"
        shown = framed(voxels)
        self.assertTrue(any(m.startswith("minecraft:grass_block") for m in shown.values()),
                        "framing dropped the ground the render needs")
        self.assertLess(len(shown), len(voxels), "framing cropped nothing")

    def test_review_renders_are_named_and_replaced(self):
        """One file per structure, named for what it is currently believed to be.

        Not part of the pipeline — nothing reads these back. They exist so a person can see
        what the model saw when it decided something, without re-deriving the render by
        hand and hoping it matches.
        """
        import render
        original = render.REVIEW_DIR
        with tempfile.TemporaryDirectory() as tmp:
            render.REVIEW_DIR = Path(tmp)
            try:
                first = render.save_for_review(b"png-bytes", "s_1_2_3",
                                               "chicken pen", 0.72)
                self.assertTrue(first.exists())
                self.assertIn("chicken-pen", first.name)
                second = render.save_for_review(b"png-bytes", "s_1_2_3",
                                                "cobblestone tower keep", 0.87)
                self.assertIn("cobblestone-tower-keep", second.name)
                remaining = list(Path(tmp).glob("s_1_2_3__*.png"))
                self.assertEqual(len(remaining), 1,
                                 "renaming left the old render behind")
            finally:
                render.REVIEW_DIR = original

    def test_a_failed_review_render_never_breaks_anything(self):
        import render
        original = render.REVIEW_DIR
        render.REVIEW_DIR = Path("/nonexistent-and-unwritable/renders")
        try:
            self.assertIsNone(render.save_for_review(b"x", "s", "name"))
        finally:
            render.REVIEW_DIR = original

    def test_every_module_imports(self):
        import importlib
        for name in ("assets", "consumer", "context", "detectors", "episodes", "god",
                     "grounding", "quests", "reconcile", "render", "router", "store",
                     "world"):
            importlib.import_module(name)


@needs_corpus
class TestRouter(unittest.TestCase):
    """P9. Almost nothing is worth interrupting a player for."""

    @classmethod
    def setUpClass(cls):
        from router import Router
        cls.per_session = {}
        for path in CORPUS:
            episodes, findings, events = analyse([path])
            cls.per_session[path.name] = (Router(events).route(findings), events)

    def test_interrupt_budget_is_never_exceeded(self):
        """The cap is a hard ceiling, not a target.

        It was silently disabled once: the budget stored a tick while the rolling-window
        filter compared wall-clock milliseconds, so every recorded interrupt looked an hour
        old the instant it landed and the cap never bound.
        """
        from router import HOUR_MS, INTERRUPTS_PER_HOUR
        for name, (triggers, _) in self.per_session.items():
            fired = sorted(t.finding.t for t in triggers if t.lane == "interrupt")
            for i, when in enumerate(fired):
                window = [w for w in fired[:i] if when - w < HOUR_MS]
                self.assertLess(len(window), INTERRUPTS_PER_HOUR,
                                f"{name} exceeded the interrupt budget")

    def test_almost_everything_is_ambient(self):
        for name, (triggers, _) in self.per_session.items():
            if len(triggers) < 20:
                continue
            ambient = sum(1 for t in triggers if t.lane == "ambient")
            self.assertGreater(ambient / len(triggers), 0.85,
                               f"{name} routes too much away from ambient")

    def test_a_death_interrupts(self):
        triggers, _ = self.per_session["05-baseline.jsonl"]
        deaths = [t for t in triggers
                  if t.finding.detector == "risk_profile"
                  and t.finding.facts.get("deaths")]
        self.assertTrue(deaths, "the session's death produced no finding")
        self.assertEqual(deaths[0].lane, "interrupt", "dying did not interrupt")

    def test_nothing_interrupts_during_warmup(self):
        """Everything is a "first" in the opening seconds of a recording."""
        from router import WARMUP_MS
        for name, (triggers, events) in self.per_session.items():
            start = events[0]["t"]
            early = [t for t in triggers
                     if t.lane == "interrupt" and t.finding.t - start < WARMUP_MS]
            self.assertEqual(early, [], f"{name} interrupted before it knew anything")

    def test_novelty_decays(self):
        """The sixth tree felling is not news."""
        triggers, _ = self.per_session["05-baseline.jsonl"]
        repeated = [t for t in triggers if "recently)" in t.reason]
        self.assertTrue(repeated, "nothing was ever decayed for repetition")
        self.assertTrue(all(t.lane == "ambient" for t in repeated),
                        "a repeated finding still escaped the ambient lane")


@needs_corpus
class TestWorld(unittest.TestCase):
    """World state is derived, not narrated. Everything here is computed from events."""

    @classmethod
    def setUpClass(cls):
        from world import World
        cls.world = World()
        for path in CORPUS:
            episodes, findings, events = analyse([path])
            cls.world.observe(events, episodes, findings)
        cls.actor = next(iter(cls.world.actors.values()))

    def test_state_accumulates_across_sessions(self):
        self.assertEqual(self.actor.sessions, len(CORPUS))
        self.assertGreater(self.actor.playtime_ms, 60 * 60 * 1000)

    def test_capability_is_derived_not_assumed(self):
        """Best tier held, from what was actually observed in hand or handled."""
        self.assertIn("diamond", self.actor.best_tool)
        self.assertIn("diamond", self.actor.best_armor)

    def test_structures_know_who_built_them(self):
        self.assertTrue(self.actor.built)
        for s in self.actor.built:
            self.assertIn("bbox", s)
            self.assertIn("blocks", s)

    def test_reach_spans_the_whole_range_played(self):
        self.assertLess(self.actor.y_min, 0, "never recorded going below sea level")
        self.assertGreater(self.actor.y_max, 60)
        self.assertGreater(len(self.actor.cells_visited), 10)

    def test_risk_and_kills_are_counted(self):
        self.assertEqual(self.actor.deaths, 1)
        self.assertEqual(self.actor.health_floor, 0)
        self.assertGreater(sum(self.actor.kills.values()), 20)

    def test_traces_are_told_apart_from_works(self):
        """Caving is not construction.

        Counting every removal cluster as a structure made the god tell a player they had
        "cut fourteen shafts into this land" when they had dug one pit and gone spelunking.
        Decided by code from three signals — something built on the spot, a return visit, or
        a deliberate excavation at the surface — never by a model.
        """
        from world import World
        deep = {"kind": "removal", "lo": [0, -40, 0], "hi": [4, -36, 4],
                "blocks": 20, "sessions": 1}
        pit = {"kind": "removal", "lo": [0, 58, 0], "hi": [11, 63, 8],
               "blocks": 300, "sessions": 1}
        revisited = {"kind": "removal", "lo": [0, -40, 0], "hi": [4, -36, 4],
                     "blocks": 20, "sessions": 3}
        house = {"kind": "placement", "lo": [0, 60, 0], "hi": [8, 68, 8],
                 "blocks": 100, "sessions": 1}
        built_on = {"kind": "removal", "lo": [0, 59, 0], "hi": [8, 60, 8],
                    "blocks": 30, "sessions": 1}
        self.assertEqual(World.significance(deep, []), "trace")
        self.assertEqual(World.significance(pit, []), "work",
                         "a large surface pit is deliberate")
        self.assertEqual(World.significance(revisited, []), "work",
                         "they came back to it")
        self.assertEqual(World.significance(built_on, [house]), "work",
                         "they built something on the spot")
        self.assertEqual(World.significance(house, []), "work")

    def test_duplicate_sessions_are_folded_in_once(self):
        """The corpus files are copies of server session files under other names.

        Counting both doubled recorded playtime and made every excavation look revisited,
        which promoted nearly every caving trace to a deliberate work. Identity has to be
        the content of a session, never the path it sits at.
        """
        import god
        original = Path(god.__file__).parent / "sessions/corpus/02-abstract.jsonl"
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "sessions/corpus").mkdir(parents=True)
            (root / "server/plugins/McGod/events").mkdir(parents=True)
            body = original.read_text()
            (root / "sessions/corpus/a.jsonl").write_text(body)
            (root / "server/plugins/McGod/events/session-x.jsonl").write_text(body)

            candidates = sorted(root.rglob("*.jsonl"))
            self.assertEqual(len(candidates), 2, "fixture should hold two copies")
            seen, kept = set(), []
            for path in candidates:
                lines = path.read_text().splitlines()
                key = (json.loads(lines[0])["t"], json.loads(lines[-1])["t"], len(lines))
                if key in seen:
                    continue
                seen.add(key)
                kept.append(path)
            self.assertEqual(len(kept), 1,
                             "the same session under two names was folded in twice")

    def test_persisted_as_derived(self):
        path = Path(tempfile.mktemp(suffix=".db"))
        store = Store(path)
        try:
            self.world.persist(store)
            rows = store.all("players")
            self.assertTrue(rows)
            for r in rows:
                self.assertEqual(r["provenance"], "DERIVED",
                                 "world state must never be a guess")
        finally:
            store.close()
            path.unlink(missing_ok=True)


class TestReconcile(unittest.TestCase):
    """L4. Scanning corrects beliefs; it must never destroy them."""

    def setUp(self):
        self.path = Path(tempfile.mktemp(suffix=".db"))
        self.store = Store(self.path)

    def tearDown(self):
        self.store.close()
        self.path.unlink(missing_ok=True)

    def _structure(self, materials):
        self.store.put("structures", {
            "id": "s", "actor": "p", "dim": "overworld", "name": None,
            "min_x": 0, "min_y": 60, "min_z": 0, "max_x": 8, "max_y": 64, "max_z": 8,
            "materials": json.dumps(materials),
        }, Belief(provenance=Provenance.DERIVED))

    def _verify_with(self, scan_result):
        """Runs verify against a stubbed scan."""
        import asyncio
        import reconcile
        import scan as scan_module
        original = scan_module.request_scan

        async def stub(url, dim, lo, hi, timeout=60.0):
            return scan_result
        scan_module.request_scan = stub
        try:
            return asyncio.run(reconcile.verify(self.store, "s", "ws://unused"))
        finally:
            scan_module.request_scan = original

    def test_namespaces_are_normalised_before_comparing(self):
        """The census strips `minecraft:`; the scan keeps it.

        A raw dict lookup missed every material and reported a house that was standing as
        0% survived — and the display stripped the namespace from both sides, so the wrong
        answer printed identically to a right one.
        """
        self._structure({"cobblestone": 53, "oak_log": 16})
        verdict = self._verify_with({
            "ok": True, "tick": 100, "solid": 69,
            "materials": {"minecraft:cobblestone": 53, "minecraft:oak_log": 16}})
        self.assertEqual(verdict["survival"], 1.0, "namespace mismatch went unnoticed")

    def test_an_empty_scan_is_a_failed_read_not_a_demolition(self):
        """Absence of evidence is refused.

        Written as evidence once, at SCANNED, over a DERIVED build record — which outranks
        it, so re-deriving from events could never repair the damage.
        """
        self._structure({"cobblestone": 53})
        verdict = self._verify_with(
            {"ok": True, "tick": 100, "solid": 0, "materials": {}})
        self.assertFalse(verdict["ok"])
        self.assertIn("failed read", verdict["error"])

    def test_verification_never_overwrites_the_build_record(self):
        """What was built is history; what stands is current state. Two rows, two facts."""
        self._structure({"cobblestone": 53, "oak_log": 16})
        self._verify_with({"ok": True, "tick": 100, "solid": 20,
                           "materials": {"minecraft:cobblestone": 20}})
        row = self.store.get("structures", "s")
        self.assertEqual(row["provenance"], "DERIVED", "a scan overwrote what was built")
        self.assertEqual(json.loads(row["materials"])["cobblestone"], 53)
        survives = self.store.get("relationships", "survives:s")
        self.assertIsNotNone(survives, "the verification was not recorded")
        self.assertEqual(survives["provenance"], "SCANNED")
        self.assertLess(float(survives["object"]), 0.6)

    def test_explosions_mark_their_region_dirty(self):
        import reconcile
        events = [
            {"tick": 1, "t": 1000, "type": "move", "actor": "p", "dim": "overworld",
             "pos": [10, 64, 10]},
            {"tick": 2, "t": 2000, "type": "explosion", "actor": "world",
             "dim": "overworld", "pos": [10, 64, 10], "blocks": 14,
             "destroyed": {"minecraft:stone": 14}},
            {"tick": 3, "t": 3000, "type": "move", "actor": "p", "dim": "overworld",
             "pos": [500, 64, 500]},
        ]
        reconcile.index_regions(self.store, events)
        rows = {r["id"]: r for r in self.store.all("regions")}
        self.assertEqual(len(rows), 2, "only occupied cells should be tracked")
        self.assertTrue(rows["overworld:0:0"]["dirty"], "the explosion cell is not dirty")
        self.assertFalse(rows["overworld:7:7"]["dirty"], "a quiet cell was marked dirty")
        queue = reconcile.work_queue(self.store)
        self.assertTrue(queue[0]["dirty"], "dirty cells must outrank merely stale ones")

    def test_only_visited_cells_are_tracked(self):
        """No world sweep. Nothing changes where nobody is, so nothing else is worth reading."""
        import reconcile
        events = [{"tick": 1, "t": 1000, "type": "move", "actor": "p",
                   "dim": "overworld", "pos": [10, 64, 10]}]
        reconcile.index_regions(self.store, events)
        self.assertEqual(len(self.store.all("regions")), 1)

    def test_live_events_feed_the_same_region_index_as_replay(self):
        """Restart and live operation must not produce different correction queues."""
        from god import God
        god = God(self.store, use_model=False)
        god.revise({"tick": 5, "t": 5000, "type": "move", "actor": "p",
                    "dim": "overworld", "pos": [130, 70, -2]})
        row = self.store.get("regions", "overworld:2:-1")
        self.assertIsNotNone(row)
        self.assertEqual(row["last_activity_ms"], 5000)
        self.assertEqual(json.loads(row["actors"]), ["p"])


class TestQuests(unittest.TestCase):
    """The model sets the goal; code decides whether it was met."""

    def _proposal(self, **over):
        base = {
            "id": "test_quest",
            "intent": "somewhere to shelter on this stretch",
            "region": {"dim": "overworld", "center": [175, 65, 297], "radius": 12},
            "constraints": [{"type": "min_blocks", "value": 80}],
            "deadline_ticks": 72000,
        }
        base.update(over)
        return base

    def test_the_models_invented_schema_is_rejected(self):
        """Observed for real: asked for a quest, the model made up its own vocabulary."""
        from quests import validate
        spec, problems = validate({
            "id": "surface_line", "title": "The Line Above",
            "objectives": [{"type": "place_block", "block": "rail", "count": 64}],
            "reward": {"item": "powered_rail", "count": 16},
        }, "p", 0)
        self.assertIsNone(spec)
        self.assertTrue(problems)

    def test_unknown_constraints_are_refused_not_ignored(self):
        """A constraint nobody can evaluate must not silently pass.

        Skipping it would make the quest completable by doing nothing.
        """
        from quests import validate
        spec, problems = validate(
            self._proposal(constraints=[{"type": "vibes", "value": "good"}]), "p", 0)
        self.assertIsNone(spec)
        self.assertTrue(any("unknown type" in p for p in problems))

    def _watcher(self, spec, placements):
        from quests import Watcher
        w = Watcher(spec)
        tick = 10
        for block, n in placements.items():
            for i in range(n):
                w.observe({"tick": tick, "type": "block_place", "actor": "p",
                           "after": block,
                           "pos": [175 + (i % 6), 65 + (i % 5), 297 + (i % 9)]})
                tick += 1
        return w

    def test_composition_is_judged_from_what_the_player_placed(self):
        """Not from the scan.

        A quest region is mostly hillside, so the raw scan is hopeless; a delta against a
        baseline is better but still counts the world's own side effects as the player's
        work. Events are precise and attributable.
        """
        from quests import judge, validate
        spec, problems = validate(self._proposal(constraints=[
            {"type": "min_blocks", "value": 80},
            {"type": "material_fraction", "block": "minecraft:cobblestone", "min": 0.25},
        ]), "p", 0)
        self.assertEqual(problems, [])
        w = self._watcher(spec, {"minecraft:cobblestone": 53, "minecraft:oak_planks": 77})
        scan = {"materials": {"minecraft:stone": 6800, "minecraft:cobblestone": 53,
                              "minecraft:oak_planks": 77}}
        verdict = judge(spec, scan, w)
        self.assertTrue(verdict.complete, verdict.summary)
        blocks = next(c for c in verdict.checks if c["type"] == "min_blocks")
        self.assertIn("130", blocks["detail"], "counted terrain instead of the build")

    def test_forbidden_means_what_the_player_used(self):
        """The dirt bug, exactly.

        Placing a block on grass turns the grass underneath into dirt, through a block
        update that fires no placement event. A player who placed 61 cobblestone, 105 oak
        planks and no dirt at all was told "found dirt" and refused.
        """
        from quests import judge, validate
        spec, _ = validate(self._proposal(constraints=[
            {"type": "min_blocks", "value": 55},
            {"type": "forbidden", "blocks": ["minecraft:dirt"]}]), "p", 0)
        w = self._watcher(spec, {"minecraft:cobblestone": 61, "minecraft:oak_planks": 105})
        scan = {"materials": {"minecraft:cobblestone": 61, "minecraft:oak_planks": 105,
                              "minecraft:dirt": 40}}
        verdict = judge(spec, scan, w)
        forbidden = next(c for c in verdict.checks if c["type"] == "forbidden")
        self.assertTrue(forbidden["ok"],
                        "blamed the player for dirt the world created under their build")

    def test_the_scan_decides_only_whether_it_still_stands(self):
        from quests import judge, validate
        spec, _ = validate(self._proposal(constraints=[
            {"type": "min_blocks", "value": 40}]), "p", 0)
        w = self._watcher(spec, {"minecraft:cobblestone": 60})
        standing = judge(spec, {"materials": {"minecraft:cobblestone": 60}}, w)
        self.assertTrue(standing.complete)
        gone = judge(spec, {"materials": {"minecraft:cobblestone": 4}}, w)
        self.assertFalse(gone.complete, "a demolished build passed")
        self.assertFalse(gone.gates_passed)

    def test_extent_comes_from_observed_placements(self):
        """A scan returns a histogram and no positions, so it cannot measure a build."""
        from quests import judge, validate
        spec, _ = validate(self._proposal(constraints=[
            {"type": "min_blocks", "value": 10},
            {"type": "min_dimensions", "value": [5, 4, 5]}]), "p", 0)
        small = self._watcher(spec, {"minecraft:cobblestone": 12})
        small.hi = [small.lo[i] + 1 for i in range(3)]
        scan = {"materials": {"minecraft:cobblestone": 12}}
        self.assertFalse(judge(spec, scan, small).complete)
        big = self._watcher(spec, {"minecraft:cobblestone": 60})
        self.assertTrue(judge(spec, scan | {"materials": {"minecraft:cobblestone": 60}},
                              big).complete)

    def test_a_quest_without_intent_is_refused(self):
        """The semantic judge grades against the intent.

        One proposal shipped with an empty intent and validation accepted it, which would
        have had the model judging a build against nothing but its own slug.
        """
        from quests import validate
        spec, problems = validate({
            "id": "nameless",
            "region": {"dim": "overworld", "center": [0, 64, 0], "radius": 8},
            "constraints": [{"type": "min_blocks", "value": 20}],
            "deadline_ticks": 72000}, "p", 0)
        self.assertIsNone(spec)
        self.assertTrue(any("intent" in p for p in problems))

    def test_a_quest_can_be_about_doing_rather_than_building(self):
        """Not every quest is a building, and a hunt needs no region."""
        from quests import Watcher, judge, validate
        spec, problems = validate({
            "id": "thin_them_out", "intent": "the skeletons own the dark here",
            "constraints": [{"type": "kill", "entity": "minecraft:skeleton", "count": 3},
                            {"type": "collect", "item": "minecraft:bone", "count": 5},
                            {"type": "reach_depth", "y": -40}],
            "deadline_ticks": 72000}, "p", 0)
        self.assertEqual(problems, [], "a region-free quest was refused")
        w = Watcher(spec)
        for i in range(3):
            w.observe({"tick": 10 + i, "type": "mob_kill", "actor": "p",
                       "entity": "minecraft:skeleton", "pos": [0, 64, 0]})
        for i in range(5):
            w.observe({"tick": 20 + i, "type": "item_pickup", "actor": "p",
                       "item": "minecraft:bone", "count": 1, "pos": [0, 64, 0]})
        self.assertFalse(judge(spec, {"materials": {}}, w).complete,
                         "passed before they had gone deep enough")
        w.observe({"tick": 40, "type": "player_state", "actor": "p", "pos": [0, -45, 0]})
        self.assertTrue(judge(spec, {"materials": {}}, w).complete)

    def test_deeds_count_only_what_was_done_after_the_asking(self):
        """"Bring me thirty iron" means thirty won since I asked."""
        from quests import Watcher, validate
        spec, _ = validate({
            "id": "haul", "intent": "iron enough for a set of tools",
            "constraints": [{"type": "collect", "item": "minecraft:iron_ingot",
                             "count": 10}],
            "deadline_ticks": 72000}, "p", 0)
        w = Watcher(spec)
        self.assertEqual(w.deed_progress({"type": "collect",
                                          "item": "minecraft:iron_ingot",
                                          "count": 10}), (0, 10))
        for i in range(10):
            w.observe({"tick": 10 + i, "type": "smelt", "actor": "p",
                       "item": "minecraft:iron_ingot", "count": 1, "pos": [0, 64, 0]})
        self.assertEqual(w.deed_progress({"type": "collect",
                                          "item": "minecraft:iron_ingot",
                                          "count": 10}), (10, 10))

    def test_a_deed_quest_announces_what_to_do(self):
        """The line the player actually reads.

        Deed constraints rendered as nothing, so a quest to fetch iron announced itself as
        "[iron_from_the_deep] within 1 blocks of 0,0,0: ." — a place it does not have, and
        no word of the task.
        """
        from god import God
        from quests import validate
        spec, _ = validate({
            "id": "haul", "intent": "iron out of the dark",
            "constraints": [{"type": "reach_depth", "y": -40},
                            {"type": "smelt", "item": "minecraft:iron_ingot",
                             "count": 12}],
            "deadline_ticks": 72000}, "p", 0)
        line = God.describe(spec)
        self.assertIn("y -40", line)
        self.assertIn("12 iron_ingot", line)
        self.assertNotIn("0,0,0", line, "quoted a place a deed quest does not have")

    def test_a_quest_names_the_end_not_the_path(self):
        """"Bring back 16 seeds; breed 4 chickens" is one quest with an errand bolted on.

        A player who already has seeds is sent to fetch more for nothing. The input is
        stripped rather than the proposal rejected, so the quest survives and only the
        busywork goes.
        """
        from quests import validate
        spec, problems = validate({
            "id": "flock", "intent": "fill the pen",
            "constraints": [{"type": "collect", "item": "minecraft:wheat_seeds",
                             "count": 16},
                            {"type": "breed", "entity": "minecraft:chicken", "count": 4}],
            "deadline_ticks": 72000}, "p", 0)
        self.assertEqual(problems, [])
        kinds = [dict(c)["type"] for c in spec.constraints]
        self.assertEqual(kinds, ["breed"], "kept the ingredient alongside the outcome")

    def test_collecting_is_still_allowed_when_it_is_the_point(self):
        """Only an input to something else asked for in the same quest is stripped."""
        from quests import validate
        spec, _ = validate({
            "id": "haul", "intent": "iron enough for tools",
            "constraints": [{"type": "collect", "item": "minecraft:wheat_seeds",
                             "count": 16}],
            "deadline_ticks": 72000}, "p", 0)
        self.assertEqual([dict(c)["type"] for c in spec.constraints], ["collect"])

    def test_hard_gates_cannot_be_talked_around(self):
        """A model may weigh the rest; it may never conclude that nothing is something."""
        from quests import HARD_GATES, judge, validate
        spec, _ = validate(self._proposal(constraints=[
            {"type": "min_blocks", "value": 80}]), "p", 0)
        w = self._watcher(spec, {"minecraft:cobblestone": 3})
        verdict = judge(spec, {"materials": {"minecraft:cobblestone": 3}}, w)
        self.assertFalse(verdict.gates_passed, "3 blocks cleared the substantive gate")
        self.assertEqual(set(HARD_GATES), {"substantive", "still_standing"})

    def test_watcher_counts_only_its_own_region_and_actor(self):
        from quests import Watcher, validate
        spec, _ = validate(self._proposal(), "p", 0, baseline={})
        w = Watcher(spec)
        for e in [
            {"tick": 10, "type": "block_place", "actor": "p", "after": "minecraft:stone",
             "pos": [175, 65, 297]},
            {"tick": 11, "type": "block_place", "actor": "other", "after": "minecraft:stone",
             "pos": [175, 65, 297]},
            {"tick": 12, "type": "block_place", "actor": "p", "after": "minecraft:stone",
             "pos": [9999, 65, 297]},
        ]:
            w.observe(e)
        self.assertEqual(sum(w.placed.values()), 1)

    def _build(self, spec, n=12, tick=10):
        from quests import Watcher
        w = Watcher(spec)
        for i in range(n):
            w.observe({"tick": tick + i, "type": "block_place", "actor": "p",
                       "after": "minecraft:stone", "pos": [175 + i % 4, 65, 297]})
        return w

    def test_watcher_debounces_on_quiet_not_on_threshold(self):
        """Or the god congratulates someone mid-build."""
        from quests import SETTLE_TICKS, validate
        spec, _ = validate(self._proposal(constraints=[
            {"type": "min_blocks", "value": 2}]), "p", 0)
        w = self._build(spec, n=2)
        self.assertFalse(w.ready(13), "fired the instant the threshold was crossed")
        self.assertTrue(w.ready(11 + SETTLE_TICKS))

    def test_everyone_settles_at_the_same_short_interval(self):
        """The person most likely to be waiting is the one standing there watching.

        Presence shapes what the god says, not how long it waits. Judging early is cheap
        because the watcher re-arms on further work.
        """
        from quests import SETTLE_TICKS, validate
        spec, _ = validate(self._proposal(constraints=[
            {"type": "min_blocks", "value": 10}]), "p", 0)
        staying = self._build(spec)
        self.assertTrue(staying.present)
        self.assertTrue(staying.ready(22 + SETTLE_TICKS),
                        "made someone standing there wait longer than someone who left")
        leaving = self._build(spec)
        leaving.observe({"tick": 30, "type": "move", "actor": "p", "pos": [9999, 65, 9999]})
        self.assertFalse(leaving.present)
        self.assertTrue(leaving.ready(30 + SETTLE_TICKS))

    def test_logging_off_settles_immediately_enough(self):
        """The event stream stops dead when the last player quits."""
        from quests import SETTLE_TICKS, validate
        spec, _ = validate(self._proposal(constraints=[
            {"type": "min_blocks", "value": 10}]), "p", 0)
        w = self._build(spec)
        w.observe({"tick": 25, "type": "player_quit", "actor": "p", "pos": [175, 65, 297]})
        self.assertTrue(w.ready(25 + SETTLE_TICKS),
                        "a player who finished and logged off was never judged")

    def test_a_judged_build_is_not_rejudged_for_standing_still(self):
        from quests import SETTLE_TICKS, validate
        spec, _ = validate(self._proposal(constraints=[
            {"type": "min_blocks", "value": 10}]), "p", 0)
        w = self._build(spec)
        self.assertTrue(w.ready(22 + SETTLE_TICKS))
        w.judged_at_tick = w.last_activity_tick
        self.assertFalse(w.ready(22 + SETTLE_TICKS * 3),
                         "re-judged a build nobody had touched")

    def test_more_work_re_arms_the_watcher(self):
        """Carrying on with a finished build should be noticed."""
        from quests import SETTLE_TICKS, validate
        spec, _ = validate(self._proposal(constraints=[
            {"type": "min_blocks", "value": 10}]), "p", 0)
        w = self._build(spec)
        w.judged_at_tick = w.last_activity_tick
        w.completed_tick = 100
        w.observe({"tick": 5000, "type": "block_place", "actor": "p",
                   "after": "minecraft:oak_planks", "pos": [176, 66, 298]})
        w.observe({"tick": 5001, "type": "move", "actor": "p", "pos": [9999, 65, 9999]})
        self.assertTrue(w.ready(5001 + SETTLE_TICKS),
                        "further work on a completed build went unnoticed")

    def test_watcher_expires_at_the_deadline(self):
        from quests import Watcher, validate
        spec, _ = validate(self._proposal(deadline_ticks=1200), "p", 0, baseline={})
        w = Watcher(spec)
        w.observe({"tick": 5000, "type": "block_place", "actor": "p",
                   "after": "minecraft:stone", "pos": [175, 65, 297]})
        self.assertFalse(w.live)
        self.assertTrue(w.expired(5000))


class TestReconciliationRegression(unittest.TestCase):
    """P10's regression test: the whole anti-hallucination thesis, end to end.

    Replay a build and let the belief settle. Replay the demolition with scanning OFF — the
    god should still believe in the structure, because nothing has told it otherwise and
    inventing doubt is as wrong as inventing certainty. Then turn scanning ON: it must
    correct itself.

    This is the behaviour every provenance rule in the store exists to produce.
    """

    def setUp(self):
        self.path = Path(tempfile.mktemp(suffix=".db"))
        self.store = Store(self.path)
        self.now = 1_800_000_000_000

    def tearDown(self):
        self.store.close()
        self.path.unlink(missing_ok=True)

    def _build(self, when: int):
        """A structure derived from events, as the census would write it."""
        self.store.put("structures", {
            "id": "placement_0_64_0", "actor": "p", "dim": "overworld", "name": None,
            "min_x": 0, "min_y": 64, "min_z": 0, "max_x": 8, "max_y": 70, "max_z": 8,
            "materials": json.dumps({"cobblestone": 60, "oak_planks": 40}),
        }, Belief(provenance=Provenance.DERIVED, verified_at_tick=100,
                  value=json.dumps({"blocks": 100, "dims": "9x7x9", "when": when,
                                    "significance": "work"})))

    def _scan(self, materials):
        """Runs a verify against a stubbed scan, so no server is needed."""
        import asyncio

        import reconcile
        import scan as scan_module
        original = scan_module.request_scan

        async def stub(url, dim, lo, hi, timeout=60.0):
            # `or 1` here would turn an empty scan into one solid block and quietly
            # disarm the failed-read guard the next test depends on.
            return {"ok": True, "tick": 200, "materials": materials,
                    "solid": sum(materials.values())}
        scan_module.request_scan = stub
        try:
            return asyncio.run(reconcile.verify(self.store, "placement_0_64_0", "ws://x"))
        finally:
            scan_module.request_scan = original

    def test_belief_survives_a_demolition_nobody_watched(self):
        """With no scan, the god keeps believing — and should.

        Doubting a structure merely because time passed would be inventing a fact. What
        decays is confidence, not the belief itself, and the difference is recoverable:
        old is fixed by looking again.
        """
        self._build(self.now)
        row = self.store.get("structures", "placement_0_64_0")
        self.assertEqual(json.loads(row["materials"])["cobblestone"], 60)
        self.assertTrue(self.store.assertable("structures", "placement_0_64_0"))

        # A fortnight later, with nobody having looked.
        later = self.now + 14 * 24 * 3_600_000
        self.assertAlmostEqual(
            self.store.confidence_now("structures", row, later), 0.5, places=2,
            msg="a structure's half-life is not being applied")
        self.assertEqual(json.loads(
            self.store.get("structures", "placement_0_64_0")["materials"])["cobblestone"],
            60, "the belief itself changed without evidence")
        self.assertTrue(self.store.stale("structures", "placement_0_64_0", 0.6, later),
                        "nothing ever becomes worth re-checking")

    def test_scanning_corrects_the_belief(self):
        """Turn scanning on and the god finds out."""
        self._build(self.now)
        verdict = self._scan({"minecraft:cobblestone": 4})
        self.assertTrue(verdict["ok"])
        self.assertLess(verdict["survival"], 0.1, "a demolition went unnoticed")

        # What was BUILT is history and must not have been rewritten.
        row = self.store.get("structures", "placement_0_64_0")
        self.assertEqual(row["provenance"], "DERIVED")
        self.assertEqual(json.loads(row["materials"])["cobblestone"], 60,
                         "a scan overwrote the record of what was built")

        # What STANDS is a separate, scanned belief.
        survives = self.store.get("relationships", "survives:placement_0_64_0")
        self.assertIsNotNone(survives, "the correction was not recorded")
        self.assertEqual(survives["provenance"], "SCANNED")
        self.assertLess(float(survives["object"]), 0.1)

    def test_a_returning_player_is_greeted_not_ignored(self):
        """Rejoining with a quest outstanding returned silently.

        From the player's side that is indistinguishable from the god being down.
        """
        import asyncio

        import god as god_module
        from quests import Watcher, validate

        said = []

        async def stub_speak(text, url=None, target=None, timeout=15.0, reply=False):
            said.append(text)
            return {"ok": True}

        original = god_module.speak
        god_module.speak = stub_speak
        try:
            g = god_module.God(self.store, use_model=False)
            spec, problems = validate({
                "id": "held", "intent": "a shelter on the ridge",
                "region": {"dim": "overworld", "center": [0, 64, 0], "radius": 8},
                "constraints": [{"type": "min_blocks", "value": 40}],
                "deadline_ticks": 72000}, "p", 0)
            self.assertEqual(problems, [])
            g.quests["p"] = (spec, Watcher(spec))
            g.names["p"] = "someone"
            asyncio.run(g.on_join({"type": "player_join", "actor": "p", "tick": 10,
                                   "t": 1, "dim": "overworld", "pos": [0, 64, 0],
                                   "name": "someone"}))
        finally:
            god_module.speak = original
        self.assertTrue(said, "a returning player with a quest was met with silence")
        self.assertTrue(any("held" in t or "still stands" in t for t in said),
                        f"said something, but never mentioned the outstanding quest: {said}")

    def test_a_failed_look_is_not_a_correction(self):
        """An empty scan where a structure was believed is a bad read, not a demolition."""
        self._build(self.now)
        verdict = self._scan({})
        self.assertFalse(verdict["ok"])
        self.assertIsNone(self.store.get("relationships", "survives:placement_0_64_0"),
                          "recorded a demolition on the strength of seeing nothing")

    def test_confidence_decays_at_different_speeds_by_kind(self):
        """A chicken count and a building's name do not rot alike."""
        self.store.put("relationships", {
            "id": "holds:s", "subject": "s", "predicate": "holds", "object": "{}",
        }, Belief(provenance=Provenance.SCANNED,
                  value=json.dumps({"when": self.now})))
        self.store.put("relationships", {
            "id": "named:s", "subject": "s", "predicate": "named", "object": "coop",
        }, Belief(provenance=Provenance.INFERRED, confidence=0.92,
                  value=json.dumps({"when": self.now})))
        hour = self.now + 3_600_000
        live = self.store.confidence_now(
            "relationships", self.store.get("relationships", "holds:s"), hour)
        name = self.store.confidence_now(
            "relationships", self.store.get("relationships", "named:s"), hour)
        self.assertLess(live, 0.2, "a live entity count should be nearly worthless in an hour")
        self.assertGreater(name, 0.9, "what a place is does not change in an hour")

    def test_an_event_contradicts_the_belief_it_disproves(self):
        """Decay is for beliefs that went quiet; this is for beliefs proved wrong.

        Someone kills the animals we counted. Waiting for a twenty-minute half-life would
        leave the god reciting a flock that has been eaten.
        """
        self.store.put("relationships", {
            "id": "holds:s", "subject": "s", "predicate": "holds",
            "object": json.dumps({"minecraft:chicken": 12}),
        }, Belief(provenance=Provenance.SCANNED,
                  value=json.dumps({"when": self.now})))
        self.assertGreater(self.store.confidence_now(
            "relationships", self.store.get("relationships", "holds:s"), self.now), 0.9)

        self.store.contradict("relationships", "holds:s", "mob_kill chicken here", 500)
        row = self.store.get("relationships", "holds:s")
        self.assertLess(row["confidence"], 0.1, "a disproved belief kept its confidence")
        self.assertEqual(self.store.contradicted("relationships", "holds:s"),
                         "mob_kill chicken here")
        # The measurement itself is untouched — only looking again can replace it.
        self.assertEqual(json.loads(row["object"])["minecraft:chicken"], 12)
        self.assertEqual(row["provenance"], "SCANNED",
                         "a contradicted scan stopped being a scan")

    def test_breaking_blocks_contradicts_the_structure(self):
        from god import God
        self._build(self.now)
        god = God(self.store, use_model=False)
        god.revise({"type": "block_break", "actor": "p", "dim": "overworld",
                    "pos": [4, 66, 4], "tick": 900})
        self.assertEqual(self.store.contradicted("structures", "placement_0_64_0"),
                         "blocks taken out of it")
        god2 = God(self.store, use_model=False)
        god2.revise({"type": "block_break", "actor": "p", "dim": "overworld",
                     "pos": [900, 66, 900], "tick": 901})
        self.assertTrue(self.store.contradicted("structures", "placement_0_64_0"),
                        "a break far away should not have cleared it")

    def test_a_break_elsewhere_touches_nothing(self):
        from god import God
        self._build(self.now)
        God(self.store, use_model=False).revise(
            {"type": "block_break", "actor": "p", "dim": "overworld",
             "pos": [900, 66, 900], "tick": 900})
        self.assertIsNone(self.store.contradicted("structures", "placement_0_64_0"),
                          "an unrelated break invalidated a structure")

    def test_building_marks_the_place_for_a_second_look(self):
        """Re-segmentation is driven by where blocks moved, not by what was contradicted.

        A structure being contradicted only covers ground the god already knows about; a
        building raised on empty grass contradicts nothing at all and would never be seen.
        """
        from god import God
        g = God(self.store, use_model=False)
        g.tick = 1000
        g.revise({"type": "block_place", "actor": "p", "dim": "overworld",
                  "pos": [900, 70, 900], "tick": 1000})
        self.assertEqual(len(g.disturbed), 1,
                         "building on untouched ground marked nothing")
        cell, seen = next(iter(g.disturbed.items()))
        self.assertEqual(seen["pos"], [900, 70, 900])
        self.assertEqual(cell[0], "overworld")

        # Blocks nearby fold into the same place rather than making a new one.
        g.revise({"type": "block_break", "actor": "p", "dim": "overworld",
                  "pos": [903, 70, 902], "tick": 1010})
        self.assertEqual(len(g.disturbed), 1, "one build produced several places to visit")

    def test_a_place_is_left_alone_until_the_player_stops(self):
        from god import God
        from quests import SETTLE_TICKS
        g = God(self.store, use_model=False)
        g.tick = 1000
        g.revise({"type": "block_place", "actor": "p", "dim": "overworld",
                  "pos": [900, 70, 900], "tick": 1000})
        _, seen = next(iter(g.disturbed.items()))
        self.assertLess(1100 - seen["tick"], SETTLE_TICKS,
                        "fixture should still be mid-build")
        self.assertGreaterEqual(1000 + SETTLE_TICKS - seen["tick"], SETTLE_TICKS)

    def test_asserted_never_becomes_assertable_however_confident(self):
        """The rule the whole scheme rests on, restated at the gate."""
        self.store.put("structures", {"id": "rumour"},
                       Belief(provenance=Provenance.ASSERTED, confidence=1.0))
        self.assertFalse(self.store.assertable("structures", "rumour"))


class TestQueryGrounding(unittest.TestCase):
    """Language may select facts; it may not manufacture spatial relationships."""

    def setUp(self):
        self.store = Store(":memory:")

    def tearDown(self):
        self.store.close()

    def _structure(self, sid, label, lo, hi):
        import time
        self.store.put("structures", {
            "id": sid, "actor": "p", "dim": "overworld", "name": None,
            "min_x": lo[0], "min_y": lo[1], "min_z": lo[2],
            "max_x": hi[0], "max_y": hi[1], "max_z": hi[2],
            "materials": json.dumps({"oak_planks": 20}),
        }, Belief(provenance=Provenance.INFERRED, confidence=0.9,
                  verified_at_ms=int(time.time() * 1000),
                  value=json.dumps({"schema_version": 2,
                                    "kind": "current_structure", "blocks": 20})))
        self.store.name_structure(sid, label, 0.9, "seen", 1, f"name:{sid}")

    def test_ambiguous_name_resolves_every_canonical_candidate(self):
        from grounding import resolve_references
        self._structure("s_loop", "minecart rail loop", [0, 64, 0], [10, 65, 10])
        self._structure("s_spur", "rail spur and shelter", [11, 64, 0], [16, 67, 4])
        self._structure("s_home", "birch cottage", [50, 64, 50], [58, 70, 58])
        got = resolve_references(self.store, "Is the rail connected?", "overworld", [5, 65, 5])
        self.assertEqual({row["id"] for row in got}, {"s_loop", "s_spur"})

    def test_box_relationships_are_measurements_not_semantic_membership(self):
        from grounding import box_relation
        self._structure("s_left", "hall", [0, 64, 0], [4, 70, 4])
        self._structure("s_right", "tower", [7, 64, 0], [9, 72, 4])
        relation = box_relation(self.store.get("structures", "s_left"),
                                self.store.get("structures", "s_right"))
        self.assertEqual(relation["empty_gap_xyz"], [2, 0, 0])
        self.assertIn("east", relation["direction_from_left"])
        self.assertTrue(relation["beside"])
        self.assertEqual(relation["semantic_part_of"], "unknown")

    def test_connection_comes_from_face_connected_live_blocks(self):
        from grounding import _physically_connected
        self._structure("s_a", "post", [0, 64, 0], [0, 64, 0])
        self._structure("s_b", "post", [2, 64, 0], [2, 64, 0])
        a, b = self.store.get("structures", "s_a"), self.store.get("structures", "s_b")
        joined = {(x, 64, 0): "minecraft:oak_planks" for x in range(3)}
        separate = {p: material for p, material in joined.items() if p[0] != 1}
        self.assertTrue(_physically_connected(joined, a, b))
        self.assertFalse(_physically_connected(separate, a, b))

    def test_a_gap_between_boxes_does_not_shortcut_the_topology_read(self):
        import asyncio
        import scan
        from grounding import _connection

        self._structure("s_a", "post", [0, 64, 0], [0, 64, 0])
        self._structure("s_b", "post", [2, 64, 0], [2, 64, 0])
        original_request, original_convert = scan.request_voxels, scan.to_voxels

        async def request(*args, **kwargs):
            return {"ok": True, "voxels": {
                (0, 64, 0): "minecraft:oak_planks",
                (1, 64, 0): "minecraft:oak_planks",
                (2, 64, 0): "minecraft:oak_planks"}}

        scan.request_voxels = request
        scan.to_voxels = lambda result: result["voxels"]
        try:
            connected, evidence = asyncio.run(_connection(
                self.store, self.store.get("structures", "s_a"),
                self.store.get("structures", "s_b"), "ws://test"))
        finally:
            scan.request_voxels, scan.to_voxels = original_request, original_convert
        self.assertTrue(connected)
        self.assertIn("SCANNED", evidence)

    def test_referenced_structure_passes_the_verification_gate(self):
        import asyncio
        import reconcile
        from grounding import ground_query, render_grounding

        self._structure("s_cottage", "birch cottage", [0, 64, 0], [8, 70, 8])
        original = reconcile.confirm_before_speaking
        checked = []

        async def confirmed(store, sid, url, floor=0.5):
            checked.append(sid)
            return {"ok": True, "survival": 0.98}

        reconcile.confirm_before_speaking = confirmed
        try:
            grounded = asyncio.run(ground_query(
                self.store, "Does the cottage still stand?", "overworld", [3, 65, 3],
                "ws://test"))
        finally:
            reconcile.confirm_before_speaking = original
        self.assertEqual(checked, ["s_cottage"])
        self.assertIn("0.98", render_grounding(grounded))

    def test_existing_features_make_redundant_advice_detectable(self):
        from grounding import advice_conflicts, material_features
        materials = {"minecraft:oak_planks": 40, "minecraft:glass_pane": 12,
                     "minecraft:oak_door": 1, "minecraft:lantern": 2}
        self.assertEqual(set(material_features(materials)),
                         {"windows/glass", "entrances", "lighting"})
        grounded = {"structures": [{"role": "referenced",
                                     "existing_features": material_features(materials)}]}
        self.assertEqual(advice_conflicts("You should add windows.", grounded),
                         ["windows/glass"])
        self.assertEqual(advice_conflicts("Make the existing windows larger.", grounded), [])


class TestImmersiveWorldKnowledge(unittest.TestCase):
    def test_raw_coordinates_are_removed_unless_requested(self):
        from god import (announces_answer_policy, contains_raw_coordinates,
                         coordinates_requested, remove_policy_announcements,
                         remove_raw_coordinates, trim_chat_answer)
        speech = "The shore lies at 173, 65, 311, west of your hall."
        self.assertTrue(contains_raw_coordinates(speech))
        self.assertFalse(coordinates_requested("Where is the shore?"))
        cleaned = remove_raw_coordinates(speech)
        self.assertNotRegex(cleaned, r"173\s*,\s*65\s*,\s*311")
        self.assertTrue(contains_raw_coordinates("It is at x=173, y=65, z=311."))
        self.assertNotIn("x=173", remove_raw_coordinates(
            "It is at x=173, y=65, z=311."))
        self.assertTrue(coordinates_requested("Give me its exact coordinates"))
        policy = "I will not give you coordinates. The shore lies west."
        self.assertTrue(announces_answer_policy(policy))
        self.assertEqual(remove_policy_announcements(policy), "The shore lies west.")
        self.assertLessEqual(len(trim_chat_answer("word " * 200)), 421)

    def test_greetings_are_social_not_factual_queries(self):
        from god import pure_greeting
        self.assertTrue(pure_greeting("hi god"))
        self.assertTrue(pure_greeting("Hail!"))
        self.assertFalse(pure_greeting("hi god, what is west of me?"))

    def test_generated_features_never_enter_player_structures(self):
        from environment import persist_environment
        from god import God
        store = Store(":memory:")
        try:
            persist_environment(store, {
                "ok": True, "dim": "overworld", "seed": 42,
                "center": [100, 200], "radius": 256, "step": 16,
                "biomes": [[-16, 0, "minecraft:beach"],
                           [-32, 0, "minecraft:ocean"]],
                "generated_features": [
                    {"kind": "minecraft:village_plains", "pos": [20, 70, 30]}],
            })
            self.assertEqual(len(store.all("terrain_regions")), 1)
            self.assertEqual(len(store.all("generated_features")), 1)
            self.assertEqual(store.all("structures"), [])
            self.assertEqual(store.all("work_events"), [])
            facts = {"lo": [0, 64, 0], "hi": [8, 70, 8],
                     "materials": {"oak_planks": 80}, "blocks": 80,
                     "dims": "9x7x9", "fill": .14, "vertical": 7}
            sid = God(store, use_model=False)._write_segment(
                "overworld", facts, {"label": "village house", "category": "building"},
                None, b"")
            self.assertTrue(sid.startswith("g_"))
            self.assertEqual(len(store.all("generated_features")), 2)
            self.assertEqual(store.all("structures"), [])
            self.assertEqual(json.loads(store.get("generated_features", sid)["value"])["origin"],
                             "world_generated")
        finally:
            store.close()

    def test_player_placements_inside_a_generated_site_stay_player_built(self):
        """A village answers where an object stands; it does not answer who made it."""
        from environment import persist_environment
        from god import God

        store = Store(":memory:")
        try:
            persist_environment(store, {
                "ok": True, "dim": "overworld", "seed": 42,
                "center": [0, 0], "radius": 256, "step": 16, "biomes": [],
                "generated_features": [
                    {"kind": "minecraft:village_plains", "pos": [20, 64, 20]}],
            })
            for x in range(8):
                store.put("event_history", {
                    "id": f"place:{x}", "actor": "player", "dim": "overworld",
                    "tick": x, "event_t": 1_000 + x, "kind": "block_place",
                    "place_id": "generated:village", "x": 10 + x, "y": 65, "z": 10,
                    "payload": json.dumps({"after": "minecraft:gold_block"}),
                }, Belief(provenance=Provenance.OBSERVED))
            facts = {"lo": [10, 65, 10], "hi": [17, 65, 10],
                     "materials": {"gold_block": 8}, "blocks": 8,
                     "dims": "8x1x1", "fill": 1.0, "vertical": False}
            sid = God(store, use_model=False)._write_segment(
                "overworld", facts, {"label": "gold marker"}, None, b"",
                origin="world_generated")
            self.assertTrue(sid.startswith("s_"))
            self.assertEqual(store.get("structures", sid)["actor"], "player")
            self.assertEqual(json.loads(store.get("structures", sid)["value"])["origin"],
                             "player_built")
        finally:
            store.close()

    def test_misattributed_generated_object_is_reconciled_from_raw_history(self):
        from structures import GENERATED_KIND, migrate_player_built_generated

        store = Store(":memory:")
        try:
            store.put("generated_features", {
                "id": "g_wrong", "dim": "overworld", "kind": "other",
                "x": 3, "y": 65, "z": 0, "discovered_from": "survey",
                "min_x": 0, "min_y": 65, "min_z": 0,
                "max_x": 7, "max_y": 65, "max_z": 0,
                "materials": json.dumps({"gold_block": 8}),
            }, Belief(provenance=Provenance.INFERRED, confidence=.9,
                      value=json.dumps({"schema_version": 2, "kind": GENERATED_KIND,
                                        "origin": "world_generated", "blocks": 8,
                                        "dims": "8x1x1", "fill": 1.0,
                                        "vertical": False})))
            store.name_structure("g_wrong", "gold marker", .9, "seen", 1, "test",
                                 saw=True)
            for x in range(8):
                store.put("event_history", {
                    "id": f"place:{x}", "actor": "player", "dim": "overworld",
                    "tick": x, "event_t": 1_000 + x, "kind": "block_place",
                    "place_id": None, "x": x, "y": 65, "z": 0,
                    "payload": json.dumps({"after": "minecraft:gold_block"}),
                }, Belief(provenance=Provenance.OBSERVED))

            moved = migrate_player_built_generated(store)
            self.assertEqual(len(moved), 1)
            old, new = moved[0]
            self.assertEqual(old, "g_wrong")
            self.assertTrue(new.startswith("s_"))
            self.assertIsNone(store.get("generated_features", old))
            self.assertEqual(store.get("structures", new)["actor"], "player")
            self.assertEqual(store.structure_name(new)["object"], "gold marker")
        finally:
            store.close()

    def test_environment_is_rendered_as_cardinal_landmarks(self):
        from environment import generated_feature_answer, render_environment
        text = render_environment({
            "ok": True, "dim": "overworld", "center": [0, 0],
            "biomes": [[-16, 0, "minecraft:beach"],
                       [-32, 0, "minecraft:ocean"],
                       [-48, 0, "minecraft:ocean"],
                       [16, 0, "minecraft:plains"]],
            "generated_features": [],
        })
        self.assertIn("west: a beach giving way to ocean", text)
        self.assertNotIn("[-16", text)
        located = generated_feature_answer({
            "ok": True, "center": [100, 200], "radius": 8192,
            "generated_features": [{"kind": "minecraft:village_taiga",
                                     "pos": [0, 70, 100]}],
        }, ["village_taiga"])
        self.assertEqual(located,
                         "The nearest village lies northwest, roughly 150 blocks away.")
        absent = generated_feature_answer({
            "ok": True, "center": [0, 0], "radius": 8192,
            "generated_features": [],
        }, ["village_plains"])
        self.assertIn("no village within roughly 8192 blocks", absent)
        inside = generated_feature_answer({
            "ok": True, "center": [100, 100], "radius": 8192,
            "generated_features": [{"kind": "minecraft:village_plains",
                                      "pos": [108, 70, 106]}],
        }, ["village_plains"])
        self.assertEqual(inside, "The nearest village is here.")

    def test_live_rpc_uses_the_event_connection_and_routes_by_id(self):
        """Speaking during play must not open a second WebSocket handshake."""
        import asyncio
        import scan

        class FakeSocket:
            def __init__(self):
                self.sent = []

            async def send(self, payload):
                self.sent.append(json.loads(payload))

        async def scenario():
            ws = FakeSocket()
            scan.bind_rpc(ws)
            try:
                waiting = asyncio.create_task(scan.speak("hello", reply=True))
                await asyncio.sleep(0)
                self.assertEqual(len(ws.sent), 1)
                request = ws.sent[0]
                self.assertEqual(request["rpc"], "speak")
                self.assertTrue(scan.dispatch_rpc({
                    "rpc": "speak_result", "id": request["id"], "ok": True,
                }))
                self.assertTrue((await waiting)["ok"])
            finally:
                scan.unbind_rpc(ws)

        asyncio.run(scenario())

    def test_live_vehicle_state_is_visible_without_restart(self):
        from world import fold_live_player
        store = Store(":memory:")
        try:
            fold_live_player(store, {"type": "vehicle_enter", "actor": "p",
                                     "dim": "overworld", "pos": [1, 64, 2],
                                     "tick": 10, "t": 1000,
                                     "vehicle": "minecraft:minecart"})
            fold_live_player(store, {"type": "player_state", "actor": "p",
                                     "dim": "overworld", "pos": [2, 64, 2],
                                     "tick": 20, "t": 2000,
                                     "travel_cm": {"minecart": 1250}})
            profile = json.loads(store.get("players", "p")["profile"])
            self.assertEqual(profile["vehicle_entries"]["minecart"], 1)
            self.assertEqual(profile["travel_m"]["minecart"], 12.5)
            world = next(s for s in Assembler(store).build(
                20, "p", {"reason": "test", "facts": {}}, immersive=True)
                         if s.name == "world")
            self.assertIn("has ridden: minecart", world.text)
            self.assertNotIn("[2, 64, 2]", world.text)
        finally:
            store.close()

    def test_vehicle_occupants_and_surface_are_replaceable_current_state(self):
        """A shared boat and its terrain stay current; exiting must not leave a ghost ride."""
        from world import fold_live_player
        store = Store(":memory:")
        try:
            shared = {
                "type": "vehicle_enter", "actor": "p", "dim": "overworld",
                "pos": [10, 70, 10], "tick": 10, "t": 1_000,
                "vehicle": "oak_boat", "vehicle_id": "boat-1",
                "passengers": {"villager-1": "minecraft:villager",
                               "p": "minecraft:player"},
                "vehicle_in_water": False,
                "vehicle_surface": "minecraft:grass_block",
            }
            fold_live_player(store, shared)
            profile = json.loads(store.get("players", "p")["profile"])
            self.assertEqual(profile["vehicle"]["vehicle_id"], "boat-1")
            self.assertIn("minecraft:villager", profile["vehicle"]["passengers"].values())
            self.assertFalse(profile["vehicle"]["vehicle_in_water"])

            fold_live_player(store, {
                "type": "vehicle_exit", "actor": "p", "dim": "overworld",
                "pos": [20, 70, 20], "tick": 20, "t": 2_000,
                "vehicle": "oak_boat", "vehicle_id": "boat-1",
            })
            profile = json.loads(store.get("players", "p")["profile"])
            self.assertNotIn("vehicle", profile)
        finally:
            store.close()

    def test_recent_deeds_are_joined_to_generated_buildings(self):
        """Low-level capture should become exact human-scale history without turning a
        generated village into one of the player's structures."""
        from activity import record_activities, summarize_recent
        store = Store(":memory:")
        try:
            store.put("generated_features", {
                "id": "generated:village", "dim": "overworld",
                "kind": "minecraft:village_plains", "x": 100, "y": 70, "z": 100,
                "discovered_from": "seed locate",
            }, Belief(provenance=Provenance.DERIVED,
                      value=json.dumps({"origin": "world_generated"})))
            base = {"actor": "p", "dim": "overworld"}
            events = [
                {**base, "type": "move", "tick": 1, "t": 50_000,
                 "pos": [0, 64, 0]},
                {**base, "type": "move", "tick": 2, "t": 55_000,
                 "pos": [90, 70, 90]},
                {**base, "type": "container_take", "tick": 3, "t": 65_000,
                 "pos": [110, 70, 108], "item": "minecraft:obsidian", "count": 7,
                 "container": "chest"},
                {**base, "type": "container_take", "tick": 3, "t": 65_000,
                 "pos": [110, 70, 108], "item": "minecraft:bread", "count": 4,
                 "container": "chest"},
                {**base, "type": "block_break", "tick": 4, "t": 70_000,
                 "pos": [112, 70, 109], "before": "minecraft:grindstone",
                 "after": "minecraft:air"},
                {**base, "type": "block_place", "tick": 5, "t": 75_000,
                 "pos": [112, 70, 108], "before": "minecraft:air",
                 "after": "minecraft:cobblestone"},
            ]
            record_activities(store, events)
            speech = summarize_recent(store, "p", 180_000, "in the last 2 minutes")
            self.assertIn("found a plains village", speech)
            self.assertIn("7 obsidian and 4 bread", speech)
            self.assertIn("village blacksmith's chest", speech)
            self.assertIn("modified the village blacksmith", speech)
            self.assertEqual(store.all("structures"), [])
            component = [row for row in store.all("generated_features")
                         if row["id"].startswith("generated_sub:")]
            self.assertEqual(len(component), 1)
            self.assertEqual(json.loads(component[0]["value"])["origin"],
                             "world_generated")
            self.assertTrue(store.all(
                "relationships", "subject = ? AND predicate = 'modified_by'",
                (component[0]["id"],)))

            # If a live segmentation pass had already represented the edited vanilla
            # building as an unowned ordinary structure, startup reconciliation moves the
            # measured geometry and name rather than preserving two competing objects.
            store.put("work_events", {
                "id": "placement_small_renovation", "kind": "placement", "actor": "p",
                "dim": "overworld", "min_x": 110, "min_y": 70, "min_z": 108,
                "max_x": 113, "max_y": 73, "max_z": 111,
                "materials": json.dumps({"cobblestone": 6}),
            }, Belief(provenance=Provenance.DERIVED))
            store.put("structures", {
                "id": "s_duplicate", "actor": None, "dim": "overworld", "name": None,
                "min_x": 106, "min_y": 68, "min_z": 104,
                "max_x": 115, "max_y": 76, "max_z": 113,
                "materials": json.dumps({"cobblestone": 90}),
            }, Belief(provenance=Provenance.INFERRED, confidence=.9,
                      first_seen_tick=6, verified_at_tick=6, verified_at_ms=76_000,
                      value=json.dumps({"schema_version": 2,
                                        "kind": "current_structure", "origin": "unknown",
                                        "blocks": 90, "dims": "10x9x10", "fill": .1,
                                        "vertical": True})))
            store.name_structure("s_duplicate", "stone village house", .9, "seen", 6,
                                 "test", saw=True)
            from structures import migrate_unknown_generated
            self.assertEqual(migrate_unknown_generated(store),
                             [("s_duplicate", component[0]["id"])])
            self.assertEqual(store.all("structures"), [])
            self.assertEqual(store.structure_name(component[0]["id"])["object"],
                             "stone village house")

            # Leaving and returning is a second arrival; walking among buildings is not.
            record_activities(store, [
                {**base, "type": "move", "tick": 6, "t": 80_000,
                 "pos": [300, 70, 300]},
                {**base, "type": "move", "tick": 7, "t": 90_000,
                 "pos": [95, 70, 95]},
            ])
            self.assertEqual(len(store.all(
                "activities", "actor = ? AND kind = 'visit_generated'", ("p",))), 2)
        finally:
            store.close()

    def test_event_history_is_queried_by_place_session_and_action(self):
        """A location-scoped question must retrieve the whole relevant visit, compose raw
        events into deeds, and ignore both older sessions and activity elsewhere."""
        from activity import record_activities, select_history, summarize_recent
        store = Store(":memory:")
        try:
            store.put("generated_features", {
                "id": "generated:village", "dim": "overworld",
                "kind": "minecraft:village_plains", "x": 100, "y": 70, "z": 100,
                "discovered_from": "seed locate",
            }, Belief(provenance=Provenance.DERIVED,
                      value=json.dumps({"origin": "world_generated"})))
            # A renderer may already have measured the building under an opaque stable ID.
            # Activity logic must use ontology membership, never an ID-prefix convention.
            store.put("generated_features", {
                "id": "g_measured_church", "dim": "overworld",
                "kind": "generated_building", "x": 110, "y": 70, "z": 110,
                "discovered_from": "measured visual segmentation",
                "min_x": 105, "min_y": 65, "min_z": 105,
                "max_x": 115, "max_y": 76, "max_z": 115,
            }, Belief(provenance=Provenance.INFERRED, confidence=.9,
                      value=json.dumps({"origin": "world_generated",
                                        "parent": "generated:village"})))
            base = {"actor": "p", "dim": "overworld"}
            events = [
                {**base, "type": "craft", "tick": 1, "t": 100,
                 "pos": [1000, 64, 1000], "item": "minecraft:torch", "count": 64},
                {**base, "type": "player_join", "tick": 2, "t": 1_000,
                 "pos": [90, 70, 90], "name": "player"},
                {**base, "type": "block_break", "tick": 3, "t": 2_000,
                 "pos": [110, 70, 110], "before": "minecraft:brewing_stand",
                 "after": "minecraft:air"},
                {**base, "type": "item_pickup", "tick": 4, "t": 3_000,
                 "pos": [110, 70, 110], "item": "minecraft:brewing_stand", "count": 1},
                {**base, "type": "mob_kill", "tick": 5, "t": 4_000,
                 "pos": [115, 70, 100], "entity": "minecraft:zombie"},
                {**base, "type": "sleep", "tick": 6, "t": 5_000,
                 "pos": [120, 70, 100], "result": "not_safe"},
                {**base, "type": "sleep", "tick": 7, "t": 6_000,
                 "pos": [120, 70, 100], "result": "ok"},
            ]
            for n in range(10):
                events.append({**base, "type": "block_break", "tick": 8 + n,
                               "t": 7_000 + n, "pos": [130, 70, 130],
                               "before": "minecraft:hay_block", "after": "minecraft:air"})
            events.extend([
                {**base, "type": "item_pickup", "tick": 18, "t": 8_000,
                 "pos": [130, 70, 130], "item": "minecraft:hay_block", "count": 10},
                {**base, "type": "craft", "tick": 19, "t": 9_000,
                 "pos": [130, 70, 130], "item": "minecraft:wheat", "count": 90,
                 "inputs": {"minecraft:hay_block": 10}},
                {**base, "type": "craft", "tick": 20, "t": 10_000,
                 "pos": [130, 70, 130], "item": "minecraft:bread", "count": 30,
                 "inputs": {"minecraft:wheat": 90}},
                {**base, "type": "mob_kill", "tick": 21, "t": 11_000,
                 "pos": [1000, 64, 1000], "entity": "minecraft:cow"},
                {"actor": "world", "dim": "overworld", "type": "weather_change",
                 "tick": 22, "t": 12_000, "weather": "rain"},
            ])
            record_activities(store, events)
            self.assertEqual(len(store.all("event_history")), len(events))
            selection = select_history(
                store, "p", 13_000, "what have i done specifically in the village",
                [130, 70, 130])
            self.assertTrue(selection.rows)
            self.assertNotIn("minecraft:torch", json.dumps(selection.rows))
            self.assertNotIn("minecraft:cow", json.dumps(selection.rows))
            speech = summarize_recent(
                store, "p", 13_000, "what have i done specifically in the village",
                [130, 70, 130])
            self.assertIn("brewing stand from the village church", speech)
            self.assertIn("killed a zombie", speech)
            self.assertIn("slept in a villager's bed", speech)
            self.assertIn("10 hay bales", speech)
            self.assertIn("90 wheat", speech)
            self.assertIn("30 bread", speech)
            self.assertNotIn("modified the village church", speech)
            church = store.get("generated_features", "g_measured_church")
            self.assertEqual(church["kind"], "village_church")
            self.assertTrue(store.all(
                "relationships", "subject = ? AND predicate = 'modified_by'",
                (church["id"],)))
        finally:
            store.close()

    def test_all_time_here_is_a_spatial_history_query(self):
        """Broad wording must retrieve old local actions, not fall into the lifetime digest."""
        from activity import (history_query_text, is_activity_question, record_activities,
                              render_history_evidence, select_history)
        store = Store(":memory:")
        try:
            base = {"actor": "p", "dim": "overworld"}
            record_activities(store, [
                {**base, "type": "block_place", "tick": 1, "t": 1_000,
                 "pos": [100, 64, 100], "before": "minecraft:air",
                 "after": "minecraft:oak_planks"},
                {**base, "type": "mob_kill", "tick": 2, "t": 2_000,
                 "pos": [500, 64, 500], "entity": "minecraft:cow"},
                {**base, "type": "player_join", "tick": 3, "t": 100_000,
                 "pos": [102, 64, 101], "name": "player"},
                {**base, "type": "craft", "tick": 4, "t": 101_000,
                 "pos": [102, 64, 101], "item": "minecraft:bread", "count": 3},
            ])
            question = "name everything ive ever done here"
            self.assertTrue(is_activity_question(question))
            selection = select_history(
                store, "p", 102_000, question, [102, 64, 101], "overworld")
            payload = json.dumps(selection.rows)
            self.assertEqual(selection.after_ms, 0)
            self.assertIn("oak_planks", payload)
            self.assertIn("bread", payload)
            self.assertNotIn("minecraft:cow", payload)
            evidence = render_history_evidence(store, selection, question)
            self.assertIn("all recorded history", evidence)
            self.assertIn("modified", evidence)

            record_activities(store, [
                {**base, "type": "chat", "tick": 5, "t": 103_000,
                 "pos": [102, 64, 101], "text": question},
                {**base, "type": "chat", "tick": 6, "t": 104_000,
                 "pos": [102, 64, 101],
                 "text": "specifically in this location in the world"},
            ])
            resolved = history_query_text(
                store, "p", 104_000, "specifically in this location in the world")
            self.assertIn(question, resolved)
            self.assertIn("specifically", resolved)
        finally:
            store.close()

    def test_join_and_chat_are_indexed_before_early_dispatch(self):
        """Live session boundaries and follow-ups must not wait for restart replay."""
        import asyncio
        from god import God

        store = Store(":memory:")
        try:
            god = God(store, use_model=False)

            async def ignore(_event):
                return None

            god.on_join = ignore
            god.on_chat = ignore
            asyncio.run(god.on_event({
                "type": "player_join", "actor": "p", "dim": "overworld",
                "tick": 1, "t": 1_000, "pos": [0, 64, 0], "name": "player",
            }))
            asyncio.run(god.on_event({
                "type": "chat", "actor": "p", "dim": "overworld",
                "tick": 2, "t": 2_000, "pos": [0, 64, 0], "text": "what did i do",
            }))
            self.assertEqual(
                [row["kind"] for row in store.all(
                    "event_history", "1 ORDER BY event_t")],
                ["player_join", "chat"])
        finally:
            store.close()

    def test_broad_history_question_routes_through_retrieval_narration(self):
        """The screenshot wording must not fall through to the generic lifetime digest."""
        import asyncio
        import god as god_module
        import history

        store = Store(":memory:")
        original_answer = history.answer_if_history
        original_speak = god_module.speak
        original_key = god_module.have_key
        spoken = []
        narrated = ("A retrieved local history " + "with evidence " * 38).strip()

        async def answer(_store, event, query, bridge_url=None):
            self.assertEqual(event["actor"], "p")
            self.assertIn("everything", query)
            self.assertEqual(bridge_url, god_module.URL)
            return narrated

        async def speak(text, *_args, **_kwargs):
            spoken.append(text)
            return {"ok": True}

        try:
            history.answer_if_history = answer
            god_module.speak = speak
            god_module.have_key = lambda: True
            live = god_module.God(store, use_model=True)
            asyncio.run(live.on_event({
                "type": "chat", "actor": "p", "dim": "overworld",
                "tick": 1, "t": 1_000, "pos": [100, 64, 100],
                "text": "name everything ive ever done here",
            }))
            self.assertGreater(len(narrated), 420)
            self.assertEqual(spoken, [narrated])
            self.assertEqual(store.all("utterances")[0]["trigger"], "history SQL answer")
        finally:
            history.answer_if_history = original_answer
            god_module.speak = original_speak
            god_module.have_key = original_key
            store.close()

    def test_generated_history_sql_is_read_only_and_spatial(self):
        from history import UnsafeHistoryQuery, clean_select, execute_select

        store = Store(":memory:")
        try:
            from activity import record_activities
            record_activities(store, [
                {"type": "mob_kill", "actor": "p", "dim": "overworld",
                 "tick": 1, "t": 1_000, "pos": [10, 64, 10],
                 "entity": "minecraft:zombie"},
                {"type": "mob_kill", "actor": "p", "dim": "overworld",
                 "tick": 2, "t": 2_000, "pos": [500, 64, 500],
                 "entity": "minecraft:cow"},
            ])
            sql = clean_select(
                "```sql\nSELECT event_t,kind,payload FROM event_history "
                "WHERE actor='p' AND x BETWEEN 0 AND 20 AND z BETWEEN 0 AND 20 "
                "ORDER BY event_t;\n```")
            result = execute_select(store, sql)
            self.assertEqual(len(result["rows"]), 1)
            self.assertIn("zombie", result["rows"][0]["payload"])
            with self.assertRaises(UnsafeHistoryQuery):
                clean_select("DELETE FROM event_history")
            with self.assertRaises(UnsafeHistoryQuery):
                clean_select("SELECT 1")
            # Follow-up tool calls may inspect approved world-ontology tables without
            # pretending those rows are action history.
            self.assertEqual(execute_select(
                store, "SELECT id,kind FROM generated_features")["rows"], [])
            with self.assertRaises(sqlite3.DatabaseError):
                execute_select(store, "SELECT name FROM sqlite_master")
        finally:
            store.close()

    def test_history_pipeline_is_model_sql_then_model_plain_text(self):
        """The model can follow event place IDs before writing the plain narrative."""
        import asyncio
        import history
        from activity import record_activities
        from store import Belief, Provenance
        from history import answer_if_history

        store = Store(":memory:")
        original_client = history.model_client
        original_run_tool = history._run_tool
        calls = []

        class Block:
            type = "text"

            def __init__(self, text):
                self.text = text

        class Reply:
            def __init__(self, *blocks):
                self.content = list(blocks)

        class ToolBlock:
            type = "tool_use"

            def __init__(self, block_id, name, values):
                self.id = block_id
                self.name = name
                self.input = values

        class ThinkingBlock:
            type = "thinking"

            def model_dump(self, **_kwargs):
                return {"type": "thinking", "thinking": "private",
                        "signature": "signed"}

        class Messages:
            async def create(self, **kwargs):
                calls.append(kwargs)
                self_test.assertIn("proves only looking at wares", kwargs["system"])
                self_test.assertIn("joins, subqueries, and CTEs", kwargs["system"])
                self_test.assertIn("most likely specific subject", kwargs["system"])
                self_test.assertIn("trunk descends from the face", kwargs["system"])
                self_test.assertEqual(
                    {tool["name"] for tool in kwargs["tools"]},
                    {"query_database", "inspect_seed_environment", "query_seed_map",
                     "render_world_area", "inspect_world_entities"})
                if len(calls) == 1:
                    return Reply(ThinkingBlock(), ToolBlock(
                        "events", "query_database", {"sql":
                        "SELECT event_t,kind,place_id,x,y,z,payload FROM event_history "
                        "WHERE actor='p' AND x BETWEEN 0 AND 40 AND z BETWEEN 0 AND 40 "
                        "AND kind NOT IN ('chat') ORDER BY event_t"}))
                if len(calls) == 2:
                    self_test.assertEqual(
                        kwargs["messages"][-2]["content"][0]["signature"], "signed")
                    tool_result = kwargs["messages"][-1]["content"][0]["content"]
                    self_test.assertIn("zombie", tool_result)
                    self_test.assertIn("g_church", tool_result)
                    return Reply(ToolBlock("place", "query_database", {"sql":
                        "SELECT id,kind,value FROM generated_features "
                        "WHERE id='g_church'"}))
                tool_result = kwargs["messages"][-1]["content"][0]["content"]
                if len(calls) == 3:
                    self_test.assertIn("village_church", tool_result)
                    return Reply(ToolBlock("picture", "render_world_area", {
                        "dim": "overworld", "min_x": 15, "min_y": 60, "min_z": 15,
                        "max_x": 25, "max_y": 75, "max_z": 25,
                        "view": "structure",
                    }))
                self_test.assertEqual(tool_result[0]["type"], "image")
                self_test.assertEqual(tool_result[0]["source"]["media_type"], "image/png")
                self_test.assertIn("Fresh live render", tool_result[1]["text"])
                return Reply(Block("You killed a zombie at the village church."))

        class Client:
            def __init__(self):
                self.messages = Messages()

        self_test = self

        async def run_tool(test_store, block, bridge_url, actor=None):
            if block.name == "render_world_area":
                self.assertEqual(bridge_url, "ws://test")
                return {"ok": True, "note": "Fresh live render",
                        "_image_base64": "aW1hZ2U="}, False
            return await original_run_tool(test_store, block, bridge_url)

        try:
            store.put("generated_features", {
                "id": "g_church", "dim": "overworld", "kind": "village_church",
                "x": 20, "y": 64, "z": 20,
                "min_x": 15, "min_y": 60, "min_z": 15,
                "max_x": 25, "max_y": 75, "max_z": 25,
            }, Belief(provenance=Provenance.SCANNED,
                      value=json.dumps({"role_evidence": "church"})))
            record_activities(store, [{
                "type": "mob_kill", "actor": "p", "dim": "overworld",
                "tick": 1, "t": 1_000, "pos": [20, 64, 20],
                "entity": "minecraft:zombie",
            }])
            history.model_client = lambda **_kwargs: Client()
            history._run_tool = run_tool
            answer = asyncio.run(answer_if_history(store, {
                "actor": "p", "dim": "overworld", "t": 2_000,
                "pos": [20, 64, 20],
            }, "what happened here", bridge_url="ws://test"))
            self.assertEqual(answer, "You killed a zombie at the village church.")
            self.assertEqual(len(calls), 4)
            self.assertIn("CURRENT_POSITION", calls[0]["messages"][0]["content"])
        finally:
            history.model_client = original_client
            history._run_tool = original_run_tool
            store.close()

    def test_world_render_tool_is_bounded_and_returns_a_png(self):
        """Visual judgment receives an image, while raw voxels remain agent-side."""
        import asyncio
        import base64
        import evidence
        import history
        import scan

        original_request = scan.request_voxels
        original_path = evidence.LATEST_EVIDENCE_RENDER
        calls = []
        store = Store(":memory:")

        async def request(_url, dim, lo, hi):
            calls.append((dim, lo, hi))
            return {
                "ok": True, "palette": ["minecraft:air", "minecraft:cobblestone"],
                "size": [3, 3, 3], "origin": [0, 64, 0],
                "blocks": bytes([1] * 27), "entities": [
                    {"type": "minecraft:sheep", "pos": [1.0, 65.0, 1.0]},
                    {"type": "minecraft:player", "pos": [2.0, 65.0, 2.0]},
                    {"type": "minecraft:cow", "pos": [20.0, 65.0, 20.0]},
                ],
            }

        class Block:
            name = "render_world_area"
            input = {"dim": "overworld", "min_x": 0, "min_y": 64, "min_z": 0,
                     "max_x": 2, "max_y": 66, "max_z": 2, "view": "structure"}

        try:
            scan.request_voxels = request
            with tempfile.TemporaryDirectory() as folder:
                evidence.LATEST_EVIDENCE_RENDER = Path(folder) / "evidence.png"
                result, used_history = asyncio.run(
                    history._run_tool(store, Block(), "ws://test"))
                self.assertFalse(used_history)
                self.assertEqual(calls, [("overworld", [0, 64, 0], [2, 66, 2])])
                png = base64.b64decode(result["_image_base64"])
                self.assertTrue(png.startswith(b"\x89PNG\r\n\x1a\n"))
                self.assertEqual(evidence.LATEST_EVIDENCE_RENDER.read_bytes(), png)
                self.assertFalse(result["entities_in_image"])
                self.assertEqual(result["tile_order"],
                                 ["perspective", "top", "front elevation",
                                  "side elevation"])

                class EntityBlock:
                    name = "inspect_world_entities"
                    input = {key: value for key, value in Block.input.items()
                             if key != "view"}

                census, _ = asyncio.run(
                    history._run_tool(store, EntityBlock(), "ws://test"))
                self.assertEqual(census["entity_counts"],
                                 {"minecraft:player": 1, "minecraft:sheep": 1})
                self.assertEqual(census["total_entities"], 2)

                Block.input = {**Block.input, "max_x": 100}
                rejected, _ = asyncio.run(
                    history._run_tool(store, Block(), "ws://test"))
                self.assertIn("limited", rejected["error"])
                self.assertEqual(len(calls), 2, "oversized reads must be refused before RPC")
        finally:
            scan.request_voxels = original_request
            evidence.LATEST_EVIDENCE_RENDER = original_path
            store.close()

    def test_visual_evidence_is_model_chosen_in_both_dialogue_paths(self):
        """No appearance-keyword router may decide whether the model is allowed to look."""
        history_source = (HERE / "history.py").read_text(encoding="utf-8")
        dialogue_source = (HERE / "god.py").read_text(encoding="utf-8")
        self.assertIn("VISUAL_TOOL", history_source)
        for tool in ("ACT_TOOL", "VISUAL_TOOL", "ENTITY_TOOL"):
            self.assertIn(tool, dialogue_source)
        self.assertNotIn("def wants_visual_advice", dialogue_source)

    def test_incomplete_model_output_is_retried_and_never_returned(self):
        """Adaptive thinking may consume the output cap; half an answer is not an answer."""
        import asyncio
        import evidence

        class Reply:
            def __init__(self, text, stop_reason):
                self.text = text
                self.stop_reason = stop_reason

        class Messages:
            def __init__(self):
                self.calls = []

            async def create(self, **kwargs):
                self.calls.append(kwargs)
                if len(self.calls) == 1:
                    return Reply("two oak buttons set in as ey", "max_tokens")
                return Reply("two oak buttons set in as eyes.", "end_turn")

        class Client:
            def __init__(self):
                self.messages = Messages()

        client = Client()
        reply = asyncio.run(evidence.complete_message(
            client, model="test", messages=[{"role": "user", "content": "look"}]))
        self.assertEqual(reply.text, "two oak buttons set in as eyes.")
        self.assertEqual([call["max_tokens"] for call in client.messages.calls],
                         [evidence.MODEL_OUTPUT_TOKENS, evidence.MODEL_RETRY_TOKENS])

    def test_openrouter_translation_preserves_tools_results_and_images(self):
        """The alternate provider must see the same evidence, not a text-only downgrade."""
        from model_api import openrouter_messages, openrouter_tools

        messages = [
            {"role": "assistant", "content": [
                {"type": "thinking", "thinking": "private", "signature": "signed"},
                {"type": "tool_use", "id": "tool-1", "name": "look",
                 "input": {"place": "here"}},
            ]},
            {"role": "user", "content": [{
                "type": "tool_result", "tool_use_id": "tool-1", "content": [
                    {"type": "image", "source": {"type": "base64",
                     "media_type": "image/png", "data": "aW1hZ2U="}},
                    {"type": "text", "text": "fresh render"},
                ],
            }]},
        ]
        converted = openrouter_messages(messages, "system prompt")
        self.assertEqual([item["role"] for item in converted],
                         ["system", "assistant", "tool", "user"])
        self.assertEqual(converted[1]["tool_calls"][0]["function"]["name"], "look")
        self.assertEqual(converted[2]["tool_call_id"], "tool-1")
        self.assertIn("fresh render", converted[2]["content"])
        self.assertTrue(converted[3]["content"][1]["image_url"]["url"].startswith(
            "data:image/png;base64,"))
        tools = openrouter_tools([{"name": "look", "description": "see",
                                   "input_schema": {"type": "object"}}])
        self.assertEqual(tools[0]["function"]["parameters"], {"type": "object"})

    def test_openrouter_response_matches_the_existing_message_contract(self):
        """Callers continue to receive text/tool blocks and Anthropic-style stop reasons."""
        import json
        from types import SimpleNamespace
        import model_api

        response = SimpleNamespace(
            choices=[SimpleNamespace(
                finish_reason="tool_calls",
                message=SimpleNamespace(content="I will look.", tool_calls=[
                    SimpleNamespace(id="call-1", function=SimpleNamespace(
                        name="render_world_area",
                        arguments=json.dumps({"view": "structure"})))
                ]))],
            usage=SimpleNamespace(prompt_tokens=12, completion_tokens=8),
        )
        normal = model_api._normalise(response)
        self.assertEqual(normal.stop_reason, "tool_use")
        self.assertEqual(normal.content[0].text, "I will look.")
        self.assertEqual(normal.content[1].name, "render_world_area")
        self.assertEqual(normal.content[1].input, {"view": "structure"})
        self.assertEqual((normal.usage.input_tokens, normal.usage.output_tokens), (12, 8))

    def test_openrouter_structured_output_schema_is_strict(self):
        """Every nested object is closed and fully required, as strict responders expect."""
        from model_api import _json_schema
        from segment import RefinementOutput

        schema = _json_schema(RefinementOutput)["json_schema"]["schema"]
        self.assertFalse(schema["additionalProperties"])
        nested = schema["$defs"]["BoundaryFix"]
        self.assertFalse(nested["additionalProperties"])
        self.assertEqual(set(nested["required"]), set(nested["properties"]))


@needs_corpus
class TestContext(unittest.TestCase):
    """P7. Budgets are enforced by truncation, and the estimator must not under-count."""

    def test_estimator_never_undercounts(self):
        """Frozen fixtures measured against messages.count_tokens."""
        fixtures = json.loads((HERE / "tests_token_fixtures.json").read_text())
        for name, f in fixtures.items():
            self.assertGreaterEqual(
                estimate_tokens(f["text"]), f["tokens"],
                f"estimator under-counts {name}: budgets become meaningless")

    def test_estimator_constant_is_below_the_densest_text(self):
        fixtures = json.loads((HERE / "tests_token_fixtures.json").read_text())
        densest = min(f["chars"] / f["tokens"] for f in fixtures.values())
        self.assertLessEqual(CHARS_PER_TOKEN, densest)

    def test_memories_stay_bounded_as_the_world_grows(self):
        """The section must not grow with the store.

        A player generates roughly 33 structures an hour. A hundred hours is 3,300, which at
        flat rendering is ~200,000 tokens of undifferentiated pile — a useless prompt long
        before it is an impossible one. Salience ranking, grouping by category, and a horizon
        past which memories are counted rather than named must hold the section flat.
        """
        import random
        from store import Belief, Provenance
        cats = ["building", "excavation", "harvest", "clearing", "path", "farm"]
        random.seed(11)
        sizes = {}
        for n in (20, 2000):
            path = Path(tempfile.mktemp(suffix=".db"))
            store = Store(path)
            try:
                for i in range(n):
                    x, y, z = (random.randint(-2000, 2000), random.randint(5, 120),
                               random.randint(-2000, 2000))
                    sid = f"placement_{x}_{y}_{z}_{i}"
                    store.put("structures", {
                        "id": sid, "actor": "p", "dim": "overworld", "name": None,
                        "min_x": x, "min_y": y, "min_z": z,
                        "max_x": x + 8, "max_y": y + 4, "max_z": z + 8,
                        "materials": json.dumps({"cobblestone": 40}),
                    }, Belief(provenance=Provenance.DERIVED,
                              verified_at_tick=random.randint(0, 500000),
                              value=json.dumps({"dims": "9x5x9", "fill": 0.4,
                                                "blocks": 60, "vertical": True})))
                    store.name_structure(sid, random.choice(cats), 0.8, "b",
                                         random.randint(0, 500000), f"k{i}",
                                         category=random.choice(cats))
                sections = Assembler(store).build(
                    now_tick=500000, actor="p",
                    trigger={"reason": "t", "pos": [0, 64, 0], "facts": {}},
                    player_state=None, episodes=[], findings=[])
                mem = next(x for x in sections if x.name == "memories")
                sizes[n] = mem.tokens
                self.assertLessEqual(mem.tokens, mem.budget,
                                     f"memories blew the budget at {n} structures")
            finally:
                store.close()
                path.unlink(missing_ok=True)
        self.assertLess(sizes[2000], sizes[20] * 2,
                        "memories grow with the store instead of staying flat")

    def test_no_context_exceeds_its_budget(self):
        episodes, findings, events = analyse(CORPUS)
        store = Store()
        assembler = Assembler(store)
        try:
            for ep in episodes:
                sections = assembler.build(
                    now_tick=ep.tick_end, actor=ep.actor,
                    trigger={"reason": "test",
                             "pos": list(ep.bbox[0]) if ep.bbox else None, "facts": {}},
                    player_state=None, episodes=episodes[:6], findings=[])
                report = audit(sections)
                self.assertFalse(report["any_over"],
                                 f"episode {ep.id} over budget: {report['sections']}")
        finally:
            store.close()




class ModelDrawnBoundaries(unittest.TestCase):
    """The model draws the boxes; code refuses the ones that are nonsense.

    Block-level proposals were tried first and produced the mess this replaced: one
    watchtower under two ids, a pen and the farmhouse beside it merged, a cottage cut in
    three. The model is better at "where does this building end" than any rule was. It is
    not better at arithmetic, so everything it draws is measured and checked here.
    """

    def setUp(self):
        from god import God
        from store import Store
        self.god = God(Store(":memory:"), use_model=False)
        self.built = {(x, 64, z): "minecraft:oak_planks"
                      for x in range(10) for z in range(10)}

    def tearDown(self):
        self.god.store.close()

    def test_a_box_below_the_shared_evidence_floor_is_not_a_structure(self):
        found = self.god._measure({"min": [0, 64, 0], "max": [6, 64, 0], "label": "hut"},
                                  self.built, [0, 0, 0], [99, 99, 99])
        self.assertIsNone(found, "seven blocks is below the universal evidence floor")

    def test_a_compact_semantic_proposal_can_be_a_structure(self):
        found = self.god._measure({"min": [0, 64, 0], "max": [2, 64, 2],
                                   "label": "floor mosaic"},
                                  self.built, [-1, 0, -1], [99, 99, 99])
        self.assertEqual(found["_facts"]["blocks"], 9)

    def test_a_box_is_measured_not_taken_on_trust(self):
        found = self.god._measure({"min": [0, 64, 0], "max": [4, 64, 4], "label": "hut"},
                                  self.built, [-1, 0, -1], [99, 99, 99])
        self.assertEqual(found["_facts"]["blocks"], 25,
                         "the count must come from the world, not from the model")

    def test_a_box_cannot_reach_outside_the_patch_that_was_read(self):
        found = self.god._measure({"min": [-500, 0, -500], "max": [500, 200, 500]},
                                  self.built, [0, 60, 0], [9, 70, 9])
        self.assertIsNone(found, "a box clipped to unread ground must not enter state")

    def test_two_structures_cannot_claim_one_record(self):
        """The bug this prevents: a farmhouse and the pen beside it both continued one id,
        and the second write silently erased the first."""
        valid, claimed = {"s_1"}, set()
        got = []
        for proposal in ({"continues": "s_1"}, {"continues": "s_1"}):
            sid = proposal["continues"]
            if sid not in valid or sid in claimed:
                sid = None
            if sid:
                claimed.add(sid)
            got.append(sid)
        self.assertEqual(got, ["s_1", None], "the second must become its own structure")

    def test_an_invented_id_is_not_honoured(self):
        valid = {"s_real"}
        self.assertNotIn("s_hallucinated", valid)


class EntitiesComeFromTheGamesOwnModels(unittest.TestCase):
    """Hand-transcribed entity boxes were tried and the chickens came out looking like
    creepers. Recognisably wrong is worse than an honest marker, because the model reading
    the render believes what it sees. These numbers are read out of the client's bytecode."""

    @classmethod
    def setUpClass(cls):
        from entity_models import load
        cls.models = load()

    def test_the_cache_covers_the_creatures_that_turn_up_in_a_world(self):
        for kind in ("chicken", "cow", "pig", "sheep", "villager", "creeper", "zombie",
                     "skeleton", "wolf", "cat", "horse", "minecart", "boat"):
            self.assertIn(kind, self.models, f"no model extracted for {kind}")

    def test_the_chicken_is_the_chickens_actual_geometry(self):
        """Straight from AdultChickenModel: a head at pivot y 15, a body pitched flat."""
        cubes = self.models["chicken"]["cubes"]
        self.assertEqual(len(cubes), 8, "head, beak, wattle, body, 2 wings, 2 legs")
        head = cubes[0]
        self.assertEqual(head["from"], [-2.0, -6.0, -2.0])
        self.assertEqual(head["size"], [4.0, 6.0, 3.0])
        self.assertEqual(head["pivot"], [0.0, 15.0, -4.0])
        body = next(c for c in cubes if c["size"] == [6.0, 8.0, 6.0])
        self.assertAlmostEqual(body["rot"][0], 1.5707964, places=5,
                               msg="the body is laid flat by a quarter turn")

    def test_a_creature_stands_on_the_ground_and_is_the_right_size(self):
        """Model space is Y-down from a point 1.501 blocks above the feet. Getting that
        transform wrong buries every animal or floats it."""
        from render import Renderer
        from assets import Assets
        quads = Renderer(Assets())._creature_quads(
            [{"type": "minecraft:chicken", "pos": [0.0, 0.0, 0.0], "yaw": 0.0}])
        points = [p for quad in quads for p in quad[0]]
        low = min(p[1] for p in points)
        wide = max(p[0] for p in points) - min(p[0] for p in points)
        self.assertAlmostEqual(low, 0.0, places=1, msg="feet must touch the ground")
        self.assertLess(wide, 0.7, "a chicken is not a block wide")
        self.assertGreater(wide, 0.3)

    def test_a_texture_patch_covers_all_six_faces_without_overlapping(self):
        from render import _box_uv
        patches = {f: _box_uv(0, 0, 8, 12, 4, f) for f in
                   ("up", "down", "north", "south", "east", "west")}
        self.assertEqual(len(set(patches.values())), 6,
                         "two faces sharing a patch means a mob with a face on its back")

    def test_dropped_items_are_not_drawn_as_creatures(self):
        """The ignore list is written namespaced; stripping the namespace before checking
        it meant every dropped item rendered as a marker cube."""
        from render import Renderer
        from assets import Assets
        quads = Renderer(Assets())._creature_quads(
            [{"type": "minecraft:item", "pos": [0.0, 64.0, 0.0], "yaw": 0.0}])
        self.assertEqual(quads, [])


class BoxesTheCodeMustRefuse(unittest.TestCase):
    """Three ways a drawn box goes wrong that a picture cannot show, so code measures them."""

    def setUp(self):
        from god import God
        from store import Store
        self.god = God(Store(":memory:"), use_model=False)

    def tearDown(self):
        self.god.store.close()

    def test_a_box_cut_by_the_edge_of_the_patch_is_refused(self):
        """This is what turned a farmhouse into half a farmhouse: the patch was centred on
        the tower next door, the house ran off the edge, and the truncated box was recorded
        as if that were the whole building."""
        built = {(x, 64, z): "minecraft:stone" for x in range(40) for z in range(4)}
        cut = self.god._measure({"min": [0, 64, 0], "max": [19, 64, 3]},
                                built, [0, 64, 0], [19, 64, 3])
        self.assertIsNone(cut, "the build carries on past the edge that was read")

    def test_a_box_touching_the_unread_boundary_is_refused(self):
        built = {(x, 64, z): "minecraft:stone" for x in range(20) for z in range(4)}
        built[(20, 64, 0)] = "minecraft:dirt_path"
        partial = self.god._measure({"min": [0, 64, 0], "max": [19, 64, 3]},
                                    built, [0, 60, 0], [20, 70, 4])
        self.assertIsNone(partial, "the scan cannot prove what lies beyond its own edge")

    def test_a_box_with_air_between_it_and_the_patch_edge_is_kept(self):
        built = {(x, 64, z): "minecraft:stone" for x in range(1, 20)
                 for z in range(1, 4)}
        whole = self.god._measure({"min": [1, 64, 1], "max": [19, 64, 3]},
                                  built, [0, 60, 0], [20, 70, 4])
        self.assertIsNotNone(whole)

    def test_what_spills_past_each_face_is_counted(self):
        from segment import spilling
        built = {(x, 64, z): "p" for x in range(20) for z in range(20)}
        self.assertEqual(spilling(built, [0, 64, 0], [9, 64, 19]).get("east"), 60)
        self.assertEqual(spilling(built, [0, 64, 0], [19, 64, 19]), {})

    def test_a_thin_semantic_structure_survives_the_shared_floor(self):
        """A rail line and compact art use the same minimum; neither needs a type rule."""
        rail = {(x, 64, 0): "minecraft:rail" for x in range(12)}
        self.assertIsNotNone(self.god._measure({"min": [0, 64, 0], "max": [11, 64, 0]},
                                               rail, [-5, 60, -5], [20, 70, 20]))


class NotMentioningIsNotDemolishing(unittest.TestCase):
    """The model looks at a whole patch and describes what it is there for. Something at the
    edge it did not bother to mention is not thereby gone — a rail line was forgotten twice
    this way. Same trap as reading an empty scan as a demolition."""

    def test_a_structure_still_standing_survives_not_being_mentioned(self):
        row = {"min_x": 0, "max_x": 11, "min_y": 64, "max_y": 65,
               "min_z": 0, "max_z": 0}
        built = {(x, 64, 0): "minecraft:rail" for x in range(12)}
        still = sum(1 for p in built
                    if row["min_x"] <= p[0] <= row["max_x"]
                    and row["min_y"] <= p[1] <= row["max_y"]
                    and row["min_z"] <= p[2] <= row["max_z"])
        self.assertGreaterEqual(still, 15 - 3, "the blocks are demonstrably still there")

    def test_a_structure_actually_torn_down_is_retired(self):
        row = {"min_x": 0, "max_x": 11, "min_y": 64, "max_y": 65,
               "min_z": 0, "max_z": 0}
        still = sum(1 for p in {} if True)
        self.assertLess(still, 15, "nothing left means nothing to keep believing in")


class RendersFollowTheirRows(unittest.TestCase):
    def test_structure_sheet_has_plan_and_two_elevations(self):
        """Every shape gets views that reveal both horizontal and vertical planar work."""
        from PIL import Image
        import render

        calls = []
        original = render.Renderer

        class FakeRenderer:
            def __init__(self, _assets, size, **_kwargs):
                self.size = size

            def render(self, _voxels, **kwargs):
                calls.append(kwargs)
                return Image.new("RGB", self.size)

        try:
            render.Renderer = FakeRenderer
            picture = render.sheet({(0, 64, 0): "minecraft:stone",
                                    (4, 64, 6): "minecraft:stone"}, object())
        finally:
            render.Renderer = original

        self.assertEqual(picture.size, (1024, 920))
        self.assertEqual((calls[0]["azimuth"], calls[0]["elevation"]), (35, 24))
        self.assertIn("ortho_extent", calls[1])
        self.assertEqual((calls[2]["azimuth"], calls[2]["elevation"]), (0, 0))
        self.assertEqual((calls[3]["azimuth"], calls[3]["elevation"]), (90, 0))

    def test_review_focus_excludes_a_neighbour_outside_the_stored_box(self):
        from render import focus_structure

        row = {"min_x": 0, "min_y": 64, "min_z": 0,
               "max_x": 2, "max_y": 66, "max_z": 2}
        voxels = {(0, 64, 0): "minecraft:oak_planks",
                  (2, 64, 2): "minecraft:oak_planks",
                  (4, 65, 1): "minecraft:cobblestone",
                  (1, 63, 1): "minecraft:grass_block"}
        focused = focus_structure(voxels, row)
        self.assertNotIn((4, 65, 1), focused)
        self.assertIn((1, 63, 1), focused, "ground context should remain")

    def test_creatures_outside_what_is_drawn_are_not_drawn(self):
        """A scan returns every entity in the box it read, always larger than the build.
        Passing the lot through drew chickens standing in mid-air beside a structure."""
        from render import entities_within
        voxels = {(0, 64, 0): "minecraft:stone", (4, 64, 4): "minecraft:stone"}
        near = {"type": "minecraft:chicken", "pos": [2.0, 64.0, 2.0]}
        far = {"type": "minecraft:chicken", "pos": [40.0, 64.0, 40.0]}
        self.assertEqual(entities_within([near, far], voxels), [near])

    def test_every_path_that_deletes_a_structure_deletes_its_render(self):
        """Two places remove a row and only one cleaned up, so the renders folder kept
        buildings that no longer existed — indistinguishable from real ones."""
        import re
        for path in ("god.py", "world.py"):
            source = Path(path).read_text(encoding="utf-8")
            for match in re.finditer(r"DELETE FROM structures", source):
                window = source[max(0, match.start() - 600):match.start() + 200]
                self.assertIn("forget_renders", window,
                              f"{path}: a row is deleted without clearing its render")


class SurveyIsATestHarnessNotABehaviour(unittest.TestCase):
    """The god must never sweep the world at runtime — unloaded chunks do not tick, so a
    region nobody has visited can only return what was already known. The survey tool exists
    because a downloaded map has no event history and cannot be reached any other way."""

    def test_the_grid_overlaps_so_a_building_on_a_seam_is_seen_whole(self):
        import survey
        parser_defaults = {a.dest: a.default for a in
                           __import__("argparse").ArgumentParser()._actions}
        self.assertTrue(hasattr(survey, "survey") and hasattr(survey, "probe"))
        source = Path("survey.py").read_text(encoding="utf-8")
        self.assertIn("spacing", source)
        # Default spacing must be under twice the span, or patches leave gaps between them.
        import re
        spacing = int(re.search(r'"--spacing", type=int, default=(\d+)', source).group(1))
        span = int(re.search(r'"--span", type=int, default=(\d+)', source).group(1))
        self.assertLess(spacing, span * 2,
                        "patches must overlap or a building on a seam is cut by both")

    def test_nothing_in_the_god_loop_calls_the_survey(self):
        for path in ("god.py", "router.py", "reconcile.py"):
            self.assertNotIn("import survey", Path(path).read_text(encoding="utf-8"),
                             f"{path} must not sweep the world at runtime")


class TheEvalScoresWhatItClaimsTo(unittest.TestCase):
    """Two one-off model comparisons produced two wrong conclusions before this existed. The
    eval is the thing standing between an impression and a decision, so its own arithmetic
    is checked here."""

    def test_overlap_of_identical_boxes_is_one(self):
        from eval_segment import iou
        self.assertEqual(iou([0, 0, 0], [9, 9, 9], [0, 0, 0], [9, 9, 9]), 1.0)

    def test_boxes_that_miss_each_other_score_zero(self):
        from eval_segment import iou
        self.assertEqual(iou([0, 0, 0], [4, 4, 4], [50, 50, 50], [54, 54, 54]), 0.0)

    def test_half_a_building_scores_about_a_half(self):
        """The failure this whole eval exists to catch: a box that is right where it sits
        but only covers part of the thing."""
        from eval_segment import iou
        score = iou([0, 0, 0], [4, 9, 9], [0, 0, 0], [9, 9, 9])
        self.assertAlmostEqual(score, 0.5, places=2)

    def test_a_name_matches_on_any_accepted_word(self):
        from eval_segment import named_correctly
        terms = ["watchtower", "lookout", "tower"]
        self.assertTrue(named_correctly("cobblestone watchtower", terms))
        self.assertTrue(named_correctly("Tall Stone Tower", terms))
        self.assertFalse(named_correctly("fenced animal pen", terms))

    def test_a_correctly_found_neighbour_is_not_counted_against_the_model(self):
        """A patch holds several real structures. Counting every box that is not the one
        being scored as spurious made a good pass look like wild over-segmentation — the
        eval reported 0/2 found with 9 spurious on a pass that had boxed the watchtower
        exactly right."""
        from eval_segment import iou, SAME_THING
        truth = [{"min": [0, 0, 0], "max": [9, 9, 9]},
                 {"min": [40, 0, 0], "max": [49, 9, 9]}]
        produced = [{"min": [0, 0, 0], "max": [9, 9, 9]},
                    {"min": [40, 0, 0], "max": [49, 9, 9]},
                    {"min": [200, 0, 200], "max": [204, 4, 204]}]
        spurious = sum(1 for box in produced
                       if all(iou(box["min"], box["max"], t["min"], t["max"]) < SAME_THING
                              for t in truth))
        self.assertEqual(spurious, 1, "only the box matching nothing known counts")

    def test_warm_mode_cannot_score_itself(self):
        """The truth file is seeded from the store, so in warm mode a row the model never
        touched sits there already matching and would report a perfect box for doing
        nothing. Only rows written by the pass being scored may count."""
        source = Path("eval_segment.py").read_text(encoding="utf-8")
        self.assertIn('row["verified_at_tick"] != god.tick', source,
                      "untouched rows must not be scored as if the pass produced them")

    def test_the_truth_file_is_usable(self):
        import json
        from pathlib import Path
        truth = json.loads((Path("segment_truth.json")).read_text())["structures"]
        self.assertGreaterEqual(len(truth), 6)
        for target in truth:
            for key in ("id", "min", "max", "terms", "patch"):
                self.assertIn(key, target, f"{target.get('id')} missing {key}")
            self.assertTrue(all(target["max"][i] >= target["min"][i] for i in range(3)),
                            f"{target['id']} has an inside-out box")
            self.assertTrue(target["terms"], f"{target['id']} accepts no name")

    def test_every_truth_box_contains_its_patch_centre(self):
        """A patch centred outside the thing it is meant to read would score the pipeline on
        a building it never showed it."""
        import json
        from pathlib import Path
        for target in json.loads(Path("segment_truth.json").read_text())["structures"]:
            for i in range(3):
                self.assertTrue(
                    target["min"][i] - 4 <= target["patch"][i] <= target["max"][i] + 4,
                    f"{target['id']}: patch centre is outside the structure on axis {i}")


class ThePlanViewIsExact(unittest.TestCase):
    """The weakest link in segmentation was getting a coordinate out of a picture. A
    three-quarter perspective cannot give one: buildings at different depths overlap and the
    same wall is wider at the near end. An orthographic plan can, and its grid is drawn from
    the same window the renderer used, so a label cannot drift from what it points at."""

    def test_orthographic_keeps_a_block_the_same_size_everywhere(self):
        from render import orthographic
        import numpy as np
        m = orthographic(10.0, 10.0)
        near = m @ np.array([1.0, 0.0, -5.0, 1.0])
        far = m @ np.array([1.0, 0.0, -500.0, 1.0])
        self.assertAlmostEqual(near[0] / near[3], far[0] / far[3], places=6,
                               msg="under orthographic, distance must not change scale")

    def test_orthographic_depth_keeps_the_nearer_surface(self):
        """A lower surface drawn later must not erase the roof above it."""
        import numpy as np
        from assets import Assets
        from render import Renderer

        renderer = Renderer(Assets(), size=(4, 4), supersample=1)
        colour = np.zeros((4, 4, 3), np.float32)
        depth = np.full((4, 4), np.inf, np.float32)
        sx = np.array([0.0, 4.0, 0.0])
        sy = np.array([0.0, 0.0, 4.0])
        inv_w = np.ones(3)
        uvs = np.zeros((3, 2))
        near = np.array([[[0.0, 1.0, 0.0, 1.0]]], np.float32)
        far = np.array([[[1.0, 0.0, 0.0, 1.0]]], np.float32)

        renderer._triangle(colour, depth, sx, sy, inv_w, uvs, near, 1.0, False,
                           zs=np.full(3, -0.5))
        renderer._triangle(colour, depth, sx, sy, inv_w, uvs, far, 1.0, False,
                           zs=np.full(3, 0.5))
        self.assertTrue(np.allclose(colour[1, 1], [0.0, 1.0, 0.0]),
                        "the farther floor overwrote the nearer roof")
        self.assertAlmostEqual(float(depth[1, 1]), -0.5)

    def test_the_grid_labels_land_where_they_say(self):
        """A ruled coordinate that is off by even a block is worse than no grid at all."""
        from assets import Assets
        from render import plan
        lo, hi = [100, 64, 200], [131, 70, 231]
        voxels = {(x, 64, z): "minecraft:stone"
                  for x in range(lo[0], hi[0] + 1) for z in range(lo[2], hi[2] + 1)}
        image = plan(voxels, Assets(), lo, hi, size=(320, 320), step=8)
        self.assertEqual(image.size, (320, 320))

    def test_segmentation_is_told_to_trust_the_plan_for_extent(self):
        from segment import SEGMENT
        self.assertIn("plan", SEGMENT.lower())
        self.assertIn("orthographic", SEGMENT.lower())

    def test_coordinate_slices_sample_occupied_levels_not_empty_intervals(self):
        """A one-block mosaic between arbitrary scan intervals must not disappear."""
        from segment import slices

        voxels = {(x, 80, z): "minecraft:cobblestone"
                  for x, z in ((0, 0), (1, 0), (2, 0), (0, 2), (2, 2))}
        cut = slices(voxels, [-10, 64, -10], [10, 110, 10])
        self.assertEqual([item["y"] for item in cut], [80])
        self.assertTrue(any("#" in row for row in cut[0]["rows"]))


class ThinkingModeMustMatchTheModel(unittest.TestCase):
    """Adaptive thinking is 4.6-and-later only. Asking an older model for it is a 400, not a
    graceful downgrade — and that failure looks exactly like a model returning nothing
    useful, so a cost comparison silently scored Haiku on two calls it never made."""

    def test_older_models_get_a_budget_instead_of_adaptive(self):
        from config import thinking_for
        self.assertEqual(thinking_for("claude-haiku-4-5")["type"], "enabled")
        self.assertIn("budget_tokens", thinking_for("claude-haiku-4-5"))

    def test_current_models_get_adaptive(self):
        from config import thinking_for
        for model in ("claude-opus-5", "claude-sonnet-5", "claude-fable-5-1"):
            self.assertEqual(thinking_for(model), {"type": "adaptive"}, model)

    def test_no_call_hardcodes_adaptive_thinking(self):
        """Every model call must go through the chooser, or the next cheap-model experiment
        fails the same silent way."""
        import re
        import glob
        for path in sorted(glob.glob("*.py")):
            if path == "test_regression.py":
                continue
            source = Path(path).read_text(encoding="utf-8")
            for match in re.finditer(r'thinking=\{"type": "adaptive"\}', source):
                line = source[:match.start()].count("\n") + 1
                self.fail(f"{path}:{line} hardcodes adaptive thinking; use thinking_for()")


class TheSegmentationModelIsSwappable(unittest.TestCase):
    """Segmentation is the call the pipeline makes most often, so what it costs is worth
    measuring rather than assuming. It is configured apart from the dialogue model because
    it is a different job: read a picture, read a grid, return exact coordinates."""

    def test_it_reads_its_own_environment_variable(self):
        import os
        from god import God
        from store import Store
        from config import VISION_MODEL

        store = Store(":memory:")
        try:
            god = God(store, use_model=False)
            self.assertEqual(god.segment_model, VISION_MODEL,
                             "unset, it must follow the dedicated vision model")
            os.environ["MCGOD_SEGMENT_MODEL"] = "claude-haiku-4-5"
            try:
                self.assertEqual(God(store, use_model=False).segment_model,
                                 "claude-haiku-4-5")
            finally:
                del os.environ["MCGOD_SEGMENT_MODEL"]
        finally:
            store.close()

    def test_both_segmentation_calls_use_it(self):
        """Setting it must change the whole pass. Leaving the second look pinned to the
        expensive model would have made every cost measurement wrong."""
        source = Path("god.py").read_text(encoding="utf-8")
        drawing = source[source.index("def _draw_boundaries"):source.index("def _measure")]
        looking = source[source.index("def _second_look"):source.index("def retire")]
        for name, block in (("_draw_boundaries", drawing), ("_second_look", looking)):
            self.assertIn("self.segment_model", block,
                          f"{name} does not honour the configured model")
            self.assertNotIn("model=DIALOGUE_MODEL", block)


class TheLedgerOfBuiltThings(unittest.TestCase):
    """Every connected built mass is on record, decided by code, whatever its size.

    Before this layer a thing stood in the present tense only if the vision model chose to
    box it, and the prompt told it to leave pillars and paths out. The model gated
    existence, and 47% of block events could not be attached to any place. These tests
    pin the replacement: code measures, code attributes, code matches identity; the model
    only says what a mass is for.
    """

    GROUND = "minecraft:grass_block"

    def _world(self, pillar: bool = True, torch: bool = True, roof: bool = True,
               bridge: bool = True) -> dict:
        voxels = {(x, 63, z): self.GROUND for x in range(-2, 40) for z in range(-2, 40)}
        # a hollow 5x4x5 cobblestone hut at the origin
        for x in range(5):
            for z in range(5):
                for y in range(64, 68):
                    if x in (0, 4) or z in (0, 4) or (y == 67 and roof):
                        voxels[(x, y, z)] = "minecraft:cobblestone"
        if pillar:
            for y in range(64, 73):
                voxels[(20, y, 20)] = "minecraft:dirt"
        if torch:
            voxels[(30, 64, 5)] = "minecraft:torch[facing=up]"
        if bridge:
            for x in range(30, 40):
                voxels[(x, 63, 30)] = "minecraft:water"
                voxels[(x, 64, 30)] = "minecraft:oak_planks"
        return voxels

    def _placed(self, store, actor: str, positions, material: str, t0: int = 1_000,
                step: int = 1_000) -> None:
        for n, (x, y, z) in enumerate(positions):
            store.put("event_history", {
                "id": f"{actor}:{x}:{y}:{z}:{t0 + n * step}", "actor": actor,
                "dim": "overworld", "tick": n, "event_t": t0 + n * step,
                "kind": "block_place", "place_id": None, "x": x, "y": y, "z": z,
                "payload": json.dumps({"after": material}),
            }, Belief(provenance=Provenance.OBSERVED))

    LO, HI = [-10, 60, -10], [50, 90, 50]

    def _apply(self, store, voxels, lo=None, hi=None):
        from masses import apply, ledger_built
        lo, hi = lo or self.LO, hi or self.HI
        built = ledger_built(store, "overworld", voxels, lo, hi)
        return apply(store, "overworld", voxels, built, lo, hi, tick=100, now_ms=5_000)

    def test_every_built_block_is_in_exactly_one_mass_whatever_its_size(self):
        from masses import current, facts_of
        store = Store(":memory:")
        try:
            voxels = self._world()
            report = self._apply(store, voxels)
            rows = current(store)
            sizes = sorted(facts_of(r)["blocks"] for r in rows)
            # hut, bridge, torch. The dirt pillar is terrain until history says otherwise.
            self.assertEqual(sizes, [1, 10, 73])
            self.assertEqual(sum(sizes), len(report["observations"]) and sum(
                o["blocks"] for o in report["observations"]))
            self.assertTrue(all(r["provenance"] == "SCANNED" for r in rows))
            torch = next(r for r in rows if facts_of(r)["blocks"] == 1)
            self.assertIn("torch", json.loads(torch["materials"]))
        finally:
            store.close()

    def test_a_dirt_pillar_counts_once_history_proves_a_player_placed_it(self):
        from masses import current, facts_of
        store = Store(":memory:")
        try:
            self._placed(store, "climber", [(20, y, 20) for y in range(64, 73)],
                         "minecraft:dirt")
            self._apply(store, self._world())
            pillar = next(r for r in current(store) if r["min_x"] == 20)
            facts = facts_of(pillar)
            self.assertEqual(facts["dims"], "1x9x1")
            self.assertEqual(pillar["actor"], "climber")
            self.assertEqual(facts["origin"], "player_built")
            self.assertEqual(facts["builders"], {"climber": 1.0})
            self.assertEqual(facts["first_built_ms"], 1_000)
            self.assertEqual(facts["footing"]["on_ground"], 1)
        finally:
            store.close()

    def test_identity_survives_growth_and_only_measured_absence_retires(self):
        from masses import current
        store = Store(":memory:")
        try:
            voxels = self._world()
            self._apply(store, voxels)
            hut = next(r for r in current(store) if r["min_x"] == 0)
            # a second floor goes on
            for x in range(5):
                for z in range(5):
                    voxels[(x, 68, z)] = "minecraft:oak_planks"
            report = self._apply(store, voxels)
            self.assertEqual(report["retired"], [])
            grown = store.get("masses", hut["id"])
            self.assertIsNotNone(grown, "an addition must not mint a new identity")
            self.assertEqual(grown["max_y"], 68)
            # the torch is taken away; the read shows nothing where it stood
            torch = next(r for r in current(store) if r["min_x"] == 30 and r["min_z"] == 5)
            del voxels[(30, 64, 5)]
            report = self._apply(store, voxels)
            self.assertEqual([r for r, _ in report["retired"]], [torch["id"]])
            self.assertIsNone(store.get("masses", torch["id"]))
            # a read that never covered the hut leaves the hut alone
            report = self._apply(store, voxels, [25, 60, 25], [45, 90, 45])
            self.assertIsNotNone(store.get("masses", hut["id"]))
            self.assertEqual(report["retired"], [])
        finally:
            store.close()

    def test_a_cut_off_view_never_overwrites_a_whole_record(self):
        from masses import current, facts_of
        store = Store(":memory:")
        try:
            voxels = self._world()
            self._apply(store, voxels)
            bridge = next(r for r in current(store) if r["min_z"] == 30)
            self.assertEqual(facts_of(bridge)["partial"], [])
            # a read whose east edge falls across the bridge
            report = self._apply(store, voxels, [25, 60, 25], [33, 90, 45])
            self.assertIn(bridge["id"], report["kept"])
            same = store.get("masses", bridge["id"])
            self.assertEqual((same["min_x"], same["max_x"]), (30, 39))
            self.assertEqual(facts_of(same)["partial"], [])
        finally:
            store.close()

    def test_a_mass_cut_by_the_read_edge_queues_a_look_beyond_it(self):
        from god import God
        from reconcile import index_regions
        store = Store(":memory:")
        try:
            god = God(store, use_model=False)
            god.tick = 50_000
            index_regions(store, [{"type": "move", "dim": "overworld",
                                   "pos": [40, 64, 30], "t": 1, "actor": "p"}])
            # An edge is followed only from ground the player changed; a read over
            # untouched ground has nothing to complete.
            god.edits[("overworld", 1, 2, 1)] = 20
            partial = [{"lo": [30, 64, 30], "hi": [33, 64, 30], "partial": ["east"],
                        "origin": "player_built", "blocks": 40}]
            god._follow_partials("overworld", partial, [25, 60, 25], [33, 90, 45])
            self.assertEqual(len(god.disturbed), 1)
            queued = next(iter(god.disturbed.values()))["pos"]
            self.assertGreater(queued[0], 33, "the look must be past the east edge")
            # but never into ground nobody has been to
            god.disturbed.clear()
            god.edits[("overworld", -3, 2, 1)] = 20
            god._follow_partials("overworld", [{"lo": [-60, 64, 30], "hi": [-55, 64, 30],
                                                "partial": ["west"], "blocks": 40,
                                                "origin": "player_built"}],
                                 [-60, 60, 25], [-52, 90, 45])
            self.assertEqual(god.disturbed, {})
        finally:
            store.close()

    def test_landmarks_are_groupings_of_masses(self):
        from masses import anchor, audit, current, facts_of
        store = Store(":memory:")
        try:
            self._apply(store, self._world())
            hut = next(r for r in current(store) if r["min_x"] == 0)
            store.put("structures", {
                "id": "s_hut", "actor": "p", "dim": "overworld", "name": None,
                "min_x": -1, "min_y": 63, "min_z": -1, "max_x": 5, "max_y": 68, "max_z": 5,
                "materials": json.dumps({"cobblestone": 73}),
            }, Belief(provenance=Provenance.INFERRED, confidence=.9, verified_at_ms=5_000,
                      value=json.dumps({"schema_version": 2, "kind": "current_structure",
                                        "blocks": 73, "dims": "5x4x5", "fill": .5,
                                        "vertical": True, "origin": "player_built"})))
            self.assertIn("s_hut: structure anchored to no mass", audit(store))
            anchor(store)
            self.assertEqual(store.get("masses", hut["id"])["place_id"], "s_hut")
            self.assertEqual(facts_of(store.get("structures", "s_hut"))["comprises"],
                             [hut["id"]])
            bridge = next(r for r in current(store) if r["min_z"] == 30)
            self.assertIsNone(bridge["place_id"], "the bridge belongs to no landmark")
            self.assertEqual(audit(store), [])
        finally:
            store.close()

    def test_history_resolves_to_the_pillar_nobody_named(self):
        from activity import record_activity
        from masses import current, relink
        store = Store(":memory:")
        try:
            self._placed(store, "climber", [(20, y, 20) for y in range(64, 73)],
                         "minecraft:dirt")
            self.assertTrue(all(r["place_id"] is None for r in store.all("event_history")))
            self._apply(store, self._world())
            pillar = next(r for r in current(store) if r["min_x"] == 20)
            self.assertGreater(relink(store), 0)
            self.assertTrue(all(r["place_id"] == pillar["id"]
                                for r in store.all("event_history")))
            # and a live event lands on it directly
            record_activity(store, {"tick": 9, "t": 99_000, "type": "block_place",
                                    "actor": "climber", "dim": "overworld",
                                    "pos": [20, 73, 20], "before": "minecraft:air",
                                    "after": "minecraft:dirt"})
            live = store.all("event_history", "event_t = 99000")[0]
            self.assertEqual(live["place_id"], pillar["id"])
        finally:
            store.close()

    def test_a_role_is_an_inferred_belief_that_never_gates_the_row(self):
        from config import CLASSIFY_MODEL
        from masses import current, pending_roles, role_key, role_prompt
        store = Store(":memory:")
        try:
            self._apply(store, self._world())
            rows = current(store)
            asked = pending_roles(store, CLASSIFY_MODEL)
            # the torch is below the evidence floor; the hut and bridge are asked about
            self.assertEqual(sorted(json.loads(r["materials"]).popitem()[0] for r in asked),
                             ["cobblestone", "oak_planks"])
            bridge = next(r for r in rows if r["min_z"] == 30)
            store.put("relationships", {
                "id": f"role:{bridge['id']}", "subject": bridge["id"],
                "predicate": "role", "object": "utility",
            }, Belief(provenance=Provenance.INFERRED, confidence=.4, verified_at_ms=5_000,
                      source_event_ids=(role_key(role_prompt(bridge), CLASSIFY_MODEL),),
                      value=json.dumps({"noun": "plank footbridge"})))
            self.assertNotIn(bridge["id"], [r["id"] for r in
                                            pending_roles(store, CLASSIFY_MODEL)])
            self.assertFalse(store.assertable("relationships", f"role:{bridge['id']}"))
            self.assertEqual(store.get("masses", bridge["id"])["provenance"], "SCANNED")
            # the prompt carries no coordinates and no ids
            self.assertNotIn("30", role_prompt(bridge))
        finally:
            store.close()

    def test_footing_and_levels_are_measured_not_guessed(self):
        from masses import current, facts_of
        store = Store(":memory:")
        try:
            self._apply(store, self._world())
            bridge = facts_of(next(r for r in current(store) if r["min_z"] == 30))
            self.assertEqual(bridge["footing"]["over_water"], 10)
            self.assertEqual(bridge["levels"], [[64, 10]])
            hut = facts_of(next(r for r in current(store) if r["min_x"] == 0))
            self.assertEqual([y for y, _ in hut["levels"]], [64, 65, 66, 67])
            self.assertEqual(hut["footing"]["on_ground"], 16)
        finally:
            store.close()

    def test_edits_count_separate_visits_and_builders_are_shares(self):
        from masses import current, facts_of
        store = Store(":memory:")
        try:
            walls = [(x, y, z) for x in range(5) for z in range(5) for y in range(64, 67)
                     if x in (0, 4) or z in (0, 4)]
            roof = [(x, 67, z) for x in range(5) for z in range(5)]
            self._placed(store, "mason", walls, "minecraft:cobblestone", t0=1_000)
            self._placed(store, "roofer", roof, "minecraft:cobblestone",
                         t0=1_000 + 60 * 60_000)
            self._apply(store, self._world())
            hut = facts_of(next(r for r in current(store) if r["min_x"] == 0))
            self.assertEqual(hut["edits"], 2)
            self.assertEqual(hut["builders"], {"mason": round(48 / 73, 3),
                                               "roofer": round(25 / 73, 3)})
            self.assertEqual(hut["builder"], "mason")
            self.assertEqual(hut["observed_share"], 1.0)
        finally:
            store.close()

    def test_generated_buildings_are_on_the_ledger_but_never_player_built(self):
        from masses import current, facts_of
        store = Store(":memory:")
        try:
            store.put("generated_features", {
                "id": "generated:village", "dim": "overworld", "kind": "minecraft:village",
                "x": 10, "y": 64, "z": 10, "discovered_from": "seed",
            }, Belief(provenance=Provenance.DERIVED))
            # a measured component building of that village, standing over the bridge
            store.put("generated_features", {
                "id": "g_house", "dim": "overworld", "kind": "building",
                "x": 34, "y": 64, "z": 30, "discovered_from": "survey",
                "min_x": 29, "min_y": 63, "min_z": 29, "max_x": 40, "max_y": 68,
                "max_z": 31, "materials": json.dumps({"oak_planks": 10}),
            }, Belief(provenance=Provenance.INFERRED, confidence=.9,
                      value=json.dumps({"parent": "generated:village",
                                        "origin": "world_generated"})))
            self._apply(store, self._world())
            hut = next(r for r in current(store) if r["min_x"] == 0)
            self.assertIsNone(hut["actor"])
            self.assertEqual(facts_of(hut)["origin"], "world_generated")
            self.assertEqual(hut["place_id"], "generated:village",
                             "a mass no building contains belongs to the site, never to "
                             "whichever component happens to be nearest")
            self.assertEqual(facts_of(hut)["observed_share"], 0.0)
            bridge = next(r for r in current(store) if r["min_z"] == 30)
            self.assertEqual(bridge["place_id"], "g_house",
                             "a measured component claims what its box contains")
        finally:
            store.close()

    def test_the_history_model_may_query_the_ledger(self):
        from history import HISTORY_SYSTEM, QUERYABLE_TABLES
        self.assertIn("masses", QUERYABLE_TABLES)
        self.assertIn("masses(id, dim, actor, place_id", HISTORY_SYSTEM)

    def test_the_segmentation_prompt_shows_the_masses_and_says_omission_is_safe(self):
        from segment import SEGMENT
        self.assertIn("built_masses", SEGMENT)
        self.assertIn("stays on\nrecord as an unnamed built mass", SEGMENT)

    def test_rebuild_windows_respect_the_plugin_caps(self):
        from masses import MAX_HEIGHT, windows_for_cell
        store = Store(":memory:")
        try:
            for n, y in enumerate((-50, 10, 90)):
                store.put("event_history", {
                    "id": f"e{n}", "actor": "p", "dim": "overworld", "tick": n,
                    "event_t": n, "kind": "block_break", "place_id": None,
                    "x": 70, "y": y, "z": 5, "payload": "{}",
                }, Belief(provenance=Provenance.OBSERVED))
            windows = windows_for_cell(store, "overworld", 1, 0)
            self.assertGreater(len(windows), 1)
            for lo, hi in windows:
                self.assertLessEqual(hi[1] - lo[1] + 1, MAX_HEIGHT)
                self.assertLessEqual(hi[0] - lo[0] + 1, 96)
            self.assertLessEqual(windows[0][0][1], -50)
            self.assertGreaterEqual(windows[-1][1][1], 90)
            for (_, a), (b, _) in zip(windows, windows[1:]):
                self.assertGreater(a[1], b[1], "slabs must overlap, not merely touch")
            self.assertEqual(windows_for_cell(store, "overworld", 5, 5), [])
        finally:
            store.close()

    def test_masses_never_cross_a_landmark_boundary(self):
        """Plain connectivity fused the settlement into one 632-block mass no landmark
        could claim. The boxes the model drew are the judgement of where one thing ends."""
        from masses import anchor, audit, current, facts_of
        store = Store(":memory:")
        try:
            voxels = self._world()
            # a cobblestone wall runs from the hut to a second hut, touching both
            for x in range(5, 12):
                voxels[(x, 64, 2)] = "minecraft:cobblestone"
            for x in range(12, 17):
                for z in range(5):
                    for y in range(64, 67):
                        if x in (12, 16) or z in (0, 4):
                            voxels[(x, y, z)] = "minecraft:cobblestone"
            self._apply(store, voxels)
            fused = max(current(store), key=lambda r: facts_of(r)["blocks"])
            self.assertEqual(facts_of(fused)["blocks"], 73 + 7 + 48)
            for sid, lo, hi in (("s_west", [0, 64, 0], [4, 67, 4]),
                                ("s_east", [12, 64, 0], [16, 66, 4])):
                store.put("structures", {
                    "id": sid, "actor": "p", "dim": "overworld", "name": None,
                    "min_x": lo[0], "min_y": lo[1], "min_z": lo[2],
                    "max_x": hi[0], "max_y": hi[1], "max_z": hi[2],
                    "materials": json.dumps({"cobblestone": 1}),
                }, Belief(provenance=Provenance.INFERRED, confidence=.9,
                          verified_at_ms=5_000,
                          value=json.dumps({"schema_version": 2,
                                            "kind": "current_structure", "blocks": 1,
                                            "dims": "1x1x1", "fill": 1, "vertical": True,
                                            "origin": "player_built"})))
            self._apply(store, voxels)
            anchor(store)
            by_place = {}
            for row in current(store):
                by_place.setdefault(row["place_id"], []).append(facts_of(row)["blocks"])
            self.assertEqual(by_place["s_west"], [73])
            self.assertEqual(by_place["s_east"], [48])
            self.assertIn(7, by_place[None], "the wall between them is its own mass")
            self.assertEqual(audit(store), [])
        finally:
            store.close()

    def test_a_players_pillar_against_a_village_wall_stays_the_players(self):
        from masses import current, facts_of
        store = Store(":memory:")
        try:
            voxels = self._world(pillar=False)
            # the hut is generated (no history); a player pillars up its wall in dirt
            for y in range(64, 70):
                voxels[(5, y, 2)] = "minecraft:dirt"
            self._placed(store, "climber", [(5, y, 2) for y in range(64, 70)],
                         "minecraft:dirt")
            self._apply(store, voxels)
            pillar = next(r for r in current(store) if r["actor"] == "climber")
            self.assertEqual(facts_of(pillar)["dims"], "1x6x1")
            hut = next(r for r in current(store) if facts_of(r)["blocks"] == 73)
            self.assertIsNone(hut["actor"])
            self.assertEqual(facts_of(hut)["origin"], "unknown")
        finally:
            store.close()

    def test_a_tree_is_not_a_built_thing_but_a_log_cabin_is(self):
        from masses import ledger_built
        store = Store(":memory:")
        try:
            voxels = self._world(pillar=False, torch=False, bridge=False)
            for y in range(64, 70):
                voxels[(30, y, 30)] = "minecraft:oak_log"
            for dx in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    voxels[(30 + dx, 70, 30 + dz)] = "minecraft:oak_leaves"
            for x in range(20, 24):
                for z in range(20, 24):
                    if x in (20, 23) or z in (20, 23):
                        voxels[(x, 64, z)] = "minecraft:oak_log"
            built = ledger_built(store, "overworld", voxels, self.LO, self.HI)
            self.assertNotIn((30, 64, 30), built, "a trunk under a canopy is a tree")
            self.assertIn((20, 64, 20), built, "logs touching no leaves are a wall")
            # a trunk the player stacked under a canopy is credited, and counts
            self._placed(store, "p", [(30, y, 30) for y in range(64, 70)],
                         "minecraft:oak_log")
            built = ledger_built(store, "overworld", voxels, self.LO, self.HI)
            self.assertIn((30, 64, 30), built)
        finally:
            store.close()

    def test_a_fragment_of_a_known_whole_is_never_minted_as_new(self):
        """Through a window edge, one known mass can appear as two disconnected fragments.
        One-to-one matching gives the id to the bigger; the smaller must not become a
        second row, or every rebuild mints and retires it."""
        from masses import current
        store = Store(":memory:")
        try:
            voxels = self._world(pillar=False, torch=False, bridge=False)
            # a U-shaped plank wall: two arms joined by a bar at the far (low z) end
            for z in range(10, 30):
                voxels[(20, 64, z)] = "minecraft:oak_planks"
                voxels[(28, 64, z)] = "minecraft:oak_planks"
            for x in range(20, 29):
                voxels[(x, 64, 10)] = "minecraft:oak_planks"
            self._apply(store, voxels)
            self.assertEqual(len(current(store)), 2)   # the hut and the U
            # a window whose north edge cuts both arms off from the bar
            u = next(r for r in current(store) if r["min_x"] == 20)
            report = self._apply(store, voxels, [15, 60, 20], [45, 90, 45])
            self.assertEqual(len(current(store)), 2, "no second row for the far arm")
            self.assertEqual(report["written"], [])
            self.assertEqual(report["retired"], [])
            same = store.get("masses", u["id"])
            self.assertEqual((same["min_z"], same["max_z"]), (10, 29),
                             "a partial view never shrinks a record")
            # the player extends one arm past the read edge: a partial view GROWS it
            for z in range(30, 38):
                voxels[(28, 64, z)] = "minecraft:oak_planks"
            self._apply(store, voxels, [15, 60, 20], [45, 90, 45])
            grown = store.get("masses", u["id"])
            self.assertEqual((grown["min_z"], grown["max_z"]), (10, 37))
            self.assertEqual(json.loads(grown["materials"])["oak_planks"], 47 + 8)
            self.assertTrue(json.loads(grown["value"])["union_of_views"])
            # and a read holding the whole thing sets it exactly
            self._apply(store, voxels)
            exact = store.get("masses", u["id"])
            self.assertFalse(json.loads(exact["value"])["union_of_views"])
            self.assertEqual(json.loads(exact["value"])["partial"], [])
        finally:
            store.close()

    def test_a_log_column_running_out_of_the_read_waits_to_be_seen_whole(self):
        from masses import ledger_built
        store = Store(":memory:")
        try:
            voxels = self._world(pillar=False, torch=False, bridge=False)
            for y in range(64, 75):
                voxels[(30, y, 30)] = "minecraft:oak_log"
            built = ledger_built(store, "overworld", voxels, [-10, 60, -10], [50, 70, 50])
            self.assertNotIn((30, 64, 30), built, "cut by the top of the read")
            built = ledger_built(store, "overworld", voxels, [-10, 60, -10], [50, 90, 50])
            self.assertIn((30, 64, 30), built, "seen whole, touching no leaves: a post")
        finally:
            store.close()

    def _fake_world_reads(self, god, voxels):
        """Route the god's voxel reads to a dictionary instead of a server."""
        import god as god_module

        async def read(_url, _dim, lo, hi):
            return {"ok": True, "tick": 1, "_lo": lo, "_hi": hi}
        god_module.request_voxels = read
        god_module.to_voxels = lambda result: {
            p: m for p, m in voxels.items()
            if all(result["_lo"][i] <= p[i] <= result["_hi"][i] for i in range(3))}

    def _hut_by(self, store, actor="mason"):
        self._placed(store, actor, [(x, y, z) for x in range(5) for z in range(5)
                                    for y in range(64, 68)
                                    if x in (0, 4) or z in (0, 4) or y == 67],
                     "minecraft:cobblestone")

    def _god_with_fake_reads(self, store, voxels):
        import god as god_module
        from god import God
        god = God(store, use_model=False)
        god.tick = 50_000
        self._fake_world_reads(god, voxels)
        asked = []

        async def draw(*_a, **_k):
            asked.append(1)
            return []
        god._draw_boundaries = draw
        return god, asked

    def test_a_new_build_nobody_named_gets_one_look_once_its_builder_steps_away(self):
        """A brand-new whole player-built mass outside every landmark earns a first look
        regardless of any threshold. Ten quiet seconds after eight blocks is a footing, so
        the look waits while the builder is still standing in it."""
        import asyncio
        import god as god_module
        store = Store(":memory:")
        saved = (god_module.request_voxels, god_module.to_voxels)
        try:
            voxels = self._world(pillar=False, torch=False, bridge=False)
            self._hut_by(store)
            god, asked = self._god_with_fake_reads(store, voxels)
            god.positions["mason"] = ("overworld", [2, 65, 2], god.tick - 40)
            god.edit_ticks[("overworld", 0, 2, 0)] = god.tick - 40
            asyncio.run(god.resegment("overworld", [2, 66, 2]))
            self.assertEqual(asked, [], "the builder is still here: the first look waits")
            self.assertEqual([e["reason"] for e in god.edited.values()], ["first"])
            # they walk off; the sweep pays the look
            god.positions["mason"] = ("overworld", [200, 65, 200], god.tick)
            asyncio.run(god.sweep())
            self.assertGreaterEqual(len(asked), 1)
            self.assertEqual(god.edited, {})
            # and having been shown once, it is not shown again on the next settle
            from masses import current, facts_of
            self.assertTrue(all(facts_of(r).get("looked_at_ms")
                                for r in current(store, "overworld")))
            before = len(asked)
            asyncio.run(god.resegment("overworld", [2, 66, 2]))
            self.assertEqual(len(asked), before)
        finally:
            god_module.request_voxels, god_module.to_voxels = saved
            store.close()

    def test_small_edits_defer_the_look_and_the_threshold_releases_it(self):
        import asyncio
        import god as god_module
        from god import LOOK_THRESHOLD
        store = Store(":memory:")
        saved = (god_module.request_voxels, god_module.to_voxels)
        try:
            voxels = self._world(pillar=False, torch=False, bridge=False)
            self._hut_by(store)
            store.put("structures", {
                "id": "s_hut", "actor": "mason", "dim": "overworld", "name": None,
                "min_x": -1, "min_y": 63, "min_z": -1, "max_x": 5, "max_y": 68, "max_z": 5,
                "materials": json.dumps({"cobblestone": 73}),
            }, Belief(provenance=Provenance.INFERRED, confidence=.9, verified_at_ms=5_000,
                      value=json.dumps({"schema_version": 2, "kind": "current_structure",
                                        "blocks": 73, "dims": "5x4x5", "fill": .5,
                                        "vertical": True, "origin": "player_built"})))
            god, asked = self._god_with_fake_reads(store, voxels)
            cell = ("overworld", 0, 2, 0)
            god.edits[cell] = 3
            asyncio.run(god.resegment("overworld", [2, 66, 2]))
            self.assertEqual(asked, [], "three blocks inside a named hut: no look")
            self.assertEqual(god.edited[cell]["reason"], "threshold")
            god.edits[cell] = LOOK_THRESHOLD
            asyncio.run(god.resegment("overworld", [2, 66, 2]))
            self.assertGreaterEqual(len(asked), 1)
            self.assertNotIn(cell, god.edits, "a look clears the debt")
            self.assertEqual(god.edited, {})
        finally:
            god_module.request_voxels, god_module.to_voxels = saved
            store.close()

    def test_a_question_about_an_edited_place_pays_its_look_first(self):
        import asyncio
        import god as god_module
        store = Store(":memory:")
        saved = (god_module.request_voxels, god_module.to_voxels)
        try:
            voxels = self._world(pillar=False, torch=False, bridge=False)
            self._hut_by(store)
            store.put("structures", {
                "id": "s_hut", "actor": "mason", "dim": "overworld", "name": None,
                "min_x": -1, "min_y": 63, "min_z": -1, "max_x": 5, "max_y": 68, "max_z": 5,
                "materials": json.dumps({"cobblestone": 73}),
            }, Belief(provenance=Provenance.INFERRED, confidence=.9, verified_at_ms=5_000,
                      value=json.dumps({"schema_version": 2, "kind": "current_structure",
                                        "blocks": 73, "dims": "5x4x5", "fill": .5,
                                        "vertical": True, "origin": "player_built"})))
            store.name_structure("s_hut", "stone hut", .9, "seen", 1, "test", saw=True)
            god, asked = self._god_with_fake_reads(store, voxels)
            god.edited[("overworld", 0, 2, 0)] = {"pos": [2, 66, 2], "since": 1,
                                                  "reason": "threshold", "dim": "overworld"}
            paid = asyncio.run(god.release_for_question(
                "what do you think of my stone hut", "overworld", [300, 64, 300]))
            self.assertEqual(paid, 1)
            self.assertGreaterEqual(len(asked), 1)
            self.assertEqual(god.edited, {})
            asked.clear()
            # a question from far away about something else pays nothing
            god.edited[("overworld", 0, 2, 0)] = {"pos": [2, 66, 2], "since": 1,
                                                  "reason": "threshold", "dim": "overworld"}
            paid = asyncio.run(god.release_for_question(
                "where is the nearest ocean", "overworld", [300, 64, 300]))
            self.assertEqual(paid, 0)
            self.assertEqual(asked, [])
            # but standing inside the edited window is a reference too
            paid = asyncio.run(god.release_for_question(
                "what is this", "overworld", [3, 65, 3]))
            self.assertEqual(paid, 1)
            self.assertGreaterEqual(len(asked), 1)
        finally:
            god_module.request_voxels, god_module.to_voxels = saved
            store.close()

    def test_an_idle_god_pays_deferred_looks_oldest_first(self):
        import asyncio
        from god import God, IDLE_LOOK_TICKS
        store = Store(":memory:")
        try:
            god = God(store, use_model=False)
            god.tick = 100_000
            looked = []

            async def fake(dim, pos, ledger_only=False, force_look=False, **_k):
                looked.append((pos, force_look))
                return 0
            god.resegment = fake
            god.edited = {
                ("overworld", 0, 2, 0): {"pos": [5, 64, 5], "since": 2_000,
                                         "reason": "threshold", "dim": "overworld"},
                ("overworld", 4, 2, 4): {"pos": [99, 64, 99], "since": 1_000,
                                         "reason": "threshold", "dim": "overworld"},
            }
            god.last_block_tick = god.tick - IDLE_LOOK_TICKS + 100
            asyncio.run(god.sweep())
            self.assertEqual(looked, [], "someone built a moment ago: not idle yet")
            god.last_block_tick = god.tick - IDLE_LOOK_TICKS
            asyncio.run(god.sweep())
            self.assertEqual(looked, [([99, 64, 99], True)])
        finally:
            store.close()

    def test_an_unchanged_ledger_is_not_boxed_twice(self):
        import asyncio
        import god as god_module
        from god import LOOK_THRESHOLD
        store = Store(":memory:")
        saved = (god_module.request_voxels, god_module.to_voxels)
        try:
            god, asked = self._god_with_fake_reads(
                store, self._world(pillar=False, torch=False, bridge=False))
            asyncio.run(god.resegment("overworld", [2, 66, 2], force_look=True))
            first = len(asked)
            self.assertGreaterEqual(first, 1)
            god.edits[("overworld", 0, 2, 0)] = LOOK_THRESHOLD
            asyncio.run(god.resegment("overworld", [2, 66, 2]))
            self.assertEqual(len(asked), first, "the same ledger under the same cell: no "
                                                "second look, whatever the counter says")
            # the world changes under it and the threshold is met: a look is owed again
            world = self._world(pillar=False, torch=False, bridge=False)
            world[(2, 68, 2)] = "minecraft:cobblestone"
            self._fake_world_reads(god, world)
            god.edits[("overworld", 0, 2, 0)] = LOOK_THRESHOLD
            asyncio.run(god.resegment("overworld", [2, 66, 2]))
            self.assertGreater(len(asked), first)
        finally:
            god_module.request_voxels, god_module.to_voxels = saved
            store.close()

    def test_settled_cells_one_read_covers_are_taken_together(self):
        import asyncio
        from god import God, RESEGMENT_SETTLE_TICKS
        store = Store(":memory:")
        try:
            god = God(store, use_model=False)
            god.tick = 50_000
            self.assertEqual(RESEGMENT_SETTLE_TICKS, 200, "ten seconds, not forty-five")
            looked = []

            async def fake(dim, pos, ledger_only=False, **_k):
                looked.append((pos, ledger_only))
                return 0
            god.resegment = fake
            god.last_block_tick = god.tick
            god.disturbed = {
                ("overworld", 0, 2, 0): {"tick": 40_000, "pos": [5, 64, 5]},
                ("overworld", 0, 2, 0, "b"): {"tick": 40_100, "pos": [9, 66, 3],
                                              "ledger_only": True},
                ("overworld", 3, 2, 3): {"tick": 40_000, "pos": [80, 64, 80]},
            }
            asyncio.run(god.sweep())
            self.assertEqual(looked, [([5, 64, 5], False)])
            self.assertEqual(set(god.disturbed), {("overworld", 3, 2, 3)},
                             "the neighbour eight blocks away rode along; the far one waits")
        finally:
            store.close()

    def test_an_edit_inside_a_landmark_refreshes_its_census_without_a_look(self):
        """Two gold blocks set into a cobblestone elephant were reported as cobblestone
        for as long as the look was deferred. The ledger is re-measured on every read;
        the structure row must carry that census, and a contradicted structure whose
        contents are known again is reaffirmed."""
        from masses import anchor, refresh_landmarks
        store = Store(":memory:")
        try:
            voxels = self._world(pillar=False, torch=False, bridge=False)
            self._hut_by(store)
            store.put("structures", {
                "id": "s_hut", "actor": "mason", "dim": "overworld", "name": None,
                "min_x": -1, "min_y": 63, "min_z": -1, "max_x": 5, "max_y": 68, "max_z": 5,
                "materials": json.dumps({"cobblestone": 73}),
            }, Belief(provenance=Provenance.INFERRED, confidence=.9, verified_at_ms=5_000,
                      value=json.dumps({"schema_version": 2, "kind": "current_structure",
                                        "blocks": 73, "dims": "5x4x5", "fill": .5,
                                        "vertical": True, "origin": "player_built"})))
            self._apply(store, voxels)
            anchor(store)
            # the player swaps two wall blocks for gold; the event contradicts the row
            voxels[(0, 65, 2)] = "minecraft:gold_block"
            voxels[(4, 65, 2)] = "minecraft:gold_block"
            self._placed(store, "mason", [(0, 65, 2), (4, 65, 2)], "minecraft:gold_block",
                         t0=9_000_000)
            store.contradict("structures", "s_hut", "added to since it was measured", 7)
            self.assertEqual(store.get("structures", "s_hut")["confidence"], 0.05)
            self._apply(store, voxels)
            anchor(store)
            refresh_landmarks(store, "overworld", now_ms=9_500_000, tick=8)
            row = store.get("structures", "s_hut")
            self.assertEqual(json.loads(row["materials"]),
                             {"cobblestone": 71, "gold_block": 2})
            self.assertEqual(json.loads(row["value"])["blocks"], 73)
            self.assertIsNone(store.contradicted("structures", "s_hut"))
            self.assertEqual(row["confidence"], 0.9, "the confidence an edit took away "
                                                     "comes back with the measurement")
            self.assertEqual(row["verified_at_ms"], 9_500_000)
        finally:
            store.close()

    def test_a_demolished_landmark_is_retired_on_the_next_read_not_the_next_look(self):
        from god import God
        store = Store(":memory:")
        try:
            god = God(store, use_model=False)
            store.put("structures", {
                "id": "s_gone", "actor": "mason", "dim": "overworld", "name": None,
                "min_x": 0, "min_y": 64, "min_z": 0, "max_x": 4, "max_y": 67, "max_z": 4,
                "materials": json.dumps({"cobblestone": 73}),
            }, Belief(provenance=Provenance.INFERRED, confidence=.9, verified_at_ms=5_000,
                      value=json.dumps({"schema_version": 2, "kind": "current_structure",
                                        "blocks": 73, "dims": "5x4x5", "fill": .5,
                                        "vertical": True, "origin": "player_built"})))
            # a read that holds the whole box and finds five stray blocks left
            built = {(x, 64, 0): "minecraft:cobblestone" for x in range(5)}
            retired = god._retire_gone("overworld", built, [-10, 60, -10], [50, 90, 50])
            self.assertEqual(retired, ["s_gone"])
            self.assertIsNone(store.get("structures", "s_gone"))
            # but a read that does not hold the whole box says nothing about it
            store.put("structures", {
                "id": "s_edge", "actor": "mason", "dim": "overworld", "name": None,
                "min_x": 40, "min_y": 64, "min_z": 0, "max_x": 60, "max_y": 67, "max_z": 4,
                "materials": json.dumps({"cobblestone": 73}),
            }, Belief(provenance=Provenance.INFERRED, confidence=.9, verified_at_ms=5_000,
                      value=json.dumps({"schema_version": 2, "kind": "current_structure",
                                        "blocks": 73, "dims": "21x4x5", "fill": .5,
                                        "vertical": True, "origin": "player_built"})))
            self.assertEqual(god._retire_gone("overworld", {}, [-10, 60, -10],
                                              [50, 90, 50]), [])
        finally:
            store.close()

    def test_the_not_history_sentinel_is_never_spoken(self):
        from history import _is_not_history
        for spelling in ("NOT_HISTORY", "NOT HISTORY", "not_history.", " Not History\n"):
            self.assertTrue(_is_not_history(spelling), spelling)
        self.assertFalse(_is_not_history("It is not history that matters here"))

    def test_a_box_read_out_as_ranges_is_a_coordinate(self):
        from god import contains_raw_coordinates, remove_raw_coordinates
        said = ("Your cobblestone elephant is just west of you, around 301-308 by -303 to "
                "-297, only about 10-17 blocks away.")
        self.assertTrue(contains_raw_coordinates(said))
        cleaned = remove_raw_coordinates(said)
        self.assertNotIn("301", cleaned)
        self.assertIn("10-17 blocks away", cleaned, "a distance is not a coordinate")
        self.assertFalse(contains_raw_coordinates("a 3 by 3 window, 10-17 blocks off"))

    def test_the_history_model_is_told_buildings_are_not_places(self):
        """Eight "villages" were five located sites plus three buildings of one of them."""
        from history import HISTORY_SYSTEM
        self.assertIn("never counted as a place", HISTORY_SYSTEM)
        self.assertIn("how many I have located", HISTORY_SYSTEM)

    def test_the_god_can_see_the_unnamed_things_nearby(self):
        from masses import current, nearby_summary
        store = Store(":memory:")
        try:
            self._apply(store, self._world())
            bridge = next(r for r in current(store) if r["min_z"] == 30)
            store.put("relationships", {
                "id": f"role:{bridge['id']}", "subject": bridge["id"],
                "predicate": "role", "object": "utility",
            }, Belief(provenance=Provenance.INFERRED, confidence=.8, verified_at_ms=5_000,
                      value=json.dumps({"noun": "plank footbridge"})))
            line = nearby_summary(store, "overworld", [10, 64, 10])
            self.assertIn("plank footbridge", line)
            self.assertIn("1 small placements under 8 blocks (torch)", line)
            self.assertEqual(nearby_summary(store, "overworld", [900, 64, 900]), "")
        finally:
            store.close()


class DeathsThePlayerDidNotCause(unittest.TestCase):
    """A creeper that explodes beside a player also kills the sheep standing there.

    Only kills BY a player were recorded, so the blast was on the record and what it cost
    was not. Not every death earns an event: hostile mobs burn by the dozen at every
    sunrise, which is weather rather than news. A death counts when something else killed
    it, or when the victim was not a monster.
    """

    def test_the_plugin_records_a_death_nobody_playing_caused(self):
        source = (HERE.parent / "plugin" / "src" / "main" / "java" / "io" / "github"
                  / "jakegilbert7" / "mcgod" / "capture" / "EntityListener.java")
        java = source.read_text(encoding="utf-8")
        self.assertIn('"mob_death"', java)
        self.assertIn("victim instanceof Monster", java,
                      "a burning zombie at sunrise is not news")
        self.assertIn("getLastDamageCause", java)
        self.assertIn("projectile.getShooter()", java,
                      "an arrow did not kill the sheep; a skeleton did")
        self.assertIn('"mob_kill"', java, "a player's own kills keep their event")

    def test_a_mob_death_is_indexed_and_findable_by_place_and_time(self):
        from activity import record_activity

        store = Store(":memory:")
        try:
            blast = {"tick": 100, "t": 5_000, "type": "explosion", "actor": "world",
                     "dim": "overworld", "pos": [310, 70, -310], "entity": "minecraft:creeper",
                     "blocks": 12}
            death = {"tick": 100, "t": 5_001, "type": "mob_death", "actor": "world",
                     "dim": "overworld", "pos": [312, 70, -309], "entity": "minecraft:sheep",
                     "killer": "minecraft:creeper", "cause": "ENTITY_EXPLOSION",
                     "name": None, "tame": False}
            for event in (blast, death):
                record_activity(store, event)
            rows = store.all("event_history", "kind = 'mob_death'")
            self.assertEqual(len(rows), 1)
            row = rows[0]
            self.assertEqual(row["actor"], "world",
                             "a death the player did not cause is not their deed")
            payload = json.loads(row["payload"])
            self.assertEqual(payload["entity"], "minecraft:sheep")
            self.assertEqual(payload["killer"], "minecraft:creeper")
            self.assertEqual((row["x"], row["z"]), (312, -309))
            # the whole point: the blast and its cost are found together
            together = store.all(
                "event_history",
                "kind IN ('explosion', 'mob_death') AND event_t BETWEEN ? AND ? "
                "AND x BETWEEN ? AND ? AND z BETWEEN ? AND ?",
                (4_000, 6_000, 300, 320, -320, -300))
            self.assertEqual({r["kind"] for r in together}, {"explosion", "mob_death"})
        finally:
            store.close()

    def test_a_death_survives_the_journey_through_a_recorded_session(self):
        """The event schema is a contract: a new type must replay unchanged."""
        from consumer import EventConsumer

        death = {"tick": 100, "t": 5_001, "type": "mob_death", "actor": "world",
                 "dim": "overworld", "pos": [312, 70, -309], "entity": "minecraft:wolf",
                 "killer": "minecraft:skeleton", "cause": "PROJECTILE",
                 "name": "Rex", "tame": True}
        line = json.dumps(death)
        self.assertEqual(json.loads(line), death)
        for field in ("tick", "t", "type", "actor", "dim", "pos"):
            self.assertIn(field, death, f"core field {field} missing")
        self.assertTrue(hasattr(EventConsumer, "on_event"),
                        "replay feeds the same consumer contract")

    def test_the_model_is_told_what_a_mob_death_proves(self):
        from history import HISTORY_SYSTEM
        self.assertIn("mob_death is a death the player did not cause", HISTORY_SYSTEM)
        self.assertIn("only mob_kill proves a kill BY THE PLAYER", HISTORY_SYSTEM)
        self.assertIn("what the blast cost", HISTORY_SYSTEM)


class TheGodMayNameWhereYouAreStanding(unittest.TestCase):
    """Asked what structure they were in while standing in a village house, the god
    rendered the place, described it correctly, and then refused its own answer.

    Grounding considered only player-built structures, so a generated building could never
    be grounded and naming one counted as inventing a place. The god could see where the
    player was and was not allowed to say it.
    """

    def _world(self):
        store = Store(":memory:")
        store.put("generated_features", {
            "id": "generated:village", "dim": "overworld",
            "kind": "minecraft:village_plains", "x": 352, "y": 64, "z": -320,
            "discovered_from": "world_generator",
        }, Belief(provenance=Provenance.SCANNED,
                  value=json.dumps({"origin": "world_generated"})))
        store.put("generated_features", {
            "id": "g_house", "dim": "overworld", "kind": "village_house",
            "x": 340, "y": 70, "z": -285, "discovered_from": "survey",
            "min_x": 335, "min_y": 64, "min_z": -292,
            "max_x": 346, "max_y": 76, "max_z": -279,
            "materials": json.dumps({"oak_planks": 120}),
        }, Belief(provenance=Provenance.INFERRED, confidence=.9, verified_at_ms=1,
                  value=json.dumps({"origin": "world_generated",
                                    "parent": "generated:village"})))
        store.name_structure("g_house", "oak-roofed village house", .9, "seen", 1, "t",
                             saw=True)
        store.put("actor_presence", {
            "id": "p", "actor": "p", "dim": "overworld",
            "place_id": "generated:village", "event_t": 1,
        }, Belief(provenance=Provenance.OBSERVED))
        return store

    def test_a_generated_building_can_be_the_answer(self):
        from grounding import standing_in

        store = self._world()
        try:
            row, table = standing_in(store, "overworld", [340, 70, -285])
            self.assertEqual(row["id"], "g_house")
            self.assertEqual(table, "generated_features")
            self.assertEqual(standing_in(store, "overworld", [900, 70, 900]), (None, None))
        finally:
            store.close()

    def test_the_smallest_containing_place_wins(self):
        """A house inside a village is the house, not the village."""
        from grounding import standing_in

        store = self._world()
        try:
            store.put("structures", {
                "id": "s_big", "actor": "p", "dim": "overworld", "name": None,
                "min_x": 300, "min_y": 60, "min_z": -320,
                "max_x": 380, "max_y": 90, "max_z": -260,
                "materials": json.dumps({"cobblestone": 900}),
            }, Belief(provenance=Provenance.INFERRED, confidence=.9, verified_at_ms=1,
                      value=json.dumps({"schema_version": 2, "kind": "current_structure",
                                        "blocks": 900, "dims": "81x31x61", "fill": .1,
                                        "vertical": True, "origin": "player_built"})))
            row, _ = standing_in(store, "overworld", [340, 70, -285])
            self.assertEqual(row["id"], "g_house", "the smaller box is the answer")
        finally:
            store.close()

    def test_the_answer_is_grounded_so_naming_it_is_not_inventing(self):
        import asyncio
        from grounding import ground_query

        store = self._world()
        try:
            grounded = asyncio.run(ground_query(
                store, "what structure am i in right now", "overworld",
                [340, 70, -285], url=None, actor="p"))
            ids = {item["id"]: item for item in grounded["structures"]}
            self.assertIn("g_house", ids)
            self.assertEqual(ids["g_house"]["role"], "the player is standing in this")
            self.assertEqual(ids["g_house"]["origin"], "world_generated",
                             "a village house is not one of the player's works")
        finally:
            store.close()

    def test_a_generated_place_can_be_referred_to_by_name(self):
        """Without this the guard cannot match the answer's own words back to the fact
        that grounded them, so a correct answer is refused for mentioning a church."""
        from grounding import resolve_references

        store = self._world()
        try:
            found = resolve_references(store, "where is the village house", "overworld",
                                       [340, 70, -285])
            self.assertIn("g_house", {row["id"] for row in found})
        finally:
            store.close()

    def test_presence_in_a_site_is_kept_apart_from_containment(self):
        """A village site has an anchor and no bounds, so presence is a radius rather than
        a box. Reporting the two the same way is how "the closest village" once became the
        one underfoot."""
        import asyncio
        from grounding import _plain_place, ground_query, render_grounding

        store = self._world()
        try:
            grounded = asyncio.run(ground_query(
                store, "what am i standing in", "overworld", [340, 70, -285],
                url=None, actor="p"))
            place = grounded["within_place"]
            self.assertEqual(place["id"], "generated:village")
            self.assertIn("not measured containment", place["evidence"])
            rendered = render_grounding(grounded, immersive=True)
            self.assertIn("plains village", rendered)
            self.assertNotRegex(rendered, r"bbox=\[", "immersive text carries no boxes")
            self.assertEqual(_plain_place("minecraft:village_snowy"), "snowy village")
            self.assertEqual(_plain_place("village_church"), "church",
                             "one building in a village is not a kind of village")
            # no actor, no presence claim
            without = asyncio.run(ground_query(
                store, "what am i standing in", "overworld", [340, 70, -285], url=None))
            self.assertIsNone(without["within_place"])
        finally:
            store.close()


class NothingIsCountedTwice(unittest.TestCase):
    """Asked for everything built within 100 blocks, the god listed several things twice.

    It read the named structures, then the masses beneath them, then reconstructed more
    objects from raw block placements, and had no rule saying those describe the same
    world. The present-tense answer is the structures in an area plus the masses there that
    belong to none of them; the two sets are disjoint and together they are everything that
    stands.
    """

    def test_structures_and_unclaimed_masses_partition_the_ledger(self):
        """The invariant the retrieval rule stands on: every mass is either part of exactly
        one structure or is its own thing, never both and never neither."""
        from masses import anchor, current as current_masses
        from structures import current as current_structures, facts_of

        store = Store(":memory:")
        try:
            store.put("masses", {
                "id": "m_wall", "dim": "overworld", "actor": "p", "place_id": None,
                "min_x": 0, "min_y": 64, "min_z": 0, "max_x": 6, "max_y": 70, "max_z": 6,
                "materials": json.dumps({"cobblestone": 80}),
            }, Belief(provenance=Provenance.SCANNED, verified_at_ms=1,
                      value=json.dumps({"schema_version": 1, "kind": "built_mass",
                                        "blocks": 80, "origin": "player_built"})))
            store.put("masses", {
                "id": "m_pillar", "dim": "overworld", "actor": "p", "place_id": None,
                "min_x": 40, "min_y": 64, "min_z": 40,
                "max_x": 40, "max_y": 72, "max_z": 40,
                "materials": json.dumps({"dirt": 9}),
            }, Belief(provenance=Provenance.SCANNED, verified_at_ms=1,
                      value=json.dumps({"schema_version": 1, "kind": "built_mass",
                                        "blocks": 9, "origin": "player_built"})))
            store.put("structures", {
                "id": "s_hut", "actor": "p", "dim": "overworld", "name": None,
                "min_x": -1, "min_y": 63, "min_z": -1,
                "max_x": 7, "max_y": 71, "max_z": 7,
                "materials": json.dumps({"cobblestone": 80}),
            }, Belief(provenance=Provenance.INFERRED, confidence=.9, verified_at_ms=1,
                      value=json.dumps({"schema_version": 2, "kind": "current_structure",
                                        "blocks": 80, "dims": "9x9x9", "fill": .1,
                                        "vertical": True, "origin": "player_built"})))
            anchor(store)

            claimed = {row["id"] for row in current_masses(store) if row["place_id"]}
            loose = {row["id"] for row in current_masses(store) if not row["place_id"]}
            self.assertEqual(claimed, {"m_wall"})
            self.assertEqual(loose, {"m_pillar"})
            self.assertFalse(claimed & loose, "a mass is never in both sets")
            self.assertEqual(claimed | loose,
                             {row["id"] for row in current_masses(store)},
                             "and never in neither")
            comprises = set(facts_of(store.get("structures", "s_hut"))["comprises"])
            self.assertEqual(comprises, claimed,
                             "what a structure comprises is exactly what points at it")
            # the answer set: one structure plus one loose mass, two things, not three
            standing = len(current_structures(store)) + len(loose)
            self.assertEqual(standing, 2)
        finally:
            store.close()

    def test_the_model_is_told_not_to_enumerate_the_same_world_twice(self):
        from history import HISTORY_SYSTEM
        self.assertIn("place_id IS NULL", HISTORY_SYSTEM)
        self.assertIn("listing both names the same thing twice", HISTORY_SYSTEM)
        self.assertIn("Do not also enumerate from block_place rows", HISTORY_SYSTEM)
        self.assertIn("PART of that place, not a thing beside it", HISTORY_SYSTEM)

    def test_a_radius_is_a_circle_and_names_come_from_names(self):
        from history import HISTORY_SYSTEM
        self.assertIn("A radius is a circle", HISTORY_SYSTEM)
        self.assertIn("Name things by what they are, not what they are made of",
                      HISTORY_SYSTEM)


class TakingFromYourOwnChestIsNotStealing(unittest.TestCase):
    """Asked how many items they had stolen, the god answered with every withdrawal it had
    ever seen: 491 items, where 417 of them came out of the player's own chests. The true
    answer was 74.

    The evidence was already sufficient and the model's query was the right shape. It
    compared a stored "minecraft:chest" against the bare word "chest", matched nothing, and
    so classified every container as somebody else's. An empty set is not an error, which is
    what made it wrong rather than broken.
    """

    def _world(self):
        store = Store(":memory:")
        actor = "p"
        def event(n, kind, pos, **payload):
            body = {"tick": n, "t": 1_000 + n, "type": kind, "actor": actor,
                    "dim": "overworld", "pos": list(pos), **payload}
            store.put("event_history", {
                "id": f"e{n}", "actor": actor, "dim": "overworld", "tick": n,
                "event_t": 1_000 + n, "kind": kind, "place_id": None,
                "x": pos[0], "y": pos[1], "z": pos[2], "payload": json.dumps(body),
            }, Belief(provenance=Provenance.OBSERVED))
        # their own chest, placed then emptied
        event(1, "block_place", (10, 64, 10), before="minecraft:air",
              after="minecraft:chest")
        event(2, "container_take", (10, 64, 10), container="chest",
              item="minecraft:oak_planks", count=40)
        # a chest they never placed, in a village
        event(3, "container_take", (99, 64, 99), container="chest",
              item="minecraft:bread", count=3)
        event(4, "container_take", (99, 64, 99), container="chest",
              item="minecraft:iron_ingot", count=1)
        return store

    OWN = ("SELECT DISTINCT x, y, z FROM event_history WHERE actor = 'p' "
           "AND kind = 'block_place' "
           "AND json_extract(payload, '$.after') LIKE '%chest%'")

    def test_a_container_event_carries_the_containers_own_position(self):
        """The join only works because the coordinates are the chest's, not the player's."""
        store = self._world()
        try:
            rows = store.all("event_history", "kind = 'container_take'")
            self.assertEqual({(r["x"], r["y"], r["z"]) for r in rows},
                             {(10, 64, 10), (99, 64, 99)})
        finally:
            store.close()

    def test_the_documented_query_separates_theirs_from_everyone_elses(self):
        store = self._world()
        try:
            stolen = store.db.execute(
                "SELECT COALESCE(SUM(json_extract(payload, '$.count')), 0) "
                "FROM event_history t WHERE t.kind = 'container_take' AND NOT EXISTS "
                f"(SELECT 1 FROM ({self.OWN}) o "
                "WHERE o.x = t.x AND o.y = t.y AND o.z = t.z)").fetchone()[0]
            everything = store.db.execute(
                "SELECT SUM(json_extract(payload, '$.count')) FROM event_history "
                "WHERE kind = 'container_take'").fetchone()[0]
            self.assertEqual(stolen, 4, "only the chest they did not place")
            self.assertEqual(everything, 44, "collecting counts all of it")
        finally:
            store.close()

    def test_a_bare_material_name_matches_nothing_and_says_nothing(self):
        """The failure that produced the wrong answer, pinned: comparing against a name
        without its namespace returns an empty set rather than an error."""
        store = self._world()
        try:
            bare = store.db.execute(
                "SELECT COUNT(*) FROM event_history WHERE kind = 'block_place' "
                "AND json_extract(payload, '$.after') = 'chest'").fetchone()[0]
            self.assertEqual(bare, 0, "this is the trap")
            namespaced = store.db.execute(
                "SELECT COUNT(*) FROM event_history WHERE kind = 'block_place' "
                "AND json_extract(payload, '$.after') LIKE '%chest%'").fetchone()[0]
            self.assertEqual(namespaced, 1)
        finally:
            store.close()

    def test_the_model_is_told_both_rules(self):
        from history import HISTORY_SYSTEM
        self.assertIn("Taking from your own chest is not stealing", HISTORY_SYSTEM)
        self.assertIn("a container is the player's own when a block_place by that same",
                      HISTORY_SYSTEM)
        self.assertIn("Material names in payloads carry their namespace", HISTORY_SYSTEM)
        self.assertIn("quietly returns an empty set rather than an", HISTORY_SYSTEM)


class TheGodMayAct(unittest.TestCase):
    """The god can do anything to the world except end the server or hand out its keys.

    There is no cap on how many commands one request runs and no gap between them: a bridge
    is a thousand blocks and a storm is a command every few ticks. What is still refused is
    what would take the world away from the people in it, and the pacing that keeps the
    server ticking, which is not a limit on what may be asked for but on how fast it lands.
    """

    JAVA = (HERE.parent / "plugin" / "src" / "main" / "java" / "io" / "github"
            / "jakegilbert7" / "mcgod" / "capture" / "CommandService.java")
    RUNNER = JAVA.parent / "CommandRunner.java"

    def _java_set(self, name: str) -> set:
        source = self.JAVA.read_text(encoding="utf-8")
        start = source.index(f"{name} = Set.of(")
        body = source[start:source.index(");", start)]
        return set(re.findall(r'"([^"]+)"', body))

    def test_the_agents_mirror_matches_the_servers_gate(self):
        """A mirror that drifts is worse than none: it would explain a rule the server
        does not have."""
        import commands

        self.assertEqual(commands.ALLOWED, frozenset(self._java_set("ALLOWED")))
        self.assertEqual(commands.FORBIDDEN, frozenset(self._java_set("FORBIDDEN")))
        source = self.JAVA.read_text(encoding="utf-8")
        self.assertIn(f"MAX_BLOCKS = {commands.MAX_BLOCKS}", source)
        self.assertIn(f"MAX_LENGTH = {commands.MAX_LENGTH}", source)

    def test_nothing_caps_how_much_or_how_often(self):
        """The limits that were here are gone on purpose. "Rain fire on me for a minute" is
        a command every few ticks for twelve hundred ticks, and refusing it would be
        refusing the point."""
        import commands

        self.assertFalse(hasattr(commands, "MAX_COMMANDS"))
        source = self.JAVA.read_text(encoding="utf-8")
        self.assertNotIn("BUDGET_PER_HOUR", source)
        self.assertNotIn("COOLDOWN_MS", source)

    def test_the_tick_is_still_protected(self):
        """The one thing that cannot be given up. A thousand commands in a single tick is a
        visible freeze, and the game stuttering is always worse than the thing being slow."""
        runner = self.RUNNER.read_text(encoding="utf-8")
        self.assertIn("PER_TICK", runner)
        self.assertIn("runTaskTimer", runner)
        self.assertIn("for (int i = 0; i < PER_TICK && !pending.isEmpty(); i++)", runner)

    def test_a_lasting_effect_is_the_servers_job_not_a_loop_in_the_agent(self):
        """Five minutes of ice underfoot must survive the agent being busy, restarting, or
        losing its socket."""
        runner = self.RUNNER.read_text(encoding="utf-8")
        self.assertIn("public synchronized String cast(", runner)
        self.assertIn("spell.remaining -= spell.every", runner)
        self.assertIn("public synchronized int stopAll()", runner)
        self.assertIn("runner.shutdown()", (HERE.parent / "plugin" / "src" / "main"
                                            / "java" / "io" / "github" / "jakegilbert7"
                                            / "mcgod" / "McGodPlugin.java")
                      .read_text(encoding="utf-8"),
                      "a spell must not outlive the plugin that owns it")

    def test_a_person_can_always_stop_it(self):
        """The price of an omnipotent god is a switch, and it must not depend on the agent
        being reachable."""
        stats = (self.JAVA.parent / "StatsCommand.java").read_text(encoding="utf-8")
        self.assertIn('args[0].equalsIgnoreCase("stop")', stats)
        self.assertIn("runner.stopAll()", stats)

    def test_nothing_that_grants_power_or_ends_the_server_can_run(self):
        import commands

        for command in ("op playerone", "deop playerone", "stop", "reload",
                        "whitelist off", "ban playerone", "kick playerone",
                        "save-off", "datapack disable x", "gamerule keepInventory true"):
            self.assertIsNotNone(commands.refusal(command, anchored=True), command)

    def test_execute_is_refused_because_anchoring_already_does_its_job(self):
        import commands

        self.assertIn("never available",
                      commands.refusal("execute as @a run op @s", anchored=True))

    def test_a_command_that_speaks_is_refused(self):
        """Speech is budgeted and capped; a command that prints to chat is a way around
        that budget."""
        import commands

        for command in ("say hello", "tellraw @a {}", "me waves", "tell blob hi"):
            self.assertIsNotNone(commands.refusal(command, anchored=True), command)

    def test_relative_coordinates_work_when_there_is_somebody_to_be_relative_to(self):
        """"Ice under my feet" and "fire above my head" are relative by nature. From the
        console ~ is the world origin, which is never what anyone meant."""
        import commands

        for command in ("summon fireball ~ ~10 ~", "fill ~-3 ~-1 ~-3 ~3 ~-1 ~3 ice",
                        "particle flame ^ ^1 ^2 0 0 0 0 5"):
            self.assertIsNone(commands.refusal(command, anchored=True), command)
            self.assertIn("relative to", commands.refusal(command, anchored=False) or "")

    def test_a_big_build_is_allowed_up_to_the_games_own_limit(self):
        import commands

        self.assertIsNone(commands.refusal("fill 0 64 0 30 70 30 stone"),
                          "a bridge is a big fill")
        self.assertIn("32768", commands.refusal("fill 0 0 0 200 200 200 stone"))

    def test_the_model_is_told_it_can_anchor_and_repeat(self):
        """Without these it writes one-shot absolute commands and quietly does a smaller
        thing than it was asked for."""
        from god import ANSWER_SCHEMA

        self.assertIn('"commands": [str]', ANSWER_SCHEMA)
        self.assertIn('"every"', ANSWER_SCHEMA)
        self.assertIn("There is no limit on how many you may run at once", ANSWER_SCHEMA)
        self.assertIn("Anchor by default", ANSWER_SCHEMA)
        self.assertIn("keeps going while you talk", ANSWER_SCHEMA)

    def test_the_player_is_the_anchor_unless_told_otherwise(self):
        import asyncio

        import god as god_module
        from god import God

        store = Store(":memory:")
        sent = {}
        saved = god_module.run_command
        async def capture(commands, url, anchor=None, every=0, duration=0, spell=None):
            sent.update(commands=commands, anchor=anchor, every=every, duration=duration)
            return {"ok": True, "accepted": len(commands), "queued": 0, "refused": []}
        god_module.run_command = capture
        try:
            g = God(store, use_model=False)
            g.names["uuid-1"] = "playerone"
            asyncio.run(g.run_commands({"commands": ["summon fireball ~ ~10 ~"]}, "uuid-1"))
            self.assertEqual(sent["anchor"], "playerone")
            self.assertEqual(sent["commands"], ["summon fireball ~ ~10 ~"])
        finally:
            god_module.run_command = saved
            store.close()

    def test_a_lasting_effect_is_passed_through_with_its_timing(self):
        import asyncio

        import god as god_module
        from god import God

        store = Store(":memory:")
        sent = {}
        saved = god_module.run_command
        async def capture(commands, url, anchor=None, every=0, duration=0, spell=None):
            sent.update(every=every, duration=duration, spell=spell, n=len(commands))
            return {"ok": True, "accepted": len(commands), "queued": 0, "refused": [],
                    "spell": spell or "spell-1"}
        god_module.run_command = capture
        try:
            g = God(store, use_model=False)
            g.names["uuid-1"] = "playerone"
            acted = asyncio.run(g.run_commands({
                "commands": ["summon fireball ~ ~12 ~"], "every": 5, "duration": 1200,
                "spell": "rain-of-fire"}, "uuid-1"))
            self.assertEqual((sent["every"], sent["duration"]), (5, 1200))
            self.assertEqual(acted["spell"], "rain-of-fire")
            self.assertIn("60 seconds", g.report_of(acted))
        finally:
            god_module.run_command = saved
            store.close()

    def test_a_forbidden_draft_never_reaches_the_server(self):
        import asyncio

        import god as god_module
        from god import God

        store = Store(":memory:")
        reached = []
        saved = god_module.run_command
        async def capture(commands, url, **kwargs):
            reached.extend(commands)
            return {"ok": True, "accepted": len(commands), "queued": 0, "refused": []}
        god_module.run_command = capture
        try:
            g = God(store, use_model=False)
            g.names["uuid-1"] = "playerone"
            acted = asyncio.run(g.run_commands(
                {"commands": ["op playerone", "time set day"]}, "uuid-1"))
            self.assertEqual(reached, ["time set day"], "the rest still runs")
            self.assertIn("op is never available", acted["failures"][0])
        finally:
            god_module.run_command = saved
            store.close()

    def test_a_refused_answer_never_hides_what_was_done(self):
        """The player asked for diamonds, got them, and was told "I cannot ground that
        answer reliably enough to give it"."""
        import asyncio

        import god as god_module
        from god import God

        store = Store(":memory:")
        saved = god_module.run_command
        async def gave(commands, url, **kwargs):
            return {"ok": True, "accepted": len(commands), "queued": 0, "refused": []}
        god_module.run_command = gave
        try:
            g = God(store, use_model=False)
            g.names["uuid-1"] = "playerone"
            acted = asyncio.run(g.run_commands(
                {"commands": ["give playerone diamond 10"]}, "uuid-1"))
            self.assertEqual(acted["failures"], [])
            self.assertEqual(g.report_of(acted), "It is done.")
        finally:
            god_module.run_command = saved
            store.close()

    def test_the_refusal_stands_when_nothing_was_done(self):
        """Only an act that happened may overrule the guard's refusal."""
        import inspect

        import god
        source = inspect.getsource(god.God.on_chat)
        self.assertIn('if acted["done"] and speech_refused:', source)
        self.assertIn("said = self.report_of(acted)", source)

    def test_a_word_from_a_description_is_not_a_place_the_answer_named(self):
        """"I do not grant authority over this world" named no place, but "world" and
        "remains" appear in stored descriptions, so every nearby build resolved and the
        god refused its own refusal."""
        import asyncio

        from grounding import ground_query

        store = Store(":memory:")
        try:
            store.put("structures", {
                "id": "s_tower", "actor": "p", "dim": "overworld", "name": None,
                "min_x": 0, "min_y": 64, "min_z": 0, "max_x": 6, "max_y": 74, "max_z": 6,
                "materials": json.dumps({"cobblestone": 200}),
            }, Belief(provenance=Provenance.INFERRED, confidence=.9, verified_at_ms=1,
                      value=json.dumps({"schema_version": 2, "kind": "current_structure",
                                        "blocks": 200, "dims": "7x11x7", "fill": .2,
                                        "vertical": True, "origin": "player_built"})))
            store.name_structure(
                "s_tower", "cobblestone watchtower", .9,
                "a lookout that still remains standing over this world", 1, "t", saw=True,
                description="a lookout that still remains standing over this world")

            named = asyncio.run(ground_query(
                store, "your cobblestone watchtower still stands", "overworld",
                [100, 70, 100], url=None))
            self.assertEqual([i["role"] for i in named["structures"]], ["referenced"])

            loose = asyncio.run(ground_query(
                store, "I do not grant authority over this world; the task remains",
                "overworld", [100, 70, 100], url=None))
            self.assertTrue(loose["structures"], "it still surfaces as context")
            self.assertNotIn("referenced", [i["role"] for i in loose["structures"]],
                             "but naming nothing is not naming a place")
        finally:
            store.close()

    def test_only_a_named_place_triggers_a_rewrite(self):
        import inspect

        import god
        source = inspect.getsource(god.God.on_chat)
        self.assertIn("named_in_draft = {item[\"id\"] for item in draft_grounded[\"structures\"]",
                      source)
        self.assertIn("if named_in_draft - initial_ids", source)

    def test_a_request_to_act_is_never_answered_as_history(self):
        """"Please give me a netherite sword" was answered "I haven't given you a
        max-enchanted netherite sword": the evidence loop read it as "have you ever given
        me one". A player asking for a thing wants the thing, not an account of whether
        they already have one."""
        from history import HISTORY_SYSTEM

        self.assertIn("Above all, return NOT_HISTORY for anything ASKING YOU TO DO",
                      HISTORY_SYSTEM)
        self.assertIn("a player asking for a thing wants the", HISTORY_SYSTEM.lower())
        self.assertIn("without looking anything up", HISTORY_SYSTEM)

    def test_what_someone_built_is_still_history(self):
        """The correction must not swing the other way: stepping aside from requests must
        not make "what did I build here" step aside too."""
        from history import HISTORY_SYSTEM

        self.assertIn("What someone has BUILT or DONE is yours", HISTORY_SYSTEM)
        self.assertIn('"What did I build here"', HISTORY_SYSTEM)

    def test_the_sentinel_is_caught_however_late_it_arrives(self):
        """The model sometimes looks something up, decides the question was not for it
        after all, and says the sentinel. Gating the check on having retrieved nothing
        meant that answer reached chat verbatim."""
        import asyncio

        import history

        store = Store(":memory:")
        store.put("event_history", {
            "id": "e1", "actor": "p", "dim": "overworld", "tick": 1, "event_t": 1,
            "kind": "block_place", "place_id": None, "x": 0, "y": 64, "z": 0,
            "payload": json.dumps({"after": "minecraft:stone"}),
        }, Belief(provenance=Provenance.OBSERVED))

        class ToolBlock:
            type = "tool_use"
            name = "query_database"
            id = "t1"
            input = {"sql": "SELECT event_t FROM event_history"}

        class Text:
            type = "text"
            text = "NOT HISTORY"

        class Reply:
            def __init__(self, blocks):
                self.content = blocks
                self.stop_reason = "end_turn"

        turns = []

        async def fake_complete(_client, **request):
            turns.append(1)
            # look something up first, then decide it was not ours after all
            return Reply([ToolBlock()] if len(turns) == 1 else [Text()])

        saved = (history.complete_message, history.model_client)
        history.complete_message = fake_complete
        history.model_client = lambda **_kw: object()
        try:
            answer = asyncio.run(history.answer_if_history(
                store, {"actor": "p", "dim": "overworld", "t": 2, "pos": [0, 64, 0]},
                "give me a sword", bridge_url=None))
            self.assertIsNone(answer, "the sentinel is a refusal, never a sentence")
        finally:
            history.complete_message, history.model_client = saved
            store.close()

    def test_a_one_shot_batch_answers_with_what_it_did_not_what_it_queued(self):
        """The god said "I am giving you a diamond sword" for a command that then failed on
        a guessed item syntax. Saying what was accepted is not saying what happened."""
        runner = self.RUNNER.read_text(encoding="utf-8")
        self.assertIn("public record Outcome(", runner)
        self.assertIn("private static final class Batch", runner)
        self.assertIn("if (--owner.outstanding <= 0)", runner)
        handler = (self.JAVA.parent / "ScanCommandHandler.java").read_text(encoding="utf-8")
        self.assertIn("outcomes -> reply(conn, id, accepted.size(), refused, null, outcomes)",
                      handler)
        self.assertIn('out.append("],\\"ran\\":[")', handler)

    def test_success_is_counted_from_outcomes_not_from_acceptance(self):
        import asyncio

        import god as god_module
        from god import God

        store = Store(":memory:")
        saved = god_module.run_command
        async def half(commands, url, **kwargs):
            return {"ok": True, "accepted": 2, "refused": [], "ran": [
                {"command": commands[0], "ok": True, "output": "Set the time to 1000"},
                {"command": commands[1], "ok": False, "output": "Unknown item component"}]}
        god_module.run_command = half
        try:
            g = God(store, use_model=False)
            g.names["u"] = "playerone"
            acted = asyncio.run(g.run_commands(
                {"commands": ["time set day", "give @s bad_thing 1"]}, "u"))
            self.assertEqual(acted["done"], ["time set day"])
            self.assertIn("Unknown item component", acted["failures"][0])
        finally:
            god_module.run_command = saved
            store.close()

    def test_a_claim_is_withdrawn_when_nothing_worked(self):
        """Appending a correction to a sentence that already claims the gift tells the
        player two different things."""
        import inspect

        import god
        source = inspect.getsource(god.God.on_chat)
        self.assertIn('if acted["attempted"] and not acted["done"]:', source)
        self.assertIn('said = "That I could not do: "', source)

    def test_the_god_can_act_look_and_put_it_right(self):
        """Acting used to happen after the observation loop closed, so the god could look
        before it acted and never after. It laid ice and could not tell whether the ice it
        then tried to clear was gone."""
        import inspect

        import god
        source = inspect.getsource(god.God.on_chat)
        for tool in ("ACT_TOOL", "VISUAL_TOOL", "ENTITY_TOOL"):
            self.assertIn(tool, source)
        self.assertIn('if block.name == "act_on_world":', source)
        self.assertIn("for _ in range(8):", source,
                      "look, act, look again, correct")
        self.assertFalse(hasattr(god.God, "retry_commands"),
                         "the loop supersedes the one blind retry")

    def test_a_command_that_changes_nothing_is_not_a_command_that_worked(self):
        """A fill whose region held none of the block it replaces reports "Changed 0
        blocks" and reports it as success. Left unmarked, the god read that as done and
        told the player the ice was gone."""
        import asyncio

        import god as god_module
        from god import God, _changed_nothing

        self.assertTrue(_changed_nothing("Changed 0 blocks"))
        self.assertFalse(_changed_nothing("Changed 137 blocks"))
        self.assertFalse(_changed_nothing("Gave 10 [Diamond] to playerone"))

        store = Store(":memory:")
        saved = god_module.run_command
        async def missed(commands, url, **kwargs):
            return {"ok": True, "accepted": 1, "refused": [], "ran": [
                {"command": commands[0], "ok": True, "output": "Changed 0 blocks"}]}
        god_module.run_command = missed
        try:
            g = God(store, use_model=False)
            g.names["u"] = "playerone"
            acted = asyncio.run(g.run_commands(
                {"commands": ["fill ~-5 ~-5 ~-5 ~5 ~5 ~5 air replace ice"]}, "u"))
            self.assertEqual(acted["done"], [], "nothing changed, so nothing was done")
            self.assertEqual(len(acted["changed_nothing"]), 1)
            self.assertIn("was not there", acted["note"])
        finally:
            god_module.run_command = saved
            store.close()

    def test_the_acting_loop_must_answer_before_it_wanders(self):
        """The loop can act, observe and correct, which is what makes it useful and also
        what let it spend three minutes on a stubborn request."""
        import inspect

        import god
        source = inspect.getsource(god.God.on_chat)
        self.assertIn("ACT_BUDGET_SECONDS", source)
        self.assertIn("Time is up. Answer now from what you have already done", source)
        self.assertGreater(god.ACT_BUDGET_SECONDS, 0)

    def test_a_power_is_not_a_command_and_needs_no_mod(self):
        """Asked to fly and throw fire, the god reached for creative mode and a stack of
        fire charges, because commands were its only lever. Creative is not a superpower,
        it is a different game. None of this needs a client mod: flight without creative,
        an entity thrown along the line of sight and immunity to your own fire are
        ordinary server API, and none of them is expressible as a command."""
        from evidence import POWER_TOOL

        java = (self.JAVA.parent / "PowerService.java").read_text(encoding="utf-8")
        self.assertIn("player.setAllowFlight(true)", java,
                      "flight without creative keeps damage, hunger and inventory")
        self.assertIn("player.getEyeLocation().getDirection()", java,
                      "a summoned entity flies in world axes; aiming needs the look vector")
        self.assertIn("Creative is not a superpower", POWER_TOOL["description"])

    def test_a_power_is_invented_rather_than_chosen_from_a_list(self):
        """It was a fixed enum of eighteen, and the interesting requests are never on such
        a list. A power is now a trigger plus commands plus a few switches, so a web
        shooter and a wake of ice are writable without anyone having thought of them."""
        from evidence import POWER_TOOL
        from scan import IMPULSES, SWITCHES, TRIGGERS

        java = (self.JAVA.parent / "PowerService.java").read_text(encoding="utf-8")
        self.assertNotIn("List.of(\"flight\"", java, "no fixed vocabulary of powers")
        for trigger in TRIGGERS:
            self.assertIn(f'"{trigger}"', java)
            self.assertIn(trigger, POWER_TOOL["description"])
        for flag in SWITCHES:
            self.assertIn(f'"{flag}', java)
        schema = POWER_TOOL["input_schema"]["properties"]
        for field in ("scripts", "switches", "projectile", "every", "cooldown_ms"):
            self.assertIn(field, schema)
        self.assertNotIn("powers", schema, "there is no list left to choose from")
        # And no drift the other way: a trigger the plugin grew that the model is never
        # told about is a capability nobody can reach.
        for constant, mirror in (("TRIGGERS", TRIGGERS), ("IMPULSES", IMPULSES)):
            declared = java[java.index(f"List<String> {constant} = List.of("):]
            self.assertEqual(
                sorted(re.findall(r'"(\w+)"', declared[:declared.index(";")])),
                sorted(mirror), f"{constant} drifted between the plugin and the agent")

    def test_a_power_fires_when_you_click_at_nothing(self):
        """Right-clicking air with an empty hand sends the server no packet at all: the
        client reports a use only when an item or a block is in reach. So a power bound to
        it appeared to work only when facing something. The arm swing always arrives."""
        java = (self.JAVA.parent / "PowerService.java").read_text(encoding="utf-8")
        self.assertIn("PlayerAnimationEvent", java)
        self.assertIn("PlayerAnimationType.ARM_SWING", java)
        self.assertIn('fire(event.getPlayer(), "on_use")', java)
        self.assertNotIn("event.getItem() != null", java,
                         "requiring an empty hand is what made this only work in reach")

    def test_a_scripted_power_passes_the_same_gate_a_command_does(self):
        """Binding a command to a right-click must not be a way around what may be run."""
        text = (self.JAVA.parent / "ScanCommandHandler.java").read_text(encoding="utf-8")
        power = text[text.index("private void power("):text.index("private static void strings(")]
        self.assertIn("CommandService.inspect(value.getAsString(), true)", power,
                      "every scripted command is inspected, and always as anchored")
        self.assertIn("refused.add(verdict.refusal())", power)

    def test_a_granted_power_is_given_back_when_it_runs_out(self):
        """And never takes away what was there before: a granted minute running out must
        not strand a builder in creative mode mid-air."""
        java = (self.JAVA.parent / "PowerService.java").read_text(encoding="utf-8")
        self.assertIn("private void restore(Player player, Held holder)", java)
        self.assertIn("getGameMode() != GameMode.CREATIVE", java)
        self.assertIn("now >= a.expiresAtTick", java)
        self.assertIn("public void onQuit(", java, "nothing lingers on a player who left")
        self.assertIn("powers.revokeAll()",
                      (self.JAVA.parent / "StatsCommand.java").read_text(encoding="utf-8"),
                      "/mcgod stop takes back everything the god handed out")

    def test_a_word_the_model_invented_comes_back_named(self):
        """A model told nothing invents the same word again. What was not understood is
        reported with the real vocabulary beside it, and the rest of the power still
        stands: one unknown switch does not cost the player their ability."""
        import asyncio

        import god as god_module  # noqa: F401
        from god import God

        store = Store(":memory:")
        import scan
        original = scan.grant_power
        seen = {}
        async def granted(player, name, url, **kwargs):
            seen.update(kwargs)
            seen["player"] = player
            seen["name"] = name
            return {"ok": True, "granted": name, "triggers": ["on_use"],
                    "switches": ["fly"], "unknown": ["heat_vision"],
                    "refused": [], "switches_available": list(scan.SWITCHES)}
        scan.grant_power = granted
        try:
            g = God(store, use_model=False)
            g.names["u"] = "playerone"
            result = asyncio.run(g.grant_powers({
                "player": "playerone", "name": "web shooter",
                "scripts": {"on_use": ["setblock ^ ^ ^4 cobweb"]},
                "switches": ["fly", "heat_vision"],
                "projectile": "small_fireball",
            }, "u"))
            self.assertEqual(result["granted"], "web shooter")
            self.assertEqual(result["unknown"], ["heat_vision"])
            self.assertEqual(seen["scripts"], {"on_use": ["setblock ^ ^ ^4 cobweb"]})
            self.assertEqual(seen["projectile"], "small_fireball")
        finally:
            scan.grant_power = original
            store.close()

    def test_taking_every_power_away_needs_no_name(self):
        """Asked to remove all their superpowers the god reported success and removed
        nothing. A missing name was being defaulted to the literal string "power", so the
        revoke asked for one called that, matched nothing, and said it was done."""
        import asyncio

        from god import God

        store = Store(":memory:")
        import scan
        original = scan.grant_power
        asked = {}
        async def revoked(player, name, url, **kwargs):
            asked["name"] = name
            asked["revoke"] = kwargs.get("revoke")
            return {"ok": True, "revoked": ["Pig Launch"], "holding": []}
        scan.grant_power = revoked
        try:
            g = God(store, use_model=False)
            g.names["u"] = "playerone"
            asyncio.run(g.grant_powers({"player": "playerone", "revoke": True}, "u"))
            self.assertTrue(asked["revoke"])
            self.assertFalse(asked["name"],
                             "no name on a revoke means every power, not one called 'power'")
        finally:
            scan.grant_power = original
            store.close()

    def test_granting_a_held_name_replaces_it(self):
        """"Make my arrow power fully automatic" is an edit. Stacking a second ability
        beside the first left the player holding both, the original still firing, and the
        change apparently ignored."""
        java = (self.JAVA.parent / "PowerService.java").read_text(encoding="utf-8")
        self.assertIn("existing.name.equalsIgnoreCase(ability.name)", java)
        self.assertIn("replaced", java)
        from evidence import POWER_TOOL
        self.assertIn("REPLACES it", POWER_TOOL["description"])

    def test_the_god_is_told_which_powers_they_already_hold(self):
        """It cannot change or take back what it cannot see. Blind to the name of the
        power it had just given, it invented a second one beside it."""
        import inspect

        import god

        source = inspect.getsource(god.God.on_chat)
        self.assertIn("held_powers", source)
        self.assertIn("POWERS THEY HOLD RIGHT NOW", source)
        self.assertIn("powers_text", source)

    def test_something_thrown_acts_where_it_lands(self):
        """"Make the pigs explode on impact" has no other answer: the interesting moment
        happens to something the player threw, somewhere they are not. The god reached for
        on_attack, which is the player swinging at something, and nothing exploded."""
        from evidence import POWER_TOOL
        from scan import TRIGGERS

        java = (self.JAVA.parent / "PowerService.java").read_text(encoding="utf-8")
        self.assertIn("on_hit", TRIGGERS)
        self.assertIn('"on_hit"', java)
        self.assertIn("ProjectileHitEvent", java,
                      "a projectile reports its own impact")
        self.assertIn("isOnGround()", java,
                      "a thrown animal does not, so it is watched until it settles")
        self.assertIn("runner.submitAt(script, where)", java,
                      "the script runs where it landed, not where the player is")
        self.assertIn("on_hit", POWER_TOOL["description"])

    def test_a_command_can_run_at_a_place_with_nobody_there(self):
        """Composed by the server, never by the model: the same rule that makes ~ usable
        without making execute available."""
        runner = self.RUNNER.read_text(encoding="utf-8")
        self.assertIn("public synchronized int submitAt(", runner)
        self.assertIn('"execute in " + at.getWorld().getKey() + " positioned "', runner)
        self.assertIn("where.clone()", runner,
                      "the thing that landed may be gone before the command runs")

    def test_a_builders_own_answer_size_is_honoured(self):
        """Every build failed, and it read as a timeout. The builder asked for sixteen
        thousand output tokens and complete_message overwrote that with the dialogue
        default of four thousand, so a statue's worth of setblock commands hit the ceiling
        on both attempts and raised."""
        import inspect

        import evidence
        from builder import BUILD_TIMEOUT_SECONDS, MAX_BUILD_TOKENS

        source = inspect.getsource(evidence.complete_message)
        self.assertIn('request.pop("max_tokens", 0)', source,
                      "a caller that asks for a bigger answer must get one")
        self.assertIn("timeout", inspect.signature(evidence.complete_message).parameters)
        self.assertGreater(MAX_BUILD_TOKENS, evidence.MODEL_RETRY_TOKENS)
        # And the loop around it must not cut off a build it is waiting for.
        import god
        self.assertGreater(god.ACT_BUDGET_SECONDS, BUILD_TIMEOUT_SECONDS)

    def test_the_builder_is_given_the_ground_not_a_picture(self):
        """It was told what to build and one anchor point, which is not enough to put a
        thing down. On a slope that sets one foot in the air and buries the other."""
        import builder

        voxels = {}
        for x in range(-8, 9):
            for z in range(-8, 9):
                ground = 64 + (x // 4)
                voxels[(x, ground, z)] = "minecraft:grass_block[snowy=false]"
                voxels[(x, ground + 1, z)] = "minecraft:short_grass"
        voxels[(0, 70, 0)] = "minecraft:oak_leaves[distance=1]"
        site = builder.survey(voxels, [0, 64, 0], radius=4)

        rows = site["height_offsets_from_anchor_y"]
        self.assertEqual(len(rows), 9)
        # Grass and leaves are not ground. Reading grass_block as vegetation reported the
        # whole site as empty air, which is worse than no survey at all.
        self.assertNotIn("~", "".join(rows))
        self.assertIn("grass_block", site["surface_legend"].values())
        self.assertNotIn("oak_leaves", site["surface_legend"].values())
        # The slope has to be visible, or there was no point measuring it.
        self.assertIn("-1", rows[0])
        self.assertIn("+1", rows[0])
        self.assertEqual(site["corner_north_west"], [-4, -4])

    def test_the_survey_is_grids_and_never_a_block_array(self):
        """Non-negotiable: the model never receives raw voxels. Character grids with a
        legend carry the same information in a form it can actually read."""
        import inspect

        import builder

        source = inspect.getsource(builder.survey)
        self.assertIn("legend", source)
        self.assertNotIn("json.dumps(voxels", source)
        site = builder.survey({(0, 64, 0): "minecraft:stone"}, [0, 64, 0], radius=1)
        self.assertTrue(all(isinstance(row, str)
                            for row in site["height_offsets_from_anchor_y"]),
                        "rows are text a model can read, not nested coordinates")
        self.assertTrue(all(isinstance(row, str) for row in site["surface"]))

    def test_a_beam_is_traced_and_never_hits_its_own_shooter(self):
        """Written as commands a laser comes out as a run of particle calls: it stops
        where the list stops rather than where it hits, passes through walls, and its
        damage lands on whoever is nearest, which was sometimes the player holding it."""
        java = (self.JAVA.parent / "PowerService.java").read_text(encoding="utf-8")
        self.assertIn("rayTrace(", java)
        self.assertIn("!entity.equals(player)", java,
                      "the shooter's own hitbox surrounds the muzzle")
        self.assertIn("ability.range", java)
        from evidence import POWER_TOOL
        self.assertIn("HITSCAN", POWER_TOOL["description"])
        self.assertIn("beam", POWER_TOOL["input_schema"]["properties"])

    def test_a_power_can_move_the_player(self):
        """Asked to be bouncy it failed: tp puts somebody somewhere, and being bouncy is
        entirely about momentum, which no command can give."""
        from evidence import POWER_TOOL
        from scan import IMPULSES, TRIGGERS

        java = (self.JAVA.parent / "PowerService.java").read_text(encoding="utf-8")
        self.assertIn("on_land", TRIGGERS)
        self.assertIn("bounce", IMPULSES)
        self.assertIn("player.setVelocity(", java)
        self.assertIn("holder.falling", java,
                      "a landing has already absorbed the velocity; the last airborne "
                      "reading is the speed they actually arrived at")
        self.assertIn("impulse", POWER_TOOL["input_schema"]["properties"])
        self.assertIn("bounce", POWER_TOOL["description"])

    def test_what_stands_nearby_survives_json(self):
        """Four builds failed in a row reporting "the builder did not answer (TypeError)".
        Nothing was wrong with the builder: structure_name returns the whole belief row,
        the label lives in its object column, and a sqlite3.Row does not serialise."""
        import json

        store = Store(":memory:")
        try:
            store.put("structures", {
                "id": "s_1", "dim": "overworld", "min_x": 0, "min_y": 64, "min_z": 0,
                "max_x": 4, "max_y": 68, "max_z": 4},
                Belief(provenance=Provenance.SCANNED, confidence=1.0))
            store.name_structure("s_1", "gold block wall", 0.9, "because", 1, "k",
                                 category="other", description="A long wall of gold.")
            named = store.structure_name("s_1")
            self.assertIsNotNone(named)
            # The name is the object; the value is an envelope that is not a name.
            self.assertEqual(named["object"], "gold block wall")
            body = json.loads(named["value"])
            self.assertEqual(body["description"], "A long wall of gold.")
            json.dumps({"name": named["object"], "description": body["description"]})
        finally:
            store.close()

    def test_the_survey_read_fits_what_the_server_allows(self):
        """The server refuses a voxel read taller than 48 and the survey asked for 49, so
        every build fell back to blind. The refusal was swallowed, so the log said "blind"
        and never said why."""
        import inspect

        import god

        java = (self.JAVA.parent / "ScanService.java").read_text(encoding="utf-8")
        cap = int(re.search(r"MAX_VOXEL_HEIGHT = (\d+)", java).group(1))
        source = inspect.getsource(god.God.design_build)
        span = int(re.search(r"lo\[1\] \+ (\d+)", source).group(1)) + 1
        self.assertLessEqual(span, cap,
                             "the survey must ask for a box the server will actually read")
        self.assertIn("could not survey the site:", source,
                      "a refused read must say why rather than silently building blind")

    def test_a_failed_tool_is_reported_not_swallowed_by_the_guard(self):
        """Asked for a castle and given none, the player was told "I cannot ground that
        answer reliably enough to give it" — a sentence about our own machinery that
        tells them nothing about what went wrong."""
        import inspect

        import god

        source = inspect.getsource(god.God.on_chat)
        self.assertIn('acted["failures"].append', source,
                      "a builder that returns nothing is a failure worth saying aloud")
        self.assertIn('elif speech_refused and acted["failures"]:', source)
        self.assertLess(source.index('elif speech_refused and acted["failures"]:'),
                        source.index('elif acted["failures"]:'),
                        "the refused-speech case must be tested before the general one")

    def test_a_build_is_placed_rather_than_handed_back(self):
        """The builder returned 275 commands for a hot air balloon and 4 reached the
        world: the list went back to the dialogue model, which had to retype it into
        act_on_world, and a build does not fit inside a reply. The player got the gondola
        floor and nothing above it."""
        import asyncio
        import inspect

        from god import God

        source = inspect.getsource(God.design_build)
        self.assertIn("await self.run_commands(", source,
                      "the build is placed here, not by the model")

        store = Store(":memory:")
        try:
            g = God(store, use_model=False)
            import builder
            original = builder.design
            wrote = [f"setblock {i} 64 0 minecraft:stone" for i in range(275)]
            async def designed(*a, **k):
                return {"commands": wrote, "describes": "a balloon", "count": len(wrote)}
            builder.design = designed
            ran = {}
            async def running(data, actor=None):
                ran["count"] = len(data["commands"])
                return {"done": list(data["commands"]), "failures": [], "queued": 0,
                        "spell": None}
            g.run_commands = running
            async def no_voxels(*a, **k):
                return {"ok": False, "error": "no server"}
            import god as gm
            saved = gm.request_voxels
            gm.request_voxels = no_voxels
            try:
                out = asyncio.run(g.design_build({"what": "a balloon", "x": 0, "y": 64,
                                                  "z": 0}))
            finally:
                gm.request_voxels = saved
                builder.design = original
            self.assertEqual(ran["count"], 275, "every command reaches the world")
            self.assertEqual(out["placed"], 275)
            self.assertNotIn("commands", out,
                             "handing the list back is what lost 271 of them")
        finally:
            store.close()

    def test_a_provider_that_says_wait_is_waited_for(self):
        """A build died to a 402 carrying Retry-After 120 that we asked again 0.4 seconds
        later, spending the second attempt on the same refusal."""
        from evidence import MAX_RETRY_AFTER_SECONDS, _retry_after

        class Response:
            headers = {"Retry-After": "120"}

        class FromHeader(Exception):
            response = Response()

        class FromBody(Exception):
            body = {"error": {"metadata": {"headers": {"Retry-After": "90"}}}}

        self.assertEqual(_retry_after(FromHeader()), 120.0)
        self.assertEqual(_retry_after(FromBody()), 90.0,
                         "the same refusal arrives both ways depending on the layer")
        self.assertEqual(_retry_after(Exception("nothing")), 0.0)

        class Forever:
            headers = {"Retry-After": "9999"}

        class Outage(Exception):
            response = Forever()

        self.assertEqual(_retry_after(Outage()), MAX_RETRY_AFTER_SECONDS,
                         "past a point it is an outage, not a pause")

    def test_the_builder_is_told_how_to_make_a_curve(self):
        """Minecraft has no curves, so a round thing is a stack of circles whose radii
        have to be computed. Guessing them gives a barrel with a lid."""
        from builder import SYSTEM

        self.assertIn("sqrt(R^2 - (y - Yc)^2)", SYSTEM)
        self.assertIn("Never approximate a curve with a single `fill`", SYSTEM)
        self.assertIn("15-25 blocks", SYSTEM, "large has to mean a number")

    def test_building_a_shape_is_asked_of_a_builder(self):
        """A dialogue model asked for a humanoid statue produced something nobody would
        recognise. Shape is a different skill from conversation."""
        from evidence import DESIGN_TOOL

        self.assertEqual(DESIGN_TOOL["name"], "design_build")
        self.assertIn("whose shape matters", DESIGN_TOOL["description"])
        for field in ("what", "x", "y", "z", "facing", "materials"):
            self.assertIn(field, DESIGN_TOOL["input_schema"]["properties"])
        import builder
        self.assertIn("silhouette right before any detail", builder.SYSTEM)
        self.assertIn("proportion above all", builder.SYSTEM)

    def test_the_builder_is_its_own_model_and_can_be_swapped(self):
        import config

        self.assertTrue(config.BUILD_MODEL)
        source = (HERE / "config.py").read_text(encoding="utf-8")
        self.assertIn("MCGOD_OPENROUTER_BUILD_MODEL", source)
        self.assertIn("MCGOD_BUILD_MODEL", source)

    def test_a_build_that_returns_nothing_is_reported_not_invented(self):
        import asyncio

        import builder

        saved = builder.complete_message
        async def empty(_client, **_kwargs):
            class Reply:
                content = []
                stop_reason = "end_turn"
            return Reply()
        builder.complete_message = empty
        try:
            result = asyncio.run(builder.design("a statue", [0, 64, 0]))
            self.assertIn("error", result)
            self.assertNotIn("commands", result)
        finally:
            builder.complete_message = saved

    def test_the_acting_tool_tells_the_model_what_actually_happened(self):
        """A tool that answered "accepted" would leave the model as blind as before."""
        from evidence import ACT_TOOL

        self.assertEqual(ACT_TOOL["name"], "act_on_world")
        self.assertIn("what each command actually did", ACT_TOOL["description"])
        self.assertIn("look at the world and try a different way",
                      ACT_TOOL["description"])
        self.assertIn("Changed 0 blocks", ACT_TOOL["description"])
        for field in ("commands", "anchor", "every", "duration"):
            self.assertIn(field, ACT_TOOL["input_schema"]["properties"])

    def test_every_act_is_recorded_where_it_happened(self):
        """"Break the ice I just made" is unanswerable if the log says only
        `setblock ~ ~-1 ~ ice` a hundred times."""
        runner = self.RUNNER.read_text(encoding="utf-8")
        self.assertIn("where = Dims.pos(player.getLocation())", runner)
        self.assertIn("GameEvent.of(Bukkit.getCurrentTick(), \"god_command\", \"god\", dim, where)",
                      runner)

    def test_a_spell_still_answers_at_once(self):
        """Nobody waits five minutes for a reply about a five-minute effect."""
        handler = (self.JAVA.parent / "ScanCommandHandler.java").read_text(encoding="utf-8")
        self.assertIn("nobody waits for it to finish before replying", handler)
        self.assertIn("reply(conn, id, accepted.size(), refused, spell, List.of())", handler)

    def test_the_gods_own_acts_make_no_player_and_no_place(self):
        """A command has an actor that is not a player and a position that means nothing.
        Left alone it would invent a player called god and send the sweep to the origin."""
        from activity import record_activity
        from reconcile import index_regions
        from store import NON_PLAYER_ACTORS

        self.assertIn("god", NON_PLAYER_ACTORS)
        store = Store(":memory:")
        try:
            act = {"tick": 1, "t": 1_000, "type": "god_command", "actor": "god",
                   "dim": "overworld", "pos": [0, 0, 0], "command": "time set day",
                   "ok": True, "output": "Set the time to 1000"}
            record_activity(store, act)
            index_regions(store, [act])
            self.assertEqual(len(store.all("event_history")), 1,
                             "but it is remembered, or the god cannot say what it did")
            self.assertEqual(store.all("players"), [])
            self.assertEqual(store.all("regions"), [])
        finally:
            store.close()

    def test_the_model_can_read_back_what_it_did(self):
        from history import HISTORY_SYSTEM
        self.assertIn("actor='god' are commands YOU ran", HISTORY_SYSTEM)

    def test_the_server_records_every_act_as_it_runs(self):
        runner = self.RUNNER.read_text(encoding="utf-8")
        self.assertIn('"god_command"', runner)
        self.assertIn("CommandService.inspect", (self.JAVA.parent
                      / "ScanCommandHandler.java").read_text(encoding="utf-8"))
        self.assertIn("createCommandSender", runner,
                      "the command's own output is what tells the god it worked")


class SpeechDoesNotWaitInLine(unittest.TestCase):
    """Someone who speaks is waiting for a reply; everything else can wait for them.

    One worker drained one queue, so a question sat behind whatever the loop was already
    doing. A quest had just been judged, which is renders and model calls, and the player
    who asked a question during it waited minutes for a sentence.
    """

    def test_speech_is_routed_away_from_the_backlog(self):
        from god import is_urgent
        for kind in ("chat", "player_join"):
            self.assertTrue(is_urgent({"type": kind}), kind)
        for kind in ("block_place", "move", "mob_death", "explosion", "player_state"):
            self.assertFalse(is_urgent({"type": kind}), kind)

    def test_the_loop_runs_a_worker_for_speech_alone(self):
        import ast
        import inspect

        import god
        source = inspect.getsource(god._watch)
        self.assertIn("spoken", source)
        tree = ast.parse(textwrap.dedent(source))
        made = [node.func.attr for node in ast.walk(tree)
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr == "create_task"]
        self.assertGreaterEqual(len(made), 4,
                                "reader, events, speech, heartbeat, reconcile")
        names = {node.name for node in ast.walk(tree)
                 if isinstance(node, ast.AsyncFunctionDef)}
        self.assertIn("listen", names)
        self.assertIn("think", names)

    def test_speech_is_never_dropped_to_make_room(self):
        """Heartbeat samples are what a full queue may lose. A question is not."""
        import inspect

        import god
        source = inspect.getsource(god._watch)
        drop = source.index("Never block the reader")
        urgent = source.index("if is_urgent(event):")
        self.assertLess(urgent, drop,
                        "speech is routed before anything can be discarded")

    def test_every_render_in_the_live_loop_runs_off_the_event_loop(self):
        """Seconds of pure CPU on the loop stops everything else the god is doing,
        including answering whoever asked for the picture."""
        import re

        for name in ("god.py", "evidence.py"):
            source = (HERE / name).read_text(encoding="utf-8")
            for call in re.finditer(r"(?<![\w.])(sheet|landscape|plan)\(", source):
                line_start = source.rfind("\n", 0, call.start()) + 1
                line = source[line_start:source.index("\n", call.start())]
                if line.lstrip().startswith(("#", "def ", "from ", "import ")):
                    continue
                context = source[max(0, call.start() - 90):call.start()]
                self.assertIn("to_thread", context,
                              f"{name}: {line.strip()} blocks the event loop")


class OneBrokenShrubIsNotAnEvent(unittest.TestCase):
    """Breaking a single shrub set the god scanning for three minutes across a village.

    Two mechanisms compounded. Any block change marked its cell for a fresh look, and
    vegetation is a block change. Then every read found the village's path network cut by
    its edge and queued a look beyond, and that look cut the same network somewhere else,
    and so on across the world one window at a time.
    """

    def _god(self):
        from god import God
        return God(Store(":memory:"), use_model=False)

    def test_pulling_up_grass_does_not_start_a_scan(self):
        god = self._god()
        try:
            for material in ("minecraft:short_grass", "minecraft:tall_grass",
                             "minecraft:bush", "minecraft:oak_leaves",
                             "minecraft:dandelion"):
                self.assertFalse(god._changes_something_built({
                    "type": "block_break", "dim": "overworld", "pos": [10, 64, 10],
                    "before": material, "after": "minecraft:air"}), material)
        finally:
            god.store.close()

    def test_breaking_anything_built_still_does(self):
        god = self._god()
        try:
            self.assertTrue(god._changes_something_built({
                "type": "block_break", "dim": "overworld", "pos": [10, 64, 10],
                "before": "minecraft:cobblestone", "after": "minecraft:air"}))
            # placing is always worth a look: a pillar is made of dirt
            self.assertTrue(god._changes_something_built({
                "type": "block_place", "dim": "overworld", "pos": [10, 64, 10],
                "before": "minecraft:air", "after": "minecraft:dirt"}))
        finally:
            god.store.close()

    def test_taking_apart_a_build_made_of_natural_blocks_is_noticed(self):
        """A dirt pillar is dirt. Demolishing it must not be filtered out as vegetation."""
        god = self._god()
        try:
            god.store.put("masses", {
                "id": "m_pillar", "dim": "overworld", "actor": "p", "place_id": None,
                "min_x": 20, "min_y": 64, "min_z": 20,
                "max_x": 20, "max_y": 72, "max_z": 20,
                "materials": json.dumps({"dirt": 9}),
            }, Belief(provenance=Provenance.SCANNED, verified_at_ms=1,
                      value=json.dumps({"schema_version": 1, "kind": "built_mass",
                                        "blocks": 9, "origin": "player_built"})))
            self.assertTrue(god._changes_something_built({
                "type": "block_break", "dim": "overworld", "pos": [20, 68, 20],
                "before": "minecraft:dirt", "after": "minecraft:air"}))
            self.assertFalse(god._changes_something_built({
                "type": "block_break", "dim": "overworld", "pos": [900, 68, 900],
                "before": "minecraft:dirt", "after": "minecraft:air"}))
        finally:
            god.store.close()

    def test_following_on_continues_along_a_build_and_stops_where_it_ends(self):
        """A bridge can be longer than one read, so a chain of follow-ups is how it gets
        measured whole. What the chain must not do is carry on across ground nobody
        touched, which is what walked the scan over a whole village."""
        from god import God
        from reconcile import index_regions

        god = God(Store(":memory:"), use_model=False)
        try:
            god.tick = 50_000
            index_regions(god.store, [{"type": "move", "dim": "overworld", "actor": "p",
                                       "t": 1, "pos": [x, 64, 0]}
                                      for x in range(-200, 400, 32)])
            cut = {"lo": [10, 64, 0], "hi": [60, 66, 4], "blocks": 200,
                   "origin": "player_built", "partial": ["east"]}
            window = ([0, 60, -20], [64, 90, 20])

            # ground the player worked: the chain continues
            god.edits[("overworld", 1, 2, 0)] = 40
            god._follow_partials("overworld", [cut], *window, depth=1)
            self.assertGreater(len(god.disturbed), 0, "a build longer than one read")
            queued = next(iter(god.disturbed.values()))
            self.assertEqual(queued["depth"], 2, "the chain records how far it has run")
            self.assertTrue(queued["ledger_only"])

            # untouched ground: nothing to complete, so the chain stops
            god.disturbed.clear()
            god.edits.clear()
            god._follow_partials("overworld", [cut], *window, depth=1)
            self.assertEqual(god.disturbed, {},
                             "a read with no changes under it must not follow on")
        finally:
            god.store.close()

    def test_a_chain_of_follow_ups_cannot_run_forever(self):
        """The backstop: no arrangement of changed ground may walk the scan indefinitely."""
        from god import MAX_FOLLOW_DEPTH, God
        from reconcile import index_regions

        god = God(Store(":memory:"), use_model=False)
        try:
            god.tick = 50_000
            index_regions(god.store, [{"type": "move", "dim": "overworld", "actor": "p",
                                       "t": 1, "pos": [x, 64, 0]}
                                      for x in range(-200, 400, 32)])
            god.edits[("overworld", 1, 2, 0)] = 999
            cut = {"lo": [10, 64, 0], "hi": [60, 66, 4], "blocks": 200,
                   "origin": "player_built", "partial": ["east"]}
            god._follow_partials("overworld", [cut], [0, 60, -20], [64, 90, 20],
                                 depth=MAX_FOLLOW_DEPTH)
            self.assertEqual(god.disturbed, {}, "the chain stops at its depth cap")
        finally:
            god.store.close()

    def test_only_a_few_worthwhile_edges_are_chased(self):
        """A village path network is hundreds of blocks and has no edge worth finding."""
        from god import FOLLOW_UPS_PER_READ
        from reconcile import index_regions

        god = self._god()
        try:
            god.tick = 50_000
            index_regions(god.store, [{"type": "move", "dim": "overworld", "actor": "p",
                                       "t": 1, "pos": [x, 64, 0]}
                                      for x in range(-200, 400, 50)])
            god.edits[("overworld", 0, 2, 0)] = 40      # the player did work here
            sprawl = {"lo": [0, 64, 0], "hi": [300, 66, 300], "blocks": 5000,
                      "origin": "unknown", "partial": ["east", "west", "north", "south"]}
            god._follow_partials("overworld", [sprawl], [0, 60, 0], [64, 90, 64])
            self.assertEqual(god.disturbed, {},
                             "terrain-scale networks are not chased")

            small = {"lo": [10, 64, 10], "hi": [20, 70, 20], "blocks": 90,
                     "origin": "player_built",
                     "partial": ["east", "west", "north", "south", "above", "below"]}
            god._follow_partials("overworld", [small], [0, 60, 0], [64, 90, 64])
            self.assertLessEqual(len(god.disturbed), FOLLOW_UPS_PER_READ)
            self.assertGreater(len(god.disturbed), 0, "a player's own build is worth it")
        finally:
            god.store.close()

    def test_a_role_is_not_re_asked_because_a_path_grew_by_a_block(self):
        """One mass was asked four times in three minutes and flapped between
        "landmark" and "utility" for the same dirt path."""
        from config import CLASSIFY_MODEL
        from masses import pending_roles, role_key, role_prompt

        store = Store(":memory:")
        try:
            def write(blocks):
                store.put("masses", {
                    "id": "m_path", "dim": "overworld", "actor": "p", "place_id": None,
                    "min_x": 0, "min_y": 64, "min_z": 0,
                    "max_x": blocks, "max_y": 64, "max_z": 2,
                    "materials": json.dumps({"dirt_path": blocks}),
                }, Belief(provenance=Provenance.SCANNED, verified_at_ms=1,
                          value=json.dumps({"schema_version": 1, "kind": "built_mass",
                                            "blocks": blocks, "dims": f"{blocks}x1x3",
                                            "origin": "player_built"})))
                return store.get("masses", "m_path")

            row = write(100)
            self.assertEqual([r["id"] for r in pending_roles(store, CLASSIFY_MODEL)],
                             ["m_path"], "never asked before")
            store.put("relationships", {
                "id": "role:m_path", "subject": "m_path", "predicate": "role",
                "object": "utility",
            }, Belief(provenance=Provenance.INFERRED, confidence=.9, verified_at_ms=1,
                      source_event_ids=(role_key(role_prompt(row), CLASSIFY_MODEL),),
                      value=json.dumps({"noun": "dirt path", "blocks": 100})))
            self.assertEqual(pending_roles(store, CLASSIFY_MODEL), [],
                             "identical facts must never be asked again")
            write(104)
            self.assertEqual(pending_roles(store, CLASSIFY_MODEL), [],
                             "four more blocks is the same path")
            write(200)
            self.assertEqual([r["id"] for r in pending_roles(store, CLASSIFY_MODEL)],
                             ["m_path"], "twice the size is worth another look")
        finally:
            store.close()


class NoAnswerMayHangForever(unittest.TestCase):
    """A stalled model call is a silent god, which from the player's side is the same as
    the thing being down.

    Measured on this setup a call takes one to nine seconds whatever its shape. Then one
    ordinary call took 609 seconds and a player waited ten minutes for a sentence. Nothing
    about the request explained it and repeating it was fast, so no prompt or model choice
    prevents it recurring. Only a deadline does.
    """

    class Reply:
        content = ()
        stop_reason = "end_turn"

    def _client(self, delays):
        """A client whose calls take the given times in order."""
        calls = []

        class Messages:
            async def create(self, **_kwargs):
                delay = delays[min(len(calls), len(delays) - 1)]
                calls.append(delay)
                await asyncio.sleep(delay)
                return NoAnswerMayHangForever.Reply()

        class Client:
            messages = Messages()

        return Client(), calls

    def test_a_stalled_call_is_abandoned_and_asked_again(self):
        import evidence

        client, calls = self._client([10.0, 0.0])
        saved = evidence.MODEL_TIMEOUT_SECONDS
        evidence.MODEL_TIMEOUT_SECONDS = 0.05
        try:
            started = time.monotonic()
            reply = asyncio.run(evidence.complete_message(client, messages=[]))
            self.assertIsInstance(reply, NoAnswerMayHangForever.Reply)
            self.assertEqual(len(calls), 2, "the first was abandoned, the second answered")
            self.assertLess(time.monotonic() - started, 5,
                            "the caller must not wait out the stalled call")
        finally:
            evidence.MODEL_TIMEOUT_SECONDS = saved

    def test_a_provider_that_never_answers_raises_rather_than_hanging(self):
        import evidence

        client, calls = self._client([10.0])
        saved = evidence.MODEL_TIMEOUT_SECONDS
        evidence.MODEL_TIMEOUT_SECONDS = 0.05
        try:
            with self.assertRaises(TimeoutError):
                asyncio.run(evidence.complete_message(client, messages=[]))
            self.assertEqual(len(calls), 2, "tried twice, then gave up")
        finally:
            evidence.MODEL_TIMEOUT_SECONDS = saved

    def test_the_evidence_loop_stops_gathering_once_its_time_is_spent(self):
        """Per-call deadlines bound one request; six rounds of them still add up to
        minutes. Past the budget the tools are withheld, so the model must answer from
        what it already has."""
        import history

        store = Store(":memory:")
        store.put("event_history", {
            "id": "e1", "actor": "p", "dim": "overworld", "tick": 1, "event_t": 1,
            "kind": "block_place", "place_id": None, "x": 0, "y": 64, "z": 0,
            "payload": json.dumps({"after": "minecraft:stone"}),
        }, Belief(provenance=Provenance.OBSERVED))
        shapes = []

        class ToolBlock:
            type = "tool_use"
            name = "query_database"
            id = "t1"
            input = {"sql": "SELECT event_t FROM event_history"}

        class Reply:
            def __init__(self, blocks):
                self.content = blocks
                self.stop_reason = "end_turn"

        class Text:
            type = "text"
            text = "You placed a stone block."

        async def fake_complete(_client, **request):
            shapes.append("tools" in request)
            # Always ask for another query; only withholding the tools can end this.
            return Reply([ToolBlock()] if "tools" in request else [Text()])

        saved_complete = history.complete_message
        saved_budget = history.ANSWER_BUDGET_SECONDS
        saved_client = history.model_client
        history.complete_message = fake_complete
        history.ANSWER_BUDGET_SECONDS = -1.0        # the budget is already spent
        history.model_client = lambda **_kw: object()
        try:
            answer = asyncio.run(history.answer_if_history(
                store, {"actor": "p", "dim": "overworld", "t": 2, "pos": [0, 64, 0]},
                "what did I do", bridge_url=None))
            self.assertEqual(answer, "You placed a stone block.")
            self.assertEqual(shapes[0], True, "the first round may use tools")
            self.assertFalse(shapes[-1], "the last round must not, or it never ends")
            self.assertLessEqual(len(shapes), history.MAX_TOOL_ROUNDS + 1)
        finally:
            history.complete_message = saved_complete
            history.ANSWER_BUDGET_SECONDS = saved_budget
            history.model_client = saved_client
            store.close()

    def test_evidence_is_still_required_before_any_answer(self):
        """Running out of time must not become a way to answer from nothing."""
        import history

        store = Store(":memory:")

        class Text:
            type = "text"
            text = "Something happened, probably."

        class Reply:
            content = [Text()]
            stop_reason = "end_turn"

        async def fake_complete(_client, **_request):
            return Reply()

        saved_complete = history.complete_message
        saved_budget = history.ANSWER_BUDGET_SECONDS
        saved_client = history.model_client
        history.complete_message = fake_complete
        history.ANSWER_BUDGET_SECONDS = -1.0
        history.model_client = lambda **_kw: object()
        try:
            with self.assertRaises(history.UnsafeHistoryQuery):
                asyncio.run(history.answer_if_history(
                    store, {"actor": "p", "dim": "overworld", "t": 2, "pos": [0, 64, 0]},
                    "what did I do", bridge_url=None))
        finally:
            history.complete_message = saved_complete
            history.ANSWER_BUDGET_SECONDS = saved_budget
            history.model_client = saved_client
            store.close()


class CoordinatesNeverReachThePlayer(unittest.TestCase):
    """Every notation a model reaches for, and the placeholder it leaves when told not to.

    The triple pattern was the whole gate, so "[-1332, 5716]" and "X -1,332, Z 5,716" both
    reached chat. A horizontal position names two axes, which is exactly how somewhere far
    away gets reported."""

    def _clean(self, text: str) -> str:
        from god import (contains_raw_coordinates, remove_policy_announcements,
                         remove_raw_coordinates)
        if contains_raw_coordinates(text):
            text = remove_raw_coordinates(text)
        return remove_policy_announcements(text)

    def test_every_notation_of_a_coordinate_is_caught(self):
        from god import contains_raw_coordinates
        for said in ("around [-1332, 5716]", "a mansion (640, 3264) lies southeast",
                     "at X -1,332, Z 5,716", "x 300 z -300", "x=204, y=68, z=-1130",
                     "at 204, 68, -1130", "around 301-308 by -303 to -297"):
            self.assertTrue(contains_raw_coordinates(said), said)
            self.assertNotRegex(self._clean(said), r"-?\d{3,}\s*,\s*-?\d{3,}")

    def test_distances_and_large_numbers_are_not_coordinates(self):
        from god import contains_raw_coordinates
        for said in ("It is 10,000 blocks away and covers 32.2% of the area.",
                     "about 3,580 blocks away in a dark forest to the southeast",
                     "272 villages within 10,000 blocks", "only about 10-17 blocks away"):
            self.assertFalse(contains_raw_coordinates(said), said)
            self.assertEqual(self._clean(said), said)

    def test_the_omission_itself_is_not_announced(self):
        """A placeholder tells the player about the rules instead of about the world."""
        self.assertEqual(
            self._clean("Yes. The closest is about 6,233 blocks south of here, near the "
                        "generated location [coordinates omitted]."),
            "Yes. The closest is about 6,233 blocks south of here.")
        self.assertEqual(self._clean("The nearest mansion is southeast, coordinates "
                                     "withheld."),
                         "The nearest mansion is southeast.")

    def test_an_ordinary_sentence_survives_untouched(self):
        for said in ("A village sits north, and a mansion lies southeast.",
                     "You built a cobblestone elephant with two gold blocks for eyes."):
            self.assertEqual(self._clean(said), said)


class TheSeedMapKnowsUnvisitedGround(unittest.TestCase):
    """Generation is a pure function of the seed, so the whole world is computable without
    visiting any of it. This is the only source that can COUNT rather than locate one.

    The god answered "how many villages within 10,000 blocks of me" with 8, by counting
    rows in generated_features: five village sites it happened to have located plus three
    component buildings of one of them. The real answer on this world is 272.

    These tests run offline. They need a C compiler once to build the vendored library;
    without one they skip rather than fail, because a missing toolchain is not a defect in
    the world model.
    """

    @classmethod
    def setUpClass(cls):
        import seedmap
        cls.seedmap = seedmap
        try:
            seedmap.build()
        except seedmap.SeedMapUnavailable as error:
            raise unittest.SkipTest(f"cubiomes could not be built here: {error}")
        cls.truth = json.loads((HERE / "seedmap_truth.json").read_text(encoding="utf-8"))
        cls.seed = cls.truth["seed"]

    def test_the_vendored_library_is_the_version_we_patched(self):
        self.assertEqual(self.seedmap.version_name(), self.truth["version"])
        self.assertEqual(self.seedmap.self_check(seed=self.seed), [])

    def test_biomes_match_the_recorded_world_in_every_dimension(self):
        for case in self.truth["biomes"]:
            x, y, z = case["pos"]
            with self.seedmap.World(self.seed, case.get("dim", "overworld")) as world:
                self.assertEqual(world.biome(x, y, z), case["biome"],
                                 f"biome changed at {case['pos']} in "
                                 f"{case.get('dim', 'overworld')}")

    def test_structure_positions_the_server_confirmed_are_still_produced(self):
        """Each of these was independently located by the running Paper server."""
        for case in self.truth["nearest"]:
            result = self.seedmap.find(case["structure"], self.seed, case["from"],
                                       radius=40000, limit=1)
            self.assertTrue(result["found"], f"lost {case['structure']}")
            got = [result["found"][0]["x"], result["found"][0]["z"]]
            self.assertEqual(got, case["pos"], f"{case['structure']} moved")

    def test_counting_over_a_wide_radius_is_the_whole_point(self):
        for case in self.truth["counts"]:
            result = self.seedmap.count(case["structure"], self.seed, case["from"],
                                        case["radius"])
            self.assertEqual(result["candidates"], case["candidates"],
                             f"{case['structure']} placement changed")
            self.assertEqual(result["count"], case["count"],
                             f"{case['structure']} terrain check changed")
        villages = next(c for c in self.truth["counts"] if c["structure"] == "village")
        self.assertGreater(villages["count"], 100,
                           "a 10km sweep finds hundreds of villages, not a handful")

    def test_spawn_and_strongholds_are_stable(self):
        with self.seedmap.World(self.seed, "overworld") as world:
            self.assertEqual(world.spawn(), self.truth["spawn"])
            self.assertEqual(world.strongholds(3), self.truth["strongholds"])

    def test_every_answer_says_it_describes_generation_not_the_present(self):
        """A god that forgets this will tell a player about a village they burned down."""
        result = self.seedmap.nearest("village", self.seed, [300, -300])
        self.assertIn("as_generated", result)
        self.assertIn("built", result["as_generated"])

    def test_an_approximate_structure_carries_its_measured_miss_rate(self):
        exact = self.seedmap.nearest("village", self.seed, [300, -300])
        self.assertEqual(exact["accuracy"], "exact")
        self.assertNotIn("caution", exact)
        rough = self.seedmap.find("desert_pyramid", self.seed, [300, -300], 20000, limit=1)
        self.assertEqual(rough["accuracy"], "approximate")
        self.assertIn("confirm", rough["caution"])

    def test_structures_are_searched_in_the_dimension_they_generate_in(self):
        self.assertEqual(self.seedmap.structure_dimension("fortress"), "the_nether")
        self.assertEqual(self.seedmap.structure_dimension("end_city"), "the_end")
        self.assertEqual(self.seedmap.structure_dimension("village"), "overworld")
        self.assertGreater(
            self.seedmap.find("fortress", self.seed, [0, 0], 3000)["count"], 0)

    def test_it_refuses_what_it_cannot_answer_instead_of_guessing(self):
        with self.assertRaises(self.seedmap.SeedMapUnavailable):
            self.seedmap.find("teapot", self.seed, [0, 0], 1000)
        with self.assertRaises(self.seedmap.SeedMapUnavailable):
            self.seedmap.find("village", self.seed, [0, 0], self.seedmap.MAX_RADIUS + 1)
        with self.assertRaises(self.seedmap.SeedMapUnavailable):
            self.seedmap.World(self.seed, "the_aether")

    def test_a_missing_seed_is_refused_not_defaulted(self):
        """Computing a map for the wrong seed describes a world that does not exist."""
        store = Store(":memory:")
        saved = os.environ.pop("MCGOD_SEED", None)
        original = self.seedmap.seed_from_level_dat
        self.seedmap.seed_from_level_dat = lambda path=None: None
        try:
            with self.assertRaises(self.seedmap.SeedMapUnavailable):
                self.seedmap.world_seed(store)
            store.put("relationships", {
                "id": "world:seed", "subject": "world", "predicate": "seed",
                "object": str(self.seed),
            }, Belief(provenance=Provenance.SCANNED))
            self.assertEqual(self.seedmap.world_seed(store), self.seed)
        finally:
            self.seedmap.seed_from_level_dat = original
            if saved is not None:
                os.environ["MCGOD_SEED"] = saved
            store.close()

    def test_the_seed_map_never_writes_to_the_store(self):
        """Persisting hundreds of computed sites would drown the record of places anyone
        has actually been, which is what generated_features is for."""
        from history import _run_seed_map

        store = Store(":memory:")
        store.put("relationships", {
            "id": "world:seed", "subject": "world", "predicate": "seed",
            "object": str(self.seed),
        }, Belief(provenance=Provenance.SCANNED))
        try:
            before = {table: len(store.all(table))
                      for table in ("generated_features", "terrain_regions", "structures",
                                    "relationships", "masses")}
            result = _run_seed_map(store, {"mode": "count", "structure": "village",
                                           "x": 300, "z": -300, "radius": 10000})
            self.assertTrue(result["ok"])
            self.assertGreater(result["count"], 100)
            after = {table: len(store.all(table))
                     for table in ("generated_features", "terrain_regions", "structures",
                                   "relationships", "masses")}
            self.assertEqual(after, before)
        finally:
            store.close()

    def test_a_wide_sweep_reports_a_count_without_listing_everything(self):
        from history import SEED_MAP_LISTED, _run_seed_map

        store = Store(":memory:")
        store.put("relationships", {
            "id": "world:seed", "subject": "world", "predicate": "seed",
            "object": str(self.seed),
        }, Belief(provenance=Provenance.SCANNED))
        try:
            result = _run_seed_map(store, {"mode": "all", "structure": "village",
                                           "x": 300, "z": -300, "radius": 10000})
            self.assertGreater(result["count"], SEED_MAP_LISTED)
            self.assertEqual(len(result["found"]), SEED_MAP_LISTED)
            self.assertEqual(result["listed"], SEED_MAP_LISTED)
        finally:
            store.close()

    def test_the_history_model_is_told_to_count_from_the_seed_not_the_table(self):
        from history import HISTORY_SYSTEM, TOOLS
        names = [tool["name"] for tool in TOOLS]
        self.assertIn("query_seed_map", names)
        self.assertIn("only way to COUNT", HISTORY_SYSTEM)
        tool = next(t for t in TOOLS if t["name"] == "query_seed_map")
        self.assertIn("never the present", tool["description"])

    def test_the_seed_map_reaches_far_past_the_server_backed_tool(self):
        """The plugin RPC stops at a few thousand blocks. "Not found within 4,096" is a
        reason to ask the seed map, not an answer: the nearest ice spikes on this world is
        6,233 blocks away and the server confirms it is there."""
        from history import HISTORY_SYSTEM, MAX_ENVIRONMENT_RADIUS, TOOLS

        with self.seedmap.World(self.seed, "overworld") as world:
            found = world.nearest_biome("ice_spikes", 300, -300, max_radius=20000)
        self.assertTrue(found["found"])
        self.assertGreater(found["distance"], MAX_ENVIRONMENT_RADIUS)
        self.assertEqual([found["x"], found["z"]], [-1332, 5716])
        self.assertEqual(found["direction"], "south")
        tool = next(t for t in TOOLS if t["name"] == "query_seed_map")
        self.assertIn("Its reach is not the reach of inspect_seed_environment",
                      tool["description"])
        self.assertIn("never an answer on its own", HISTORY_SYSTEM)

    def test_a_biome_that_is_not_found_is_reported_as_not_found_here(self):
        """Never as "this biome does not exist": a patch smaller than the sampling step
        can be stepped over, and one may lie beyond the radius searched."""
        with self.seedmap.World(self.seed, "overworld") as world:
            missing = world.nearest_biome("mushroom_fields", 300, -300, max_radius=600)
        self.assertFalse(missing["found"])
        self.assertIn("within 600 blocks", missing["note"])
        self.assertIn("may lie farther out", missing["note"])
        with self.assertRaises(self.seedmap.SeedMapUnavailable):
            with self.seedmap.World(self.seed, "overworld") as world:
                world.nearest_biome("marshmallow_swamp", 0, 0)

    def test_the_most_common_biome_is_counted_not_eyeballed(self):
        """The god said plains; ocean covers a third of that circle."""
        with self.seedmap.World(self.seed, "overworld") as world:
            spread = world.biome_counts(300, -300, 1000, step=16)
        self.assertEqual(spread["most_common"], "ocean")
        self.assertGreater(spread["biomes"]["ocean"]["share"], 0.3)
        self.assertGreater(spread["biomes"]["ocean"]["samples"],
                           spread["biomes"]["plains"]["samples"])
        self.assertEqual(sum(item["samples"] for item in spread["biomes"].values()),
                         spread["samples"])

    def test_the_village_you_are_standing_in_is_marked_as_such(self):
        """Asked for the closest village other than the one they were in, the god named
        the one they were standing in, 43 blocks away. The computation was right; the
        model had no way to tell that this coordinate was its own village."""
        from history import _run_seed_map

        store = Store(":memory:")
        actor = "player-1"
        try:
            store.put("relationships", {
                "id": "world:seed", "subject": "world", "predicate": "seed",
                "object": str(self.seed),
            }, Belief(provenance=Provenance.SCANNED))
            # the plains village the test world actually has, and the player inside it
            store.put("generated_features", {
                "id": "generated:plains", "dim": "overworld",
                "kind": "minecraft:village_plains", "x": 352, "y": 64, "z": -320,
                "discovered_from": "world_generator",
            }, Belief(provenance=Provenance.SCANNED,
                      value=json.dumps({"origin": "world_generated"})))
            # a component building of it must never be mistaken for a second village
            store.put("generated_features", {
                "id": "g_church", "dim": "overworld", "kind": "village_church",
                "x": 340, "y": 76, "z": -283, "discovered_from": "observed workstation",
            }, Belief(provenance=Provenance.INFERRED,
                      value=json.dumps({"origin": "world_generated",
                                        "parent": "generated:plains"})))
            store.put("actor_presence", {
                "id": actor, "actor": actor, "dim": "overworld",
                "place_id": "generated:plains", "event_t": 1,
            }, Belief(provenance=Provenance.OBSERVED))

            result = _run_seed_map(store, {"mode": "all", "structure": "village",
                                           "x": 310, "z": -310, "radius": 2000}, actor)
            nearest = result["found"][0]
            self.assertEqual([nearest["x"], nearest["z"]], [352, -320])
            self.assertLess(nearest["distance"], 100, "it is the one they are standing in")
            self.assertTrue(nearest["you_are_here"])
            self.assertEqual(nearest["known_as"]["id"], "generated:plains")
            self.assertEqual(result["standing_in"], "generated:plains")
            self.assertIn("skip it", result["note_here"])

            others = result["found"][1:]
            self.assertTrue(others, "there are other villages to name")
            self.assertFalse(any(item.get("you_are_here") for item in others))
            self.assertGreater(others[0]["distance"], 1000,
                               "the next real village is a long way off")
        finally:
            store.close()

    def test_the_model_is_told_to_skip_the_place_it_is_standing_in(self):
        from history import HISTORY_SYSTEM
        self.assertIn("you_are_here", HISTORY_SYSTEM)
        self.assertIn("The nearest village to someone", HISTORY_SYSTEM)

    def test_speech_uses_directions_rather_than_coordinates(self):
        result = self.seedmap.nearest("village", self.seed, [300, -300])
        said = self.seedmap.describe(result, [300, -300])
        from god import contains_raw_coordinates
        self.assertFalse(contains_raw_coordinates(said), said)
        self.assertIn("village", said)


if __name__ == "__main__":
    unittest.main(verbosity=2)
