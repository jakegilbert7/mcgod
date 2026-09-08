#!/usr/bin/env python3
"""L5: assemble the god's context.

Every section has a hard token budget, enforced by truncation. Nothing is ever dumped
wholesale — the store will eventually hold far more than fits, and a system that grows its
prompt until it breaks fails silently at the worst moment.

Two rules shape this module:

Context is rebuilt fresh from the store on every call. Conversation history is never
accumulated, because the god's own past speculation would become indistinguishable from
evidence within a few turns.

Provenance and staleness are rendered into the prompt text. The model has to be able to see
its own uncertainty; a fact it read from a scan two minutes ago and a guess it made an hour
ago must not look the same on the page.
"""

from __future__ import annotations

import collections
import json
import time
from dataclasses import dataclass, field

from store import INFERRED_THRESHOLD, Belief, Provenance, Store

# Budgets from CLAUDE.md. The sum is the ceiling on any single call.
BUDGETS = {
    "persona": 800,
    "player_state": 300,
    "world": 900,
    "trigger": 600,
    "digest": 1400,
    "memories": 3000,
    "quests": 500,
    "tools": 600,
    "said": 500,
}
TOTAL_BUDGET = sum(BUDGETS.values())


#: Chars per token, chosen to over-count rather than under-count.
#:
#: Calibrated against messages.count_tokens on real text from this system, and the obvious
#: chars/4 rule is badly wrong here. Measured densities:
#:
#:     prose             3.44 chars/token
#:     slice grid        3.47
#:     structure facts   2.48
#:     a digest line     1.94   <- densest, and the digest is made of these
#:
#: Coordinates, brackets, signed numbers and material keys tokenize far denser than prose.
#: A first pass at chars/4 measured the full assembled context 41% under; a second at
#: chars/2.2 still under-counted the digest. An estimate that under-counts makes every
#: budget check meaningless in the one direction that hurts, so the constant sits at or
#: below the densest text measured, not at the average.
#:
#: Cost of the margin: prose over-counts by ~1.8x. That is the right trade — over-counting
#: truncates a little early, under-counting blows the context window in production.
#: Fixtures in tests_token_fixtures.json pin this; re-measure with
#: `python3 classify.py --check-tokens` whenever the prompt shape changes.
CHARS_PER_TOKEN = 1.9


def estimate_tokens(text: str) -> int:
    """Rough token count, deliberately an upper bound.

    This stays local on purpose. The accurate count is `client.messages.count_tokens`, but
    that is a network round trip, and spending one per section per call to decide how much
    to truncate would cost more than the truncation saves. Calibrate it instead:
    `python3 classify.py --check-tokens` reports this estimate against the real count, and
    the estimate is only trustworthy while it stays at or above it. Never use tiktoken —
    it is OpenAI's tokenizer and undercounts Claude badly.
    """
    if not text:
        return 0
    return int(len(text) / CHARS_PER_TOKEN) + 1


def _age(now_tick: int, then_tick: int | None) -> str:
    """Human-readable staleness, so the model can weigh how old a belief is."""
    if then_tick is None:
        return "age unknown"
    ticks = max(0, now_tick - then_tick)
    minutes = ticks / 20 / 60
    if minutes < 1:
        return "just now"
    if minutes < 60:
        return f"{minutes:.0f}m ago"
    return f"{minutes / 60:.1f}h ago"


def stamp(provenance: str, confidence: float, now_tick: int, verified: int | None,
          decayed: float | None = None, contradicted: str | None = None,
          verified_ms: int | None = None) -> str:
    """The provenance tag that precedes every retrieved fact.

    Staleness is rendered, not merely tracked. A belief nobody has confirmed in a long while
    is not wrong — it is old — and the god can only hedge appropriately if it can see which
    of its facts have gone quiet.
    """
    tag = provenance
    if provenance == Provenance.INFERRED.name:
        tag += f" {confidence:.2f}"
    if provenance == Provenance.ASSERTED.name:
        tag += ", NEVER assert as fact"
    if contradicted:
        tag += f", KNOWN OUT OF DATE: {contradicted}"
    elif decayed is not None and decayed < 0.5:
        tag += f", unconfirmed — {decayed:.0%} sure it still holds"
    if verified_ms:
        age_ms = max(0, int(time.time() * 1000) - verified_ms)
        minutes = age_ms / 60_000
        age = ("just now" if minutes < 1 else f"{minutes:.0f}m ago"
               if minutes < 60 else f"{minutes / 60:.1f}h ago")
    else:
        age = _age(now_tick, verified)
    return f"[{tag}, {age}]"


def _facts(row) -> dict:
    try:
        return json.loads(row["value"] or "{}")
    except (TypeError, ValueError):
        return {}


def _label(row, names) -> str | None:
    named = names.get(row["id"])
    return named["object"] if named else None


def _category(row, names) -> str | None:
    """The grouping key. Free-text labels vary; the category is what code can rely on."""
    named = names.get(row["id"])
    if not named:
        return None
    try:
        return json.loads(named["value"] or "{}").get("category")
    except (TypeError, ValueError):
        return None


def _named(row, names, store: Store | None = None) -> str:
    """A structure's model-given name, carrying its own separate provenance.

    The name and geometry are separate beliefs, so the two cannot share one tag. A
    name below the assertability threshold is marked, because the god must be able to tell
    a guess it should hedge from a measurement it can state.
    """
    named = names.get(row["id"])
    if not named:
        return f'{row["id"]} (unnamed — describe it by its materials, do not invent a name)'
    confidence = (store.confidence_now("relationships", named) if store else
                  float(named["confidence"]))
    assertable = (named["provenance"] != Provenance.ASSERTED.name
                  and (named["provenance"] != Provenance.INFERRED.name
                       or confidence >= INFERRED_THRESHOLD))
    # Spelled out rather than flagged. A terse marker was ignored: the god read a 0.72
    # guess and told the player "your fields are fenced and tilled" as though it had seen
    # it. Naming a place is the most confident-sounding thing it does, so the prompt says
    # in words whose guess it is.
    hedge = "" if assertable else " — SAY THIS AS A GUESS OR NOT AT ALL"
    return (f'the one you called "{named["object"]}" '
            f'({named["confidence"]:.2f} sure{hedge}) [{row["id"]}]')


@dataclass
class Section:
    name: str
    budget: int
    lines: list[str] = field(default_factory=list)
    truncated: int = 0

    @property
    def text(self) -> str:
        body = "\n".join(self.lines)
        if self.truncated:
            body += f"\n… {self.truncated} more omitted (budget)"
        return body

    @property
    def tokens(self) -> int:
        return estimate_tokens(self.text)

    def fit(self) -> None:
        """Drops lines from the end until the section fits its budget."""
        while self.lines and estimate_tokens(self.text) > self.budget:
            self.lines.pop()
            self.truncated += 1


PERSONA = """\
You are a god watching a Minecraft world. You perceive it only through the summaries below.

Rules you must obey:
- State as fact only what is marked SCANNED, OBSERVED or DERIVED. Hedge anything INFERRED.
  Never assert anything marked ASSERTED — that is your own past speech, not evidence.
- The NAME of a place is always a guess of yours, never an observation. Measurements are
  facts; what a thing is FOR is inference. Say "what I take to be your coop" and not "your
  coop", and never refer to something you have no record of at all — if you cannot point to
  a structure in the list below, do not mention it.
- You do not decide whether a player succeeded at a quest. Code decides; you narrate.
- If you have not been told something, you do not know it. Do not fill gaps.
- Rewards are in-game only.
Reply with JSON: {"speech": str, "quest_spec": object|null, "actions": [], "beliefs_proposed": []}"""

TOOLS = """\
scan_region(dim, min[3], max[3]) -> histogram, block-entity census, character-grid slices
recall(subject) -> stored beliefs about a structure, player or episode, with provenance
issue_quest(spec) -> constraint spec; code compiles it to watchers and judges completion
speak(text) -> address a player
grant(item, count) -> in-game reward, subject to per-hour budget"""


class Assembler:
    """Builds one context from the store plus whatever triggered this call."""

    def __init__(self, store: Store) -> None:
        self.store = store

    def build(self, now_tick: int, actor: str, trigger: dict,
              player_state: dict | None = None,
              episodes: list = (), findings: list = (),
              immersive: bool = False) -> list[Section]:
        return [
            self._persona(),
            self._player_state(now_tick, actor, player_state, immersive),
            self._world(now_tick, actor, immersive),
            self._trigger(trigger),
            self._digest(now_tick, episodes, immersive),
            self._memories(now_tick, actor, trigger, immersive),
            self._quests(now_tick, actor),
            self._said(now_tick, actor),
            self._tools(),
        ]

    def _persona(self) -> Section:
        s = Section("persona", BUDGETS["persona"], PERSONA.splitlines())
        s.fit()
        return s

    def _player_state(self, now_tick: int, actor: str, state: dict | None,
                      immersive: bool = False) -> Section:
        s = Section("player_state", BUDGETS["player_state"])
        if not state:
            s.lines.append(f"{actor[:8]}: no recent state sample")
        else:
            where = (f" at {state.get('pos')}" if not immersive else "")
            s.lines.append(
                f"{actor[:8]} {stamp('OBSERVED', 1.0, now_tick, state.get('tick'))} "
                f"{state.get('health')}hp food {state.get('food')} "
                f"lvl {state.get('level')}{where} in "
                f"{str(state.get('biome', '?')).replace('minecraft:', '')}")
            s.lines.append(
                f"  {state.get('weather', '?')}, light {state.get('light', '?')}, "
                f"time {state.get('time', '?')}, holding "
                f"{str(state.get('held') or 'nothing').replace('minecraft:', '')}")
        s.fit()
        return s

    def _world(self, now_tick: int, focus_actor: str,
               immersive: bool = False) -> Section:
        """The state of the world and everyone in it.

        Not a digest of what happened — a description of what *is*. Who exists, what they
        are equipped to do, where they operate, how much of the world they have moved. The
        god reads state and draws its own conclusions; nothing here narrates.

        Every figure is DERIVED: deterministic computation over observed events. A structure
        knows who placed it because the event that created it named an actor, so ownership
        is observation rather than inference.
        """
        s = Section("world", BUDGETS["world"])
        rows = self.store.all("players")
        if not rows:
            s.lines.append("no actors on record")
            s.fit()
            return s

        for row in rows:
            try:
                p = json.loads(row["profile"] or "{}")
            except (TypeError, ValueError):
                continue
            who = row["id"][:8] + (" (this player)" if row["id"] == focus_actor else "")
            s.lines.append(
                f"[{row['provenance']}] {who}: {p.get('playtime_min')}min over "
                f"{p.get('sessions')} sessions, {p.get('episodes')} episodes")
            location = (f"{p.get('last_pos')} in {row['dim']}" if not immersive
                        else f"in {row['dim']}")
            s.lines.append(
                f"  now: {location}, {p.get('health')}hp "
                f"food {p.get('food')} level {p.get('level')}, wearing "
                + (", ".join(v.replace('minecraft:', '')
                             for k, v in (p.get('equipment') or {}).items() if k != 'held')
                   or "nothing"))
            s.lines.append(
                f"  capable of: {str(p.get('best_tool')).replace('minecraft:', '')} / "
                f"{str(p.get('best_armor')).replace('minecraft:', '')}, "
                f"{p.get('distinct_materials')} distinct materials handled, "
                f"{len(p.get('advancements') or [])} advancements")
            if immersive:
                s.lines.append(f"  operates across {p.get('cells_visited')} known regions")
            else:
                s.lines.append(
                    f"  operates: y {p.get('y_range')}, {p.get('cells_visited')} region cells, "
                    f"home cell {p.get('home_cell')}")
            s.lines.append(
                f"  has done: placed {p.get('placed')} broke {p.get('broken')} "
                f"(build ratio {p.get('build_ratio')}), {p.get('structures_built')} "
                f"structures, {p.get('deaths')} deaths, floor {p.get('health_floor')}hp")
            kills = p.get("kills") or {}
            if kills:
                s.lines.append("  has killed: " + ", ".join(
                    f"{v} {k.replace('minecraft:', '')}" for k, v in
                    sorted(kills.items(), key=lambda kv: -kv[1])[:6]))
            travel = p.get("travel_m") or {}
            entries = p.get("vehicle_entries") or {}
            if travel or entries:
                rides = []
                for vehicle in sorted(set(travel) | set(entries)):
                    detail = []
                    if entries.get(vehicle):
                        detail.append(f"entered {entries[vehicle]} time(s)")
                    if travel.get(vehicle):
                        detail.append(f"travelled {travel[vehicle]}m")
                    rides.append(f"{vehicle}: " + ", ".join(detail))
                s.lines.append("  has ridden: " + "; ".join(rides))
        s.fit()
        return s

    def _trigger(self, trigger: dict) -> Section:
        s = Section("trigger", BUDGETS["trigger"])
        s.lines.append(f"Why you are awake: {trigger.get('reason', 'unspecified')}")
        for key, value in trigger.get("facts", {}).items():
            s.lines.append(f"  {key}: {value}")
        s.fit()
        return s

    def _digest(self, now_tick: int, episodes, immersive: bool = False) -> Section:
        """Most recent episodes first; the budget decides how far back it reaches."""
        s = Section("digest", BUDGETS["digest"])
        for ep in sorted(episodes, key=lambda e: e.tick_end, reverse=True):
            # Rank by magnitude. Counter.most_common on an all-negative delta returns the
            # least-negative first, which shows the god the three most trivial materials and
            # none of the ones that matter.
            biggest = sorted(ep.net.items(), key=lambda kv: -abs(kv[1]))[:2]
            net = " ".join(f"{v:+d}{k.replace('minecraft:', '')}"
                           for k, v in biggest) or "-"
            bbox = ep.bbox
            where = (f"@{bbox[0][0]},{bbox[0][1]},{bbox[0][2]}"
                     if bbox and not immersive else "-")
            s.lines.append(
                f"{_age(now_tick, ep.tick_end)}: {ep.duration_s / 60:.1f}m "
                f"+{ep.gross_placed}/-{ep.gross_broken} {net} {where} "
                f"walk{ep.path:.0f}")
        s.fit()
        return s

    #: How many memories are rendered in full before detail drops away.
    FULL_DETAIL = 3
    #: How many get a single line after that. The rest are counted, not listed.
    BRIEF_DETAIL = 15
    #: How many memories are eligible to be grouped and named at all.
    #:
    #: Grouping alone is bounded but not selective: with ten thousand structures every
    #: category collapses into one line reading "1263x tree felling, nearest at (109,47,-958)"
    #: — true, a thousand blocks away, and useless. Only the most salient few hundred are
    #: worth naming; past that the honest summary is a number.
    CONSIDER = 200

    def _memories(self, now_tick: int, actor: str, trigger: dict,
                  immersive: bool = False) -> Section:
        """Retrieval by salience, rendered at a detail level that falls off with rank.

        Rendering every memory in full is the waste that matters. The building a player is
        standing in and a tree they felled forty minutes ago three hundred blocks away cost
        the same 60 tokens, and only one of them is worth that. Three tiers — full, one line,
        a counted summary — hold several times as many memories in fewer tokens, and they
        degrade gracefully: at three thousand structures the tail becomes a sentence instead
        of an impossible prompt.

        The tail is aggregated by name, because six separate tree fellings are one fact about
        this player, not six.

        No tier drops provenance or staleness. A model that cannot see which memories are
        scanned and which are guesses cannot hedge correctly, and hedging correctly is the
        entire job.
        """
        s = Section("memories", BUDGETS["memories"])
        focus = trigger.get("pos")
        names = self.store.all_names()

        scored, traces = [], 0
        for row in self.store.all("structures"):
            provenance = Provenance[row["provenance"]]
            facts = _facts(row)
            # Caving leaves hundreds of small removal clusters that are not works. They stay
            # on record and stay queryable — they are facts — but listing them alongside
            # buildings made the god tell a player they had "cut fourteen shafts into this
            # land" when they had dug one pit and gone spelunking.
            if facts.get("significance") == "trace":
                traces += 1
                continue
            # Confidence decay already uses epoch milliseconds and survives server restarts.
            # Bare server ticks made an old row from a prior session look newer than a row
            # observed after restart.
            recency = max(0.05, self.store.confidence_now("structures", row))
            quality = 1.0 - (provenance.value / len(Provenance))
            near = 1.0
            if focus and row["min_x"] is not None:
                cx = (row["min_x"] + row["max_x"]) / 2
                cz = (row["min_z"] + row["max_z"]) / 2
                near = 1.0 / (1.0 + (abs(focus[0] - cx) + abs(focus[2] - cz)) / 128)
            # A 331-block excavation is more worth recalling than a 10-block one, but size
            # should modulate rank rather than decide it.
            size = 0.5 + 0.5 * min(1.0, (facts.get("blocks") or 0) / 100.0)
            scored.append((recency * quality * near * size, row))
        scored.sort(key=lambda p: p[0], reverse=True)

        for _, row in scored[:self.FULL_DETAIL]:
            s.lines.append(self._memory_full(row, names, now_tick, immersive))

        # Below full detail, memories sharing a name collapse into one line. Six separate
        # tree fellings are one fact about this player, not six, and rendering them
        # separately spends the budget proving the same thing over and over.
        rest = [row for _, row in scored[self.FULL_DETAIL:self.CONSIDER]]
        distant = [row for _, row in scored[self.CONSIDER:]]
        groups: dict[str, list] = {}
        for row in rest:
            key = _category(row, names) or _label(row, names) or row["id"]
            groups.setdefault(key, []).append(row)

        shown = 0
        overflow: list = []
        for label, members in groups.items():
            if shown >= self.BRIEF_DETAIL:
                overflow.extend(members)
                continue
            shown += 1
            if len(members) == 1:
                s.lines.append(self._memory_brief(members[0], names, now_tick, immersive))
            else:
                s.lines.append(self._memory_grouped(label, members, names, now_tick,
                                                    immersive))

        if overflow:
            counts = collections.Counter(_label(row, names) or "unnamed" for row in overflow)
            summary = ", ".join(f"{n}x {label}" for label, n in counts.most_common(6))
            s.lines.append(f"and {len(overflow)} more further off: {summary}")
        if distant:
            s.lines.append(f"({len(distant)} older or more distant structures are on record "
                           f"but not recalled here; use recall() to look one up)")
        if traces:
            s.lines.append(f"(also {traces} scattered excavations left from caving — "
                           f"passages, not works. Do not count them as things they built.)")
        # The ledger beneath the landmarks: pillars, bridges, scaffolds and torch lines
        # stand in the world too, and a god that cannot see them will call a player's
        # escape pillar nothing at all. One line, near the player, never itemised.
        try:
            from masses import nearby_summary
            nearby = nearby_summary(self.store, trigger.get("dim") or "overworld", focus)
        except Exception:  # noqa: BLE001 - a summary failing must not cost the prompt
            nearby = ""
        if nearby:
            s.lines.append(nearby)

        if not s.lines:
            s.lines.append("nothing recalled")
        s.fit()
        return s

    def _living(self, row) -> str:
        """What is alive inside a structure, if anything has looked recently.

        Direct observation, and the only thing that separates a pen full of chickens from
        an empty one — the event stream sees a breeding, never a flock.
        """
        held = self.store.get("relationships", f"holds:{row['id']}")
        if not held:
            return ""
        try:
            counts = json.loads(held["object"])
        except (TypeError, ValueError):
            return ""
        alive = {k: v for k, v in counts.items()
                 if not k.endswith((":item", ":experience_orb", ":arrow"))}
        if not alive:
            return ""
        listed = ", ".join(f"{v} {k.replace('minecraft:', '')}"
                           for k, v in sorted(alive.items(), key=lambda kv: -kv[1])[:4])
        wrong = self.store.contradicted("relationships", f"holds:{row['id']}")
        if wrong:
            return (f" — last counted {listed}, but {wrong}, so that number is stale "
                    f"[SCANNED, out of date]")
        return f" — holds {listed} [SCANNED]"

    def _memory_full(self, row, names, now_tick: int, immersive: bool = False) -> str:
        facts = _facts(row)
        materials = json.loads(row["materials"] or "{}")
        top = ", ".join(f"{v} {k.replace('minecraft:', '')}"
                        for k, v in list(materials.items())[:4])
        shape = ""
        if facts.get("dims"):
            shape = (f"{facts['dims']}, fill {facts.get('fill')}"
                     f"{', vertical' if facts.get('vertical') else ''}")
        # Size from the ledger, which is re-measured on every read; the row's own count
        # waits for the next boxing pass and can lag a build in progress.
        try:
            from masses import landmark_blocks
            now_blocks = landmark_blocks(self.store, row)
        except Exception:  # noqa: BLE001
            now_blocks = None
        if now_blocks is not None and now_blocks != (facts.get("blocks") or 0):
            shape = (shape + ", " if shape else "") + f"now {now_blocks} blocks"
        head = stamp(row["provenance"], row["confidence"], now_tick,
                     row["verified_at_tick"],
                     self.store.confidence_now("structures", row),
                     self.store.contradicted("structures", row["id"]),
                     row["verified_at_ms"])
        by = f" by {row['actor'][:8]}" if row["actor"] else ""
        where = (f" at ({row['min_x']},{row['min_y']},{row['min_z']})"
                 if not immersive else "")
        return (f"{head} {_named(row, names, self.store)}{by}{where}: {shape} — {top}"
                f"{self._living(row)}")

    def _memory_grouped(self, label: str, members, names, now_tick: int,
                        immersive: bool = False) -> str:
        """Several memories that share a name, as one line.

        The freshest one keeps its coordinates so the god can still point at something; the
        rest become a count. Confidence is reported as the range across the group, so a set
        containing one shaky guess does not read as uniformly certain.
        """
        freshest = max(members, key=lambda r: r["verified_at_ms"] or 0)
        # Name the group by what the freshest member is actually called, so the god has a
        # phrase to use rather than an internal bucket name.
        label = _label(freshest, names) or label
        confs = [names[r["id"]]["confidence"] for r in members if r["id"] in names]
        spread = ""
        if confs:
            lo, hi = min(confs), max(confs)
            spread = f" (name {lo:.2f}" + (f"-{hi:.2f}" if hi > lo else "") + ")"
        head = stamp(freshest["provenance"], freshest["confidence"], now_tick,
                     freshest["verified_at_tick"],
                     verified_ms=freshest["verified_at_ms"])
        where = (f", nearest at ({freshest['min_x']},{freshest['min_y']},"
                 f"{freshest['min_z']})" if not immersive else "")
        return f'{head} {len(members)}x "{label}"{spread}{where}'

    def _memory_brief(self, row, names, now_tick: int, immersive: bool = False) -> str:
        facts = _facts(row)
        head = stamp(row["provenance"], row["confidence"], now_tick,
                     row["verified_at_tick"], verified_ms=row["verified_at_ms"])
        by = f" by {row['actor'][:8]}" if row["actor"] else ""
        where = (f" at ({row['min_x']},{row['min_y']},{row['min_z']})"
                 if not immersive else "")
        return (f"{head} {_named(row, names, self.store)}{by} {facts.get('dims', '')}"
                f"{where}{self._living(row)}")

    def _quests(self, now_tick: int, actor: str) -> Section:
        """Active quests, and what has already been asked for.

        Each call is otherwise stateless, so the god could not see that it had asked for a
        lit stone shelter the last four times. Five proposals in a row came back as the same
        building because nothing told it what it had already said.
        """
        s = Section("quests", BUDGETS["quests"])
        rows = self.store.all("quests", "actor = ?", (actor,))
        active = [r for r in rows if r["state"] == "active"]
        done = [r for r in rows if r["state"] != "active"]

        for row in active:
            try:
                spec = json.loads(row["spec"])
            except (TypeError, ValueError):
                spec = {}
            s.lines.append(f"ACTIVE {row['id']}: {spec.get('intent') or '(no intent)'}")

        if done:
            # A distribution, not just a list. Shown a list of buildings the god reads it as
            # "not that" and picks one obvious opposite — three proposals in a row came back
            # as the same descent. A tally of what it has actually favoured is something it
            # can balance against rather than merely react to.
            build = collections.Counter()
            for row in done:
                try:
                    spec = json.loads(row["spec"])
                except (TypeError, ValueError):
                    continue
                kinds = {c.get("type") for c in spec.get("constraints", [])}
                shape = "gather" if kinds & {"collect", "craft", "smelt"} else (
                    "hunt" if kinds & {"kill", "breed"} else (
                        "journey" if kinds & {"reach_depth", "advancement"} else "build"))
                build[shape] += 1
            spread = ", ".join(f"{n} {k}" for k, n in build.most_common())
            s.lines.append(f"you have set them {len(done)} quests so far: {spread}")
            s.lines.append("balance that — do not simply pick whatever you asked least.")
            for row in sorted(done, key=lambda r: -(r["resolved_tick"] or 0))[:5]:
                try:
                    spec = json.loads(row["spec"])
                except (TypeError, ValueError):
                    spec = {}
                s.lines.append(f"  {row['state']}: {spec.get('intent') or row['id']}"[:110])
        if not s.lines:
            s.lines.append("nothing has been asked of them yet")
        s.fit()
        return s

    def _said(self, now_tick: int, actor: str) -> Section:
        """The last few things the god said, marked as its own speech and nothing more.

        Conversation history is never accumulated into the prompt, because the god's past
        speculation would become indistinguishable from evidence within a few turns. But
        answering a question needs to know what was just said, so it comes back through the
        store as ASSERTED — visible, and permanently barred from being treated as fact.
        """
        s = Section("said", BUDGETS["said"])
        for row in self.store.recent_utterances(actor):
            s.lines.append(f"[ASSERTED — your own words, never evidence, "
                           f"{_age(now_tick, row['tick'])}] {row['speech']}"[:220])
        if not s.lines:
            s.lines.append("you have not spoken to them yet")
        s.fit()
        return s

    def _tools(self) -> Section:
        s = Section("tools", BUDGETS["tools"], TOOLS.splitlines())
        s.fit()
        return s


def render(sections: list[Section]) -> str:
    return "\n\n".join(f"## {s.name}\n{s.text}" for s in sections)


def audit(sections: list[Section]) -> dict:
    """Per-section budget report. The P7 acceptance test reads this."""
    return {
        "total_tokens": sum(s.tokens for s in sections),
        "total_budget": TOTAL_BUDGET,
        "sections": {s.name: {"tokens": s.tokens, "budget": s.budget,
                              "over": s.tokens > s.budget, "truncated": s.truncated}
                     for s in sections},
        "any_over": any(s.tokens > s.budget for s in sections),
    }
