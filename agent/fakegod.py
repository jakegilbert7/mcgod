#!/usr/bin/env python3
"""P7: the fake god.

    python3 fakegod.py sessions/corpus/*.jsonl              # budget audit over the corpus
    python3 fakegod.py sessions/corpus/01-house.jsonl --show # print an assembled prompt

Assembles a real context for every episode in the corpus, then hands it to a stub model that
prints the prompt and returns canned JSON. No model is called.

Two things this is for. The stated acceptance test is that no assembled context exceeds its
budget anywhere in the corpus. The more useful one is sufficiency: read a prompt and ask
whether anything could tell what happened from it alone. That question is answerable now,
against four sessions with known ground truth, and its answer decides which detector is
worth building next — rather than guessing at twenty.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from consumer import BOLD, DIM, GREEN, RED, RESET
from context import Assembler, audit, render
from detectors import analyse
from store import Store


def latest_state(events: list[dict], tick: int) -> dict | None:
    """The most recent player_state sample at or before a tick."""
    best = None
    for e in events:
        if e["type"] == "player_state" and e["tick"] <= tick:
            if best is None or e["tick"] > best["tick"]:
                best = e
    return best


def stub_model(prompt: str) -> dict:
    """Stands in for the model. Returns the structured shape L6 will validate."""
    return {
        "speech": "(stub god says nothing)",
        "quest_spec": None,
        "actions": [],
        "beliefs_proposed": [],
    }


def validate(reply: dict) -> list[str]:
    """The gate every model reply passes before anything is persisted.

    Model output becomes a proposal, gets checked deterministically, and only then is
    written. Anything the god merely says is ASSERTED and can never be promoted, so a
    proposal claiming stronger provenance for its own speech is rejected here.
    """
    problems = []
    if not isinstance(reply.get("speech"), str):
        problems.append("speech must be a string")
    for key in ("actions", "beliefs_proposed"):
        if not isinstance(reply.get(key), list):
            problems.append(f"{key} must be a list")
    for belief in reply.get("beliefs_proposed", []) or []:
        if belief.get("provenance") not in (None, "INFERRED", "ASSERTED"):
            problems.append(
                f"model proposed provenance {belief.get('provenance')!r}; "
                "a model may only propose INFERRED or ASSERTED")
    return problems


def main(argv: list[str]) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("sessions", nargs="+", type=Path)
    p.add_argument("--show", action="store_true", help="print each assembled prompt")
    p.add_argument("--episode", type=int, help="print only this episode index (1-based)")
    args = p.parse_args(argv)

    episodes, findings, events = analyse(args.sessions)
    store = Store()
    assembler = Assembler(store)

    print(f"{BOLD}fake god{RESET} over {len(args.sessions)} sessions, "
          f"{len(episodes)} episodes, {len(findings)} findings\n")

    worst = {}
    failures = 0
    for index, ep in enumerate(episodes, 1):
        mine = [f for f in findings if f.episode_id == ep.id]
        trigger = {
            "reason": (f"{mine[0].detector} fired" if mine else "episode ended"),
            "pos": list(ep.bbox[0]) if ep.bbox else None,
            "facts": dict(mine[0].facts) if mine else {
                "placed": ep.gross_placed, "broken": ep.gross_broken},
        }
        # Only episodes up to this one exist yet; the god never sees its own future.
        history = [e for e in episodes if e.tick_end <= ep.tick_end][-6:]
        sections = assembler.build(
            now_tick=ep.tick_end, actor=ep.actor, trigger=trigger,
            player_state=latest_state(events, ep.tick_end),
            episodes=history, findings=mine)

        report = audit(sections)
        for name, info in report["sections"].items():
            worst[name] = max(worst.get(name, 0), info["tokens"])
        if report["any_over"]:
            failures += 1

        if args.show and (args.episode is None or args.episode == index):
            prompt = render(sections)
            print(f"{DIM}{'─' * 78}{RESET}")
            print(f"{BOLD}episode {index}  {ep.id}{RESET}  "
                  f"{report['total_tokens']} tokens / {report['total_budget']} budget")
            print(f"{DIM}{'─' * 78}{RESET}")
            print(prompt)
            reply = stub_model(prompt)
            problems = validate(reply)
            print(f"\n{DIM}stub reply:{RESET} {json.dumps(reply)}")
            print(f"{DIM}validation:{RESET} "
                  + (f"{GREEN}clean{RESET}" if not problems else f"{RED}{problems}{RESET}"))
            print()

    print(f"{BOLD}budget audit across every episode{RESET}")
    print(f"  {'section':14} {'peak':>6} {'budget':>7}   status")
    for name, budget in [(s.name, s.budget) for s in sections]:
        peak = worst[name]
        ok = peak <= budget
        status = f"{GREEN}ok{RESET}" if ok else f"{RED}OVER{RESET}"
        print(f"  {name:14} {peak:6d} {budget:7d}   {status}")
    total_peak = sum(worst.values())
    total_budget = sum(s.budget for s in sections)
    print(f"  {'TOTAL':14} {total_peak:6d} {total_budget:7d}")
    print()
    if failures:
        print(f"{RED}{failures} episodes assembled a context over budget{RESET}")
        return 1
    print(f"{GREEN}no assembled context exceeded its budget across the corpus{RESET}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
