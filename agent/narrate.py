#!/usr/bin/env python3
"""Ask the god to describe a session, and see how much of it actually survived.

    python3 narrate.py sessions/corpus/05-baseline.jsonl            # what the god really sees
    python3 narrate.py sessions/corpus/05-baseline.jsonl --review   # everything in the store

Two modes, because they answer different questions.

The default assembles exactly what the live system would hand the model at the end of the
session: one trigger's worth of context, budgeted, with a rolling digest of the last few
episodes. That is the honest measure of the running system.

`--review` ignores the per-trigger budget and lays out every episode and finding from the
whole session. That measures the data, not the plumbing. The gap between the two is the
cost of the rolling window, and it is worth knowing which of the two is losing information.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from config import DIALOGUE_MODEL, have_key, thinking_for
from consumer import BOLD, DIM, RED, RESET
from context import Assembler, audit, estimate_tokens, render
from detectors import analyse
from router import Router
from store import Store
from model_api import model_client

ASK = ("Describe what this player did, in order, as the god who watched it. Be specific "
       "about what you actually know. Where the record is thin or a name is marked as a "
       "guess, say so rather than inventing detail. Do not speculate about motive.")


def review_context(episodes, findings, events, store) -> str:
    """Every episode and finding from the session, budget ignored."""
    lines = ["## everything on record for this session", ""]
    for i, ep in enumerate(episodes, 1):
        net = ", ".join(f"{v:+d} {k.replace('minecraft:', '')}"
                        for k, v in sorted(ep.net.items(), key=lambda kv: -abs(kv[1]))[:4])
        bbox = ep.bbox
        where = (f"{bbox[0]}..{bbox[1]}" if bbox else "no block work")
        lines.append(f"episode {i}: {ep.duration_s/60:.1f}min, placed {ep.gross_placed} "
                     f"broke {ep.gross_broken} ({net or 'nothing net'}), {where}, "
                     f"moved {ep.path:.0f} blocks")
        for f in findings:
            if f.episode_id != ep.id:
                continue
            if f.detector in ("context_profile", "risk_profile", "rate_anomaly"):
                lines.append(f"    {f.detector}: {f.facts}")
            elif f.detector == "structure_census":
                lines.append(f"    {f.kind}: {f.facts.get('dims')} fill "
                             f"{f.facts.get('fill')} — {f.facts.get('materials')}")
        firsts = [f for f in findings if f.episode_id == ep.id and f.detector == "firsts"]
        if firsts:
            lines.append("    firsts: " + ", ".join(f.facts["value"] for f in firsts[:14]))
    return "\n".join(lines)


def main(argv: list[str]) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("session", type=Path)
    p.add_argument("--review", action="store_true",
                   help="ignore the per-trigger budget and show the whole session")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args(argv)

    episodes, findings, events = analyse([args.session])
    store = Store()
    last = episodes[-1]

    if args.review:
        prompt = review_context(episodes, findings, events, store)
    else:
        triggers = Router(events).route(findings)
        loud = [t for t in triggers if t.lane in ("interrupt", "queued")]
        trig = loud[-1] if loud else None
        state = next((e for e in reversed(events) if e["type"] == "player_state"), None)
        sections = Assembler(store).build(
            now_tick=last.tick_end, actor=last.actor,
            trigger={"reason": trig.reason if trig else "session ended",
                     "pos": list(last.bbox[0]) if last.bbox else None,
                     "facts": dict(trig.finding.facts) if trig else {}},
            player_state=state, episodes=episodes, findings=findings)
        report = audit(sections)
        # Look the section up by name. Indexing positionally broke silently the moment a
        # new section was inserted ahead of it, and reported the wrong count for a while.
        digest = next(x for x in sections if x.name == "digest")
        print(f"{DIM}per-trigger context: {report['total_tokens']} tokens, "
              f"digest holds {len(digest.lines)} of {len(episodes)} episodes"
              f"{RESET}", file=sys.stderr)
        prompt = render(sections)

    print(f"{DIM}prompt ~{estimate_tokens(prompt)} tokens{RESET}", file=sys.stderr)
    if args.dry_run:
        print(prompt)
        return 0
    if not have_key():
        print(f"{RED}no API key{RESET}", file=sys.stderr)
        return 1

    client = model_client()
    with client.messages.stream(
        model=DIALOGUE_MODEL, max_tokens=4000,
        thinking=thinking_for(DIALOGUE_MODEL),
        system=("You are a god who has watched a Minecraft world through the summaries "
                "below. You know only what they contain."),
        messages=[{"role": "user", "content": prompt + "\n\n" + ASK}],
    ) as stream:
        message = stream.get_final_message()
    for block in message.content:
        if block.type == "text":
            print(block.text)
    print(f"\n{DIM}in {message.usage.input_tokens} / out "
          f"{message.usage.output_tokens} tokens{RESET}", file=sys.stderr)
    store.close()
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
