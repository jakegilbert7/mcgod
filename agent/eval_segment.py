"""Score segmentation against known ground truth, across models and settings.

Two one-off comparisons were run before this existed and both produced a wrong conclusion —
the first because a model was 400ing and never ran at all, the second because a single patch
with one run each cannot see past run-to-run variance. Segmentation is the call the pipeline
makes most often and the one whose output becomes permanent record, so "which model, with
what context" deserves a measurement rather than an impression.

What it measures, per structure the truth file names:

  found      the pass produced a box overlapping it enough to be the same thing
  iou        how closely that box matches — the boundary quality, which is the whole job
  named      the name contains a word a person would accept for it
  spurious   boxes that match nothing in the truth set, which is how over-segmentation shows

Each run starts cold: structures inside the patch are removed from a copy of the store first,
so what is scored is segmentation itself and not the continue-what-is-already-there path.
That path is real and worth its own eval; mixing them would let a good prior hide a bad read.

    python3 eval_segment.py --dry-run
    python3 eval_segment.py --models claude-sonnet-5 --runs 3
    python3 eval_segment.py --runs 2 --plan-view both
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import shutil
import statistics
import tempfile
import time
from pathlib import Path

from structures import is_current

HERE = Path(__file__).parent
TRUTH = HERE / "segment_truth.json"
DB = HERE / "mcgod.db"

#: Overlap at which a proposal is judged to be "the same thing" as a truth structure. Low on
#: purpose: the question at this bar is whether the model saw the building at all, and box
#: quality is then reported separately as the IoU rather than folded into a pass/fail.
SAME_THING = 0.15


def iou(a_min, a_max, b_min, b_max) -> float:
    """Volume overlap of two boxes over their union."""
    lo = [max(a_min[i], b_min[i]) for i in range(3)]
    hi = [min(a_max[i], b_max[i]) for i in range(3)]
    if any(hi[i] < lo[i] for i in range(3)):
        return 0.0
    inter = 1
    for i in range(3):
        inter *= hi[i] - lo[i] + 1
    volume = lambda p, q: (q[0] - p[0] + 1) * (q[1] - p[1] + 1) * (q[2] - p[2] + 1)
    union = volume(a_min, a_max) + volume(b_min, b_max) - inter
    return inter / union if union else 0.0


def named_correctly(name: str, terms: list) -> bool:
    lowered = (name or "").lower()
    return any(term in lowered for term in terms)


async def one_run(model: str, target: dict, span: int, plan_view: bool,
                  every: list, cold: bool = True) -> dict:
    """Segment one patch cold and score what came out against that patch's truth."""
    from god import God
    from store import Store

    scratch = Path(tempfile.mkdtemp()) / "copy.db"
    shutil.copy(DB, scratch)
    store = Store(str(scratch))
    centre = target["patch"]

    # Cold, the pass has to see the building rather than inherit it. Warm is what production
    # actually does after the first look, and the difference between them turns out to be
    # enormous: cold, a model merged a whole settlement into one "stone castle with towers",
    # 40x9x31. Both are worth knowing — cold is what a downloaded world looks like.
    for row in (list(store.all("structures")) if cold else []):
        if row["min_x"] is None:
            continue
        if all(abs(row[k] - centre[i]) <= span + 20
               for i, k in enumerate(("min_x", "min_y", "min_z"))):
            store.db.execute("DELETE FROM structures WHERE id = ?", (row["id"],))
            store.db.execute("DELETE FROM relationships WHERE subject = ?", (row["id"],))
    store.db.commit()

    god = God(store, use_model=True)
    god.segment_model = model
    god.tick = 999999
    os.environ["MCGOD_PLAN_VIEW"] = "1" if plan_view else "0"

    started = time.time()
    try:
        await god.resegment("overworld", list(centre), span=span)
    except Exception as e:
        store.close()
        return {"error": f"{type(e).__name__}: {e}"}
    took = time.time() - started

    # Only what stands in the patch that was just read. Scoring against the whole store
    # counted the cottage two hundred blocks away as a spurious box on every single run,
    # which buried the real result under noise the pass had nothing to do with.
    produced = []
    for row in store.all("structures"):
        if row["min_x"] is None or not is_current(row):
            continue
        if not (row["min_x"] <= centre[0] + span and row["max_x"] >= centre[0] - span
                and row["min_z"] <= centre[2] + span and row["max_z"] >= centre[2] - span):
            continue
        # Only rows this pass actually wrote. Warm mode otherwise scores itself: the truth
        # file was seeded from the store, so a row the model never looked at sits there
        # already matching perfectly and reports IoU 1.00 for doing nothing.
        if row["verified_at_tick"] != god.tick:
            continue
        named = store.structure_name(row["id"])
        produced.append({
            "min": [row["min_x"], row["min_y"], row["min_z"]],
            "max": [row["max_x"], row["max_y"], row["max_z"]],
            "name": named["object"] if named else ""})
    store.close()

    best, best_box = 0.0, None
    for box in produced:
        score = iou(box["min"], box["max"], target["min"], target["max"])
        if score > best:
            best, best_box = score, box
    matched = best >= SAME_THING
    # Spurious means matching NOTHING known, not merely "not the one being scored". A patch
    # holds several real structures, and counting a correctly-found neighbour against the
    # model made every pass look like wild over-segmentation.
    spurious = sum(1 for box in produced
                   if all(iou(box["min"], box["max"], other["min"], other["max"])
                          < SAME_THING for other in every))
    return {"found": matched, "iou": round(best, 3), "seconds": round(took, 1),
            "named": bool(matched and named_correctly(best_box["name"], target["terms"])),
            "name": best_box["name"] if best_box else "",
            "spurious": spurious, "produced": len(produced)}


def report(rows: list) -> None:
    """One line per model per setting, plus the spread, because the spread is the point."""
    print(f"\n{'model':22} {'plan':5} {'found':>7} {'iou':>15} {'named':>7} "
          f"{'extra':>6} {'secs':>6}")
    print("-" * 74)
    for (model, plan_view), results in rows:
        good = [r for r in results if "error" not in r]
        if not good:
            print(f"{model:22} {'on' if plan_view else 'off':5}  all runs failed")
            continue
        ious = [r["iou"] for r in good]
        spread = (f"{statistics.mean(ious):.2f} ±{statistics.pstdev(ious):.2f}"
                  if len(ious) > 1 else f"{ious[0]:.2f}")
        print(f"{model:22} {'on' if plan_view else 'off':5} "
              f"{sum(r['found'] for r in good):3d}/{len(good):<3d} {spread:>15} "
              f"{sum(r['named'] for r in good):3d}/{len(good):<3d} "
              f"{statistics.mean(r['spurious'] for r in good):5.1f} "
              f"{statistics.mean(r['seconds'] for r in good):5.0f}")


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+",
                        default=["claude-opus-5", "claude-sonnet-5", "claude-haiku-4-5"])
    parser.add_argument("--runs", type=int, default=2, help="repeats per structure")
    parser.add_argument("--span", type=int, default=22)
    parser.add_argument("--only", nargs="+", help="limit to these truth ids")
    parser.add_argument("--plan-view", choices=["on", "off", "both"], default="off")
    parser.add_argument("--start", choices=["cold", "warm"], default="warm",
                        help="warm keeps what is already on record, which is what production "
                             "does; cold forgets it first, which is what a downloaded world "
                             "with no history looks like")
    parser.add_argument("--out", default="segment_eval.json")
    parser.add_argument("--dry-run", action="store_true",
                        help="print the plan and what it will cost in wall time")
    args = parser.parse_args()

    truth = json.loads(TRUTH.read_text())["structures"]
    if args.only:
        truth = [t for t in truth if t["id"] in args.only]
    settings = ([True, False] if args.plan_view == "both"
                else [args.plan_view == "on"])
    passes = len(args.models) * len(settings) * len(truth) * args.runs
    print(f"{len(args.models)} models x {len(settings)} setting(s) x {len(truth)} "
          f"structures x {args.runs} runs = {passes} passes")
    print(f"at ~90s a pass that is roughly {passes * 90 / 60:.0f} minutes of wall time "
          f"and real API spend.")
    if args.dry_run:
        for target in truth:
            print(f"  {target['id']:20} patch {target['patch']}")
        return 0

    rows, detail = [], []
    for model in args.models:
        for plan_view in settings:
            results = []
            for target in truth:
                for run in range(args.runs):
                    result = await one_run(model, target, args.span, plan_view, truth,
                                           cold=args.start == "cold")
                    result.update({"model": model, "plan_view": plan_view, "start": args.start,
                                   "structure": target["id"], "run": run})
                    results.append(result)
                    detail.append(result)
                    mark = ("ERR" if "error" in result else
                            "ok " if result["found"] else "MISS")
                    extra = ("" if "error" in result else
                             f"iou {result['iou']:.2f}  "
                             f"{'named' if result['named'] else 'misnamed'}"
                             f"  +{result['spurious']}  {result['name'][:34]}")
                    print(f"  {mark} {model:18} {target['id']:20} {extra}", flush=True)
            rows.append(((model, plan_view), results))

    report(rows)
    Path(args.out).write_text(json.dumps(detail, indent=1))
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
