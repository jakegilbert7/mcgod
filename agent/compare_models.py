"""Run one patch through several models and compare what each says is there.

Segmentation is the call the pipeline makes most often, so what it costs matters. This runs
the identical patch — same voxels, same render, same slices, same prior state — through each
model and prints what each drew, so the question "is a cheaper one good enough" is answered
by looking rather than by guessing.

Each model gets its own copy of the store, so none of them sees another's answers and the
real store is never touched.

    python3 compare_models.py --centre 161 68 323 --span 22
"""
from __future__ import annotations

import argparse
import asyncio
import json
import shutil
import tempfile
import time
from pathlib import Path

DB = Path(__file__).parent / "mcgod.db"


async def one(model: str, centre, span: int, dim: str) -> dict:
    from god import God
    from store import Store

    scratch = Path(tempfile.mkdtemp()) / "copy.db"
    shutil.copy(DB, scratch)
    store = Store(str(scratch))
    god = God(store, use_model=True)
    god.segment_model = model
    god.tick = 999999

    def snapshot() -> dict:
        return {row["id"]: (row["min_x"], row["min_y"], row["min_z"],
                            row["max_x"], row["max_y"], row["max_z"],
                            row["value"]) for row in store.all("structures")}

    before = snapshot()
    started = time.time()
    try:
        await god.resegment(dim, list(centre), span=span)
    except Exception as e:
        return {"model": model, "error": f"{type(e).__name__}: {e}"}
    took = round(time.time() - started, 1)

    # Only what THIS pass changed. A copied store carries rows from every earlier run, and
    # counting those made all three models look alike no matter what they actually did.
    after = snapshot()
    found, retired = [], [i for i in before if i not in after]
    for sid, row in after.items():
        if row == before.get(sid) or row[0] is None:
            continue
        value = json.loads(row[6] or "{}")
        named = store.structure_name(sid)
        found.append({"blocks": value.get("blocks") or 0, "dims": value.get("dims"),
                      "at": [row[0], row[1], row[2]],
                      "new": sid not in before,
                      "name": named["object"] if named else "-"})
    store.close()
    found.sort(key=lambda f: -f["blocks"])
    return {"model": model, "seconds": took, "structures": found, "retired": retired}


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--centre", nargs=3, type=int, required=True)
    parser.add_argument("--span", type=int, default=22)
    parser.add_argument("--dim", default="overworld")
    parser.add_argument("--models", nargs="+",
                        default=["claude-opus-5", "claude-sonnet-5", "claude-haiku-4-5"])
    args = parser.parse_args()

    for model in args.models:
        print(f"\n{'=' * 68}\n{model}\n{'=' * 68}")
        result = await one(model, args.centre, args.span, args.dim)
        if "error" in result:
            print(f"  FAILED: {result['error']}")
            continue
        print(f"  {result['seconds']}s, wrote {len(result['structures'])}, "
              f"retired {len(result['retired'])}")
        for found in result["structures"]:
            mark = "new" if found["new"] else "upd"
            print(f"    {mark} {found['blocks']:5d}b {found['dims']:>10}  "
                  f"({found['at'][0]},{found['at'][1]},{found['at'][2]})  {found['name']}")
        for sid in result["retired"]:
            print(f"    ret                              {sid}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
