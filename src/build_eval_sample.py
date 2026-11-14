# builds the fixed eval sample with seed 0 that every model and condition uses
# run from the repo root with python src/build_eval_sample.py

import collections
import random
import statistics

from utils import BUILT, CI_TASKS, CITIES, SHAPES, US_TASKS, answer_gap, get_options, load_task, origin, write_jsonl

N = 100
CORE_CI = ["poi2coor", "road_length"]   # the two ci tasks with a shortcut go in the core sample

rng = random.Random(0)
items = []


def check_rows(rows, name):
    # first csv column is the pandas index and should just count up from 0
    for i, r in enumerate(rows):
        assert r[""] == str(i), name


for city in CITIES:
    for shape in SHAPES:
        for kind in ["dir", "dis"]:
            # route and noroute files have the same rows so one sampled row gives 2 items
            route = load_task(city, f"{shape}_{kind}_routine")
            noroute = load_task(city, f"{shape}_{kind}_noroutine")
            check_rows(route, f"{city} {shape}_{kind}_routine")
            check_rows(noroute, f"{city} {shape}_{kind}_noroutine")
            n = len(noroute)
            assert len(route) == n

            if kind == "dir":
                picked = [(r, None) for r in rng.sample(range(n), N)]
            else:
                # sort by answer distance and cut into 4 equal chunks with ties broken by row then take 25 from each
                order = sorted(range(n), key=lambda r: (float(noroute[r][noroute[r]["answer"]]), r))
                picked = []
                for q in range(4):
                    chunk = order[q * n // 4:(q + 1) * n // 4]
                    picked += [(r, (q + 1, len(chunk))) for r in rng.sample(chunk, N // 4)]

            for r, qbin in sorted(picked):
                opts = get_options(noroute[r])
                ans = noroute[r]["answer"]
                assert get_options(route[r]) == opts and route[r]["answer"] == ans
                # cluster is the place after from and if theres none the row is its own cluster
                o = origin(noroute[r]["question"])
                cluster = f"{city}|{o}" if o else f"{city}|{shape}_{kind}/{r}"
                for is_route, rows, suffix in [(True, route, "routine"), (False, noroute, "noroutine")]:
                    task = f"{shape}_{kind}_{suffix}"
                    it = {
                        "item_id": f"{city}/{task}/{r}",   # r is the 0 based row in the v2.3 csv
                        "city": city, "task": task, "row": r,
                        "group": "SR", "shape": shape, "kind": kind, "route": is_route,
                        "core": True,
                        "question": rows[r]["question"],
                        "options": opts, "answer": ans,
                        "cluster_id": cluster,
                        "has_origin": o is not None,
                        "task_n": n,
                    }
                    if kind == "dis":
                        it["d"] = float(opts[ans])
                        it["d_bin"] = qbin[0]
                        it["bin_n"] = qbin[1]   # chunk size for reweighting later
                        it["gap"] = answer_gap(opts, ans)
                    items.append(it)

    # ci and us tasks with up to 100 rows each
    # poi2coor and road_length are core and the rest is only for the reproduction check
    for grp, tasks in [("CI", CI_TASKS), ("US", US_TASKS)]:
        for task in tasks:
            rows = load_task(city, task)
            check_rows(rows, f"{city} {task}")
            n = len(rows)
            picked = sorted(rng.sample(range(n), min(N, n)))
            for r in picked:
                items.append({
                    "item_id": f"{city}/{task}/{r}",
                    "city": city, "task": task, "row": r,
                    "group": grp, "shape": None, "kind": None, "route": None,
                    "core": task in CORE_CI,
                    "question": rows[r]["question"],
                    "options": get_options(rows[r]), "answer": rows[r]["answer"],
                    # no shared places to cluster on here so every item is its own cluster
                    "cluster_id": f"{city}|{task}/{r}",
                    "has_origin": False,
                    "task_n": n,
                })

# quick summary
print("total items:", len(items))
print("core items:", sum(it["core"] for it in items))
assert len(set(it["item_id"] for it in items)) == len(items)
for city in CITIES:
    sr = [it for it in items if it["city"] == city and it["group"] == "SR" and not it["route"]]
    no_origin = sum(not it["has_origin"] for it in sr)
    sizes = list(collections.Counter(it["cluster_id"] for it in sr).values())
    shared = sum(s for s in sizes if s > 1) / len(sr)
    print(f"{city:8} sr rows={len(sr)} no origin={no_origin} clusters={len(sizes)} "
          f"max={max(sizes)} mean={statistics.mean(sizes):.2f} in clusters >1: {shared:.0%}")

dis = [it for it in items if it["kind"] == "dis" and not it["route"]]
capped = sum(it["gap"] is not None and it["gap"] >= 0.4999 for it in dis)
print(f"distance rows: {len(dis)}, gap at the 0.5 cap: {capped} ({capped / len(dis):.0%}), "
      f"answer <= 0: {sum(it['gap'] is None for it in dis)}")

write_jsonl(BUILT / "eval" / "eval_sample.jsonl", items)
print("wrote", BUILT / "eval" / "eval_sample.jsonl")
