# rq4 items where both places can be found in the cityqa address data plus 1 km grid ids
# run from the repo root with python src/build_rq4.py

import math
import random
import re
import statistics
from collections import defaultdict

from utils import BUILT, CITIES, addresses, answer_gap, get_options, load_task, load_train, origin, write_jsonl

TASKS = ["aoi2aoi", "poi2aoi", "poi2poi"]   # rd tasks end at a road not a point so the gis baseline cant do them
COORD = re.compile(r"\(\s*(-?\d{1,3}\.\d+)\s*,\s*(-?\d{1,3}\.\d+)\s*\)")
ASK = re.compile(r"address:\s*(.*?),\s*please tell me the geographic coordinates")


def haversine(a, b):
    lng1, lat1, lng2, lat2 = map(math.radians, [a[0], a[1], b[0], b[1]])
    h = math.sin((lat2 - lat1) / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin((lng2 - lng1) / 2) ** 2
    return 2 * 6371000 * math.asin(math.sqrt(h))


def address_book(city):
    # cityqa asks address to coordinates both ways round and only beijing has a source field so match the text
    seen = defaultdict(list)
    for x in load_train(city):
        t, o = x["instruction"], x["output"]
        m, c = ASK.search(t), COORD.search(o)
        if m and c:
            seen[m.group(1).strip()].append((float(c.group(1)), float(c.group(2))))
        elif COORD.search(t) and not c and "on the" in o and "side of" in o:
            c = COORD.search(t)
            seen[o.strip().rstrip(".")].append((float(c.group(1)), float(c.group(2))))
    # drop addresses with unknown road or with points more than 200m apart since we cant tell which place it is
    book = {}
    for a, pts in seen.items():
        mid = (statistics.mean(p[0] for p in pts), statistics.mean(p[1] for p in pts))
        if "unknown road" not in a and max(haversine(p, mid) for p in pts) <= 200:
            book[a] = mid
    return book


rng = random.Random(0)
out = []
for city in CITIES:
    book = address_book(city)
    lat0 = statistics.mean(c[1] for c in book.values())
    kx = 111320 * math.cos(math.radians(lat0))   # meters per degree of longitude around this city
    found = {}
    for shape in TASKS:
        for kind in ["dir", "dis"]:
            ok = []
            for r, row in enumerate(load_task(city, f"{shape}_{kind}_noroutine")):
                addrs = [a for _, _, a in addresses(row["question"])]
                o = origin(row["question"])
                t = addrs[0] if addrs[-1] == o else addrs[-1]   # the target is the other place
                if len(addrs) == 2 and o in book and t in book:
                    ok.append((r, row, book[o], book[t]))
            found[(shape, kind)] = ok
    # 1000 per city with up to 167 per task and the big poi2poi tasks filling up the rest
    take = {k: min(len(v), 167) for k, v in found.items()}
    rest = 1000 - sum(v for (s, _), v in take.items() if s != "poi2poi")
    take[("poi2poi", "dir")] = min(len(found[("poi2poi", "dir")]), rest - rest // 2)
    take[("poi2poi", "dis")] = min(len(found[("poi2poi", "dis")]), rest // 2)

    for (shape, kind), ok in found.items():
        for r, row, o, t in sorted(rng.sample(ok, take[(shape, kind)]), key=lambda x: x[0]):
            opts = get_options(row)
            cell = f"{city}|{math.floor(o[0] * kx / 1000)}|{math.floor(o[1] * 110574 / 1000)}"
            it = {"item_id": f"{city}/{shape}_{kind}_noroutine/{r}", "city": city, "task": f"{shape}_{kind}_noroutine",
                  "row": r, "group": "SR", "shape": shape, "kind": kind, "route": False, "core": False,
                  "question": row["question"], "options": opts, "answer": row["answer"],
                  "cluster_id": cell, "grid_id": cell, "origin_xy": list(o), "target_xy": list(t), "cond": "rq4"}
            if kind == "dis":
                it["d"] = float(opts[row["answer"]])
                it["gap"] = answer_gap(opts, row["answer"])
            out.append(it)
    print(f"{city}: {len(book)} addresses, {sum(take.values())} items")

write_jsonl(BUILT / "eval" / "rq4.jsonl", out)
print("wrote", len(out), "items")
