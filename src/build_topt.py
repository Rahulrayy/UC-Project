# T_opt for the eval sample with the same questions but fair options
# run from the repo root with python src/build_topt.py

import random
from collections import defaultdict

from fair_options import FairSampler
from utils import BUILT, CITIES, SHAPES, answer_gap, load_task, read_jsonl, write_jsonl

S = 0.2   # options at least 20 percent apart since 0.25 doesnt work for the small paris tasks
LETTERS = "ABCDEFGH"
# how the original poi2coor options are made with the answer in the middle and 4 corners 0.01 away
CORNERS = [(0, 0), (-0.01, -0.01), (-0.01, 0.01), (0.01, -0.01), (0.01, 0.01)]

items = read_jsonl(BUILT / "eval" / "eval_sample.jsonl")
rng = random.Random(0)

# one sampler per city and task fit on the whole task file
samplers = {}
for city in CITIES:
    for shape in SHAPES:
        rows = load_task(city, f"{shape}_dis_noroutine")
        fs = FairSampler([r[r["answer"]] for r in rows], s=S, k=3)
        samplers[(city, f"{shape}_dis")] = fs
    # road_length has 5 options and some answers are 0 so it gets 4 distractors and the shift
    rows = load_task(city, "road_length")
    samplers[(city, "road_length")] = FairSampler([r[r["answer"]] for r in rows], s=S, k=4, shift=100)
    print(city, "samplers done")


def rule_pick(opts, task):
    # the rank the original generator always puts the answer at
    order = sorted(opts, key=lambda k: (float(opts[k]), k))
    if task == "road_length":
        return order[2]
    if task.startswith("poi2poi_dis"):
        return order[2]
    return order[1]


def coord_str(lng, lat):
    # same style as in the csvs rounded to 4 decimals
    return f"{round(lng, 4)},{round(lat, 4)}"


def centre_pick(opts):
    # the option closest to all the others which is the old rule for poi2coor
    pts = {k: tuple(map(float, v.split(","))) for k, v in opts.items()}

    def spread(k):
        return sum(abs(pts[k][0] - p[0]) + abs(pts[k][1] - p[1]) for p in pts.values())
    return min(pts, key=spread)


done = {}   # maps city task and row to the options and answer so route and noroute match
out = []
for it in items:
    if it["kind"] == "dis":
        key = (it["city"], f"{it['shape']}_dis", it["row"])
    elif it["task"] in ("road_length", "poi2coor"):
        key = (it["city"], it["task"], it["row"])
    else:
        continue

    if key not in done:
        y = it["options"][it["answer"]]
        if it["task"] == "poi2coor":
            # same 5 point pattern but the true point goes to a random spot instead of always the middle
            # sampling real coords instead would make the options km apart and way easier
            lng, lat = map(float, y.split(","))
            dx, dy = rng.choice(CORNERS)
            c = (lng - dx, lat - dy)
            vals = [coord_str(c[0] + ox, c[1] + oy) for ox, oy in CORNERS]
            assert coord_str(lng, lat) == y and y in vals
        else:
            vals = [y] + samplers[key[:2]].draw(y, rng)
        rng.shuffle(vals)
        letters = LETTERS[:len(vals)]
        opts = dict(zip(letters, vals))
        ans = letters[vals.index(y)]
        done[key] = (opts, ans)

    opts, ans = done[key]
    new = dict(it)
    new["cond"] = "topt"
    new["orig_options"] = it["options"]
    new["orig_answer"] = it["answer"]
    new["options"] = opts
    new["answer"] = ans
    if it["kind"] == "dis":
        # the gap to the closest wrong option changes with the new options so keep both
        new["orig_gap"] = it["gap"]
        new["gap"] = answer_gap(opts, ans)
    # what the old rule would pick now which is needed for the rule following rate
    new["rule_letter"] = centre_pick(opts) if it["task"] == "poi2coor" else rule_pick(opts, it["task"])
    out.append(new)

# check how often the old rule is still right which should be about 1 in the number of options
hits = defaultdict(lambda: [0, 0])
for it in out:
    if it.get("route"):
        continue   # route rows are copies of the noroute ones
    t = it["task"].replace("_noroutine", "")
    hits[t][0] += it["rule_letter"] == it["answer"]
    hits[t][1] += 1
for t, (h, n) in sorted(hits.items()):
    print(f"  {t:14} rule still right {h}/{n} = {h / n:.2f}")
h = sum(v[0] for k, v in hits.items() if "_dis" in k)
n = sum(v[1] for k, v in hits.items() if "_dis" in k)
print(f"  all distance   {h}/{n} = {h / n:.3f}")

write_jsonl(BUILT / "eval" / "topt.jsonl", out)
print("wrote", len(out), "items")
