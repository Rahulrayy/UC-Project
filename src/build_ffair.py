# builds f_fair which is f_ft with fair options on every rank rule item and on poi2coor
# run from the repo root with python src/build_ffair.py

import json
import random
import re
from collections import Counter

from fair_options import FairSampler
from utils import BUILT, NUM_OPT, is_rank_rule, load_train

OUT = BUILT / "train"
CORNERS = [(0, 0), (-0.01, -0.01), (-0.01, 0.01), (0.01, -0.01), (0.01, 0.01)]
COORD = re.compile(r"(?m)^([A-H])\. ?(-?\d+(?:\.\d+)?,-?\d+(?:\.\d+)?)[ \t]*$")

d = load_train("Beijing")
with open(OUT / "ids.json") as f:
    ids = json.load(f)
with open(OUT / "f_ft.json", encoding="utf-8") as f:
    f_ft = json.load(f)


def family(x):
    # new distractors come from the answers of the same family
    s = x["source"]
    if not s.startswith("self-reasoning"):
        return s.replace(".csv", "")
    # look at the question right before the first option
    first = NUM_OPT.search(x["instruction"]).start()
    q = x["instruction"][max(0, first - 600):first]
    if "What is the distance between" in q:
        return "reason_rd"
    return "reason_walk" if "need to walk from" in q else "reason_other"


todo = []
for i in ids["f_ft"]:
    x = d[i]
    if x["source"] != "poi2coor.csv" and not is_rank_rule(x):
        continue
    fam = family(x)
    found = list((COORD if fam == "poi2coor" else NUM_OPT).finditer(x["instruction"]))
    opts = {m.group(1): m.group(2) for m in found}
    # cityreasoning has Answer X somewhere in the reasoning and template answers start with the letter
    if fam.startswith("reason"):
        ans = re.findall(r"Answer:([A-H])\b", x["output"])[0]
    else:
        ans = x["output"][0]
    assert "".join(opts) in ("ABCD", "ABCDE") and ans in opts, i
    todo.append((i, fam, found, opts, ans))
fams = Counter(t[1] for t in todo)
print("items to rewrite:", len(todo), dict(fams))

samplers = {}
for fam in fams:
    answers = [opts[ans] for _, f2, _, opts, ans in todo if f2 == fam]
    if fam == "road_length":
        samplers[fam] = FairSampler(answers, s=0.2, k=4, shift=100)   # 5 options and some lengths are 0
    elif fam != "poi2coor":
        samplers[fam] = FairSampler(answers, s=0.2, k=3)

rng = random.Random(0)
new = {}
hits = Counter()
for i, fam, found, opts, ans in todo:
    y = opts[ans]
    if fam == "poi2coor":
        # same 5 point pattern with the true point moved to a random spot like in build_topt
        lng, lat = map(float, y.split(","))
        dx, dy = rng.choice(CORNERS)
        vals = [f"{round(lng - dx + ox, 4)},{round(lat - dy + oy, 4)}" for ox, oy in CORNERS]
        assert y in vals, i
    else:
        vals = [y] + samplers[fam].draw(y, rng)
    rng.shuffle(vals)
    new_opts = dict(zip(opts, vals))
    new_ans = list(opts)[vals.index(y)]

    text = d[i]["instruction"]
    for m, v in zip(reversed(found), reversed(vals)):   # back to front so the positions stay right
        text = text[:m.start(2)] + v + text[m.end(2):]
    out = d[i]["output"]
    if fam.startswith("reason"):
        out = out.replace(f"Answer:{ans}", f"Answer:{new_ans}")
    else:
        out = new_ans + out[1:]
    new[i] = (text, out)

    # how often the old rank rule still picks the answer which should be about 1 in the number of options
    if fam != "poi2coor":
        r = sorted(opts, key=lambda k: float(opts[k])).index(ans)
        hits[fam] += sorted(new_opts, key=lambda k: float(new_opts[k]))[r] == new_ans

for fam in sorted(hits):
    print(f"  {fam:24} old rule still right {hits[fam]}/{fams[fam]} = {hits[fam] / fams[fam]:.2f}")

# same items in the same order as f_ft and only the rewritten ones change
rows = []
for r in f_ft:
    if r["id"] in new:
        text, out = new[r["id"]]
        r = {"id": r["id"], "instruction": text, "input": "", "output": out, "history": r["history"]}
    rows.append(r)
with open(OUT / "f_fair.json", "w", encoding="utf-8") as f:
    json.dump(rows, f, ensure_ascii=False)
print("wrote f_fair", len(rows))

# add f_fair to the dataset info file for llamafactory
with open(OUT / "dataset_info.json") as f:
    info = json.load(f)
info["f_fair"] = {"file_name": "f_fair.json", "columns": info["f_ft"]["columns"]}
with open(OUT / "dataset_info.json", "w") as f:
    json.dump(info, f, indent=2)
print("done")
