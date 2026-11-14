# builds the training sets from the beijing file
# run from the repo root with python src/build_train_sets.py

import collections
import json
import random

from utils import BUILT, group, is_rank_rule, item_tokens, load_train

OUT = BUILT / "train"
N_WALK = 10000
N_GEN = 2000
MAX_TOK = 2048

d = load_train("Beijing")
print("beijing file:", len(d), "items")

# history should be question answer pairs since llamafactory wants it like that
for x in d:
    for h in x["history"]:
        assert isinstance(h, list) and len(h) == 2

groups = collections.defaultdict(list)
for i, x in enumerate(d):
    groups[group(x["source"])].append(i)
for g, idx in groups.items():
    print(f"  {g:14} {len(idx)}")

rng = random.Random(0)

# citywalk is the biggest city source so only keep 10k of the 30k
walk = rng.sample(groups["citywalk"], N_WALK)

# general slice only takes short ones then samples in proportion to each source
gen_ok = [i for i in groups["general"] if item_tokens(d[i]) <= MAX_TOK]
by_src = collections.defaultdict(list)
for i in gen_ok:
    by_src[d[i]["source"]].append(i)
gen = []
for src, idx in by_src.items():
    gen += rng.sample(idx, max(1, round(N_GEN * len(idx) / len(gen_ok))))

pool = groups["city_other"] + walk + groups["cityreasoning"] + groups["template"] + groups["spatial"] + gen

# f_ft is the pool minus R and R is as big as the template set so f_ft and f_notask have the same size
# R only comes from city data and general so cityreasoning stays identical in both models
r_src = [i for i in pool if group(d[i]["source"]) in ("city_other", "citywalk", "general")]
n_task = len(groups["template"])
R = set(rng.sample(r_src, n_task))

f_ft = [i for i in pool if i not in R]
f_notask = [i for i in pool if group(d[i]["source"]) != "template"]   # pool minus the sf template items
full = list(range(len(d)))   # whole file for the calibration run


def tok(idx):
    return sum(item_tokens(d[i]) for i in idx) / 1e6


def count(idx, g):
    return sum(group(d[i]["source"]) == g for i in idx)


print()
print(f"general items <= {MAX_TOK} tok: {len(gen_ok)}/{len(groups['general'])}, slice {len(gen)} items {tok(gen):.2f}M tok")
print(f"pool:     {len(pool)} items {tok(pool):.2f}M tok")
print(f"f_ft:     {len(f_ft)} items {tok(f_ft):.2f}M tok")
print(f"f_notask: {len(f_notask)} items {tok(f_notask):.2f}M tok")
print(f"full:     {len(full)} items {tok(full):.2f}M tok")
print("cityreasoning in f_ft / f_notask:", count(f_ft, "cityreasoning"), count(f_notask, "cityreasoning"))
for name, idx in [("pool", pool), ("f_ft", f_ft), ("f_notask", f_notask), ("full", full)]:
    rr = sum(is_rank_rule(d[i]) for i in idx)
    print(f"rank rule items in {name}: {rr} ({rr / len(idx):.1%})")

# sanity checks against the numbers in the compute plan
assert len(pool) == 78491
assert len(f_ft) == len(f_notask) == 71225
assert count(f_ft, "cityreasoning") == count(f_notask, "cityreasoning") == 7992

# save indices into the beijing file since f_fair has to reuse exactly the f_ft items later
OUT.mkdir(parents=True, exist_ok=True)
with open(OUT / "ids.json", "w") as f:
    json.dump({"seed": 0, "pool": pool, "R": sorted(R), "f_ft": f_ft, "f_notask": f_notask}, f)


def save(name, idx):
    # shuffle the order once with a fixed seed so items from one source arent all next to each other
    idx = list(idx)
    random.Random(0).shuffle(idx)
    rows = []
    for i in idx:
        x = d[i]
        rows.append({"id": i, "instruction": x["instruction"], "input": "",
                     "output": x["output"], "history": x["history"]})
    with open(OUT / f"{name}.json", "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False)
    print("wrote", name, len(rows))


save("f_ft", f_ft)
save("f_notask", f_notask)
save("full", full)

# dataset info file for llamafactory and f_fair gets added to it later
cols = {"prompt": "instruction", "query": "input", "response": "output", "history": "history"}
info = {name: {"file_name": f"{name}.json", "columns": cols} for name in ["f_ft", "f_notask", "full"]}
info_path = OUT / "dataset_info.json"
if info_path.exists():
    with open(info_path) as f:
        old = json.load(f)
    old.update(info)
    info = old
with open(info_path, "w") as f:
    json.dump(info, f, indent=2)
print("done")
