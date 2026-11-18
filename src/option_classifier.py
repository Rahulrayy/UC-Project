# option only classifier that checks whether the fair options in T_opt still give the answer away
# run from the repo root with python src/option_classifier.py

import json
import math
import re

import numpy as np
from scipy.stats import binomtest
from sklearn.ensemble import HistGradientBoostingClassifier

from utils import BUILT, NUM_OPT, is_rank_rule, load_train, read_jsonl

# pass rule set before running
# on the original options it has to reach 0.9 or more so we know it can find the trick
# on T_opt it must not beat chance with one sided binomial p below 0.05 for each kind
COORD = re.compile(r"(?m)^([A-H])\. ?(-?\d+(?:\.\d+)?,-?\d+(?:\.\d+)?)[ \t]*$")


def near(others, t):
    # is some other option about t
    return np.any(np.abs(others - t) <= 0.01 * max(1, abs(t)))


def features(vals, coord):
    # one row per option made only from the option strings
    if coord:
        pts = np.array([[float(a) for a in s.split(",")] for s in vals])
        spread = [np.abs(pts - p).sum() for p in pts]
        return np.array([[sp, sp == min(spread)] for sp in spread], dtype=float)
    v = np.array([float(s) for s in vals])
    rank = np.argsort(np.argsort(v, kind="stable"), kind="stable")
    rows = []
    for i, s in enumerate(vals):
        others = np.delete(v, i)
        whole = s.split(".")[0]
        row = [rank[i], math.log1p(abs(v[i])) - np.mean(np.log1p(np.abs(v))), len(whole) - len(whole.rstrip("0")),
               s.endswith(".0")]
        # the old options are half and double and plus 500 and so on so check for those
        row += [near(others, v[i] * m) for m in (0.1, 0.3, 0.5, 2, 3, 10)]
        row += [near(others, v[i] + o) for o in (-1000, 500)]
        rows.append(row)
    return np.array(rows, dtype=float)


def from_train(rows, sources):
    data = {"dis": [], "road_length": [], "poi2coor": []}
    for x, src in zip(rows, sources):
        if "poi2coor" in src:
            kind, pat = "poi2coor", COORD
        elif "road_length" in src:
            kind, pat = "road_length", NUM_OPT
        else:
            kind, pat = "dis", NUM_OPT
        opts = {}
        for m in pat.finditer(x["instruction"]):
            opts[m.group(1)] = m.group(2)
        # cityreasoning has Answer X in the reasoning and the template answers start with the letter
        found = re.findall(r"Answer:([A-H])\b", x["output"])
        ans = found[0] if found else x["output"][0]
        data[kind].append((opts, ans))
    return data


def from_eval(path):
    data = {"dis": [], "road_length": [], "poi2coor": []}
    for it in read_jsonl(path):
        if it.get("route"):
            continue   # route and no route copies share the options so count them once
        if it["task"] in data:
            data[it["task"]].append((it["options"], it["answer"]))
        elif it["kind"] == "dis":
            data["dis"].append((it["options"], it["answer"]))
    return data


def fit(data, kind):
    X, y = [], []
    for o, a in data:
        X.append(features([o[k] for k in sorted(o)], kind == "poi2coor"))
        y += [k == a for k in sorted(o)]
    return HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, random_state=0).fit(np.vstack(X), y)


def hits(model, data, kind):
    # the option the classifier likes most counts as its answer
    n = 0
    for o, a in data:
        letters = sorted(o)
        p = model.predict_proba(features([o[k] for k in letters], kind == "poi2coor"))[:, 1]
        n += letters[np.argmax(p)] == a
    return n


d = load_train("Beijing")
with open(BUILT / "train" / "ids.json") as f:
    ids = json.load(f)
with open(BUILT / "train" / "f_fair.json", encoding="utf-8") as f:
    fair = {r["id"]: r for r in json.load(f)}
picked = [i for i in ids["f_ft"] if d[i]["source"] == "poi2coor.csv" or is_rank_rule(d[i])]
srcs = [d[i]["source"] for i in picked]
orig_train = from_train([d[i] for i in picked], srcs)
fair_train = from_train([fair[i] for i in picked], srcs)
orig_eval = from_eval(BUILT / "eval" / "eval_sample.jsonl")
fair_eval = from_eval(BUILT / "eval" / "topt.jsonl")

ok = True
for kind, chance in [("dis", 0.25), ("road_length", 0.2), ("poi2coor", 0.2)]:
    n = len(fair_eval[kind])
    control = hits(fit(orig_train[kind], kind), orig_eval[kind], kind) / len(orig_eval[kind])
    h = hits(fit(fair_train[kind], kind), fair_eval[kind], kind)
    p = binomtest(h, n, chance, alternative="greater").pvalue   # chance of at least this many right by guessing
    ok &= control >= 0.9 and p >= 0.05
    print(f"{kind:12} original {control:.3f}  T_opt {h / n:.3f} p={p:.2f}  chance {chance}")
print("T_opt passes" if ok else "T_opt FAILS")
