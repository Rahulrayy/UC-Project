# accuracy like the paper plus cluster bootstrap and sign flip tests for the results files
# run from the repo root with python src/stats.py to test it on fake results

import numpy as np
from statsmodels.stats.multitest import multipletests

from utils import BUILT, read_jsonl


def setup(items):
    # arrays the stats need in a fixed item order
    tasks = sorted(set(it["task"] for it in items))
    clusters = sorted(set(it["cluster_id"] for it in items))
    city = np.array([it["city"] for it in items])
    s = {
        "ids": [it["item_id"] for it in items],
        "task": np.array([tasks.index(it["task"]) for it in items]),
        # distance rows were drawn 25 per quartile so each one stands for bin_n rows of the csv
        "w": np.array([it["bin_n"] / 25 if it.get("bin_n") else 1.0 for it in items]),
        "cluster": np.array([clusters.index(it["cluster_id"]) for it in items]),
        "route": np.array([bool(it.get("route")) for it in items]),
        "sr": np.array([it["group"] == "SR" for it in items]),
    }
    s["members"] = [np.flatnonzero(s["cluster"] == c) for c in range(len(clusters))]
    s["city_clusters"] = [np.unique(s["cluster"][city == c]) for c in sorted(set(city))]
    return s


def task_mean(s, c, keep, idx=None):
    # mean over tasks of the weighted accuracy inside each task like in the paper
    idx = np.arange(len(c)) if idx is None else idx
    idx = idx[keep[idx]]
    num = np.bincount(s["task"][idx], weights=s["w"][idx] * c[idx], minlength=s["task"].max() + 1)
    den = np.bincount(s["task"][idx], weights=s["w"][idx], minlength=s["task"].max() + 1)
    return np.mean(num[den > 0] / den[den > 0])


def sr(s, c, idx=None):
    # sr is the mean of the route and no route halves
    return 0.5 * task_mean(s, c, s["sr"] & s["route"], idx) + 0.5 * task_mean(s, c, s["sr"] & ~s["route"], idx)


def sr_weights(s):
    # item weights so that sr is a plain weighted sum on the full sample which the sign flip test needs
    w = np.zeros(len(s["ids"]))
    for keep in [s["sr"] & s["route"], s["sr"] & ~s["route"]]:
        tasks = np.unique(s["task"][keep])
        for t in tasks:
            sel = keep & (s["task"] == t)
            w[sel] = 0.5 / len(tasks) * s["w"][sel] / s["w"][sel].sum()
    return w


def bootstrap(s, stat, runs, B=10000, seed=0):
    # runs maps a model to a list of 0 1 vectors one per training seed
    # stage one draws the seeds again and stage two draws origin clusters again within each city
    rng = np.random.default_rng(seed)
    est = stat({m: np.mean(v, axis=0) for m, v in runs.items()}, np.arange(len(s["ids"])))
    reps = []
    for _ in range(B):
        avg = {}
        for m, v in runs.items():
            pick = rng.integers(0, len(v), size=len(v))
            avg[m] = np.mean([v[k] for k in pick], axis=0)
        picks = []
        for cl in s["city_clusters"]:
            picks += list(rng.choice(cl, size=len(cl), replace=True))
        idx = np.concatenate([s["members"][c] for c in picks])
        reps.append(stat(avg, idx))
    lo, hi = np.percentile(reps, [2.5, 97.5])
    return est, lo, hi


def signflip(s, v, B=10000, seed=0):
    # v is each item part of a paired difference and with no real difference whole clusters flip sign at random
    rng = np.random.default_rng(seed)
    per = np.bincount(s["cluster"], weights=v)
    null = np.dot(rng.choice([-1.0, 1.0], size=(B, len(per))), per)
    return per.sum(), (np.sum(np.abs(null) >= abs(per.sum()) - 1e-12) + 1) / (B + 1)


def holm(p):
    # holm correction for the secondary tests
    return multipletests(p, method="holm")[1]


def load_run(path, ids):
    # one results file as a 0 1 vector in the item order
    got = {r["item_id"]: r["correct"] for r in read_jsonl(path)}
    return np.array([got[i] for i in ids])


if __name__ == "__main__":
    # fake results on the real sample to test the code before any model has run
    items = [it for it in read_jsonl(BUILT / "eval" / "eval_sample.jsonl") if it["group"] == "SR"]
    s = setup(items)
    rng = np.random.default_rng(1)
    dis = np.array([it["kind"] == "dis" for it in items])

    def fake(p_dir, p_dis):
        # every cluster gets its own shift so its items move together
        p = np.where(dis, p_dis, p_dir) + rng.normal(0, 0.10, len(s["members"]))[s["cluster"]]
        return (rng.random(len(items)) < np.clip(p, 0, 1)).astype(float)

    f0, ft = fake(0.25, 0.25), [fake(0.30, 0.45) for _ in range(3)]
    w = sr_weights(s)
    assert abs(np.dot(w, f0) - sr(s, f0)) < 1e-9
    est, lo, hi = bootstrap(s, lambda a, idx: sr(s, a["ft"], idx) - sr(s, a["f0"], idx), {"ft": ft, "f0": [f0]}, B=2000)
    p = signflip(s, w * (np.mean(ft, axis=0) - f0), B=9999)[1]
    print(f"fake gain of about 0.125: A = {est:.3f} ci [{lo:.3f}, {hi:.3f}] p = {p:.4f}")
    rej = sum(signflip(s, w * (fake(0.3, 0.3) - fake(0.3, 0.3)), 999, k)[1] < 0.05 for k in range(200))
    print(f"no real difference: rejected {rej}/200 at 0.05 which should be about 10")
    print("holm", holm([0.01, 0.04, 0.03, 0.2]))
