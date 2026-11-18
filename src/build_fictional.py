# fictional place versions of the no route sr items so no city knowledge can help
# run from the repo root with python src/build_fictional.py

import glob
import random
import re

from utils import BUILT, CITIES, DATA, SHAPES, names_in, read_jsonl, write_jsonl

ONSETS = ["b", "br", "c", "d", "dr", "f", "g", "gr", "h", "k", "l", "m", "n", "p", "pl", "qu", "r", "s", "st", "t", "th", "v", "w"]
VOWELS = ["a", "e", "i", "o", "u", "ai", "ea", "ou"]
CODAS = ["", "", "n", "r", "l", "th", "ck", "sh", "m", "x", "nd", "rt"]
PLACE_WORDS = ["Hall", "Park", "Market", "Gardens", "House", "Center", "Library", "Cafe", "School", "Clinic",
               "Plaza", "Studio", "Tower", "Court", "Bakery", "Chapel"]
ROAD_WORDS = ["Street", "Avenue", "Road", "Lane", "Boulevard", "Way", "Drive"]

# every word in the training files and benchmark csvs so a made up word never matches a real one
words = set()
files = glob.glob(str(DATA / "CityInstruction" / "*.json")) + glob.glob(str(DATA / "CityEval" / "tasks" / "*" / "v2.3" / "*.csv"))
for path in files:
    with open(path, encoding="utf-8") as f:
        text = f.read().lower()
    for k in range(0, len(text), 20_000_000):   # in chunks that overlap a bit to save memory
        words.update(re.findall(r"[a-z]+", text[k:k + 20_000_100]))
print("real words:", len(words))

rng = random.Random(0)
used = set()


def new_word():
    while True:
        w = "".join(rng.choice(ONSETS) + rng.choice(VOWELS) + rng.choice(CODAS) for _ in range(rng.choice([2, 3])))
        if len(w) >= 5 and w not in words and w not in used:
            used.add(w)
            return w.capitalize()


def fake(q):
    # same made up name every time a real name comes back in the question
    names = {}
    for n, road in names_in(q):
        if n in names:
            continue
        core = new_word()
        if road:
            names[n] = core + " " + rng.choice(ROAD_WORDS)
        else:
            names[n] = (core + " " + rng.choice(PLACE_WORDS + [""])).strip()
    pat = "|".join(re.escape(n) for n in sorted(names, key=len, reverse=True))
    return re.sub(rf"(?<!\w)(?:{pat})(?!\w)", lambda m: names[m.group(0)], q), names


items = read_jsonl(BUILT / "eval" / "eval_sample.jsonl")
out = []
for shape in SHAPES:
    for kind in ["dir", "dis"]:
        for city in CITIES:
            pool = [it for it in items if it["city"] == city and it["task"] == f"{shape}_{kind}_noroutine"]
            for it in sorted(rng.sample(pool, 25), key=lambda x: x["row"]):   # 25 per city so 100 per task
                q, names = fake(it["question"])
                assert all(real not in q for real in names), it["item_id"]
                new = dict(it)
                new["item_id"] = "fictional/" + it["item_id"]
                new["cond"] = "fictional"
                new["orig_question"] = it["question"]
                new["question"] = q
                new["cluster_id"] = "fictional|" + it["item_id"]   # made up places arent shared so own cluster
                new["has_origin"] = False
                new["n_names"] = len(names)
                out.append(new)

print("items:", len(out))
print(" ", out[0]["orig_question"])
print(" ", out[0]["question"])
write_jsonl(BUILT / "eval" / "fictional.jsonl", out)
