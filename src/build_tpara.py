# T_para where cohere command a with a free trial key rewords the no route sr questions
# run from the repo root with python src/build_tpara.py

import csv
import json
import os
import random
import re
import time

import requests
from dotenv import load_dotenv
from rouge_score import rouge_scorer

from utils import BUILT, ROOT, addresses, names_in, read_jsonl, write_jsonl

URL = "https://api.cohere.com/v2/chat"
MODEL = "command-a-03-2025"   # another family than qwen and it answers straight away without thinking first
# the trial key goes in the env file at the repo root which git ignores so it never gets pushed
# its only needed when a question has no saved paraphrase yet
load_dotenv(ROOT / ".env")
KEY = os.environ.get("COHERE_API_KEY", "")
PER_CALL = 20
GAP = 6   # the trial key allows 20 a minute but bursts get blocked so wait a bit between requests
CACHE = ROOT / "paraphrases" / "tpara_raw_cohere.jsonl"   # every answer is saved here and it goes on github since the model cant be rerun exactly
BRACKETS = r"\([^()]*(?:\([^()]*\)[^()]*)*\)"   # an address in brackets that can have one more bracket inside
ASK = """Rewrite each of the following questions so that it asks exactly the same thing in different words.
Rules:
- Copy every name and everything in brackets exactly as written, character for character, including the brackets.
- A direction question "In which direction is A from B?" asks which way you go from B to reach A. Keep A as the place you are looking for and B as where you start. Never swap them and never ask "from which direction". For example "In which direction is the bank from the station?" can become "Standing at the station, which way is the bank?" or "Which direction is the bank, seen from the station?"
- A distance question must still ask for the walking distance along the road between the same places, or between the place and the road. For example "How many meters do I need to walk from the bank to the station along the road?" can become "What is the walking distance along the road from the bank to the station, in meters?"
- Do not answer the questions.
Return only a JSON list of strings with one rewritten question per input, in the same order.

"""
MARKERS = {"from", "at", "to", "of", "on"}   # words that put a place in the role of where you start like on for roads


def call(questions):
    # one request with a few questions and None if the answer cant be used
    prompt = ASK + "\n".join(f"{n + 1}. {q}" for n, q in enumerate(questions))
    body = {"model": MODEL, "temperature": 0, "max_tokens": 8000,   # room for 20 long beijing questions
            "messages": [{"role": "user", "content": prompt}]}
    headers = {"Authorization": "Bearer " + KEY, "User-Agent": "uc-project-script"}
    for wait in [60, 120, 240, 480]:
        try:
            r = requests.post(URL, json=body, headers=headers, timeout=120)
        except requests.RequestException as e:
            print("  network problem", e, "waiting", wait, "s")   # timeouts and dropped connections
            time.sleep(wait)
            continue
        if r.status_code in (403, 429, 500, 502, 503, 504):
            print("  http", r.status_code, "waiting", wait, "s")   # 429 and the odd 403 mean too many requests
            time.sleep(wait)
            continue
        if r.status_code != 200:
            raise SystemExit(f"http {r.status_code} {r.text[:300]}")   # wrong key or model so stop right away
        # take the part of the reply that is the actual text and give up on this batch if there isnt one
        text = ""
        for c in r.json()["message"]["content"]:
            if c["type"] == "text":
                text = c["text"]
        try:
            out = json.loads(text[text.index("["):text.rindex("]") + 1])
        except ValueError:
            return None
        return out if len(out) == len(questions) and all(isinstance(o, str) for o in out) else None
    return None


def keeps_names(orig, new):
    # every bracket block and every name has to come back exactly and the wording has to change
    blocks = [orig[i:j + 1] for i, j, _ in addresses(orig)]
    return all(b in new for b in blocks) and all(n in new for n, _ in names_in(orig)) and new.strip() != orig.strip()


def word_before(text, name):
    words = re.findall(r"[a-z]+", text[:text.find(name)].lower())
    return words[-1] if words else ""


def keeps_roles(orig, new):
    # in a direction question the start place has to stay the start and the other place the goal
    # the start should come right after from or at or to or of or on and the goal shouldnt
    if not orig.startswith("In which direction is "):
        return True
    i, j, _ = addresses(orig)[0]
    goal = orig[len("In which direction is "):i]
    rest = orig[j + 1 + len(" from "):]
    start = rest[:rest.index("(")] if "(" in rest else rest[:-1]
    return ("from which direction" not in new.lower() and word_before(new, start) in MARKERS
            and word_before(new, goal) not in MARKERS)


scorer = rouge_scorer.RougeScorer(["rougeL"])


def rouge_l(a, b):
    # rouge l f score on the text outside the brackets so it measures the template part
    return scorer.score(re.sub(BRACKETS, " ", a), re.sub(BRACKETS, " ", b))["rougeL"].fmeasure


items = read_jsonl(BUILT / "eval" / "eval_sample.jsonl")
stems = [it for it in items if it["group"] == "SR" and not it["route"]]
done = {}
stem_q = {s["item_id"]: s["question"] for s in stems}
if CACHE.exists():
    # check every saved answer again with the current rules and keep a good one if there is any
    for r in read_jsonl(CACHE):
        q = stem_q[r["item_id"]]
        r["ok"] = keeps_names(q, r["paraphrase"]) and keeps_roles(q, r["paraphrase"])
        if r["item_id"] not in done or not done[r["item_id"]]["ok"]:
            done[r["item_id"]] = r
# up to 3 passes so questions that failed the checks get another go
for attempt in range(3):
    todo = [it for it in stems if it["item_id"] not in done or not done[it["item_id"]]["ok"]]
    print(f"pass {attempt + 1}: {len(todo)} questions left")
    if todo and not KEY:
        print("no cohere key in .env so these stay out")
        break
    for k in range(0, len(todo), PER_CALL):
        batch = todo[k:k + PER_CALL]
        time.sleep(GAP)
        out = call([it["question"] for it in batch])
        if out is None:
            out = [""] * len(batch)
        with open(CACHE, "a", encoding="utf-8") as f:
            for it, new in zip(batch, out):
                good = keeps_names(it["question"], new) and keeps_roles(it["question"], new)
                r = {"item_id": it["item_id"], "paraphrase": new, "ok": good}
                done[it["item_id"]] = r
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"  {k + len(batch)}/{len(todo)} ok so far {sum(r['ok'] for r in done.values())}")

ok = {i: r["paraphrase"] for i, r in done.items() if r["ok"]}
print(f"paraphrased {len(ok)}/{len(stems)} questions")

# route items keep their route text and get the reworded question at the end
# distance items get the fair options from T_opt like the plan says
topt = {it["item_id"]: it for it in read_jsonl(BUILT / "eval" / "topt.jsonl")}
out = []
for it in items:
    if it["group"] != "SR":
        continue
    stem_id = it["item_id"].replace("_routine/", "_noroutine/")
    if stem_id not in ok:
        continue
    stem = stem_q[stem_id]
    new = dict(topt[it["item_id"]]) if it["kind"] == "dis" else dict(it)
    new["cond"] = "tpara"
    new["orig_question"] = it["question"]
    new["question"] = it["question"][:len(it["question"]) - len(stem)] + ok[stem_id]
    new["rouge_l"] = round(rouge_l(stem, ok[stem_id]), 3)
    out.append(new)
write_jsonl(BUILT / "eval" / "tpara.jsonl", out)
print("wrote", len(out), "items, mean rouge l of the template part", round(sum(r["rouge_l"] for r in out) / len(out), 3))

# 100 random ones for the two of us to check by hand
# only made once so a rerun never wipes the answers we filled in
check = ROOT / "paraphrases" / "tpara_check.csv"
if not check.exists():
    rng = random.Random(0)
    with open(check, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["item_id", "original", "paraphrase", "same_meaning_annotator_1", "same_meaning_annotator_2"])
        for i in rng.sample(sorted(ok), 100):
            w.writerow([i, stem_q[i], ok[i], "", ""])
    print("wrote tpara_check.csv for the hand check")
