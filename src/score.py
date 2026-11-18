# scores one model on one eval file with likelihood scoring or the paper generation protocol or free answers
# run from the repo root with python src/score.py --model Qwen/Qwen2.5-0.5B-Instruct --name base05 --cond sample --protocol ll

import argparse
import csv
import re

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from utils import BUILT, DATA, map_open, read_jsonl, write_jsonl

# code from paper gh
INIT_PROMPT = "The following is a multiple-choice question. Please choose the most suitable one among A, B, C and D as the answer to this question. Please output the option directly. No need for explaination.\n"


# code from paper gh
def extract_choice(gen):
    res = re.search(
        r"(?:(?:[Cc]hoose)|(?:(?:[Aa]nswer|[Cc]hoice)(?![^A-H]{0,20}?(?:n't|not))[^A-H]{0,10}?\b(?:|is|:|be))\b)[^A-H]{0,20}?\b([A-H])\b",
        gen,
    )
    if res is None:
        res = re.search(
            r"\b([A-H])\b(?![^A-H]{0,8}?(?:n't|not)[^A-H]{0,5}?(?:correct|right))[^A-H]{0,10}?\b(?:correct|right)\b",
            gen,
        )
    if res is None:
        res = re.search(r"^([A-H])(?:\.|,|:|$)", gen)
    if res is None:
        res = re.search(r"\n\s*([A-H])", gen)
    if res is None:
        return "Z"
    return res.group(1)


n_opts = {}


def format_example(it, answer_suffix, init=True):
    # same layout as format_example in the paper without the few shot examples
    # the paper counts options from the csv header so an empty E column still shows up as nan
    key = (it["city"], it["task"])
    if key not in n_opts:
        with open(DATA / "CityEval" / "tasks" / it["city"] / "v2.3" / f"{it['task']}.csv", encoding="utf-8") as f:
            n_opts[key] = 5 if "E" in next(csv.reader(f)) else 4
    text = (INIT_PROMPT if init else "") + "Question: " + it["question"]
    for k in "ABCDE"[:n_opts[key]]:
        text += f"\n{k}. {it['options'].get(k, 'nan')}"
    if answer_suffix:
        text += "\nAnswer:"
    return text


def chat(conv):
    return tok.apply_chat_template(conv, tokenize=False, add_generation_prompt=True)


def next_logprobs(texts):
    # log probs of the next token after each prompt and only the last position so it fits in memory
    out = []
    for k in range(0, len(texts), args.batch):
        enc = tok(texts[k:k + args.batch], return_tensors="pt", padding=True).to("cuda")
        with torch.no_grad():
            logits = model(**enc, logits_to_keep=1).logits[:, -1].float()
        out += list(torch.log_softmax(logits, -1).cpu())
    return out


def generate(convs, max_new=100, allowed=None):
    # greedy like temperature 0 in the paper and repetition penalty 1 like their vllm server
    # allowed is for T_open and says which tokens may come next given how many are already written
    out = []
    for k in range(0, len(convs), args.batch):
        enc = tok([chat(c) for c in convs[k:k + args.batch]], return_tensors="pt", padding=True).to("cuda")
        n = enc["input_ids"].shape[1]

        def limit(b, ids):
            return allowed(len(ids) - n)

        with torch.no_grad():
            gen = model.generate(**enc, max_new_tokens=max_new, do_sample=False, temperature=None, top_p=None,
                                 top_k=None, repetition_penalty=1.0, pad_token_id=tok.pad_token_id,
                                 prefix_allowed_tokens_fn=limit if allowed else None)
        out += tok.batch_decode(gen[:, n:], skip_special_tokens=True)
    return out


def score_ll(items):
    # the answer is the option letter with the highest log prob right after Answer
    letters = {k: tok.convert_tokens_to_ids(k) for k in "ABCDE"}
    lps = next_logprobs([chat([{"role": "user", "content": format_example(it, True)}]) for it in items])
    out = []
    for it, lp in zip(items, lps):
        scores = {k: round(lp[letters[k]].item(), 4) for k in it["options"]}
        out.append({"prediction": max(scores, key=scores.get), "scores": scores, "parse": "ok"})
    return out


def score_gen(items):
    # one round for route items and city image and semantics and three rounds for no route items like auto_multi
    many = [i for i, it in enumerate(items) if it["group"] == "SR" and not it["route"]]
    one = [i for i, it in enumerate(items) if not (it["group"] == "SR" and not it["route"])]
    rounds = {}
    outs = generate([[{"role": "user", "content": format_example(items[i], args.answer_suffix)}] for i in one])
    for i, o in zip(one, outs):
        rounds[i] = [o]

    # code from paper gh
    # round three is an empty message unless answer_suffix is on just like in their code
    q, conv = {}, {}
    for i in many:
        q[i] = format_example(items[i], False, init=False)
        conv[i] = [{"role": "user", "content": q[i] + "\nTo answer the question, please think about the navigation instruction about two locations."}]
    for step in range(3 if many else 0):
        outs = generate([conv[i] for i in many])
        for i, o in zip(many, outs):
            rounds.setdefault(i, []).append(o)
            if step == 0:
                conv[i] += [{"role": "assistant", "content": "The navigation instruction: " + o},
                            {"role": "user", "content": "Please consider the above context and answer the following question. " + q[i] + "\nLet's think step by step."}]
            else:
                conv[i] += [{"role": "assistant", "content": o},
                            {"role": "user", "content": "Thus, " + q[i] + "\nAnswer:" if args.answer_suffix else ""}]

    out = []
    for i in range(len(items)):
        final = rounds[i][-1]
        pred = extract_choice(final) if final else "No output"   # empty output counts as wrong like in the paper
        parse = "fail" if pred in ("Z", "No output") else "ok"
        out.append({"prediction": pred, "rounds": rounds[i], "parse": parse, "answer_suffix": args.answer_suffix})
    return out


def ask(it):
    return [{"role": "user", "content": it["question"] + "\n" + it["open_instruction"]}]


def score_open(items):
    # distance answers can only be digits and direction answers are the likeliest of the four words
    digits = [tok.convert_tokens_to_ids(str(n)) for n in range(10)]
    stop = tok.convert_tokens_to_ids("<|im_end|>")

    def allowed(n_new):
        return digits + [stop] if n_new > 0 else digits   # stop only after at least one digit

    dis = [it for it in items if it["kind"] == "dis"]
    dirs = [it for it in items if it["kind"] == "dir"]
    answers = {}
    for it, text in zip(dis, generate([ask(it) for it in dis], 7, allowed)):
        answers[it["item_id"]] = text
    words = {w: tok.convert_tokens_to_ids(w) for w in ["north", "south", "east", "west"]}
    for it, lp in zip(dirs, next_logprobs([chat(ask(it)) for it in dirs])):
        answers[it["item_id"]] = max(words, key=lambda w: lp[words[w]])
    out = []
    for it in items:
        letter = map_open(answers[it["item_id"]], it)
        out.append({"prediction": letter or "Z", "answer_text": answers[it["item_id"]], "parse": "ok" if letter else "fail"})
    return out


ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--adapter", default=None, help="lora adapter folder")
ap.add_argument("--name", required=True, help="short model name for the output file")
ap.add_argument("--cond", required=True, choices=["sample", "topt", "fictional", "rq4", "topen"])
ap.add_argument("--protocol", required=True, choices=["ll", "gen", "open"])
ap.add_argument("--answer_suffix", action="store_true", help="the paper flag include_answer_prompt_final")
ap.add_argument("--limit", type=int, default=0, help="only n items spread over the file for a quick test")
ap.add_argument("--batch", type=int, default=16)
args = ap.parse_args()

items = read_jsonl(BUILT / "eval" / ("eval_sample.jsonl" if args.cond == "sample" else f"{args.cond}.jsonl"))
if args.limit:
    items = items[::len(items) // args.limit][:args.limit]   # every kth item so a quick test still covers all tasks
tok = AutoTokenizer.from_pretrained(args.model, padding_side="left")
model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16).to("cuda")
if args.adapter:
    from peft import PeftModel
    model = PeftModel.from_pretrained(model, args.adapter)
model.eval()

if args.protocol == "ll":
    res = score_ll(items)
elif args.protocol == "gen":
    res = score_gen(items)
else:
    res = score_open(items)

rows = []
for it, r in zip(items, res):
    row = {"item_id": it["item_id"], "cluster_id": it["cluster_id"], "cond": args.cond, "model": args.name,
           "protocol": args.protocol, "gold": it["answer"]}
    row.update(r)
    row["correct"] = float(r["prediction"] == it["answer"])
    if "rule_letter" in it:
        row["rule_hit"] = float(r["prediction"] == it["rule_letter"])
    # e13 items have the answer value under more than one letter so they get a flag
    row["dup_answer"] = list(it["options"].values()).count(it["options"][it["answer"]]) > 1
    rows.append(row)
write_jsonl(BUILT / "results" / f"{args.name}__{args.cond}__{args.protocol}.jsonl", rows)
print(f"{args.name} {args.cond} {args.protocol}: {len(rows)} items, parse fails {sum(r['parse'] == 'fail' for r in rows)}")
