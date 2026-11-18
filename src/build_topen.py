# T_open for the sr items where the options are hidden and the model has to give the answer itself
# run from the repo root with python src/build_topen.py

from utils import BUILT, map_open, read_jsonl, write_jsonl

ASK = {
    "dis": "Answer with the distance in meters as a single number.",
    "dir": "Answer with one word: north, south, east or west.",
}

out = []
for it in read_jsonl(BUILT / "eval" / "eval_sample.jsonl"):
    if it["group"] == "SR":
        new = dict(it)
        new["cond"] = "topen"
        new["open_instruction"] = ASK[it["kind"]]
        # the true value has to map straight back to the right letter
        assert map_open(it["options"][it["answer"]], new) == it["answer"], it["item_id"]
        out.append(new)
write_jsonl(BUILT / "eval" / "topen.jsonl", out)
print("wrote", len(out), "items")
