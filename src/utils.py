import csv
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
BUILT = DATA / "built"

CITIES = ["Beijing", "London", "NewYork", "Paris"]
SHAPES = ["aoi2aoi", "aoi2rd", "poi2aoi", "poi2poi", "poi2rd"]

CI_TASKS = ["aoi2addr", "aoi_boundary_poi", "aoi_poi", "boundary_road", "districts_poi_type",
            "landmark_path", "poi2addr", "poi2coor", "poi_aoi", "road_arrived_pois",
            "road_length", "road_link", "road_od"]
US_TASKS = ["aoi2type", "aoi_group", "landmark_env", "poi2type", "type2aoi", "type2poi"]

# poi2poi csvs have huge fields
csv.field_size_limit(10**9)


def load_train(city="Beijing"):
    path = DATA / "CityInstruction" / f"citygpt-{city}-v24.6-SF-v2.3-alpaca.json"
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_task(city, task):
    path = DATA / "CityEval" / "tasks" / city / "v2.3" / f"{task}.csv"
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def get_options(row):
    # some files have an empty E column so only keep filled ones
    return {k: row[k] for k in "ABCDEFGH" if row.get(k, "") != ""}


def answer_gap(opts, ans):
    # relative gap between the answer and the closest wrong option
    y = float(opts[ans])
    others = [float(v) for k, v in opts.items() if k != ans]
    if y <= 0:
        return None
    return min(abs(o - y) for o in others) / y


CJK = re.compile(r"[\u3400-\u9fff]")


def est_tokens(text):
    # rough token count about 4 chars per token for latin and 1.4 for chinese
    # not the real tokenizer but good enough for budgeting
    c = len(CJK.findall(text))
    return (len(text) - c) / 4 + c / 1.4


def item_tokens(x):
    text = x["instruction"] + x["output"]
    for q, a in x["history"]:
        text += q + a
    return est_tokens(text)


def group(source):
    # which bucket a training item goes in when building the lean pool
    if source.endswith(".csv"):
        return "template"
    if source.startswith("self-reasoning"):
        return "cityreasoning"
    if source == "cwm_citywalk_data_citywalk":
        return "citywalk"
    if source.startswith("cwm_"):
        return "city_other"
    if source.startswith("spatial-reasoning"):
        return "spatial"
    return "general"


# finds the numeric options in training items
NUM_OPT = re.compile(r"(?:^|[\s?.:])([A-H])[.:]\s?(-?\d+(?:\.\d+)?)\s*(?=\n|$|[A-H][.:])", re.M)


def is_rank_rule(x):
    # training items where the answer rank is fixed by how the options were made
    s = x["source"]
    if s == "road_length.csv" or "_dis_" in s:
        return True
    return s.startswith("self-reasoning") and len(NUM_OPT.findall(x["instruction"])) >= 4


SIDE = re.compile(r"on the [a-z]+ side of .*", re.S)


def addresses(q):
    # finds every place address in a question
    # they look like inside Y on the east side of road X 350m from some junction
    # names can have brackets in them like Gowanus Expressway HOV so count the depth
    # instead of using a regex
    # returns start end and the address text
    out = []
    i = q.find("(")
    while i != -1:
        if q.startswith("inside ", i + 1) or q.startswith("on the ", i + 1):
            depth = 0
            j = i
            while j < len(q):
                if q[j] == "(":
                    depth += 1
                elif q[j] == ")":
                    depth -= 1
                    if depth == 0:
                        break
                j += 1
            m = SIDE.search(q[i + 1:j])
            if m:
                out.append((i, j, m.group(0).strip()))
            i = q.find("(", j + 1)
        else:
            i = q.find("(", i + 1)
    return out


def origin(question):
    # the place after from is the reference point and is used as the cluster for the stats
    # aoi2rd and poi2rd questions have no address for the road so fall back to the one named place
    addrs = addresses(question)
    if not addrs:
        return None
    for m in re.finditer(r"\bfrom ", question):
        # skip the 350m from the corner bits inside addresses
        if any(s < m.start() < e for s, e, _ in addrs):
            continue
        after = [a for s, e, a in addrs if s > m.start()]
        if after:
            return after[0]
    return addrs[0][2]


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]
