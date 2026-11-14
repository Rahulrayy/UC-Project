# Spatial Cognition or Pattern Matching? Re-examining Fine-Tuned LLMs on Urban Computing Benchmarks

Urban Computing project, Leiden University 2026-2027.

The project looks at CityGPT ([paper](https://doi.org/10.1145/3711896.3736878), [code](https://github.com/tsinghua-fib-lab/CityGPT)). In its benchmark (CityEval) the wrong options for distance questions are made from the right answer d: d/2, 2d and d+500 (or d-1000 for poi2poi). So the right answer is always the 2nd smallest option (3rd smallest for poi2poi), and a model can score well without knowing anything about the city. The goal is to see how much of CityGPT's improvement is still there once that shortcut is gone. The full plan is in the proposal PDF in `lit/`.

Below is what's done so far and how to run it.

## Setup

Python 3.12. Only the standard library is used so far, so there's nothing to pip install yet. You need around 2.5 GB of disk space (zip, unzipped data and generated files) and ~3 GB of free RAM (step 1 loads the whole Beijing training file).

```
git clone https://github.com/Rahulrayy/UC-Project.git
cd UC-Project
```

## Getting the data

The dataset is on Hugging Face: https://huggingface.co/datasets/Tianhui-Liu/CityGPT-Data. It's a single 326 MB zip. From the repo root in Git Bash / Linux / Mac:

```
curl -L -o CityGPT-Data.zip https://huggingface.co/datasets/Tianhui-Liu/CityGPT-Data/resolve/main/CityGPT-Data.zip
unzip -q CityGPT-Data.zip
mv CityGPT-Data data
rm -rf __MACOSX CityGPT-Data.zip
```

or in PowerShell:

```
curl.exe -L -o CityGPT-Data.zip https://huggingface.co/datasets/Tianhui-Liu/CityGPT-Data/resolve/main/CityGPT-Data.zip
Expand-Archive CityGPT-Data.zip -DestinationPath .
Rename-Item CityGPT-Data data
Remove-Item -Recurse -Force __MACOSX, CityGPT-Data.zip
```

After that you should have:

```
data/CityInstruction/       4 training json files, one per city (~275 MB each)
data/CityEval/tasks/        Beijing, London, NewYork, Paris, each with a v2.3 folder of 39 csv files
```

There are some .DS_Store files in there from a Mac, you can ignore them. Don't commit `data/`, it's in .gitignore anyway and the files are too big for GitHub.

## What's done so far

Three scripts, run them in this order from the repo root:

```
python src/build_train_sets.py
python src/build_eval_sample.py
python src/build_topt.py
```

Everything goes into `data/built/` (also gitignored, just rebuild it). All the random stuff uses seed 0 so you should get exactly the same files. The first script takes about 20 seconds, the other two a few seconds. The expected output is pasted under each step so you can check your run matches.

### Step 1: build_train_sets.py

Makes the training sets. Training is only done on Beijing, the other three cities are used as unseen cities (same as the transfer experiment in the paper).

- lean pool (78,491 items): all Beijing address and route questions, all CityReasoning items, the 7,266 San Francisco items written in the exact test format and the other spatial data, plus 10k of the 30k CityWalk items and 2k of the ~42k general instruction items (chat, maths and so on). This cuts training from ~62M to ~19M tokens.
- f_ft: lean pool minus 7,266 random items (only taken from the Beijing city data and the general items), so it's the same size as f_notask
- f_notask: lean pool minus the 7,266 test format items
- full: the whole Beijing file, for one calibration run

It writes `f_ft.json`, `f_notask.json` and `full.json` (alpaca format, `id` is the index in the original Beijing file), a `dataset_info.json` for LLaMA-Factory, and `ids.json` with the index lists (needed later for the fixed training set). All in `data/built/train/`.

You should see (among other lines):

```
pool:     78491 items 19.16M tok
f_ft:     71225 items 17.82M tok
f_notask: 71225 items 17.41M tok
full:     138356 items 62.11M tok
cityreasoning in f_ft / f_notask: 7992 7992
rank rule items in pool: 4697 (6.0%)
```

The token counts are only estimated from the number of characters.

### Step 2: build_eval_sample.py

Picks the fixed set of test items that every model gets evaluated on.

- spatial reasoning: 100 rows for each question type (5 types, direction and distance) per city. The route and no route csvs have the same rows, so each row gives 2 items, 8,000 in total
- for distance the 100 rows are spread over the answer size: rows are sorted by distance, split into 4 chunks and 25 are drawn from each
- poi2coor and road_length: 100 each per city since they have a similar shortcut. Together with spatial reasoning this is the "core" set (8,800 items)
- the other 17 City Image / Urban Semantics tasks: up to 100 each, only used to check that the trained model reproduces the paper's gain. 14,687 items overall
- every spatial reasoning item gets a `cluster_id`, which is the place the question is measured from. A lot of questions share places and the stats will need to account for that. The other items are each their own cluster

Output is `data/built/eval/eval_sample.jsonl` with one item per line. The main fields are `item_id` (city/task/row, where row is the 0 based row in the csv), `question`, `options` (like `{"A": "2800", ...}`), `answer`, `cluster_id` and `core`. Distance items also have `d` (the answer distance), `d_bin` (which of the 4 chunks), `bin_n` (how big that chunk is) and `gap` (relative gap to the closest wrong option).

The output should start with:

```
total items: 14687
core items: 8800
Beijing  sr rows=1000 no origin=0 clusters=685 max=5 mean=1.46 in clusters >1: 55%
London   sr rows=1000 no origin=0 clusters=686 max=6 mean=1.46 in clusters >1: 53%
NewYork  sr rows=1000 no origin=0 clusters=592 max=11 mean=1.69 in clusters >1: 63%
Paris    sr rows=1000 no origin=0 clusters=542 max=9 mean=1.85 in clusters >1: 71%
```

### Step 3: build_topt.py

Same test questions but with fair wrong options, so "pick the 2nd smallest" only works by chance. The actual sampling is in `fair_options.py`.

- distance: the wrong options are drawn from real answers of the same task and city, and all options are at least 20% apart. Just drawing them at random still put the answer in the middle ranks too often, so `fair_options.py` reweights the draw so every rank is equally likely (the comments in the file explain how)
- road_length: same idea with 5 options. Some roads have length 0, so the spacing is done on log(length + 100)
- poi2coor: in the original the answer is always the middle point and the 4 wrong options are the corners of a square around it, 0.01 degrees off in longitude and latitude. The same pattern is kept but the answer is moved to a random one of the 5 positions

Output is `data/built/eval/topt.jsonl` (4,800 items). Same fields as the eval sample plus `orig_options`, `orig_answer`, `rule_letter` (what the old rule would pick now) and `cond` = "topt". For distance items `gap` is recomputed for the new options and the old value is kept as `orig_gap`.

The output should end with this. The old rule should now only be right about 1 in 4 times (1 in 5 for road_length and poi2coor):

```
  aoi2aoi_dis    rule still right 103/400 = 0.26
  aoi2rd_dis     rule still right 100/400 = 0.25
  poi2aoi_dis    rule still right 92/400 = 0.23
  poi2coor       rule still right 79/400 = 0.20
  poi2poi_dis    rule still right 102/400 = 0.26
  poi2rd_dis     rule still right 91/400 = 0.23
  road_length    rule still right 66/400 = 0.17
  all distance   488/2000 = 0.244
wrote 4800 items
```

### utils.py

Shared stuff: paths, loaders for the json and csv files, the token estimate, grouping training items by source, the answer gap, and the parser that finds the origin place in a question. The parser counts brackets because some place names have brackets in them, e.g. "Gowanus Expressway (HOV)".

## Folders

```
lit/          paper, proposal, pitch slides
src/          the scripts
data/         dataset (not in git)
data/built/   generated files (not in git)
```
