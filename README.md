# Spatial Cognition or Pattern Matching? Re-examining Fine-Tuned LLMs on Urban Computing Benchmarks

Urban Computing project, Leiden University 2026-2027.

The project looks at CityGPT ([paper](https://doi.org/10.1145/3711896.3736878), [code](https://github.com/tsinghua-fib-lab/CityGPT)). In its benchmark (CityEval) the wrong options for distance questions are made from the right answer d: d/2, 2d and d+500 (or d-1000 for poi2poi). So the right answer is always the 2nd smallest option (3rd smallest for poi2poi), and a model can score well without knowing anything about the city. The goal is to see how much of CityGPT's improvement is still there once that shortcut is gone. The full plan is in the proposal PDF in `lit/`.

Below is what's done so far and how to run it.

## Setup

Python 3.12. The build scripts only use the standard library. The T_opt check needs numpy and scikit-learn, and scoring and training need torch, transformers and LLaMA-Factory. You need around 2.5 GB of disk space for the data and generated files, about 6 GB more for the packages and models, and ~3 GB of free RAM (step 1 loads the whole Beijing training file).

```
git clone https://github.com/Rahulrayy/UC-Project.git
cd UC-Project
```

For the scoring and training part install torch with CUDA first and then the rest:

```
pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
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

The build scripts, run them in this order from the repo root:

```
python src/build_train_sets.py
python src/build_eval_sample.py
python src/build_topt.py
python src/build_ffair.py
python src/build_topen.py
python src/build_fictional.py
python src/build_rq4.py
python src/build_tpara.py
```

Everything goes into `data/built/` (also gitignored, just rebuild it). All the random stuff uses seed 0 so you should get exactly the same files. Most take a few seconds, build_train_sets.py and build_rq4.py about 20 to 30 seconds and build_fictional.py about 2 minutes. The expected output is pasted under each step so you can check your run matches.

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

### Step 4: build_ffair.py

Makes f_fair, the fixed training set. It's f_ft with the same items in the same order, but every item where the rank trick works (2,000 distance template items, 200 road_length, 2,497 CityReasoning questions with number options) plus the 193 poi2coor items get fair options the same way as T_opt. In the CityReasoning items the reasoning stays as it is and only the letter after "Answer:" changes. It writes `data/built/train/f_fair.json` and adds f_fair to `dataset_info.json`.

It prints how often the old rule still picks the answer for each kind of item, which should be about 1 in 4 (1 in 5 for road_length, 0.243 overall), and ends with `wrote f_fair 71225`.

### Step 5: build_topen.py

T_open: the same spatial reasoning questions without the options. The model has to answer with a number of meters or one of north, south, east and west, and that answer is mapped to the closest original option (on a log scale for distances, so an option of 0 or less is never picked for a positive answer). The mapping is `map_open` in utils.py. Output is `data/built/eval/topen.jsonl` and it prints `wrote 8000 items`.

### Step 6: build_fictional.py

Fictional places: 100 no route questions per spatial reasoning task (25 per city) where every place, building and road name is swapped for a made up one. A real name that shows up twice in a question gets the same fake name both times. The made up words are checked against every word in the training files and the benchmark csvs. Options and answers stay the same, so nobody can know the answer but the rank trick still works. Output is `data/built/eval/fictional.jsonl` and it prints `real words: 245116` and `items: 1000`.

### Step 7: build_rq4.py

The RQ4 items: no route place to place questions (aoi2aoi, poi2aoi, poi2poi) where both places can be found in the address to coordinate questions of that city's CityQA data. Addresses with "unknown road" or that point to spots more than 200 m apart are dropped. 1000 per city, and each item gets the coordinates of both places and a 1 km grid cell, which the stats use instead of the origin cluster. Output is `data/built/eval/rq4.jsonl` and it ends with `wrote 4000 items`.

### Step 8: build_tpara.py

T_para: the no route spatial reasoning questions reworded by an LLM from another family than Qwen (Cohere Command A, free trial key). To make new paraphrases you need your own key: put `COHERE_API_KEY=...` in a `.env` file in the repo root (it's in .gitignore). Every name and every address in brackets has to come back exactly, and in direction questions the start place has to stay the start, otherwise the question is asked again (3 tries). Route items keep their route text with the reworded question at the end, and distance items get the T_opt options. The paraphrases are saved in `paraphrases/tpara_raw_cohere.jsonl`, which is in git because the model can't be rerun to get exactly the same text. So you don't need a key to rebuild T_para: the script reuses the saved ones and only asks the model about questions that don't have a good paraphrase yet (without a key those just stay out). It prints `paraphrased 3997/4000 questions` and `wrote 7994 items, mean rouge l of the template part 0.522`.

`paraphrases/tpara_check.csv` has 100 random paraphrases for two people to check by hand that they mean the same thing. It's only made once, so a rerun won't wipe the answers.

### option_classifier.py

Checks that T_opt really removes the trick. A classifier only sees the options (rank, size, rounding, whether another option is half or double of it and so on). Trained and tested on the original options it gets everything right, which shows it can find the trick. Trained on the fair training items and tested on T_opt it should be at chance. Takes about a minute:

```
dis          original 1.000  T_opt 0.245 p=0.69  chance 0.25
road_length  original 1.000  T_opt 0.177 p=0.88  chance 0.2
poi2coor     original 1.000  T_opt 0.225 p=0.12  chance 0.2
T_opt passes
```

### stats.py

The stats for the results files: accuracy like the paper (mean of task accuracies, SR is the mean of the route and no route halves, distance rows weighted by their quartile), a bootstrap that resamples training seeds and then origin clusters within each city, a sign flip test that flips whole clusters, and Holm correction. Running it directly tests it on fake results: it should find the planted gain (A = 0.132) and wrongly reject only about 5 percent of the time when there is no difference (11/200).

### score.py

Scores a model on one eval file and writes `data/built/results/<name>__<cond>__<protocol>.jsonl` with one line per item (prediction, gold letter, correct, and the scores or the raw answers). Three ways to score:

- ll: the option letter the model gives the highest probability right after "Answer:". This is the main one.
- gen: the paper's own way, copied from their run_eval.py. The model writes an answer and their regex picks the letter, and no route items get their three round prompt. Their third round sends an empty message unless their include_answer_prompt_final flag is on, so here that flag is `--answer_suffix` and it's off by default like in their code.
- open: for T_open. Distance answers can only be digits and direction answers are the most likely of the four words.

```
python src/score.py --model Qwen/Qwen2.5-0.5B-Instruct --name base05 --cond sample --protocol ll
python src/score.py --model Qwen/Qwen2.5-0.5B-Instruct --adapter runs/q05_f_ft --name ft05 --cond topt --protocol ll
```

`--cond` is one of sample, topt, fictional, rq4 and topen. `--limit 40` does a quick test on 40 items spread over the file.

### Training

Training uses LLaMA-Factory 0.9.5 with the settings in `configs/qwen_lora.yaml` (LoRA rank 16 on all layers, 1 epoch, lr 2e-4, max length 2048 with packing). Pick the training set and output folder on the command line:

```
llamafactory-cli train configs/qwen_lora.yaml dataset=f_ft output_dir=runs/q05_f_ft
```

The config uses Qwen2.5-0.5B-Instruct. For the real runs add `model_name_or_path=Qwen/Qwen2.5-1.5B-Instruct`. On a laptop RTX 3060 (6 GB) the 0.5B run on f_ft is 653 steps and took 3 h 42 min (final training loss 0.40). Scoring the whole eval sample (14,687 items) with ll takes about 9 minutes for the 0.5B model on the same laptop, a bit longer with an adapter. Windows has no flash attention 2, so the config uses sdpa.

### utils.py

Shared stuff: paths, loaders for the json and csv files, the token estimate, grouping training items by source, the answer gap, the parser that finds the origin place in a question, and `map_open` for the T_open answers. The parser counts brackets because some place names have brackets in them, e.g. "Gowanus Expressway (HOV)".

## Folders

```
lit/          paper, proposal, pitch slides
src/          the scripts
configs/      training settings for llamafactory
paraphrases/  the T_para paraphrases and the hand check sheet
data/         dataset (not in git)
data/built/   generated files (not in git)
```

## Data and license

The questions come from CityGPT-Data (Feng et al., KDD '25, https://huggingface.co/datasets/Tianhui-Liu/CityGPT-Data), which is under the MIT license, like the CityGPT code (https://github.com/tsinghua-fib-lab/CityGPT). The paraphrases in `paraphrases/` are derived from those questions, and the parts of `src/score.py` marked "code from paper gh" come from the CityGPT repo. If you use any of it, cite the paper.
