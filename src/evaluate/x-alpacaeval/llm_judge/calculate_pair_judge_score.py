import argparse
import json
import pdb


def read_jsonl_as_list(path: str):
    with open(path, "r", encoding="utf8") as fin:
        result = []
        for line in fin:
            data = json.loads(line.strip())
            result.append(data)
    print(f"Read {len(result)} data from {path}")
    return result


def mean(array):
    return sum(array) / len(array)


def count_results(result_list, header=""):
    total_results = len(result_list)
    model_1_wins = result_list.count("model_1")
    model_2_wins = result_list.count("model_2")
    slight_model_1_wins = result_list.count("slight_model_1")
    slight_model_2_wins = result_list.count("slight_model_2")
    ties = result_list.count("tie")
    errors = result_list.count("error")
    print(f"{header}:")
    print(
        f"Model 1 wins: {model_1_wins} ({model_1_wins / total_results:.1%}), Model 2 wins: {model_2_wins} ({model_2_wins / total_results:.1%}), slight Model 1 wins: {slight_model_1_wins} ({slight_model_1_wins / total_results:.1%}), slight Model 2 wins: {slight_model_2_wins} ({slight_model_2_wins / total_results:.1%}), Ties: {ties} ({ties / total_results:.1%}), Errors: {errors} ({errors / total_results:.1%})"
    )
    total_model_1_wins = model_1_wins + slight_model_1_wins
    total_model_2_wins = model_2_wins + slight_model_2_wins
    print(
        f"Total model 1 wins: {total_model_1_wins} ({total_model_1_wins / total_results:.1%}), Total model 2 wins: {total_model_2_wins} ({total_model_2_wins / total_results:.1%}), Diff (abs): {abs(total_model_1_wins - total_model_2_wins)} ({abs(total_model_1_wins - total_model_2_wins) / total_results:.1%})"
    )


parser = argparse.ArgumentParser()
parser.add_argument("-data", type=str, default="Original instruction file")
parser.add_argument("-results", type=str, help="GPT-4 eval result file")
args = parser.parse_args()

data = json.load(open(args.data, "r", encoding="utf-8"))
id2original = {x["id"]: x for x in data}
try:
    results = json.load(open(args.results, "r", encoding="utf-8"))
except:
    results = read_jsonl_as_list(args.results)

# skip the prompt element if exists
if "question_id" not in results[0]:
    results = results[1:]

skip_cnt = 0
scores = {"all": [], "datasets": {}, "categories": {}}
for i, example in enumerate(results):

    id_ = example["question_id"]
    try:
        original_data = id2original[id_]
    except KeyError:
        skip_cnt += 1
        continue

    model_1 = example["model_1"]
    model_2 = example["model_2"]

    g1_winner = example["g1_winner"]
    g2_winner = example["g2_winner"]
    winner_set = set([g1_winner, g2_winner])

    result = None
    if winner_set == {"model_1", "model_2"}:
        result = "tie"
    elif winner_set == {"model_1"}:
        result = "model_1"
    elif winner_set == {"model_2"}:
        result = "model_2"
    elif winner_set == {"tie", "model_1"}:
        result = "slight_model_1"
    elif winner_set == {"tie", "model_2"}:
        result = "slight_model_2"
    elif winner_set == {"tie"}:
        result = "tie"
    elif winner_set == {"error", "model_1"}:
        result = "error"
    elif winner_set == {"error", "model_2"}:
        result = "error"
    elif winner_set == {"error", "tie"}:
        result = "error"
    elif winner_set == {"error"}:
        result = "error"
    else:
        raise ValueError(f"Invalid winner set: {winner_set}")

    scores["all"].append(result)

    dataset = original_data["dataset"]
    category = original_data["category"]

    if dataset not in scores["datasets"]:
        scores["datasets"][dataset] = []
    scores["datasets"][dataset].append(result)

    if category not in scores["categories"]:
        scores["categories"][category] = []
    scores["categories"][category].append(result)


print(f"Model 1: {model_1}")
print(f"Model 2: {model_2}\n")
count_results(scores["all"], header="All")

for dataset in scores["datasets"]:
    count_results(scores["datasets"][dataset], header=f"Dataset: {dataset}")

for category in scores["categories"]:
    count_results(scores["categories"][category], header=f"Category: {category}")

print(f"Skipped {skip_cnt} examples due to not found in original data file")
