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
    average_score = mean(result_list)
    print(f"{header}: {average_score:.2f}")


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

    model = example["model"]
    score = example["score"]
    scores["all"].append(score)

    if "dataset" in original_data:
        dataset = original_data["dataset"]
        if dataset not in scores["datasets"]:
            scores["datasets"][dataset] = []
        scores["datasets"][dataset].append(score)

    if "category" in original_data:
        category = original_data["category"]
        if category not in scores["categories"]:
            scores["categories"][category] = []
        scores["categories"][category].append(score)


print(f"Model: {model}")
count_results(scores["all"], header="All")

for dataset in scores["datasets"]:
    count_results(scores["datasets"][dataset], header=f"Dataset: {dataset}")

for category in scores["categories"]:
    count_results(scores["categories"][category], header=f"Category: {category}")

print(f"Skipped {skip_cnt} examples due to not found in original data file")
