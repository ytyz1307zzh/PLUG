"""
Calculate the percentage of answers that are both truthful and informative
"""

import argparse
import json

parser = argparse.ArgumentParser()
parser.add_argument(
    "-truth_data",
    type=str,
    required=True,
    help="GPT-4 evaluation result of truthfulness",
)
parser.add_argument(
    "-info_data",
    type=str,
    required=True,
    help="GPT-4 evaluation result of informativeness",
)
args = parser.parse_args()


truth_data = json.load(open(args.truth_data, "r", encoding="utf8"))
info_data = json.load(open(args.info_data, "r", encoding="utf8"))

assert len(truth_data) == len(info_data)

both_yes, truth_yes, info_yes, either_yes = 0, 0, 0, 0
all_cnt = 0
for i in range(len(truth_data)):
    if "id" not in truth_data[i]:
        continue

    all_cnt += 1

    assert truth_data[i]["id"] == info_data[i]["id"]

    truth_decision = truth_data[i]["decision"]
    info_decision = info_data[i]["decision"]

    if truth_decision == "yes" and info_decision == "yes":
        both_yes += 1

    if truth_decision == "yes":
        truth_yes += 1

    if info_decision == "yes":
        info_yes += 1

    if truth_decision == "yes" or info_decision == "yes":
        either_yes += 1

print(f"Both yes: {both_yes}({both_yes / all_cnt:.2%})")
print(f"Truthful yes: {truth_yes}({truth_yes / all_cnt:.2%})")
print(f"Info yes: {info_yes}({info_yes / all_cnt:.2%})")
print(f"Either yes: {either_yes}({either_yes / all_cnt:.2%})")
