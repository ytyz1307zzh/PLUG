"""
Use GPT-4 to evaluate the response on TruthfulQA.
"""

import argparse
import json
import os
import sys

sys.path.append("src")

import openai
from tqdm import tqdm
from utils.gpt_utils import chatgpt_single_turn_inference


def mean(array):
    return sum(array) / len(array)


def read_jsonl_as_list(path: str):
    with open(path, "r", encoding="utf8") as fin:
        result = []
        for line in fin:
            data = json.loads(line.strip())
            result.append(data)
    return result


def format_answer_list(answer_list):
    answer_string = ""
    for i in range(len(answer_list)):
        answer = answer_list[i]
        answer_string += f"- {answer}\n"

    return answer_string.strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-data", type=str, required=True, help="The file of original data"
    )
    parser.add_argument(
        "-response", type=str, required=True, help="The file of model responses"
    )
    parser.add_argument(
        "-prompt", type=str, required=True, help="Evaluation prompt file"
    )
    parser.add_argument(
        "-output", type=str, default=None, help="The file to save results"
    )
    parser.add_argument(
        "-evaluate",
        type=str,
        choices=["informative", "truthful"],
        required=True,
        help='Which aspect to evaluate? "informativeness" or "truthfulness"?',
    )
    parser.add_argument("-model", type=str, default="gpt-4-0613")
    parser.add_argument("-max_tokens", type=int, default=1024)
    parser.add_argument("-timeout", type=int, default=30)
    parser.add_argument("-sleep", type=int, default=2)
    parser.add_argument("-api_key", type=str, required=True)
    args = parser.parse_args()

    openai.api_key = args.api_key

    prompt_info = json.load(open(args.prompt, "r", encoding="utf8"))
    data_list = json.load(open(args.data, "r", encoding="utf8"))
    total_num_data = len(data_list)
    print(f"Loaded {len(data_list)} examples from {args.data}")
    response_list = json.load(open(args.response, "r", encoding="utf8"))
    print(f"Loaded {len(response_list)} responses from {args.response}")
    assert len(data_list) == len(response_list)

    results = []
    if args.output is not None and os.path.exists(args.output):
        results = json.load(open(args.output, "r", encoding="utf8"))

        # If the results are already finished
        if len(results) >= len(data_list):
            data_list, response_list = [], []

        else:
            old_question_ids = set([x["id"] for x in results])
            print(
                f"Continue generating from {args.output}, skipping {len(old_question_ids)} old results..."
            )
            data_list = [x for x in data_list if x["id"] not in old_question_ids]
            response_list = [
                x for x in response_list if x["id"] not in old_question_ids
            ]

    for i in tqdm(range(len(data_list))):
        assert data_list[i]["id"] == response_list[i]["id"]
        data = data_list[i]
        response_info = response_list[i]

        correct_answers = data["correct_answers"]
        incorrect_answers = data["incorrect_answers"]
        instruction = data["instruction"]
        system = response_info["system"]
        response = response_info["output"]
        generator = response_info["generator"]

        if args.evaluate == "truthful":
            eval_system_prompt = prompt_info["truthfulness"]["system_prompt"]
            prompt_template = prompt_info["truthfulness"]["prompt_template"]
            prompt = prompt_template.format(
                question=instruction,
                correct_answers=format_answer_list(correct_answers),
                incorrect_answers=format_answer_list(incorrect_answers),
                response=response,
            )
        else:
            eval_system_prompt = prompt_info["informativeness"]["system_prompt"]
            prompt_template = prompt_info["informativeness"]["prompt_template"]
            prompt = prompt_template.format(
                question=instruction,
                response=response,
            )

        messages = [
            {"role": "system", "content": eval_system_prompt},
            {"role": "user", "content": prompt},
        ]

        judgment = chatgpt_single_turn_inference(
            messages=messages,
            model=args.model,
            max_tokens=args.max_tokens,
            num_return=1,
            temperature=0.0,
            top_p=0.0,
            stop=None,
            timeout=args.timeout,
            sleep=args.sleep,
        )
        judgment = judgment[0].strip()

        results.append(
            {
                "id": data["id"],
                "system": system,
                "instruction": instruction,
                "correct_answers": correct_answers,
                "incorrect_answers": incorrect_answers,
                "output": response,
                "evaluation": judgment,
                "generator": generator,
            }
        )

        if len(results) % 10 == 0 and args.output is not None:
            json.dump(
                results,
                open(args.output, "w", encoding="utf-8"),
                indent=4,
                ensure_ascii=False,
            )

    # "or" because we add a prompt info element to the end
    assert (
        len(results) == total_num_data or len(results) == total_num_data + 1
    ), f"results: {len(results)}, total_num_data: {total_num_data}"

    if args.evaluate == "truthful":
        truth_cnt, wrong_cnt, unsure_cnt, error_cnt = 0, 0, 0, 0
        for example in results:
            # skip the prompt info
            if "evaluation" not in example:
                continue
            if "[Truthful]" in example["evaluation"]:
                decision = "yes"
                truth_cnt += 1
            elif "[Not Truthful]" in example["evaluation"]:
                decision = "no"
                wrong_cnt += 1
            elif "[Not Enough Info]" in example["evaluation"]:
                decision = "unsure"
                unsure_cnt += 1
            else:
                decision = "error"
                error_cnt += 1
            example["decision"] = decision

        print(f"Truthful: {truth_cnt}({truth_cnt / total_num_data:.2%})")
        print(f"Wrong: {wrong_cnt}({wrong_cnt / total_num_data:.2%})")
        print(f"Unsure: {unsure_cnt}({unsure_cnt / total_num_data:.2%})")
        print(f"Error: {error_cnt}({error_cnt / total_num_data:.2%})")
    else:
        info_cnt, not_info_cnt, unsure_cnt, error_cnt = 0, 0, 0, 0
        for example in results:
            # skip the prompt info
            if "evaluation" not in example:
                continue
            if "[Informative]" in example["evaluation"]:
                decision = "yes"
                info_cnt += 1
            elif "[Not Informative]" in example["evaluation"]:
                decision = "no"
                not_info_cnt += 1
            elif "[Not Sure]" in example["evaluation"]:
                decision = "unsure"
                unsure_cnt += 1
            else:
                decision = "error"
                error_cnt += 1
            example["decision"] = decision

        print(f"Informative: {info_cnt}({info_cnt / total_num_data:.2%})")
        print(f"Not Informative: {not_info_cnt}({not_info_cnt / total_num_data:.2%})")
        print(f"Unsure: {unsure_cnt}({unsure_cnt / total_num_data:.2%})")
        print(f"Error: {error_cnt}({error_cnt / total_num_data:.2%})")

    if args.output is not None:
        results = sorted(results, key=lambda x: x["id"])

        # Add the prompt to the results file (if there isn't one previously)
        if "decision" in results[-1]:
            results.append(prompt_info)

        json.dump(
            results,
            open(args.output, "w", encoding="utf-8"),
            indent=4,
            ensure_ascii=False,
        )
        print(f"Saved {len(results) - 1} results to {args.output}")


if __name__ == "__main__":
    main()
