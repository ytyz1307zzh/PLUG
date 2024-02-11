
import argparse
import json
import os
import pdb
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from typing import List

import shortuuid

# add the current directory to the python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import openai
from fastchat.llm_judge.common import (
    Judge,
    MatchPair,
    MatchSingle,
    load_judge_prompts,
    play_a_match_pair,
    play_a_match_reference,
    play_a_match_single,
)
from tqdm import tqdm

NEED_REF_CATS = ["math", "coding"]


def read_jsonl_as_list(path: str):
    with open(path, "r", encoding="utf8") as fin:
        result = []
        for line in fin:
            data = json.loads(line.strip())
            result.append(data)
    return result


def save_list_as_jsonl(path: str, data: List):
    with open(path, "w", encoding="utf8") as fout:
        for instance in data:
            fout.write(json.dumps(instance, ensure_ascii=False))
            fout.write("\n")
    print(f"Saved {len(data)} data to {path}")


def read_question_file(question_file):
    question_list = json.load(open(question_file, "r", encoding="utf8"))

    results = []
    for question in question_list:
        results.append(
            {
                "question_id": question["id"],
                "category": question["category"]
                if "category" in question
                else "default",
                "turns": [question["instruction"]],
            }
        )

    return results


def read_answer_file(answer_file, model_id: str):
    answer_list = json.load(open(answer_file, "r", encoding="utf8"))

    results = {}
    for answer in answer_list:
        if "output" in answer and "response" in answer:
            raise ValueError(
                "Answer file cannot contain both `output` and `response` fields."
            )
        generation = answer["output"] if "output" in answer else answer["response"]
        results[answer["id"]] = {
            "question_id": answer["id"],
            "answer_id": shortuuid.uuid(),
            "model_id": model_id,
            "choices": [{"index": 0, "turns": [generation]}],
            "tstamp": time.time(),
        }

    return results


def make_match(
    questions,
    model_answers,
    judge,
    ref_answers=None,
    multi_turn=False,
):
    matches = []
    for q in questions:
        if multi_turn and len(q["turns"]) != 2:
            continue
        q_id = q["question_id"]
        assert len(model_answers) == 2
        m_1 = list(model_answers.keys())[0]
        m_2 = list(model_answers.keys())[1]

        a_1 = model_answers[m_1][q_id]
        a_2 = model_answers[m_2][q_id]
        if ref_answers is not None:
            ref = ref_answers[q_id]
            match = MatchPair(
                dict(q),
                m_1,
                m_2,
                a_1,
                a_2,
                judge,
                ref_answer=ref,
                multi_turn=multi_turn,
            )
        else:
            match = MatchPair(dict(q), m_1, m_2, a_1, a_2, judge, multi_turn=multi_turn)
        matches.append(match)
    return matches


def make_match_single(
    questions,
    model_answers,
    judge,
    ref_answers=None,
    multi_turn=False,
):
    matches = []
    for q in questions:
        if multi_turn and len(q["turns"]) != 2:
            continue
        q_id = q["question_id"]

        assert len(model_answers) == 1
        m = list(model_answers.keys())[0]

        a = model_answers[m][q_id]
        if ref_answers is not None:
            ref = ref_answers[q_id]
            matches.append(
                MatchSingle(dict(q), m, a, judge, ref_answer=ref, multi_turn=multi_turn)
            )
        else:
            matches.append(MatchSingle(dict(q), m, a, judge, multi_turn=multi_turn))
    return matches


def make_judge_pairwise(judge_model, judge_prompts):
    judges = {}
    judges["default"] = Judge(judge_model, judge_prompts["pair-v2"])
    judges["math"] = Judge(judge_model, judge_prompts["pair-math-v1"], ref_based=True)
    judges["default-mt"] = Judge(
        judge_model, judge_prompts["pair-v2-multi-turn"], multi_turn=True
    )
    judges["math-mt"] = Judge(
        judge_model,
        judge_prompts["pair-math-v1-multi-turn"],
        ref_based=True,
        multi_turn=True,
    )
    return judges


def make_judge_single(judge_model, judge_prompts):
    judges = {}
    judges["default"] = Judge(judge_model, judge_prompts["single-v1"])
    judges["math"] = Judge(judge_model, judge_prompts["single-math-v1"], ref_based=True)
    judges["default-mt"] = Judge(
        judge_model, judge_prompts["single-v1-multi-turn"], multi_turn=True
    )
    judges["math-mt"] = Judge(
        judge_model,
        judge_prompts["single-math-v1-multi-turn"],
        ref_based=True,
        multi_turn=True,
    )
    return judges


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--instruction_file",
        type=str,
        required=True,
        help="The path to the instructions",
    )
    parser.add_argument(
        "--answer_file",
        type=str,
        required=True,
        nargs="+",
        help="The path to the model answers (can be one or two)",
    )
    parser.add_argument(
        "--ref_answer_file",
        type=str,
        default=None,
        help="File with GPT-4 reference answers of math and coding problems",
    )
    parser.add_argument(
        "--judge_file",
        type=str,
        default="data/judge_prompts.jsonl",
        help="The file of judge prompts.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        help="The path to the output file",
    )
    parser.add_argument(
        "--overwrite_output",
        action="store_true",
        help="Whether to overwrite the output file",
    )
    parser.add_argument(
        "--continue_output",
        action="store_true",
        help="Whether to continue generating from previous results",
    )
    parser.add_argument("--judge_model", type=str, default="gpt-4-0613")
    parser.add_argument(
        "--organization",
        type=str,
        default=None,
        help="The organization of the OpenAI account",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="single",
        choices=["pair", "single", "reference"],
        help=(
            "Evaluation mode. "
            "`pair` runs pairwise comparision against a baseline. "
            "`single` runs single answer grading."
        ),
    )
    parser.add_argument("--api_key", type=str, required=True, help="OpenAI API key")
    parser.add_argument(
        "--parallel", type=int, default=1, help="The number of concurrent API calls."
    )
    parser.add_argument(
        "--first-n", type=int, help="A debug option. Only run the first `n` judgments."
    )
    args = parser.parse_args()
    openai.api_key = args.api_key
    if args.organization is not None:
        openai.organization = args.organization

    question_file = args.instruction_file
    answer_file_list = args.answer_file
    assert len(answer_file_list) in [1, 2]

    # Load questions
    questions = read_question_file(question_file)
    pred_answers = read_answer_file(answer_file_list[0], model_id=answer_file_list[0])
    model_answers = {answer_file_list[0]: pred_answers}
    if len(answer_file_list) == 2:
        baseline_answers = read_answer_file(
            answer_file_list[1], model_id=answer_file_list[1]
        )
        model_answers[answer_file_list[1]] = baseline_answers

    # Load judge
    judge_prompts = load_judge_prompts(args.judge_file)

    if args.first_n:
        questions = questions[: args.first_n]

    if args.mode == "single":
        judges = make_judge_single(args.judge_model, judge_prompts)
        play_a_match_func = play_a_match_single
        output_file = args.output_path
        make_match_func = make_match_single
    elif args.mode == "pair":
        judges = make_judge_pairwise(args.judge_model, judge_prompts)
        play_a_match_func = play_a_match_pair
        output_file = args.output_path
        make_match_func = make_match
    elif args.mode == "reference":
        judges = make_judge_pairwise(args.judge_model, judge_prompts)
        play_a_match_func = play_a_match_reference
        output_file = args.output_path
        make_match_func = make_match

    # check_data(questions, model_answers, ref_answers, models, judges)
    question_math = [q for q in questions if q["category"] in NEED_REF_CATS]
    question_default = [q for q in questions if q["category"] not in NEED_REF_CATS]

    # if no reference answers are provided, then all examples use the default prompt
    if args.ref_answer_file is None:
        question_default = sorted(
            question_default + question_math, key=lambda x: x["question_id"]
        )
        question_math = []
        ref_answers = None
    # if reference answers are provided, use the math prompt for math examples
    else:
        ref_answers = read_jsonl_as_list(args.ref_answer_file)
        ref_answers = {a["question_id"]: a for a in ref_answers}

    # Make matches
    matches = []
    matches += make_match_func(question_default, model_answers, judges["default"])
    matches += make_match_func(
        question_math,
        model_answers,
        judges["math"],
        ref_answers=ref_answers,
    )
    matches += make_match_func(
        question_default,
        model_answers,
        judges["default-mt"],
        multi_turn=True,
    )
    matches += make_match_func(
        question_math,
        model_answers,
        judges["math-mt"],
        ref_answers=ref_answers,
        multi_turn=True,
    )

    match_stat = {}
    match_stat["mode"] = args.mode
    match_stat["judge"] = args.judge_model
    match_stat["total_num_questions"] = len(questions)
    match_stat["total_num_matches"] = len(matches)
    match_stat["output_path"] = output_file

    # Show match stats and prompt enter to continue
    print("Stats:")
    print(json.dumps(match_stat, indent=4))
    # input("Press Enter to confirm...")

    if output_file and os.path.exists(output_file):
        if args.continue_output:
            assert not args.overwrite_output, "Cannot specify both arguments!"
            try:
                old_results = read_jsonl_as_list(output_file)
            except json.decoder.JSONDecodeError:
                # If the output file is already JSON, then it means we continue from a finished previous run
                old_results = json.load(open(output_file, "r", encoding="utf8"))
                # In a finished run, the first element is the prompts
                assert "prompts" in old_results[0].keys()
                old_results = old_results[1:]
                # re-save the file in JSONL, to avoid messing up the format
                save_list_as_jsonl(output_file, old_results)

            old_question_ids = set([x["question_id"] for x in old_results])
            print(
                f"Continue generating from {output_file}, skipping {len(old_question_ids)} old results..."
            )
            matches = [
                m for m in matches if m.question["question_id"] not in old_question_ids
            ]

        elif not args.overwrite_output:
            raise ValueError(
                f"The output file {output_file} already exists. "
                "Please use --overwrite_output to overwrite it."
            )
        else:
            print(f"Overwriting the output file {output_file}...")
            os.remove(output_file)

    # Play matches
    if args.parallel == 1:
        for match in tqdm(matches):
            play_a_match_func(match, output_file=output_file)
    else:

        def play_a_match_wrapper(match):
            play_a_match_func(match, output_file=output_file)

        np.random.seed(0)
        np.random.shuffle(matches)

        with ThreadPoolExecutor(args.parallel) as executor:
            for match in tqdm(
                executor.map(play_a_match_wrapper, matches), total=len(matches)
            ):
                pass

    # Re-format the output file as a JSON file
    results = read_jsonl_as_list(output_file)
    results = sorted(results, key=lambda x: x["question_id"])
    results.insert(0, {"prompts": judge_prompts})
    json.dump(
        results, open(output_file, "w", encoding="utf8"), indent=4, ensure_ascii=False
    )
    print(f"Saved {len(results)} data to {output_file}")
