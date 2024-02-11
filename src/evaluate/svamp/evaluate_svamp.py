import argparse
import json
import math
import re

INVALID_ANS = "[invalid]"
ANS_RE = re.compile(r"(\-?\d+(,\d{3})*)")


def extract_answer(completion):
    match = re.findall(ANS_RE, completion)
    if match:
        match_str = match[-1][0].replace(",", "")
        return float(match_str)
    else:
        return INVALID_ANS


def is_correct(completion: str, answer: str):
    gold = answer
    prediction = extract_answer(completion)
    assert gold != INVALID_ANS, "No ground truth answer found in the document."

    if prediction == INVALID_ANS:
        # print(f"No predicted answer found in {completion}.")
        return False

    return math.isclose(float(prediction), float(gold))


def mean(array):
    return sum(array) / len(array)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-data", type=str, required=True, help="Path to the file with input and answer."
    )
    parser.add_argument(
        "-prediction",
        type=str,
        required=True,
        help="Path to the file with model generations.",
    )
    parser.add_argument(
        "-output", type=str, default=None, help="Path to the output file."
    )
    args = parser.parse_args()

    data = json.load(open(args.data, "r", encoding="utf-8"))
    print(f"Loaded {len(data)} examples from {args.data}.")
    id2answer = {ex["id"]: ex["answer"] for ex in data}

    predictions = json.load(open(args.prediction, "r", encoding="utf-8"))
    print(f"Loaded {len(predictions)} predictions from {args.prediction}.")

    results = []
    for example in predictions:
        id_ = example["id"]
        
        # If we already used ChatGPT to extract the answer, use the extracted answer
        if "pred_answer" in example:
            raw_output = example["output"]
            output = example["pred_answer"]
        else:
            raw_output = example["output"]
            output = example["output"]

        answer = id2answer[id_]
        accuracy = int(is_correct(completion=output, answer=answer))
        results.append(
            {
                "id": id_,
                "instruction": example["instruction"],
                "output": raw_output,
                "chatgpt_extracted_answer": output if "pred_answer" in example else None,
                "pred_answer": extract_answer(output),
                "answer": answer,
                "accuracy": accuracy,
            }
        )

    mean_acc = mean([ex["accuracy"] for ex in results])
    print(f"Mean accuracy: {mean_acc:.2%}")

    if args.output is not None:
        json.dump(
            results,
            open(args.output, "w", encoding="utf-8"),
            indent=4,
            ensure_ascii=False,
        )
        print(f"Saved {len(results)} results to {args.output}.")


if __name__ == "__main__":
    main()
