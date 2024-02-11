"""
Extract the desired response from the bilingual output.
"""

import argparse
import json

LANG2PROMPT = {
    "en": "English response:",
    "zh": "中文回复：",
    "it": "Risposta in italiano:",
    "es": "Respuesta en español:",
    "ko": "한국어 응답:",
}


def extract_first_response(output, target_lang, other_lang):
    """
    Extract the first response in the output (total 2 responses).
    """
    target_prompt = LANG2PROMPT[target_lang]
    other_prompt = LANG2PROMPT[other_lang]

    if target_prompt not in output:
        return "", "not found"

    response = output[
        output.index(target_prompt) + len(target_prompt) :
    ].strip()  # remove the prompt

    if other_prompt not in response:
        return response, "monolingual"

    response = response[: response.index(other_prompt)].strip()
    return response, "ok"


def extract_second_response(output, target_lang, other_lang):
    """
    Extract the first response in the output (total 2 responses).
    """
    target_prompt = LANG2PROMPT[target_lang]
    other_prompt = LANG2PROMPT[other_lang]

    if target_prompt not in output:
        return "", "not found"

    response = output[
        output.index(target_prompt) + len(target_prompt) :
    ].strip()  # remove the prompt
    return response, "ok"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-data", type=str, required=True)
    parser.add_argument(
        "-target_lang",
        type=str,
        choices=["en", "zh", "ko", "es", "it"],
        help="The language of the desired response.",
    )
    parser.add_argument(
        "-other_lang",
        type=str,
        choices=["en", "zh", "ko", "es", "it"],
        help="The language of the other response that is co-generated in the same iteration.",
    )
    parser.add_argument(
        "-order", type=str, choices=["target_first", "other_first"], required=True
    )
    parser.add_argument("-output", type=str, required=True)
    args = parser.parse_args()

    data = json.load(open(args.data, "r", encoding="utf-8"))
    print(f"Read {len(data)} examples from {args.data}")

    results = []
    not_found_cnt, monolingual_cnt = 0, 0
    for example in data:
        output = example["output"]

        if args.order == "target_first":
            response, code = extract_first_response(
                output, args.target_lang, args.other_lang
            )
            if code == "not found":
                not_found_cnt += 1
            elif code == "monolingual":
                monolingual_cnt += 1

        elif args.order == "other_first":
            response, code = extract_second_response(
                output, args.target_lang, args.other_lang
            )
            if code == "not found":
                not_found_cnt += 1
            elif code == "monolingual":
                monolingual_cnt += 1

        else:
            raise NotImplementedError

        example["raw_output"] = output  # also save the original output
        example["output"] = response

        results.append(example)

    json.dump(
        results, open(args.output, "w", encoding="utf-8"), indent=4, ensure_ascii=False
    )
    print(f"Wrote {len(results)} examples to {args.output}")
    print(f"not found: {not_found_cnt}, monolingual: {monolingual_cnt}")


if __name__ == "__main__":
    main()
