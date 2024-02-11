"""
Split the translation result into the translated instruction and the translated response.
"""

import argparse
import json
import re

instruction_translate_dict = {
    "zh": "指令",
    "ko": "지시문",
    "it": "Istruzione",
    "es": "Instrucción",
}

response_translate_dict = {"zh": "回复", "ko": "답변", "it": "Risposta", "es": "Respuesta"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-input",
        type=str,
        default="train_chinese_all.json",
        help="Path to the translation results",
    )
    parser.add_argument(
        "-output",
        type=str,
        default="train_chinese.json",
        help="Path to the output file (with instruction and response separated)",
    )
    parser.add_argument(
        "-lang",
        type=str,
        default="zh",
        choices=["zh", "ko", "it", "es"],
        help="Language of the translation results",
    )
    args = parser.parse_args()

    data = json.load(open(args.input, "r", encoding="utf8"))
    print(f"Loaded {len(data)} examples from {args.input}")
    translation_error_cnt, match_error_cnt = 0, 0
    results = []

    for example in data:
        instruction_token = instruction_translate_dict[args.lang]
        response_token = response_translate_dict[args.lang]
        translation = (
            example["translation"][0]
            if isinstance(example["translation"], list)
            else example["translation"]
        )

        if instruction_token not in translation or response_token not in translation:
            translation_error_cnt += 1
            continue

        try:
            match_results = re.match(
                rf"{instruction_token}[:|：](.*){response_token}[:|：](.*)",
                translation,
                re.DOTALL,
            )
            instruction = match_results.group(1).strip()
            response = match_results.group(2).strip()
        except ValueError:
            match_error_cnt += 1
            continue

        results.append(
            {
                "id": example["id"],
                "source": example["source"],
                "instruction": instruction,
                "response": response,
            }
        )

    json.dump(
        results, open(args.output, "w", encoding="utf8"), ensure_ascii=False, indent=4
    )
    print(f"Wrote {len(results)} examples to {args.output}")
    print(f"Missing indicator tokens during translation: {translation_error_cnt}")
    print(f"Cannot find instructions or responses in regex: {match_error_cnt}")


if __name__ == "__main__":
    main()
