"""
Translate the instruction and the response with a translation model
"""

import argparse
import json
import os
import pdb
import sys
from copy import deepcopy
from typing import List

sys.path.append("src")

import openai
import tiktoken
from tqdm import tqdm
from utils.gpt_utils import chatgpt_single_turn_inference

LANG2TEMPLATE = {
    "zh": "将下列英文翻译成中文。保留<|user|>和<|assistant|>这两个特殊字符。\n\n<|user|> {instruction} \n\n<|assistant|> {response}",
    "ko": "다음 텍스트를 한글로 번역하시오. 두 개의 특수 문자 <|user|> 와 <|assistant|> 를 그대로 두시오.\n\n<|user|> {instruction} \n\n<|assistant|> {response}",
    "es": "Traduce el siguiente texto al español. Mantén dos tokens especiales <|user|> y <|assistant|> tal como están.\n\n<|user|> {instruction} \n\n<|assistant|> {response}",
    "it": "Traduci il seguente testo dall'inglese all'italiano. Mantieni le due parole speciali <|user|> e <|assistant|> così come sono.\n\n<|user|> {instruction} \n\n<|assistant|> {response}",
}
INSTRUCTION_PREFIX = "<|user|>"
RESPONSE_PREFIX = "<|assistant|>"
MAX_LENGTH = 2048 - 10


def mean(array):
    return sum(array) / len(array)


def translate(messages, model, tokenizer):
    assert len(messages) == 1
    token_ids = tokenizer.encode(messages[0]["content"])
    max_length = MAX_LENGTH
    if len(token_ids) > max_length:
        print(f"Warning: Truncate input length {len(token_ids)} --> {max_length}!")
        token_ids = token_ids[:max_length]
        messages[0]["content"] = tokenizer.decode(token_ids)

    translation = chatgpt_single_turn_inference(
        messages=messages,
        model=model,
        max_tokens=2048,
        num_return=1,
        temperature=0.0,
        top_p=0.0,
        stop=None,
        timeout=180,
        sleep=2,
    )

    return translation


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-input_path",
        type=str,
        default="train_english.json",
    )
    parser.add_argument(
        "-output_path",
        type=str,
        default="train_zh_all.json",
    )
    parser.add_argument(
        "-target_lang", type=str, choices=["zh", "ko", "es", "it"], default="zh"
    )
    parser.add_argument("-model", type=str, default="gpt-3.5-turbo")
    parser.add_argument("-api_key", required=True, type=str)
    args = parser.parse_args()

    input_path = args.input_path
    output_path = args.output_path

    data = json.load(open(input_path, "r", encoding="utf8"))

    print(f"{len(data)} total examples to finish!")

    openai.api_key = args.api_key
    tokenizer = tiktoken.encoding_for_model(args.model)

    translations = []

    if os.path.exists(output_path):
        translations = json.load(open(output_path, "r", encoding="utf8"))
        print(f"Loaded {len(translations)} translations from {output_path}")
        finished_ids = [ex["id"] for ex in translations]
        data = [
            ex for ex in data if ex["id"] not in finished_ids
        ]  # get the remaining examples

    print(f"Remaining examples to translate: {len(data)}")
    special_token_error = 0

    for example in tqdm(data, desc="Translating"):
        translated_example = {}

        messages = [
            {"role": "user", "content": deepcopy(LANG2TEMPLATE[args.target_lang])}
        ]
        messages[0]["content"] = messages[0]["content"].format(
            instruction=example["instruction"], response=example["response"]
        )

        translation = translate(messages, args.model, tokenizer)[0]
        try:
            translated_instruction_idx = translation.index(INSTRUCTION_PREFIX)
            translated_response_idx = translation.index(RESPONSE_PREFIX)
        except ValueError:
            special_token_error += 1
            continue
        translated_instruction = translation[
            translated_instruction_idx
            + len(INSTRUCTION_PREFIX) : translated_response_idx
        ].strip()
        translated_response = translation[
            translated_response_idx + len(RESPONSE_PREFIX) :
        ].strip()
        translated_example = {
            "id": example["id"],
            "instruction": translated_instruction,
            "response": translated_response,
        }

        translations.append(translated_example)

        # save results to a temp file every 10 iterations
        if len(translations) % 10 == 0:
            json.dump(
                translations,
                open(output_path, "w", encoding="utf8"),
                ensure_ascii=False,
                indent=4,
            )

    # sort examples based on id
    translations = sorted(translations, key=lambda x: x["id"])
    json.dump(
        translations,
        open(output_path, "w", encoding="utf8"),
        ensure_ascii=False,
        indent=4,
    )
    print(
        f"Saved {len(translations)} examples! {special_token_error} examples have special token error!"
    )


if __name__ == "__main__":
    main()
