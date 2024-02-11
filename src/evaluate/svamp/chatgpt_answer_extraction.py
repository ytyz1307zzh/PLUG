"""
SVAMP evaluation: first use chatgpt to extract the answer (this file), then calculate the accuracy (evaluate_svamp.py)
"""
import argparse
import random
import json
import math
import os
import re
import sys

sys.path.append("src/utils")
import openai
import tiktoken
from gpt_utils import chatgpt_single_turn_inference
from tqdm import tqdm

LENGTH_LIMIT = 3980

GPT_ANSWER_EXTRACT_PROMPT = """Please read the following examples, then extract the answer (arabic numerals) from the model response. If no answer is provided, write "[invalid]".

Question: 弗兰克正在阅读他最喜欢的书。这本书有555页，平均分布在5个章节中。弗兰克花了220天才读完这本书。每个章节有多少页？先逐步思考，然后给出最终的答案。

Model response: 这本书的每个章节大约有111页。\n以下是计算的详细过程：\n555页总共分布在5个章节中，这意味着每个章节的页数等于总页数除以章节数：\n555 ÷ 5 = 111页/章节。

Extracted answer: 111

Question: Dopo che Jessie ha iniziato a fare jogging tutti i giorni, ha perso 126 chilogrammi. Attualmente pesa 66 chilogrammi. Quanto pesava prima di iniziare a fare jogging? Riporta tutti i passaggi prima di rispondere alla domanda.

Model response: Jessie ha iniziato a fare jogging tutti i giorni e ha perso 126 chilogrammi. Ciò significa che in passato pesava 126 chilogrammi in più rispetto a quanto fa ora, ovvero 192 chilogrammi. Attualmente pesa 66 chilogrammi.

Extracted answer: 192

Question: 에드워드는 2권의 책을 각각 같은 가격에 구매하기 위해 6달러를 사용했습니다. 이제 그는 12달러를 가지고 있습니다. 각 책은 얼마에 팔렸을까요? 최종 답에 도달하기 위해 순차적으로 생각하세요.

Model response: 각 책이 3달러라고 가정해 봅시다.\n수학을 해보겠습니다:\n책을 사기 위해 사용한 돈 = 구매한 책 2권\n같은 가격으로 구매한 책 = 1권의 가격 * 2\n1권의 가격 * 2 = 6\n1권의 가격 = 3\n각 책은 3달러입니다.\n그러므로, 에드워드는 3달러씩 두 권의 책을 6달러로 구매했습니다. 이제 그는 12달러를 가지고 있습니다.

Extracted answer: 3

Question: Allan trajo 5 globos y Jake trajo 3 globos al parque. ¿Cuántos globos más tenía Allan que Jake en el parque? Describe tu proceso de pensamiento paso a paso antes de llegar a la respuesta final.

Model response: Para determinar cuántos globos más tenía Allan que Jake en el parque, necesitamos comparar la cantidad de globos que cada uno de ellos trajo. Según la información dada, Allan trajo 5 globos y Jake trajo 3 globos.\nEntonces, Allan tenía 2 globos más que Jake en el parque. Para resumir:\nAllan trajo 5 globos y Jake trajo 3 globos. La diferencia entre los dos es 2, con Allan teniendo más globos.

Extracted answer: 2

Question: 朱莉娅周一和7个孩子玩捉迷藏，周二和13个孩子玩捉迷藏。她周三和20个孩子玩扑克牌。她总共和多少个孩子玩捉迷藏？先逐步思考，然后给出最终的答案。

Model response: 为了确定朱莉娅总共和多少个孩子玩捉迷藏，我们需要知道她周一和周二玩的孩子数量，以及周三玩扑克牌的孩子数量。\n然而，提供的信息只说她周一和周二分别和7个和13个孩子玩捉迷藏，但没有提到她周三玩扑克牌的孩子数量。因此，没有这个信息，无法确定朱莉娅总共和多少个孩子玩捉迷藏。

Extracted answer: [invalid]

Question: {question}

Model response: {response}

Extracted answer:"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-prediction",
        type=str,
        required=True,
        help="Path to the file with model generations.",
    )
    parser.add_argument(
        "-output", type=str, required=True, help="Path to the output file."
    )
    parser.add_argument("-model", type=str, default="gpt-3.5-turbo")
    parser.add_argument("-api_key", type=str, required=True, help="OpenAI API key.")
    args = parser.parse_args()
    openai.api_key = args.api_key
    tokenizer = tiktoken.encoding_for_model(args.model)

    data = json.load(open(args.prediction, "r", encoding="utf-8"))
    print(f"Loaded {len(data)} examples from {args.prediction}.")

    results = []
    if os.path.exists(args.output):
        old_results = json.load(open(args.output, "r", encoding="utf-8"))
        old_ids = set([x["id"] for x in old_results])
        data = [x for x in data if x["id"] not in old_ids]
        print(f"Loaded {len(old_results)} old results, continue with the remaining {len(data)} examples...")
        results = old_results

    for example in tqdm(data):
        id_ = example["id"]
        question = example["instruction"]
        response = example["output"]

        # replace multiple linebreaks with a single one
        question = re.sub(r"\n+", "\n", question)
        response = re.sub(r"\n+", "\n", response)

        prompt = GPT_ANSWER_EXTRACT_PROMPT.format(question=question, response=response)

        # Long prompts are all broken generations
        prompt_tokens = tokenizer.encode(prompt)
        if len(prompt_tokens) > LENGTH_LIMIT:
            prompt = tokenizer.decode(prompt_tokens[:LENGTH_LIMIT])

        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ]

        extracted_answer = chatgpt_single_turn_inference(
            messages=messages,
            model=args.model,
            max_tokens=20,
            temperature=0,
            top_p=0,
            timeout=30,
            sleep=2,
        )
        extracted_answer = extracted_answer[0].strip()

        results.append(
            {
                "id": id_,
                "instruction": question,
                "output": response,
                "pred_answer": extracted_answer,
            }
        )

        if len(results) % 10 == 0:
            json.dump(
                results,
                open(args.output, "w", encoding="utf-8"),
                indent=4,
                ensure_ascii=False,
            )

    json.dump(
        results,
        open(args.output, "w", encoding="utf-8"),
        indent=4,
        ensure_ascii=False,
    )
    print(f"Saved {len(results)} results to {args.output}.")


if __name__ == "__main__":
    main()
