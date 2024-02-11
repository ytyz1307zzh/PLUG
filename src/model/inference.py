import argparse
import json
import logging
import os
import pdb
import random
from datetime import datetime

import torch
from accelerate import Accelerator
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    GenerationConfig,
    LlamaTokenizer,
    LlamaTokenizerFast,
    XGLMTokenizer,
    XGLMTokenizerFast,
)

random.seed(42)
logger = logging.getLogger(__name__)


def format_input_prompt(instruction, tokenizer, system=None):
    """
    This method must be consistent with encode_with_messages_format in train.py
    """
    prompt = ""

    if system is not None:
        prompt += "<|system|>\n" + system.strip() + "\n"

    prompt += "<|user|>\n" + instruction.strip() + "\n"

    if isinstance(tokenizer, LlamaTokenizer) or isinstance(
        tokenizer, LlamaTokenizerFast
    ):
        prompt += "<|assistant|>\n"
    elif isinstance(tokenizer, XGLMTokenizer) or isinstance(
        tokenizer, XGLMTokenizerFast
    ):
        prompt += "<|assistant|>"

    return prompt


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument(
        "--data",
        type=str,
        required=True,
    )
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument(
        "--precision", type=str, default="bf16", choices=["fp32", "fp16", "bf16"]
    )
    parser.add_argument(
        "--max_input_length",
        type=int,
        default=1536,
        help="Max sequence length for the instruction.",
    )
    parser.add_argument(
        "--max_output_length",
        type=int,
        default=2048,
        help="Max sequence length for generating the response.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Temperature for sampling, 0.0 means greedy decoding",
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.0,
        help="Top-p for sampling, 0.0 means greedy decoding",
    )
    parser.add_argument(
        "--repetition_penalty",
        type=float,
        default=1.0,
        help="Repetition penalty for sampling, 1.0 means no penalty",
    )
    parser.add_argument(
        "--subset",
        type=str,
        default=None,
        help="if specified, use a subset of alpaca_eval",
    )
    parser.add_argument(
        "--num_examples",
        type=int,
        default=None,
        help='If specified, only use the first "num_examples" examples in the dataset.',
    )
    parser.add_argument(
        "--overwrite_output",
        action="store_true",
        help="If specified, overwrite the original output file (if exists).",
    )
    parser.add_argument(
        "--continue_output",
        action="store_true",
        help="If specified, continue writing to the original output file (if exists).",
    )
    parser.add_argument(
        "--pre_formatted",
        action="store_true",
        help="If specified, the input instruction is already formatted. Also, only the 'instruction' field will be used as the input (so if there is a system prompt, it should be inside the 'instruction' field).",
    )
    args = parser.parse_args()

    accelerator = Accelerator()

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if accelerator.is_local_main_process else logging.WARNING,
    )

    logger.info("loading data and model...")
    # load some data
    eval_data = json.load(open(args.data, "r", encoding="utf-8"))

    # select the specified subset
    if args.subset is not None:
        eval_data = [x for x in eval_data if x["dataset"] == args.subset]

    if args.num_examples is not None:
        eval_data = eval_data[: args.num_examples]

    logger.info(f"Total evaluation data: {len(eval_data)}")

    prev_data = None
    if os.path.exists(args.output_path) and not args.overwrite_output:
        if args.continue_output:
            prev_data = json.load(open(args.output_path, "r", encoding="utf-8"))
            prev_data_ids = {x["id"] for x in prev_data}
            logger.warning(
                f"Continue writing to {args.output_path}, which already has {len(prev_data)} examples..."
            )
            eval_data = [x for x in eval_data if x["id"] not in prev_data_ids]
        else:
            logger.warning("File %s already exists, exiting...", args.output_path)
            return

    my_outputs = []

    if args.precision == "fp32":
        precision = torch.float32
    elif args.precision == "fp16":
        precision = torch.float16
    elif args.precision == "bf16":
        precision = torch.bfloat16
    else:
        raise ValueError("Unknown precision %s", args.precision)

    if "polylm" in args.model:
        model = AutoModelForCausalLM.from_pretrained(
            args.model, device_map="auto", torch_dtype=precision, trust_remote_code=True
        )
        tokenizer = AutoTokenizer.from_pretrained(
            args.model, legacy=False, use_fast=False
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.model, device_map="auto", torch_dtype=precision
        )
        tokenizer = AutoTokenizer.from_pretrained(args.model)

    # add padding token if not already there (for Llama models)
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "<pad>"})
        model.resize_token_embeddings(len(tokenizer))

    logger.info("model and data loaded!")
    logger.info("generating...")

    if args.top_p == 0.0 and args.temperature == 0.0:
        do_sample = False
    else:
        do_sample = True

    generation_config = GenerationConfig.from_pretrained(
        args.model,
        max_length=args.max_output_length,
        top_p=args.top_p,
        temperature=args.temperature,
        do_sample=do_sample,
        repetition_penalty=args.repetition_penalty,
    )
    logger.warning(
        f"[{datetime.now().strftime('%H:%M:%S')}] <GPU {accelerator.process_index}> Start generating..."
    )

    random.shuffle(eval_data)

    with accelerator.split_between_processes(eval_data) as eval_data_curr_process:

        dataloader = torch.utils.data.DataLoader(
            eval_data_curr_process, batch_size=args.batch_size, shuffle=False
        )

        with torch.inference_mode():
            for samples in tqdm(dataloader, desc=f"GPU {accelerator.process_index}"):

                if args.pre_formatted:
                    input_texts = [
                        samples["instruction"][j] for j in range(len(samples["id"]))
                    ]
                else:
                    input_texts = [
                        format_input_prompt(
                            samples["instruction"][j],
                            tokenizer,
                            system=samples["system"][j]
                            if "system" in samples
                            else None,
                        )
                        for j in range(len(samples["id"]))
                    ]

                # print the first example
                if len(my_outputs) == 0:
                    logger.info(input_texts[0])

                inputs = tokenizer(
                    input_texts,
                    return_tensors="pt",
                    max_length=args.max_input_length,
                    padding=True,
                    truncation=True,
                )
                input_ids = inputs.input_ids.to(model.device)
                attention_mask = inputs.attention_mask.to(model.device)
                outputs = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    generation_config=generation_config,
                )

                for j in range(len(samples["id"])):
                    output = outputs[j]
                    output_string = tokenizer.decode(
                        output[input_ids.size(1) :], skip_special_tokens=True
                    )
                    my_outputs.append(
                        {
                            "id": samples["id"][j].item(),
                            "category": samples["category"][j]
                            if "category" in samples
                            else "default",
                            "system": samples["system"][j]
                            if "system" in samples
                            else "",
                            "instruction": samples["instruction"][j],
                            "generator": f"{args.model}",
                            "output": output_string.strip(),
                        }
                    )

        output_path_curr_process = args.output_path + f".{accelerator.process_index}"
        json.dump(
            my_outputs,
            open(output_path_curr_process, "w", encoding="utf8"),
            indent=4,
            ensure_ascii=False,
        )

    logger.warning(
        f"[{datetime.now().strftime('%H:%M:%S')}] <GPU {accelerator.process_index}> Finished generation!"
    )

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        # concatenate outputs from all processes
        all_outputs = []
        for i in range(accelerator.num_processes):
            output_path_curr_process = args.output_path + f".{i}"
            all_outputs += json.load(
                open(output_path_curr_process, "r", encoding="utf-8")
            )
            os.remove(output_path_curr_process)

        if prev_data is not None:
            all_outputs += prev_data

        all_outputs = sorted(all_outputs, key=lambda x: x["id"])
        json.dump(
            all_outputs,
            open(args.output_path, "w", encoding="utf8"),
            indent=4,
            ensure_ascii=False,
        )
        print(f"Saved {len(all_outputs)} examples to {args.output_path}.")

        logger.info(all_outputs[0])
        # format should be something like:
        # {'instruction': 'What are the names of some famous actors that started their careers on Broadway?', 'input': '', 'output': 'Some famous actors that started their careers on Broadway are Hugh Jackman, Meryl Streep, Denzel Washington, Audra McDonald, and Lin-Manuel Miranda.', 'generator': 'gpt-3.5-turbo-0301', 'dataset': 'helpful_base', 'datasplit': 'eval'}


if __name__ == "__main__":
    main()
