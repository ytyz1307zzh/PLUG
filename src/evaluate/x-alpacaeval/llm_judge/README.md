# Code for X-AlpacaEval evaluation

The `fastchat` directory and `llm_judge.py` are adopted from [MT-Bench](https://github.com/lm-sys/FastChat) with neglectable modifications.

To run evaluation on X-AlpacaEval, use the following commands:

```bash
API_KEY=YOUR_OPENAI_API_KEY
CKPT_DIR_1=PATH_TO_BASELINE_CHECKPOINT
CKPT_DIR_2=PATH_TO_PLUG_CHECKPOINT

# --instruction_file is the JSON-formatted X-AlpacaEval data in the target language
# The file contains a list of instances, each in the following format:
# {
#     "id": 1,
#     "dataset": "helpful_base",
#     "system": "Please interpret the instruction in English, and then respond both in English and in Chinese.",
#     "instruction": "有哪些知名演员在百老汇开启职业生涯？",
# }
# For the monolingual baseline, the system prompt is "Please respond to the following user message in Chinese."

python src/evaluate/x-alpacaeval/llm_judge/llm_judge.py \
    --instruction_file data/x-alpacaeval/chinese.json \
    --answer_file \
    ${CKPT_DIR_1}/baseline-output-alpacaeval-zh.json \
    ${CKPT_DIR_2}/plug-output-alpacaeval-zh-extracted.json \
    --api_key ${API_KEY} \
    --judge_file src/evaluate/x-alpacaeval/llm_judge/fastchat/llm_judge/data/judge_prompts_focus_language.jsonl \
    --output_path ${CKPT_DIR_2}/gpt4-eval-compare-plug-baseline-alpacaeval-zh.json \
    --judge_model gpt-4-0613 \
    --mode pair \
    --continue_output

python src/evaluate/x-alpacaeval/llm_judge/calculate_pair_judge_score.py \
    -data data/x-alpacaeval/chinese.json \
    -results ${CKPT_DIR_2}/gpt4-eval-compare-plug-baseline-alpacaeval-zh.json
```
