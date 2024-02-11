# Code for X-TruthfulQA evaluation

```bash
API_KEY=YOUR_OPENAI_API_KEY
DATA_DIR=PATH_TO_TRUTHFULQA_DATA
CKPT_DIR=PATH_TO_CHECKPOINT

# Evaluating Truthfulness
python src/evaluate/truthfulqa/evaluate_truthfulqa_gpt4.py \
    -data ${DATA_DIR}/zh/truthfulqa_zh.json \
    -response ${CKPT_DIR}/truthfulqa/truthfulqa-zeroshot-zh-extracted.json \
    -prompt src/evaluate/truthfulqa/gpt4_evaluate_prompt.json \
    -output ${CKPT_DIR}/truthfulqa/truth-gpt4-eval-truthfulqa-zeroshot-zh.json \
    -evaluate truthful \
    -model gpt-4-0613 \
    -max_tokens 1024 \
    -timeout 60 \
    -sleep 2 \
    -api_key ${API_KEY}

# Evaluating Informativeness
python src/evaluate/truthfulqa/evaluate_truthfulqa_gpt4.py \
    -data ${DATA_DIR}/zh/truthfulqa_zh.json \
    -response ${CKPT_DIR}/truthfulqa/truthfulqa-zeroshot-zh-extracted.json \
    -prompt src/evaluate/truthfulqa/gpt4_evaluate_prompt.json \
    -output ${CKPT_DIR}/truthfulqa/info-gpt4-eval-truthfulqa-zeroshot-zh.json \
    -evaluate informative \
    -model gpt-4-0613 \
    -max_tokens 1024 \
    -timeout 60 \
    -sleep 2 \
    -api_key ${API_KEY}

# Combine Truthful and Informativeness evaluation
python src/evaluate/truthfulqa/combine_truthful_informative.py \
    -truth_data ${CKPT_DIR}/truthfulqa/truth-gpt4-eval-truthfulqa-zeroshot-zh.json \
    -info_data ${CKPT_DIR}/truthfulqa/info-gpt4-eval-truthfulqa-zeroshot-zh.json
```