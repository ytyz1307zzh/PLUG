# Code for X-SVAMP Evaluation

First use `chatgpt_answer_extraction.py` to extract the predicted answer from the generated chain-of-thought response. Here we use `gpt-3.5-turbo` since this is a relatively easy task. Then, use `evaluate_svamp.py` to compare the predicted answer with the ground-truth answer.

```bash
API_KEY=YOUR_OPENAI_API_KEY
DATA_DIR=PATH_TO_SVAMP_DATA
CKPT_DIR=PATH_TO_CHECKPOINT

python src/evaluate/svamp/chatgpt_answer_extraction.py \
    -prediction ${CKPT_DIR}/svamp/svamp-zeroshot-zh-extracted.json \
    -output ${CKPT_DIR}/svamp/chatgpt-extract-answer-svamp-zeroshot-zh.json \
    -model gpt-3.5-turbo \
    -api_key ${API_KEY}

python src/evaluate/svamp/evaluate_svamp.py \
    -data ${DATA_DIR}/zh/svamp_zh.json \
    -prediction ${CKPT_DIR}/svamp/chatgpt-extract-answer-svamp-zeroshot-zh.json \
    -output ${CKPT_DIR}/svamp/chatgpt-eval-svamp-zeroshot-zh.json
```

You can also directly use `evaluate_svamp.py` to evaluate the whole chain-of-thought response (without ChatGPT answer extraction), and if this is the case, the script will use regular expressions to extract the answer from the response.
