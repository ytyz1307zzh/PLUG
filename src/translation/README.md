# Training Data Translation

This directory contains code that translates the training data (instructions & responses) into the target language with ChatGPT.

1. Translate with `translate_together_chatgpt.py`. This script concatenates the instruction and response, and then translates them as a single sequence. The input JSON file should contain a list of training instances, each of which contains at least the following fields: `id`, `instruction`, and `response`.
```bash
python src/translation/translate_together_chatgpt.py \
    -input_path train_english.json \
    -output_path train_chinese_all.json \
    -target_lang zh \
    -model gpt-3.5-turbo \
    -api_key YOUR_OPENAI_KEY
```
2. Split the translated sequence into the translated instruction and the translated response with `split_translated_instruction_response.py`
```bash
python src/translation/split_translated_instruction_response.py \
    -input train_chinese_all.json \
    -output train_chinese.json \
    -lang zh
```
