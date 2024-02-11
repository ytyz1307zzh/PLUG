NUM_GPUS=8
BATCH_SIZE_PER_GPU=1
GRADIENT_ACC_STEPS=8
LR=5e-6
SEQ_LENGTH=4096
WEIGHT_DECAY=0.0
NUM_EPOCHS=4
WARMUP_RATIO=0.03
SEED=42

# Count the total batch size
TOTAL_BATCH_SIZE=$((${BATCH_SIZE_PER_GPU}*${NUM_GPUS}*${GRADIENT_ACC_STEPS}))

CKPT_NAME=YOUR_CKPT_DIR_PATH
HF_TOKEN_FILE=PATH_WITH_YOUR_HUGGINGFACE_TOKEN
TRAIN_FILE_PATH=YOUR_TRAIN_FILE_PATH  # See the released training data on Github for the format

wandb login YOUR_WANDB_KEY

# ----------------- Training -----------------

# if GPU memory is limited, use src/ds_config/zero3_offloading_accelerate.conf for --deepspeed_config_file

accelerate launch \
    --mixed_precision bf16 \
    --num_machines 1 \
    --num_processes ${NUM_GPUS} \
    --use_deepspeed \
    --deepspeed_config_file src/ds_config/zero3_no_offloading_accelerate.conf \
    --zero3_init_flag true \
    src/model/train.py \
    --train_file ${TRAIN_FILE_PATH} \
    --model_name_or_path meta-llama/Llama-2-13b-hf \
    --max_seq_length ${SEQ_LENGTH} \
    --per_device_train_batch_size ${BATCH_SIZE_PER_GPU} \
    --gradient_accumulation_steps ${GRADIENT_ACC_STEPS} \
    --learning_rate ${LR} \
    --lr_scheduler_type linear \
    --warmup_ratio ${WARMUP_RATIO} \
    --weight_decay ${WEIGHT_DECAY} \
    --num_train_epochs ${NUM_EPOCHS} \
    --gradient_checkpointing \
    --output_dir ${CKPT_NAME} \
    --preprocessing_num_workers 4 \
    --checkpointing_steps epoch \
    --save_at_epoch "4" \
    --logging_steps 1 \
    --with_tracking \
    --report_to wandb \
    --seed ${SEED} \
    --hf_token_file ${HF_TOKEN_FILE}


# ----------------- Inference -----------------

TEST_DATA_PATH=YOUR_TEST_FILE_PATH  # X-AlpacaEval test data file (Can be English or target language)

accelerate launch \
    --num_machines 1 \
    --num_processes ${NUM_GPUS} \
    src/model/inference.py \
    --model ${CKPT_NAME}/epoch_4 \
    --batch_size 1 \
    --data ${TEST_DATA_PATH} \
    --output_path ${CKPT_NAME}/epoch_4/plug-output-alpacaeval-zh.json \  # Use Chinese (zh) as an example
    --precision bf16 \
    --max_input_length 3072 \
    --max_output_length 4096 \
    --temperature 0.0 \
    --top_p 0.0 \
    --repetition_penalty 1.0


# ----------------- Post-Processing -----------------

# Extract the final response in target language from the PLUG output (using Chinese as an example)

python src/model/extract_response_from_bilingual_output.py \
    -data ${OUTPUT_DIR}/plug-output-alpacaeval-zh.json \
    -target_lang zh \
    -other_lang en \
    -order other_first \
    -output ${OUTPUT_DIR}/plug-output-alpacaeval-zh-extracted.json

# Please refer to `evaluate` directory for evaluation instructions
