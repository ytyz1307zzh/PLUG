# Instruction Tuning & Inference

This directory contains code for model training and inference. Training and inference commands are shown in `run.sh`

The training and inference code are largely adapted from [TULU](https://github.com/allenai/open-instruct). However, during our experiments, we **haven't tested nor used** the parts of code of (1) LoRA training, and (2) resuming training from a checkpoint. So make sure you test them if you would like to utilize these features (the commands in `run.sh` don't use them).
