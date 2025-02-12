# qwen2.5-r1-ft

R1-distilled SFT (Supervised Fine-Tuning) for Qwen 2.5, derived from [Qwen's official fine-tuning example](https://github.com/QwenLM/Qwen/blob/main/recipes/finetune/deepspeed/finetune_lora_multi_gpu.ipynb).

## Overview

This project implements LoRA fine-tuning for the Qwen-2.5-1.5B-Chat model using DeepSpeed with multi-GPU support. The implementation focuses on R1-style training using high-quality instruction datasets.

## Features

- Multi-GPU training support with DeepSpeed
- LoRA fine-tuning implementation
- Support for R1-style instruction datasets
- Special token handling for thought process
- BF16 mixed precision training
- Cosine learning rate scheduling

## Prerequisites

- Python 3.11+ (Recommended for optimal performance)
- Multi-GPU platform recommended for efficient training
- PyTorch with CUDA support
- Transformers
- DeepSpeed
- At least 2 CUDA-capable GPUs (16+ GB VRAM each)

## Supported Datasets

The training supports various R1-style datasets including:
- bespokelabs/Bespoke-Stratos-17k
- bespokelabs/Bespoke-Stratos-35k
- NovaSky-AI/Sky-T1_data_17k
- open-thoughts/OpenThoughts-114k

You can merge multiple datasets for better results and reduced overfitting.

## Setup

1. Download the base model:
```bash
huggingface-cli download "Qwen/Qwen2.5-1.5B-Instruct" --local-dir qwen2.5-1.5b-ins
```

2. Download your chosen dataset:
```bash
huggingface-cli download "bespokelabs/Bespoke-Stratos-17k" --local-dir dataset_stratos_17k --repo-type dataset
```

3. (Optional) Process the dataset to handle special tokens using:
```bash
python simplify_dataset.py
```

## Training

Launch training with DeepSpeed using:

```bash
torchrun --nproc_per_node 2 --nnodes 1 --node_rank 0 --master_addr localhost --master_port 6601 finetune.py \
    --model_name_or_path "qwen2.5-1.5b-ins" \
    --data_path "optimized_dataset" \
    --bf16 True \
    --output_dir "output_qwen" \
    --num_train_epochs 5 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 500 \
    --save_total_limit 3 \
    --learning_rate 1e-5 \
    --weight_decay 0.1 \
    --warmup_ratio 0.01 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1
```

Key training parameters can be adjusted based on your hardware and requirements:
- `nproc_per_node`: Number of GPUs to use
- `per_device_train_batch_size`: Batch size per GPU
- `gradient_accumulation_steps`: Steps before gradient update
- `learning_rate`: Training learning rate
- `num_train_epochs`: Number of training epochs

## Special Token Handling

The model supports special tokens for thought process:
- `<|begin_of_thought|>`
- `<|end_of_thought|>`

These can be added either through the HuggingFace Tokenizer API or by manually editing the tokenizer configuration.

## License

This project follows the licensing terms of the original Qwen model and associated datasets.
