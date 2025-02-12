# qwen2.5-r1-ft

R1-distilled SFT (Supervised Fine-Tuning) for Qwen 2.5, derived from [Qwen's official fine-tuning example](https://github.com/QwenLM/Qwen/blob/main/recipes/finetune/deepspeed/finetune_lora_multi_gpu.ipynb).

> ⚠️ **Important Note**: The original Qwen repository's code is **not compatible** with Qwen 2.5 models. This project includes necessary modifications to support Qwen 2.5. Do not use the original repository directly for Qwen 2.5 fine-tuning, as it may lead to errors or unexpected behavior.

## Overview

This project implements LoRA fine-tuning for the Qwen-2.5-1.5B-Chat model using DeepSpeed with multi-GPU support. Tongyi Qianwen is a large language model developed by Alibaba Cloud based on the Transformer architecture, trained on diverse pre-training data including internet text, specialized books, and code. This implementation focuses on R1-style training using high-quality instruction datasets.

## Features

- Multi-GPU training support with DeepSpeed
- LoRA fine-tuning implementation
- Support for R1-style instruction datasets
- Special token handling for thought process
- BF16 mixed precision training
- Cosine learning rate scheduling
- Gradient accumulation for effective batch size control
- Checkpoint saving and management
- Warm-up ratio optimization

## Model Architecture

The base model used is Qwen2.5-1.5B-Instruct, which provides:
- Strong base capabilities from extensive pre-training
- Good balance between model size and performance
- Built-in chat/instruction following capabilities
- Support for context window up to 8192 tokens

## Supported Datasets

The training supports various R1-style datasets including:
- bespokelabs/Bespoke-Stratos-17k
- bespokelabs/Bespoke-Stratos-35k
- NovaSky-AI/Sky-T1_data_17k
- open-thoughts/OpenThoughts-114k

You can merge multiple datasets for better results and reduced overfitting. The datasets are designed to enhance:
- Instruction following capabilities
- Reasoning and thought process
- Response quality and coherence
- Task-specific performance

## Training Guide

For detailed training instructions and configurations, please refer to the Jupyter notebook:
`finetune_lora_multi_gpu.ipynb`

The notebook provides comprehensive guidance on:
- Model and dataset preparation
- Special token handling and tokenizer configuration
- Training configuration with DeepSpeed
- Multi-GPU setup and optimization
- Hyperparameter tuning suggestions
- Gradient accumulation strategies
- Learning rate scheduling
- Model saving and checkpointing
- Memory optimization techniques
- Dataset processing and preparation

Key training aspects covered in the notebook:
1. Base model download and setup
2. Dataset preprocessing options
3. Special token integration
4. DeepSpeed configuration
5. Training launch and monitoring
6. Model merging and saving
7. Performance optimization tips

## Special Token Support

The model supports special tokens for thought process:
- `<|begin_of_thought|>`
- `<|end_of_thought|>`

These tokens can be integrated through:
1. HuggingFace Tokenizer API
2. Manual tokenizer configuration
3. Automated token addition during training setup

## Hardware Recommendations

For optimal training performance:
- Multiple CUDA-capable GPUs (16+ GB VRAM each)
- High-speed GPU interconnect for multi-GPU training
- Sufficient system RAM for dataset handling
- Fast storage for checkpoint saving

## License

This project follows the licensing terms of the original Qwen model and associated datasets.
