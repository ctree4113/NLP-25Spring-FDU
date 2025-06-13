# NLP-25Spring-FDU: KnowHiRA and Baseline PEFT Methods

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/release/python-3100/)
[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?logo=PyTorch&logoColor=white)](https://pytorch.org/)

Repository of the final project of Natural Language Processing 2025 Spring at Fudan University

## Overview

This repository contains the implementation of **KnowHiRA** (Knowledge-aware Hadamard-integrated Rank Adaptation), a novel parameter-efficient fine-tuning method, along with comprehensive baseline implementations for fair comparison. The project focuses on commonsense reasoning tasks and provides a unified framework for training and evaluating different PEFT approaches.

### Key Contributions

- **KnowHiRA**: A novel PEFT method that bridges high-rank expressivity with knowledge preservation
- **Comprehensive Baselines**: Unified implementation of HiRA, LoRA, FFT, and IA3 methods
- **Extensive Evaluation**: Systematic evaluation across nine commonsense reasoning benchmarks
- **Reproducible Results**: Complete experimental pipeline with detailed documentation

## Project Structure

```
NLP-25Spring-FDU/
├── KnowHiRA/                           # Main implementation directory
│   ├── README_KnowHiRA.md             # KnowHiRA-specific documentation
│   ├── README_baseline.md             # Baseline methods documentation
│   ├── env.yml                        # Conda environment configuration
│   ├── DATA_LICENSE                   # Data usage license
│   ├── train_knowhira.py              # KnowHiRA training script
│   ├── train_hira.py                  # Baseline training script
│   ├── eval_commonsense.py            # Evaluation script
│   ├── analyze_results.py             # Results analysis tool
│   ├── download_all_eval_datasets.py  # Dataset download utility
│   ├── continue_evaluation.sh         # KnowHiRA evaluation script
│   ├── run_gptneo13b_knowhira_multigpu.sh # Multi-GPU training script
│   ├── hira/                          # KnowHiRA PEFT framework
│   │   ├── peft_model.py              # KnowHiRA PEFT model wrapper
│   │   ├── mapping.py                 # KnowHiRA module mapping
│   │   ├── tuners/                    # KnowHiRA and extended methods
│   │   │   ├── knowhira.py            # KnowHiRA implementation
│   │   │   ├── lora.py                # Enhanced LoRA implementation
│   │   │   ├── adaption_prompt.py     # Adaptation prompt tuning
│   │   │   ├── prefix_tuning.py       # Prefix tuning
│   │   │   └── prompt_tuning.py       # Prompt tuning
│   │   └── utils/                     # KnowHiRA utility functions
│   ├── hira_baseline/                 # Baseline PEFT framework
│   │   ├── peft_model.py              # Baseline PEFT model wrapper
│   │   ├── mapping.py                 # Baseline module mapping
│   │   ├── tuners/                    # Baseline method implementations
│   │   │   ├── lora.py                # Standard LoRA implementation
│   │   │   ├── adaption_prompt.py     # Standard adaptation prompt
│   │   │   ├── prefix_tuning.py       # Standard prefix tuning
│   │   │   └── prompt_tuning.py       # Standard prompt tuning
│   │   └── utils/                     # Baseline utility functions
│   ├── models/                        # Model loading utilities
│   │   ├── get_models.py              # KnowHiRA model loader
│   │   └── get_models_baseline.py     # Baseline model loaders
│   ├── dataset/                       # Dataset processing
│   │   ├── dataset_hg.py              # HuggingFace dataset wrapper
│   │   ├── dataset_helper.py          # Dataset utilities
│   │   └── format_inputs.py           # Input formatting
│   ├── customized_trainer/            # Training infrastructure
│   │   ├── customized_trainer.py      # KnowHiRA trainer
│   │   └── customized_trainer_baseline.py # Baseline trainer
│   ├── baseline_test/                 # Evaluation scripts
│   │   ├── hira_all_dataset.sh        # HiRA evaluation
│   │   ├── lora_all_dataset.sh        # LoRA evaluation
│   │   ├── fft_all_dataset.sh         # FFT evaluation
│   │   ├── ia3_all_dataset.sh         # IA3 evaluation
│   │   ├── analyze_results.py         # Results analyzer
│   │   └── analyze_all_methods.sh     # Batch analysis
│   ├── ds_configs/                    # DeepSpeed configurations
│   ├── data_file/                     # Data storage
│   └── hira_baseline/                 # Baseline implementations
├── paper/                             # Research paper
│   ├── iclr2025_conference.tex        # Main paper
│   └── img/                           # Figures and images
└── README.md                          # This file
```

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/ctree4113/NLP-25Spring-FDU.git
cd NLP-25Spring-FDU/KnowHiRA

# Create conda environment from env.yml
conda env create -f env.yml
conda activate hira

# Verify installation
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
python -c "import transformers; print(f'Transformers version: {transformers.__version__}')"
```

**Note**: The `env.yml` file contains all necessary dependencies including:
- PyTorch 2.2.1 with CUDA 11.8 support
- Transformers 4.41.1
- PEFT, Accelerate, BitsAndBytes
- DeepSpeed for distributed training
- All evaluation and analysis dependencies

**Framework Structure**: The repository contains two PEFT frameworks:
- `hira/` - Enhanced framework with KnowHiRA implementation and extended methods
- `hira_baseline/` - Standard baseline framework for fair comparison with existing methods

### KnowHiRA Training

```bash
# Quick start with automated script
./run_gptneo13b_knowhira_multigpu.sh

# Or manual training
python train_knowhira.py \
    --model_name EleutherAI/gpt-neo-1.3B \
    --dataset common_170k \
    --r_ab 48 \
    --knowledge_alpha 0.7 \
    --ortho_lambda 2e-05 \
    --batch 8 \
    --lr 1e-4 \
    --epoch 3.0
```

### Baseline Training

```bash
# HiRA
python train_hira.py --peft_type=hira --model_name=EleutherAI/gpt-neo-1.3B --r_ab=64 --target_modules="q_proj,v_proj" --batch=8 --grad_acc=32 --lr=5e-5 --epoch=1 --load_bit=16 --enable_grad_ckpt --dataset=common_170k --seed=36 --warmup=100 --eval_strategy=steps --eval_steps=80 --output_folder=results_hira

# LoRA
python train_hira.py --peft_type=lora --model_name=EleutherAI/gpt-neo-1.3B --r_ab=64 --target_modules="q_proj,v_proj" --batch=8 --grad_acc=32 --lr=5e-5 --epoch=1 --load_bit=16 --enable_grad_ckpt --dataset=common_170k --seed=36 --eval_strategy=steps --eval_steps=80 --output_folder=results_lora

# Full Fine-Tuning
python train_hira.py --peft_type=fft --model_name=EleutherAI/gpt-neo-1.3B --batch=4 --grad_acc=64 --lr=1e-5 --epoch=1 --load_bit=16 --enable_grad_ckpt --dataset=common_170k --seed=36 --warmup=100 --eval_strategy=steps --eval_steps=80 --output_folder=results_fft

# IA3
python train_hira.py --peft_type=ia3 --model_name=EleutherAI/gpt-neo-1.3B --target_modules="q_proj,v_proj" --batch=32 --grad_acc=1 --lr=1e-5 --epoch=1 --load_bit=16 --enable_grad_ckpt --dataset=common_170k --eval_steps=80 --output_folder=results_ia3
```

## Evaluation

### Comprehensive Evaluation

Evaluate all methods across nine commonsense reasoning benchmarks:

```bash
# Evaluate baseline methods
bash baseline_test/hira_all_dataset.sh
bash baseline_test/lora_all_dataset.sh
bash baseline_test/fft_all_dataset.sh
bash baseline_test/ia3_all_dataset.sh

# Evaluate KnowHiRA
bash continue_evaluation.sh
```

### Results Analysis

Generate comprehensive performance analysis:

```bash
bash baseline_test/analyze_all_methods.sh
python analyze_results.py
```

## Evaluation Datasets

The framework evaluates models on nine commonsense reasoning benchmarks:

1. **BoolQ**: Boolean question answering requiring reading comprehension
2. **PIQA**: Physical interaction question answering for physical commonsense
3. **Social IQA**: Social intelligence question answering for interpersonal reasoning
4. **HellaSwag**: Sentence completion requiring situational understanding
5. **WinoGrande**: Pronoun resolution requiring world knowledge
6. **ARC-Easy**: Science question answering (easy difficulty)
7. **ARC-Challenge**: Science question answering (challenging difficulty)
8. **OpenBookQA**: Multi-hop reasoning with external knowledge
9. **CommonsenseQA**: General commonsense reasoning

## Key Features

### KnowHiRA Innovations

- **Knowledge-Guided Hadamard Updates**: Incorporates SVD-derived knowledge structure
- **Adaptive Knowledge Gating**: Dynamic control of knowledge influence
- **Orthogonal Parameter Regularization**: Maximizes effective rank utilization
- **Spectrum-Aware Initialization**: Leverages singular value distributions

### Baseline Methods

- **HiRA**: High-rank adaptation using Hadamard products
- **LoRA**: Low-rank matrix factorization
- **FFT**: Traditional full fine-tuning
- **IA3**: Lightweight scaling-based adaptation

### Experimental Framework

- **Dual Framework Design**: Separate implementations for KnowHiRA and baseline methods
- **Unified Interface**: Consistent API across all methods
- **Comprehensive Evaluation**: Systematic benchmarking
- **Reproducible Results**: Fixed seeds and detailed logging
- **Efficient Training**: Multi-GPU support and memory optimization

## Results Summary

KnowHiRA achieves superior performance across commonsense reasoning tasks:

| Method | Average Accuracy | BoolQ | PIQA | Social IQA | HellaSwag | WinoGrande | ARC-Easy | ARC-Challenge | OpenBookQA |
|--------|------------------|-------|------|------------|-----------|------------|----------|---------------|------------|
| **KnowHiRA** | **36.76%** | **55.69%** | **50.27%** | 31.88% | **25.58%** | 49.57% | **24.83%** | 24.15% | **26.40%** |
| HiRA | 34.83% | 37.83% | 50.33% | **33.57%** | 25.39% | **50.43%** | 24.20% | 23.63% | 26.40% |
| FFT | 34.22% | 37.55% | 49.62% | 33.73% | 24.88% | 50.43% | 23.82% | **24.06%** | 20.60% |
| LoRA | 34.87% | 37.83% | 50.00% | 33.73% | 25.38% | 50.43% | 24.28% | 23.63% | 26.00% |
| IA3 | 9.82% | 20.80% | 21.38% | 0.46% | 3.74% | 0.24% | 4.42% | 5.38% | 3.60% |

## Documentation

- **[KnowHiRA Documentation](KnowHiRA/README_KnowHiRA.md)**: Detailed guide for KnowHiRA usage
- **[Baseline Documentation](KnowHiRA/README_baseline.md)**: Comprehensive baseline methods guide
- **[Research Paper](paper/iclr2025_conference.pdf)**: Technical details and experimental analysis

## Acknowledgments

- Fudan University Natural Language Processing Course 2025 Spring
- Original implementations of HiRA, LoRA, and other baseline methods
- Hugging Face Transformers and PEFT libraries
- Commonsense reasoning benchmark datasets

---

**For detailed usage instructions, please refer to the specific documentation files in the KnowHiRA directory.**

