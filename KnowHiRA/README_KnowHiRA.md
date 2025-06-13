# KnowHiRA: Knowledge-aware Hadamard-integrated Rank Adaptation

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/release/python-3100/)
[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?logo=PyTorch&logoColor=white)](https://pytorch.org/)

## Overview

**KnowHiRA** (Knowledge-aware Hadamard-integrated Rank Adaptation) is a novel parameter-efficient fine-tuning (PEFT) method that bridges the fundamental trade-off between adaptation expressivity and knowledge preservation in large language models. By synergistically combining the high-rank expressivity of Hadamard products with sophisticated knowledge-aware mechanisms, KnowHiRA enables complex task-specific transformations while respecting the rich semantic structures encoded in pre-trained weights.

The method addresses a critical limitation in existing PEFT approaches: high-rank methods like HiRA excel at capturing complex patterns but may disrupt important knowledge structures, while knowledge-aware methods like KaSA maintain semantic coherence but sacrifice adaptation flexibility. KnowHiRA transcends this dichotomy through knowledge-guided multiplicative updates that achieve both objectives simultaneously.

## Key Innovations

### Knowledge-Guided Hadamard Updates

At the core of KnowHiRA lies a reformulation of standard Hadamard updates to incorporate knowledge structure derived from singular value decomposition (SVD). Rather than applying multiplicative transformations blindly, our method introduces a knowledge-guided gating matrix that modulates adaptations based on the importance hierarchy encoded in pre-trained weights:

```
ΔW = W₀ ⊙ (A · G_Σ · B)
```

where `G_Σ` is constructed from the normalized singular values of the pre-trained weights, ensuring that adaptations respect the model's inherent knowledge organization while enabling high-rank transformations.

### Adaptive Knowledge Gating

The method employs learnable gating mechanisms that dynamically control knowledge influence across different model components and tasks. This adaptive behavior allows the model to automatically balance between knowledge preservation and task-specific adaptation without requiring manual hyperparameter tuning. The gating mechanism learns to emphasize beneficial knowledge directions while de-emphasizing those that may hinder adaptation.

### Orthogonal Parameter Regularization

To maximize the effective rank of adaptations while maintaining parameter efficiency, KnowHiRA incorporates orthogonal regularization that encourages diversity in adaptation matrices. This prevents redundancy in adaptation directions and ensures optimal utilization of the limited parameter budget while improving optimization stability.

### Spectrum-Aware Initialization

The method leverages the singular value distribution of pre-trained weights to establish favorable starting conditions for optimization. This initialization strategy aligns initial adaptations with the model's knowledge structure, reducing exploration requirements and accelerating convergence while maintaining stable optimization dynamics.

## Architecture and Implementation

KnowHiRA extends the HiRA framework with minimal computational overhead while adding sophisticated knowledge-aware capabilities. The SVD computation is performed once during initialization and cached for subsequent use, while the diagonal structure of the knowledge-gating matrix ensures efficient application through element-wise operations.

The method introduces only a small number of additional parameters (specifically, the gating vector of size `r`) compared to standard HiRA, representing minimal increase in parameter overhead while providing substantial improvements in adaptation quality. The approach can be seamlessly integrated into existing transformer architectures and training pipelines.

## Installation and Setup

### Prerequisites

KnowHiRA requires Python 3.10 or higher and is built on PyTorch. The implementation leverages the Transformers library for model loading and the PEFT library for efficient training.

```bash
# Create conda environment from env.yml
conda env create -f env.yml
conda activate hira
```

### Quick Start

The simplest way to get started with KnowHiRA is through our automated training script:

```bash
# Run complete training and evaluation pipeline
./run_gptneo13b_knowhira_multigpu.sh
```

This script automatically handles model loading, training on the Commonsense-170K dataset, and evaluation across eight commonsense reasoning benchmarks.

## Usage

### Basic Training

For manual control over the training process, you can use the training script directly:

```bash
python train_knowhira.py \
    --model_name EleutherAI/gpt-neo-1.3B \
    --dataset common_170k \
    --r_ab 32 \
    --knowledge_alpha 0.6 \
    --ortho_lambda 0.0001 \
    --svd_rank_ratio 0.85 \
    --spectrum_init_scale 0.1 \
    --adaptive_gating \
    --batch 8 \
    --lr 1e-4 \
    --epoch 3.0
```

### Multi-GPU Training

For distributed training across multiple GPUs:

```bash
torchrun \
    --nproc_per_node=2 \
    --master_port=29501 \
    train_knowhira.py \
    --model_name EleutherAI/gpt-neo-1.3B \
    --dataset common_170k \
    --r_ab 48 \
    --knowledge_alpha 0.7 \
    --batch 12 \
    --grad_acc 12 \
    --lr 1e-4 \
    --epoch 4.0
```

### Model Evaluation

To evaluate a trained model on commonsense reasoning tasks:

```bash
python train_knowhira.py \
    --ckpt path/to/checkpoint \
    --dataset boolq \
    --eval_strategy no \
    --batch 8
```

## Configuration Parameters

### Core KnowHiRA Parameters

The method provides several key hyperparameters for controlling the knowledge-aware adaptation behavior:

- **`knowledge_alpha`** (default: 0.6): Controls the weight of knowledge guidance in the adaptation process. Higher values increase knowledge preservation, while lower values prioritize task-specific adaptation.

- **`ortho_lambda`** (default: 0.0001): Strength of orthogonal regularization applied to adaptation matrices. This parameter balances adaptation diversity with training stability.

- **`svd_rank_ratio`** (default: 0.85): Proportion of singular value components used for knowledge structure analysis. This controls the granularity of knowledge guidance.

- **`spectrum_init_scale`** (default: 0.1): Scaling factor for spectrum-aware initialization. Conservative values ensure stable early training dynamics.

- **`adaptive_gating`** (default: True): Enables learnable knowledge gating mechanisms for dynamic adaptation control.

### Model Architecture Parameters

- **`r_ab`**: Rank of adaptation matrices, controlling the expressivity-efficiency trade-off
- **`target_modules`**: Transformer modules to apply KnowHiRA adaptation (e.g., "q_proj,k_proj,v_proj,o_proj")
- **`init_ab`**: Initialization strategy for adaptation matrices (e.g., "kaiming,zero")

### Training Parameters

- **`batch`**: Per-device training batch size
- **`grad_acc`**: Gradient accumulation steps for effective batch size scaling
- **`lr`**: Learning rate for optimizer
- **`epoch`**: Number of training epochs

## Evaluation Framework

KnowHiRA includes comprehensive evaluation capabilities across multiple commonsense reasoning benchmarks:

- **BoolQ**: Boolean question answering requiring reading comprehension
- **PIQA**: Physical interaction question answering for physical commonsense
- **Social IQA**: Social intelligence question answering for interpersonal reasoning
- **HellaSwag**: Sentence completion requiring situational understanding
- **WinoGrande**: Pronoun resolution requiring world knowledge
- **ARC-Easy/Challenge**: Science question answering with different difficulty levels
- **OpenBookQA**: Multi-hop reasoning with external knowledge

The evaluation framework automatically generates detailed performance reports and provides analysis tools for understanding adaptation behavior across different reasoning domains.

## Dataset Management

The framework includes utilities for automatic dataset downloading and preprocessing:

```bash
# Download all evaluation datasets
python download_all_eval_datasets.py --datasets all

# Download specific datasets
python download_all_eval_datasets.py --datasets boolq piqa siqa
```

All datasets are automatically formatted for compatibility with the training and evaluation pipeline, ensuring consistent preprocessing and evaluation protocols.

## Advanced Features

### Custom Model Integration

KnowHiRA can be applied to any transformer-based language model through the modular configuration system:

```python
from models.get_models import get_knowhira_models

model, tokenizer, config = get_knowhira_models(
    model_name="your-model-name",
    r_ab=32,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    knowledge_alpha=0.6,
    ortho_lambda=0.0001,
    # Additional parameters...
)
```

### Inference and Deployment

Trained KnowHiRA models can be easily deployed for inference with minimal modifications to existing pipelines. The adapter weights are stored separately from the base model, enabling efficient model sharing and deployment strategies.

### Result Analysis

The framework provides comprehensive analysis tools for understanding model behavior and adaptation quality:

```bash
# Generate detailed performance analysis
python analyze_results.py

# Evaluate with fixed decoding parameters
bash run_fixed_evaluation.sh
```
