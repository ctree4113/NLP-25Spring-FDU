# Baseline Methods for Parameter-Efficient Fine-Tuning

This document provides comprehensive instructions for training and evaluating baseline parameter-efficient fine-tuning (PEFT) methods on commonsense reasoning tasks. The implementation supports multiple PEFT approaches including HiRA, LoRA, Full Fine-Tuning (FFT), and IA3.

## Overview

The baseline framework provides a unified interface for training and evaluating different PEFT methods on the GPT-Neo-1.3B model using the Commonsense-170K dataset. All methods are evaluated on nine commonsense reasoning benchmarks to ensure fair comparison.

## Supported Methods

### 1. HiRA (Hadamard High-Rank Adaptation)
High-rank adaptation using Hadamard products for multiplicative parameter updates.

### 2. LoRA (Low-Rank Adaptation)
Low-rank matrix factorization for efficient parameter adaptation.

### 3. FFT (Full Fine-Tuning)
Traditional fine-tuning approach that updates all model parameters.

### 4. IA3 (Infused Adapter by Inhibiting and Amplifying Inner Activations)
Lightweight adaptation through learned scaling vectors.

## Installation

```bash
# Create conda environment from env.yml
conda env create -f env.yml
conda activate hira

# Verify installation
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
python -c "import transformers; print(f'Transformers version: {transformers.__version__}')"
```

## Training

### HiRA Training
```bash
python train_hira.py \
    --peft_type=hira \
    --model_name=EleutherAI/gpt-neo-1.3B \
    --r_ab=64 \
    --target_modules="q_proj,v_proj" \
    --batch=8 \
    --grad_acc=32 \
    --lr=5e-5 \
    --epoch=1 \
    --load_bit=16 \
    --enable_grad_ckpt \
    --dataset=common_170k \
    --seed=36 \
    --warmup=100 \
    --eval_strategy=steps \
    --eval_steps=80 \
    --output_folder=results_hira
```

### LoRA Training
```bash
python train_hira.py \
    --peft_type=lora \
    --model_name=EleutherAI/gpt-neo-1.3B \
    --r_ab=64 \
    --target_modules="q_proj,v_proj" \
    --batch=8 \
    --grad_acc=32 \
    --lr=5e-5 \
    --epoch=1 \
    --load_bit=16 \
    --enable_grad_ckpt \
    --dataset=common_170k \
    --seed=36 \
    --eval_strategy=steps \
    --eval_steps=80 \
    --output_folder=results_lora
```

### Full Fine-Tuning (FFT)
```bash
python train_hira.py \
    --peft_type=fft \
    --model_name=EleutherAI/gpt-neo-1.3B \
    --batch=4 \
    --grad_acc=64 \
    --lr=1e-5 \
    --epoch=1 \
    --load_bit=16 \
    --enable_grad_ckpt \
    --dataset=common_170k \
    --seed=36 \
    --warmup=100 \
    --eval_strategy=steps \
    --eval_steps=80 \
    --output_folder=results_fft
```

### IA3 Training
```bash
python train_hira.py \
    --peft_type=ia3 \
    --model_name=EleutherAI/gpt-neo-1.3B \
    --target_modules="q_proj,v_proj" \
    --batch=32 \
    --grad_acc=1 \
    --lr=1e-5 \
    --epoch=1 \
    --load_bit=16 \
    --enable_grad_ckpt \
    --dataset=common_170k \
    --eval_steps=80 \
    --output_folder=results_ia3
```

## Evaluation

### Single Dataset Evaluation
To evaluate a trained model on a specific dataset:

```bash
python train_hira.py \
    --ckpt path/to/checkpoint \
    --dataset boolq \
    --batch=16 \
    --beam_size=8
```

### Comprehensive Evaluation
Use the provided scripts to evaluate models across all benchmark datasets:

#### HiRA Evaluation
```bash
bash baseline_test/hira_all_dataset.sh
```

#### LoRA Evaluation
```bash
bash baseline_test/lora_all_dataset.sh
```

#### FFT Evaluation
```bash
bash baseline_test/fft_all_dataset.sh
```

#### IA3 Evaluation
```bash
bash baseline_test/ia3_all_dataset.sh
```

### Results Analysis
Generate comprehensive performance analysis across all methods:

```bash
bash baseline_test/analyze_all_methods.sh
```

This will produce accuracy summaries for each method across all evaluation datasets.

## Configuration Parameters

### Core Parameters
- `--peft_type`: Type of PEFT method (hira, lora, fft, ia3)
- `--model_name`: Base model identifier (default: EleutherAI/gpt-neo-1.3B)
- `--dataset`: Training dataset (default: common_170k)
- `--r_ab`: Rank for adaptation matrices (HiRA/LoRA only)
- `--target_modules`: Modules to apply adaptation (comma-separated)

### Training Parameters
- `--batch`: Per-device batch size
- `--grad_acc`: Gradient accumulation steps
- `--lr`: Learning rate
- `--epoch`: Number of training epochs
- `--warmup`: Warmup steps
- `--weight_decay`: Weight decay coefficient

### System Parameters
- `--load_bit`: Model precision (16 for fp16, 32 for fp32)
- `--enable_grad_ckpt`: Enable gradient checkpointing for memory efficiency
- `--seed`: Random seed for reproducibility

### Evaluation Parameters
- `--eval_strategy`: Evaluation strategy (no, steps, epoch)
- `--eval_steps`: Evaluation frequency
- `--beam_size`: Beam size for generation tasks
- `--max_new_tokens`: Maximum tokens to generate

## Datasets

### Training Dataset
- **Commonsense-170K**: Large-scale commonsense reasoning dataset for fine-tuning

### Evaluation Datasets
1. **BoolQ**: Boolean question answering
2. **PIQA**: Physical interaction question answering
3. **Social IQA**: Social intelligence question answering
4. **HellaSwag**: Sentence completion
5. **WinoGrande**: Pronoun resolution
6. **ARC-Easy**: Science questions (easy)
7. **ARC-Challenge**: Science questions (challenging)
8. **OpenBookQA**: Multi-hop reasoning
9. **CommonsenseQA**: General commonsense reasoning

## Output Structure

Training outputs are organized as follows:
```
results_{method}/
├── {model}-{dataset}-{method}-lr={lr}-seed={seed}-{timestamp}/
│   ├── checkpoint-{step}/
│   ├── output.jsonl          # Training configuration
│   ├── trainer_state.json    # Training state
│   └── training_args.bin     # Training arguments
```

Evaluation outputs are saved as:
```
{checkpoint_dir}/
├── output_{dataset}_eval.jsonl    # Evaluation results
└── predictions_{dataset}.json     # Model predictions
```

## Performance Monitoring

The framework provides comprehensive logging and monitoring:

- **Training Loss**: Tracked during training with configurable evaluation frequency
- **Validation Performance**: Automatic evaluation on validation sets
- **Parameter Efficiency**: Detailed reporting of trainable vs. total parameters
- **Memory Usage**: GPU memory monitoring during training
- **Training Time**: Automatic timing of training phases

## Troubleshooting

### Common Issues

1. **Out of Memory**: Reduce batch size or enable gradient checkpointing
2. **Slow Training**: Increase gradient accumulation, reduce evaluation frequency
3. **Poor Convergence**: Adjust learning rate, increase warmup steps
4. **Checkpoint Loading**: Ensure checkpoint path and configuration match

### Memory Optimization
- Use `--load_bit=16` for half-precision training
- Enable `--enable_grad_ckpt` for gradient checkpointing
- Reduce `--batch` size and increase `--grad_acc` accordingly

## Citation

If you use these baseline implementations in your research, please cite the original papers:

```bibtex
@article{hu2021lora,
  title={LoRA: Low-Rank Adaptation of Large Language Models},
  author={Hu, Edward J and Shen, Yelong and Wallis, Phillip and Allen-Zhu, Zeyuan and Li, Yuanzhi and Wang, Shean and Wang, Lu and Chen, Weizhu},
  journal={arXiv preprint arXiv:2106.09685},
  year={2021}
}

@article{huang2024hira,
  title={HiRA: Hadamard High-Rank Adaptation for Parameter-Efficient Fine-Tuning},
  author={Huang, Jiawei and others},
  year={2024}
}
```
