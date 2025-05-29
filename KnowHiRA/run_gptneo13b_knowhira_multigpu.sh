#!/bin/bash

# ==========================================
# GPT-Neo 1.3B KnowHiRA Training Script
# ==========================================

echo "=============================================="
echo "  KnowHiRA Training (GPT-Neo 1.3B)"
echo "=============================================="

# Basic configuration
MODEL_NAME="EleutherAI/gpt-neo-1.3B"
DATASET="common_170k"
PEFT_TYPE="knowhira"
OUTPUT_FOLDER="results_knowhira_gptneo13b"

# KnowHiRA parameters
R_AB=48                                       # Moderate rank, balancing expressiveness and training stability
KNOWLEDGE_ALPHA=0.7                           # Enhanced knowledge fusion weight, but not excessive
ORTHO_LAMBDA=0.00002                          # Reduced orthogonal regularization to avoid over-constraint
SVD_RANK_RATIO=0.85                           # Maintain high SVD retention ratio
SPECTRUM_INIT_SCALE=0.05                      # Very conservative initialization for stability
TARGET_MODULES="q_proj,k_proj,v_proj,o_proj"  # Complete attention coverage

# Training parameters
BATCH_SIZE=12     # Small batch size for training stability
GRAD_ACC=12       # Increased gradient accumulation, effective batch size = 4*12*2 = 96
LR=1e-4           # Moderate learning rate, fast convergence but stable
EPOCH=4.0         # Increased training epochs for sufficient learning
SEED=42

# Multi-GPU configuration
GPU_IDS="0,1"
NUM_GPUS=2

echo "Training configuration overview:"
echo "Model: $MODEL_NAME"
echo "Method: KnowHiRA (Knowledge-aware High-rank Adaptation)"
echo "Output: $OUTPUT_FOLDER"
echo "Using GPUs: $GPU_IDS (${NUM_GPUS} cards)"
echo "Batch size: ${BATCH_SIZE} × ${GRAD_ACC} × ${NUM_GPUS} = $((BATCH_SIZE * GRAD_ACC * NUM_GPUS))"
echo "KnowHiRA Rank: $R_AB, Alpha: $KNOWLEDGE_ALPHA"
echo "Target modules: $TARGET_MODULES"
echo ""

# ====================
# Stage 1:  training
# ====================
echo "Starting training..."

# Set CUDA devices
export CUDA_VISIBLE_DEVICES=$GPU_IDS

# Use torchrun for multi-GPU training
torchrun \
    --nproc_per_node=$NUM_GPUS \
    --master_port=29503 \
    --nnodes=1 \
    --node_rank=0 \
    train_knowhira.py \
    --peft_type $PEFT_TYPE \
    --model_name $MODEL_NAME \
    --dataset $DATASET \
    --output_folder $OUTPUT_FOLDER \
    --r_ab $R_AB \
    --target_modules $TARGET_MODULES \
    --init_ab "kaiming,zero" \
    --train_ab "yy" \
    --knowledge_alpha $KNOWLEDGE_ALPHA \
    --ortho_lambda $ORTHO_LAMBDA \
    --svd_rank_ratio $SVD_RANK_RATIO \
    --spectrum_init_scale $SPECTRUM_INIT_SCALE \
    --adaptive_gating \
    --batch $BATCH_SIZE \
    --grad_acc $GRAD_ACC \
    --lr $LR \
    --epoch $EPOCH \
    --seed $SEED \
    --load_bit 16 \
    --eval_strategy "steps" \
    --eval_steps 200 \
    --save_total_limit 3 \
    --weight_decay 0.01 \
    --warmup 200 \
    --early_stop_patience 5

# Check training results
CHECKPOINT=$(find $OUTPUT_FOLDER -name "*gpt-neo*" -type d 2>/dev/null | tail -1)
if [ -z "$CHECKPOINT" ]; then
    echo "Training failed: No checkpoint found"
    exit 1
else
    echo "Training completed! Checkpoint: $(basename $CHECKPOINT)"
fi

# ====================
# Stage 2: Evaluation
# ====================
echo ""
echo "Starting evaluation..."

# Validate checkpoint
if [ ! -f "$CHECKPOINT/adapter_model.safetensors" ] && [ ! -f "$CHECKPOINT/adapter_model.bin" ]; then
    echo "Incomplete checkpoint"
    exit 1
fi

# Evaluation task list (including common_170k)
EVAL_TASKS=("boolq" "piqa" "siqa" "hellas" "winog" "arce" "arcc" "obqa" "common_170k")

echo "Evaluation tasks: ${EVAL_TASKS[*]}"
echo ""

# Single GPU evaluation
export CUDA_VISIBLE_DEVICES=0

success_count=0
for task in "${EVAL_TASKS[@]}"; do
    echo "Evaluating task: $task"
    
    # Backup original result files and check repetition issues
    EXISTING_RESULT="$CHECKPOINT/output_-1_${task}_maxT=8_eval.jsonl"
    if [ -f "$EXISTING_RESULT" ]; then
        # Check for repetition issues
        REPEAT_COUNT=$(grep -o "the correct answer is.*the correct answer is" "$EXISTING_RESULT" | wc -l 2>/dev/null || echo 0)
        if [ $REPEAT_COUNT -eq 0 ]; then
            echo "   $task already completed without repetition issues"
            ((success_count++))
            continue
        else
            echo "   $task has $REPEAT_COUNT repetition generations, needs re-evaluation"
            cp "$EXISTING_RESULT" "$EXISTING_RESULT.backup_$(date +%H%M%S)"
            echo "   Original results backed up"
        fi
    fi
    
    # Dataset path mapping
    case $task in
        "boolq") DATASET_PATH="data_file/llm_adapt/boolq" ;;
        "piqa") DATASET_PATH="data_file/llm_adapt/piqa" ;;
        "siqa") DATASET_PATH="data_file/llm_adapt/social_i_qa" ;;
        "hellas") DATASET_PATH="data_file/llm_adapt/hellaswag" ;;
        "winog") DATASET_PATH="data_file/llm_adapt/winogrande" ;;
        "arce") DATASET_PATH="data_file/llm_adapt/ARC-Easy" ;;
        "arcc") DATASET_PATH="data_file/llm_adapt/ARC-Challenge" ;;
        "obqa") DATASET_PATH="data_file/llm_adapt/openbookqa" ;;
        "common_170k") DATASET_PATH="data_file/llm_adapt/commonsense_170k" ;;
    esac
    
    TEST_FILE="$DATASET_PATH/test.json"
    if [ ! -f "$TEST_FILE" ]; then
        echo "   Dataset file not found: $TEST_FILE"
        continue
    fi
    
    echo "   Starting evaluation..."
    python train_knowhira.py \
        --ckpt $CHECKPOINT \
        --dataset $task \
        --batch 8 \
        --max_new_tokens 8 \
        --eval_strategy "no" \
        --decoding "fixed" \
        --load_order -1
    
    # Check evaluation results and verify fix effectiveness
    if [ $? -eq 0 ]; then
        # Verify fix effectiveness
        if [ -f "$EXISTING_RESULT" ]; then
            REPEAT_COUNT_AFTER=$(grep -o "the correct answer is.*the correct answer is" "$EXISTING_RESULT" | wc -l 2>/dev/null || echo 0)
            if [ $REPEAT_COUNT_AFTER -eq 0 ]; then
                echo "   $task completed"
                ((success_count++))
            else
                echo "   $task completed but still has $REPEAT_COUNT_AFTER repetition generations"
                ((success_count++))
            fi
            
            # Show prediction examples
            echo "   Prediction examples:"
            grep '"pred":' "$EXISTING_RESULT" | head -2 | cut -d'"' -f4 | sed 's/^/      /'
        else
            echo "   $task completed"
            ((success_count++))
        fi
    else
        echo "   $task failed"
    fi
    echo ""
done

echo "Evaluation completed, success: $success_count/${#EVAL_TASKS[@]} tasks"

# ====================
# Stage 3: Results analysis
# ====================
echo ""
echo "Running results analysis..."

# Show detailed results summary
echo ""
echo "Evaluation results summary:"
echo "===================="
total_repeats=0
for task in "${EVAL_TASKS[@]}"; do
    RESULT_FILE="$CHECKPOINT/output_-1_${task}_maxT=8_eval.jsonl"
    if [ -f "$RESULT_FILE" ]; then
        REPEAT_COUNT=$(grep -o "the correct answer is.*the correct answer is" "$RESULT_FILE" | wc -l 2>/dev/null || echo 0)
        total_repeats=$((total_repeats + REPEAT_COUNT))
        if [ $REPEAT_COUNT -eq 0 ]; then
            echo "$task: No repetition generation issues"
        else
            echo "$task: Still has $REPEAT_COUNT repetition generations"
        fi
    else
        echo "$task: Evaluation failed"
    fi
done

echo ""
echo "Effectiveness summary:"
echo "   Total repetition generations: $total_repeats"
if [ $total_repeats -eq 0 ]; then
    echo "   Repetition generation problem completely solved!"
else
    echo "   Further adjustment of decoding parameters needed"
fi

if [ -f "analyze_results.py" ]; then
    echo ""
    echo "Running detailed results analysis..."
    python analyze_results.py
else
    echo "Results analysis script not found"
fi

echo ""
echo "KnowHiRA training summary:"
echo "   Checkpoint directory: $(basename $CHECKPOINT)"
echo "   Evaluation success rate: $success_count/${#EVAL_TASKS[@]}"
echo "   Result file location: $CHECKPOINT"
echo "==============================================" 
