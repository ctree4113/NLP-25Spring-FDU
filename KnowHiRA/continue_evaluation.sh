#!/bin/bash

# Specify the latest complete checkpoint
CHECKPOINT="results_knowhira_gptneo13b/gpt-neo-1.3B-common_170k-knowhira-lr=1.00e-04-r_ab=48-init=kz-train=yy-kalpha=0.7-ortho=2e-05-seed=42--2025-05-24-14-05-11"

if [ ! -d "$CHECKPOINT" ]; then
    echo "Checkpoint directory does not exist: $CHECKPOINT"
    exit 1
fi

echo "Using checkpoint: $(basename $CHECKPOINT)"

# Validate checkpoint
if [ ! -f "$CHECKPOINT/adapter_model.bin" ] && [ ! -f "$CHECKPOINT/adapter_model.safetensors" ]; then
    echo "Incomplete checkpoint"
    echo "Directory contents:"
    ls -la "$CHECKPOINT/"
    exit 1
fi

echo "Checkpoint validation passed"

# Evaluation task list (including commonsense170k)
EVAL_TASKS=("boolq" "piqa" "siqa" "hellas" "winog" "arce" "arcc" "obqa" "common_170k")

echo "Evaluation tasks: ${EVAL_TASKS[*]}"
echo ""

# Single GPU evaluation
export CUDA_VISIBLE_DEVICES=0

success_count=0
for task in "${EVAL_TASKS[@]}"; do
    echo "Evaluating task: $task"
    
    # Check result files (support multiple maxT values)
    EXISTING_RESULT_16="$CHECKPOINT/output_-1_${task}_maxT=16_eval.jsonl"
    EXISTING_RESULT_8="$CHECKPOINT/output_-1_${task}_maxT=8_eval.jsonl"
    
    # Prefer maxT=16 file, use maxT=8 if not available
    if [ -f "$EXISTING_RESULT_16" ]; then
        EXISTING_RESULT="$EXISTING_RESULT_16"
    elif [ -f "$EXISTING_RESULT_8" ]; then
        EXISTING_RESULT="$EXISTING_RESULT_8"
    else
        EXISTING_RESULT=""
    fi
    
    if [ -n "$EXISTING_RESULT" ] && [ -f "$EXISTING_RESULT" ]; then
        # Check prediction quality
        echo "   Checking existing result: $(basename $EXISTING_RESULT)"
        
        # Create Python script to check prediction quality - use answer extraction logic consistent with baseline
        python3 -c "
import jsonlines
import sys
import re

def extract_answer(dataset, sentence):
    '''Answer extraction function for specific dataset formats'''
    sentence_ = sentence.strip().lower()
    
    if dataset == 'boolq':
        # boolq: GT is \"true\"/\"false\"
        pred_answers = re.findall(r'\\b(true|false)\\b', sentence_)
        if not pred_answers:
            return ''
        return pred_answers[0]
    
    elif dataset in ['piqa', 'siqa', 'arcc', 'arce', 'obqa', 'hellas', 'winog']:
        # These datasets: GT is \"solution1\"/\"solution2\" etc.
        pred_answers = re.findall(r'solution([1-5])', sentence_)
        if not pred_answers:
            return ''
        return f'solution{pred_answers[0]}'
    
    elif dataset == 'common_170k':
        # commonsense_170k: GT is complete sentences like \"the correct answer is true\"
        # More precise extraction logic to handle repetition generation issues
        
        # 1. Try to extract \"the correct answer is X\", X can be various formats
        patterns = [
            r'the correct answer is\\s+(true|false)(?:\\s|$)',
            r'the correct answer is\\s+(answer[1-5])(?:\\s|$)',
            r'the correct answer is\\s+(option[1-2])(?:\\s|$)',
            r'the correct answer is\\s+(ending[1-4])(?:\\s|$)',
            r'the correct answer is\\s+(solution[1-2])(?:\\s|$)'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, sentence_)
            if match:
                return f'the correct answer is {match.group(1)}'
        
        # 2. If no complete format, try to extract keywords and construct answer
        # Check if contains true/false
        if re.search(r'\\b(true|false)\\b', sentence_):
            bool_answer = re.findall(r'\\b(true|false)\\b', sentence_)[0]
            return f'the correct answer is {bool_answer}'
        
        # Check if contains answer1-5
        answer_match = re.findall(r'answer([1-5])', sentence_)
        if answer_match:
            return f'the correct answer is answer{answer_match[0]}'
        
        # Check if contains option1-2
        option_match = re.findall(r'option([1-2])', sentence_)
        if option_match:
            return f'the correct answer is option{option_match[0]}'
        
        # Check if contains ending1-4
        ending_match = re.findall(r'ending([1-4])', sentence_)
        if ending_match:
            return f'the correct answer is ending{ending_match[0]}'
        
        # Check if contains solution1-2
        solution_match = re.findall(r'solution([1-2])', sentence_)
        if solution_match:
            return f'the correct answer is solution{solution_match[0]}'
        
        return ''
    
    return ''

def check_prediction_quality(file_path, task_name):
    try:
        with jsonlines.open(file_path, 'r') as reader:
            data = list(reader)
        
        predictions = []
        ground_truths = []
        for item in data:
            if isinstance(item, dict) and 'pred' in item and 'gt' in item:
                predictions.append(item['pred'])
                ground_truths.append(item['gt'])
        
        if not predictions:
            return True, 'No prediction results found'
        
        # Use baseline answer extraction logic to check quality
        valid_preds = 0
        correct_preds = 0
        
        for pred, gt in zip(predictions, ground_truths):
            extracted_pred = extract_answer(task_name, pred)
            extracted_gt = extract_answer(task_name, gt)
            
            if extracted_pred:
                valid_preds += 1
                if extracted_pred == extracted_gt:
                    correct_preds += 1
        
        valid_rate = valid_preds / len(predictions) if predictions else 0
        accuracy = correct_preds / len(predictions) if predictions else 0
        
        # Quality criteria: valid prediction rate > 80% and accuracy > 10%
        if valid_rate < 0.8:
            return True, f'Valid prediction rate too low: {valid_rate:.1%}'
        elif accuracy < 0.1:
            return True, f'Accuracy too low: {accuracy:.1%}'
        else:
            return False, f'Good quality (valid rate:{valid_rate:.1%}, accuracy:{accuracy:.1%})'
            
    except Exception as e:
        return True, f'Check failed: {e}'

needs_reeval, reason = check_prediction_quality('$EXISTING_RESULT', '$task')
print(f'{needs_reeval}|{reason}')
" > /tmp/check_result.txt
        
        check_result=$(cat /tmp/check_result.txt)
        needs_reeval=$(echo "$check_result" | cut -d'|' -f1)
        reason=$(echo "$check_result" | cut -d'|' -f2)
        
        if [ "$needs_reeval" = "False" ]; then
            echo "   $task completed with good quality: $reason"
            ((success_count++))
            continue
        else
            echo "   $task needs re-evaluation: $reason"
            # Backup original result
            cp "$EXISTING_RESULT" "$EXISTING_RESULT.backup_$(date +%H%M%S)"
            echo "   Original result backed up"
        fi
    else
        echo "   $task result file does not exist, evaluation needed"
    fi
    
    # Check dataset existence
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
        echo "   $task dataset file does not exist: $TEST_FILE"
        continue
    fi
    
    echo "Starting evaluation..."
    # Use repaired decoding parameters and add prediction cleanup functionality
    python train_knowhira.py \
        --ckpt "$CHECKPOINT" \
        --dataset $task \
        --batch 16 \
        --max_new_tokens 16 \
        --eval_strategy "no" \
        --decoding "fixed" \
        --load_order -1
    
    # Check evaluation result
    if [ $? -eq 0 ]; then
        # Re-determine result file path
        NEW_RESULT_16="$CHECKPOINT/output_-1_${task}_maxT=16_eval.jsonl"
        NEW_RESULT_8="$CHECKPOINT/output_-1_${task}_maxT=8_eval.jsonl"
        
        if [ -f "$NEW_RESULT_16" ]; then
            RESULT_FILE="$NEW_RESULT_16"
        elif [ -f "$NEW_RESULT_8" ]; then
            RESULT_FILE="$NEW_RESULT_8"
        else
            RESULT_FILE=""
        fi
        
        if [ -n "$RESULT_FILE" ] && [ -f "$RESULT_FILE" ]; then
            # Validate repaired effect
            echo "   Checking result file: $(basename $RESULT_FILE)"
            
            # Check if there are cleaned pred and original_pred fields
            python3 -c "
import jsonlines
try:
    with jsonlines.open('$RESULT_FILE', 'r') as reader:
        data = list(reader)
    
    predictions = []
    has_original_pred = False
    has_clean_pred = False
    
    for item in data:
        if isinstance(item, dict) and 'pred' in item:
            predictions.append(item['pred'])
            if 'original_pred' in item:
                has_original_pred = True
                # Check if pred has been cleaned
                if len(item['pred']) < len(item.get('original_pred', '')):
                    has_clean_pred = True
    
    print(f'Prediction count: {len(predictions)}')
    print(f'Has original_pred field: {has_original_pred}')
    print(f'Has cleaned pred: {has_clean_pred}')
    
    if len(predictions) > 0:
        print('Top 3 prediction examples:')
        count = 0
        for item in data:
            if isinstance(item, dict) and 'pred' in item and count < 3:
                print(f'  {count+1}: pred=\"{item[\"pred\"]}\"')
                if 'original_pred' in item:
                    print(f'      original=\"{item[\"original_pred\"][:50]}...\"')
                count += 1
except Exception as e:
    print(f'Check failed: {e}')
"
            
            echo "   $task completed"
            ((success_count++))
        else
            echo "   $task evaluation failed: Result file not found"
        fi
    else
        echo "   $task evaluation failed"
    fi
    echo ""
done

echo "✅ Full repaired version evaluation completed, success: $success_count/${#EVAL_TASKS[@]} tasks"

# Display detailed result summary
echo ""
echo "📊 Full repaired version evaluation result summary:"
echo "===================="

# Create final quality check script
python3 -c "
import jsonlines
import os
import re

def analyze_final_quality(checkpoint_dir, tasks):
    print('Task quality analysis:')
    print('-' * 60)
    
    total_issues = 0
    total_tasks = 0
    
    for task_id in tasks:
        # Check multiple file name formats
        result_files = [
            os.path.join(checkpoint_dir, f'output_-1_{task_id}_maxT=16_eval.jsonl'),
            os.path.join(checkpoint_dir, f'output_-1_{task_id}_maxT=8_eval.jsonl')
        ]
        
        result_file = None
        for rf in result_files:
            if os.path.exists(rf):
                result_file = rf
                break
        
        if not result_file:
            print(f'{task_id:15s}: File does not exist')
            continue
        
        total_tasks += 1
        
        try:
            with jsonlines.open(result_file, 'r') as reader:
                data = list(reader)
            
            predictions = []
            has_original_pred = False
            has_clean_pred = False
            
            for item in data:
                if isinstance(item, dict) and 'pred' in item:
                    predictions.append(item['pred'])
                    if 'original_pred' in item:
                        has_original_pred = True
                        # Check if pred has been cleaned
                        if len(item['pred']) < len(item.get('original_pred', '')):
                            has_clean_pred = True
            
            if not predictions:
                print(f'{task_id:15s}: No prediction results')
                continue
            
            # Quality check
            issues = []
            
            # Check if there are cleaned predictions
            if not has_original_pred:
                issues.append('Missing original_pred')
            elif not has_clean_pred:
                issues.append('Pred not cleaned')
            
            # Repeat check
            repeat_count = 0
            for pred in predictions:
                if 'the correct answer is' in pred.lower():
                    if pred.lower().count('the correct answer is') > 1:
                        repeat_count += 1
            
            if repeat_count > 0:
                issues.append(f'Repeat:{repeat_count}')
            
            # Empty prediction check
            empty_count = sum(1 for pred in predictions if len(pred.strip()) == 0)
            if empty_count > 0:
                issues.append(f'Empty:{empty_count}')
            
            # Format check (for multiple choice questions)
            if task_id in ['boolq', 'piqa', 'siqa', 'hellas', 'winog', 'arce', 'arcc', 'obqa']:
                invalid_count = 0
                for pred in predictions:
                    pred_clean = pred.lower().strip()
                    if task_id == 'boolq':
                        if 'true' not in pred_clean and 'false' not in pred_clean:
                            invalid_count += 1
                    else:
                        if not re.search(r'solution[12345]|[12345]', pred_clean):
                            invalid_count += 1
                
                if invalid_count > len(predictions) * 0.1:
                    issues.append(f'Format:{invalid_count}')
            
            if issues:
                print(f'{task_id:15s}: {\",\".join(issues)} (Total:{len(predictions)})')
                total_issues += len(issues)
            else:
                print(f'{task_id:15s}: Good quality (Total:{len(predictions)})')
                
        except Exception as e:
            print(f'{task_id:15s}: Analysis failed - {e}')
    
    print('-' * 60)
    if total_issues == 0:
        print('All tasks prediction quality good!')
    else:
        print(f'Found {total_issues} types of problems, involving {total_tasks} tasks')
    
    return total_issues == 0

checkpoint = '$CHECKPOINT'
tasks = ['boolq', 'piqa', 'siqa', 'hellas', 'winog', 'arce', 'arcc', 'obqa', 'common_170k']
all_good = analyze_final_quality(checkpoint, tasks)

if all_good:
    print()
    print('Repair effect summary:')
    print('   All tasks have cleaned pred fields')
    print('   Repetition generation issues resolved')
    print('   Prediction format correct')
    print('   Can perform accuracy analysis!')
else:
    print()
    print('Repair effect summary:')
    print('   Some tasks still need further repair')
    print('   Suggest re-running problematic tasks')
"

echo ""
echo "📁 Result file location: $CHECKPOINT"

# Automatically run result analysis
if [ -f "analyze_results.py" ]; then
    echo ""
    echo "Running result analysis..."
    python analyze_results.py
else
    echo "Can run result analysis script: python analyze_results.py"
fi

echo "================================================"
