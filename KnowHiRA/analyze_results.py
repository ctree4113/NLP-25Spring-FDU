#!/usr/bin/env python3
"""
KnowHiRA evaluation result analysis script
"""

import json
import jsonlines
import os
import glob
from pathlib import Path

def find_latest_checkpoint():
    """Find the latest checkpoint directory"""
    pattern = "results_knowhira_gptneo13b*/gpt-neo*"
    checkpoints = glob.glob(pattern)
    if not checkpoints:
        print("No checkpoint directory found")
        return None
    
    latest = sorted(checkpoints)[-1]  # Latest checkpoint
    print(f"Analyzing checkpoint: {latest}")
    return latest

def extract_answer(dataset, sentence):
    """Answer extraction function for specific dataset formats"""
    import re
    sentence_ = sentence.strip().lower()
    
    if dataset == 'boolq':
        # boolq: GT is "true"/"false"
        pred_answers = re.findall(r'\b(true|false)\b', sentence_)
        if not pred_answers:
            return ""
        return pred_answers[0]
    
    elif dataset in ['piqa', 'siqa', 'arcc', 'arce', 'obqa', 'hellas', 'winog']:
        # These datasets: GT is "solution1"/"solution2" etc.
        pred_answers = re.findall(r'solution([1-5])', sentence_)
        if not pred_answers:
            return ""
        return f"solution{pred_answers[0]}"
    
    elif dataset == 'common_170k':
        # commonsense_170k: GT is complete sentences like "the correct answer is true"
        
        # Try to extract "the correct answer is X", X can be various formats
        patterns = [
            r'the correct answer is\s+(true|false)(?:\s|$)',
            r'the correct answer is\s+(answer[1-5])(?:\s|$)',
            r'the correct answer is\s+(option[1-2])(?:\s|$)',
            r'the correct answer is\s+(ending[1-4])(?:\s|$)',
            r'the correct answer is\s+(solution[1-2])(?:\s|$)'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, sentence_)
            if match:
                return f"the correct answer is {match.group(1)}"
        
        # Check if contains true/false
        if re.search(r'\b(true|false)\b', sentence_):
            bool_answer = re.findall(r'\b(true|false)\b', sentence_)[0]
            return f"the correct answer is {bool_answer}"
        
        # Check if contains answer1-5
        answer_match = re.findall(r'answer([1-5])', sentence_)
        if answer_match:
            return f"the correct answer is answer{answer_match[0]}"
        
        # Check if contains option1-2
        option_match = re.findall(r'option([1-2])', sentence_)
        if option_match:
            return f"the correct answer is option{option_match[0]}"
        
        # Check if contains ending1-4
        ending_match = re.findall(r'ending([1-4])', sentence_)
        if ending_match:
            return f"the correct answer is ending{ending_match[0]}"
        
        # Check if contains solution1-2
        solution_match = re.findall(r'solution([1-2])', sentence_)
        if solution_match:
            return f"the correct answer is solution{solution_match[0]}"
        
        return ""
    
    return ""

def analyze_task_result(result_file, task_name):
    """Analyze results for a single task"""
    if not os.path.exists(result_file):
        return None
    
    try:
        with jsonlines.open(result_file, 'r') as reader:
            data = list(reader)
        
        # Skip configuration info, process prediction results
        predictions = []
        ground_truths = []
        
        for item in data:
            if isinstance(item, dict) and 'pred' in item and 'gt' in item:
                predictions.append(item['pred'])
                ground_truths.append(item['gt'])
        
        if not predictions:
            return None
        
        # Calculate accuracy using answer extraction logic consistent with baseline
        correct = 0
        valid_predictions = 0
        total = len(predictions)
        
        for pred, gt in zip(predictions, ground_truths):
            # Extract predicted answer and ground truth answer
            extracted_pred = extract_answer(task_name, pred)
            extracted_gt = extract_answer(task_name, gt)
            
            if extracted_pred:
                valid_predictions += 1
                if extracted_pred == extracted_gt:
                    correct += 1
        
        accuracy = (correct / total) * 100 if total > 0 else 0
        valid_rate = (valid_predictions / total) * 100 if total > 0 else 0
        
        return {
            'total': total,
            'correct': correct,
            'valid_predictions': valid_predictions,
            'accuracy': accuracy,
            'valid_rate': valid_rate
        }
    
    except Exception as e:
        print(f"Error processing {task_name}: {e}")
        return None

def main():
    print("=" * 50)
    print("Evaluation Result Analysis")
    print("=" * 50)
    
    checkpoint_dir = find_latest_checkpoint()
    if not checkpoint_dir:
        return
    
    # Task list
    tasks = [
        ('boolq', 'BoolQ'),
        ('piqa', 'PIQA'),
        ('siqa', 'Social IQA'),
        ('hellas', 'HellaSwag'),
        ('winog', 'WinoGrande'),
        ('arce', 'ARC-Easy'),
        ('arcc', 'ARC-Challenge'),
        ('obqa', 'OpenBookQA'),
        ('common_170k', 'CommonsenseQA 170k')
    ]
    
    results = {}
    total_accuracy = 0
    completed_tasks = 0
    
    print("\nResults by task:")
    print("-" * 50)
    
    for task_id, task_name in tasks:
        # Prefer fixed version maxT=16 file, use maxT=8 file if not available
        result_file_fixed = os.path.join(checkpoint_dir, f"output_-1_{task_id}_maxT=16_eval.jsonl")
        result_file_original = os.path.join(checkpoint_dir, f"output_-1_{task_id}_maxT=8_eval.jsonl")
        
        if os.path.exists(result_file_fixed):
            result_file = result_file_fixed
        else:
            result_file = result_file_original
            
        result = analyze_task_result(result_file, task_id)
        
        if result:
            results[task_id] = result
            total_accuracy += result['accuracy']
            completed_tasks += 1
            print(f"{task_name:20s}: {result['accuracy']:6.2f}% ({result['correct']:4d}/{result['total']:4d}) [Valid rate: {result['valid_rate']:5.1f}%]")
        else:
            print(f"{task_name:20s}: Not completed or result file does not exist")
    
    print("-" * 50)
    
    if completed_tasks > 0:
        avg_accuracy = total_accuracy / completed_tasks
        print(f"Average accuracy: {avg_accuracy:.2f}% (based on {completed_tasks} completed tasks)")
    else:
        print("No completed evaluation tasks")
    
    # Save result summary
    summary_file = os.path.join(checkpoint_dir, "evaluation_summary.json")
    summary = {
        'checkpoint': checkpoint_dir,
        'completed_tasks': completed_tasks,
        'total_tasks': len(tasks),
        'average_accuracy': avg_accuracy if completed_tasks > 0 else 0,
        'task_results': results
    }
    
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    print(f"\nDetailed results saved to: {summary_file}")

if __name__ == "__main__":
    main()
