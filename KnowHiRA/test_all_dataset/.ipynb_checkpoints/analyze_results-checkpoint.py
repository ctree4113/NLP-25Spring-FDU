#!/usr/bin/env python3
"""
KnowHiRA评估结果分析脚本
"""

import json
import jsonlines
import os
import glob
from pathlib import Path
import argparse

def find_latest_checkpoint(path):
    """查找最新的检查点目录"""
    pattern = os.path.join(path, "gpt-neo*")
    checkpoints = glob.glob(pattern)
    if not checkpoints:
        print("未找到检查点目录")
        return None
    
    latest = sorted(checkpoints)[-1]  # 最新的检查点
    print(f"分析检查点: {latest}")
    return latest

def extract_answer(dataset, sentence):
    """针对每个数据集特定格式的答案提取函数"""
    import re
    sentence_ = sentence.strip().lower()
    
    if dataset == 'boolq':
        # boolq: GT是 "true"/"false"
        pred_answers = re.findall(r'(true|false)', sentence_)
        if not pred_answers:
            return ""
        return pred_answers[0]
    
    elif dataset in ['piqa', 'siqa', 'arcc', 'arce', 'obqa', 'hellas', 'winog']:
        # 这些数据集: GT是 "solution1"/"solution2"等
        pred_answers = re.findall(r'\d', sentence_)
        if not pred_answers:
            return ""
        return f"solution{pred_answers[0]}"
    
    elif dataset == 'common_170k':
        # commonsense_170k: GT是 "the correct answer is true"等完整句子
        
        # 尝试提取"the correct answer is X"，X可能是各种格式
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
        
        # 检查是否包含true/false
        if re.search(r'\b(true|false)\b', sentence_):
            bool_answer = re.findall(r'\b(true|false)\b', sentence_)[0]
            return f"the correct answer is {bool_answer}"
        
        # 检查是否包含answer1-5
        answer_match = re.findall(r'answer([1-5])', sentence_)
        if answer_match:
            return f"the correct answer is answer{answer_match[0]}"
        
        # 检查是否包含option1-2
        option_match = re.findall(r'option([1-2])', sentence_)
        if option_match:
            return f"the correct answer is option{option_match[0]}"
        
        # 检查是否包含ending1-4
        ending_match = re.findall(r'ending([1-4])', sentence_)
        if ending_match:
            return f"the correct answer is ending{ending_match[0]}"
        
        # 检查是否包含solution1-2
        solution_match = re.findall(r'solution([1-2])', sentence_)
        if solution_match:
            return f"the correct answer is solution{solution_match[0]}"
        
        return ""
    
    return ""

def analyze_task_result(result_file, task_name):
    """分析单个任务的结果"""
    if not os.path.exists(result_file):
        return None
    
    try:
        with jsonlines.open(result_file, 'r') as reader:
            data = list(reader)
        
        # 跳过配置信息，处理预测结果
        predictions = []
        ground_truths = []
        
        for item in data:
            if isinstance(item, dict) and 'pred' in item and 'gt' in item:
                predictions.append(item['pred'])
                ground_truths.append(item['gt'])
        
        if not predictions:
            return None
        
        # 使用与baseline一致的答案提取逻辑计算准确率
        correct = 0
        valid_predictions = 0
        total = len(predictions)
        
        for pred, gt in zip(predictions, ground_truths):
            # 提取预测答案和真实答案
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
        print(f"⚠️  处理 {task_name} 时出错: {e}")
        return None

def main():
    print("=" * 50)
    print("评估结果分析")
    print("=" * 50)

    parser = argparse.ArgumentParser(description="评估结果分析脚本")
    parser.add_argument(
    "--checkpoint_path",
    type=str,
    default="results_hira",  # 默认路径
    help="检查点目录的父级路径"
)
    args = parser.parse_args()
    checkpoint_dir = find_latest_checkpoint(args.checkpoint_path)
    if not checkpoint_dir:
        return
    
    # 任务列表
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
    
    print("\n各任务结果:")
    print("-" * 50)
    
    for task_id, task_name in tasks:
        # 原始代码（检查 maxT=16 和 maxT=8）
        # result_file_fixed = os.path.join(checkpoint_dir, f"output_-1_{task_id}_maxT=16_eval.jsonl")
        # result_file_original = os.path.join(checkpoint_dir, f"output_-1_{task_id}_maxT=8_eval.jsonl")
        
        # 修改后：直接使用 beam=8
        result_file = os.path.join(checkpoint_dir, f"output_-1_{task_id}_beam=8_eval.jsonl")
        
        if os.path.exists(result_file):
            result = analyze_task_result(result_file, task_id)
        else:
            print(f"{task_name:20s}: 结果文件不存在（{result_file}）")
            continue
            
        result = analyze_task_result(result_file, task_id)
        
        if result:
            results[task_id] = result
            total_accuracy += result['accuracy']
            completed_tasks += 1
            print(f"{task_name:20s}: {result['accuracy']:6.2f}% ({result['correct']:4d}/{result['total']:4d}) [有效率: {result['valid_rate']:5.1f}%]")
        else:
            print(f"{task_name:20s}: 未完成或结果文件不存在")
    
    print("-" * 50)
    
    if completed_tasks > 0:
        avg_accuracy = total_accuracy / completed_tasks
        print(f"平均准确率: {avg_accuracy:.2f}% (基于 {completed_tasks} 个已完成任务)")
    else:
        print("没有完成的评估任务")
    
    # 保存结果摘要
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
    
    print(f"\n详细结果已保存到: {summary_file}")

if __name__ == "__main__":
    main() 