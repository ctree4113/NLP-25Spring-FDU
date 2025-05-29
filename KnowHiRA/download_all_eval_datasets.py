#!/usr/bin/env python3
"""
Complete KnowHiRA evaluation dataset download script
Download all 8 commonsense reasoning evaluation datasets
Generate datasets compatible with HiRA expected format
"""

import os
import json
import argparse
from datasets import load_dataset
from tqdm import tqdm

def create_boolq_dataset():
    """Create BoolQ dataset"""
    print("Downloading BoolQ dataset...")
    
    try:
        dataset = load_dataset('google/boolq')
        os.makedirs('data_file/llm_adapt/boolq', exist_ok=True)
        
        for split_name in ['train', 'validation']:
            if split_name not in dataset:
                continue
                
            split_data = dataset[split_name]
            formatted_data = []
            
            print(f"   Processing {split_name} split ({len(split_data)} samples)")
            
            for sample in tqdm(split_data, desc=f"Processing {split_name}"):
                formatted_sample = {
                    'instruction': f"Passage: {sample['passage']}\nQuestion: {sample['question']}\nAnswer this true/false question:",
                    'output': 'true' if sample['answer'] else 'false',
                }
                formatted_data.append(formatted_sample)
            
            split_file = 'test' if split_name == 'validation' else split_name
            output_file = f'data_file/llm_adapt/boolq/{split_file}.json'
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(formatted_data, f, ensure_ascii=False, indent=2)
            
            print(f"   {split_file}.json saved successfully ({len(formatted_data)} samples)")
        
        print("   BoolQ download completed!")
        return True
        
    except Exception as e:
        print(f"   BoolQ download failed: {e}")
        return False

def create_piqa_dataset():
    """Create PIQA dataset"""
    print("Downloading PIQA dataset...")
    
    try:
        dataset = load_dataset('ybisk/piqa', trust_remote_code=True)
        os.makedirs('data_file/llm_adapt/piqa', exist_ok=True)
        
        for split_name in ['train', 'validation']:
            if split_name not in dataset:
                continue
                
            split_data = dataset[split_name]
            formatted_data = []
            
            print(f"   Processing {split_name} split ({len(split_data)} samples)")
            
            for sample in tqdm(split_data, desc=f"Processing {split_name}"):
                goal = sample['goal']
                sol1 = sample['sol1']
                sol2 = sample['sol2']
                correct = sample['label']  # 0 or 1
                
                formatted_sample = {
                    'instruction': f"Goal: {goal}\n\nSolution1: {sol1}\n\nSolution2: {sol2}\n\nAnswer format: solution1/solution2",
                    'output': f'solution{correct + 1}',
                }
                formatted_data.append(formatted_sample)
            
            split_file = 'test' if split_name == 'validation' else split_name
            output_file = f'data_file/llm_adapt/piqa/{split_file}.json'
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(formatted_data, f, ensure_ascii=False, indent=2)
            
            print(f"   {split_file}.json saved successfully ({len(formatted_data)} samples)")
        
        print("   PIQA download completed!")
        return True
        
    except Exception as e:
        print(f"   PIQA download failed: {e}")
        return False

def create_siqa_dataset():
    """Create Social IQa dataset"""
    print("Downloading Social IQa dataset...")
    
    try:
        dataset = load_dataset('social_i_qa')
        os.makedirs('data_file/llm_adapt/social_i_qa', exist_ok=True)
        
        for split_name in ['train', 'validation']:
            if split_name not in dataset:
                continue
                
            split_data = dataset[split_name]
            formatted_data = []
            
            print(f"   Processing {split_name} split ({len(split_data)} samples)")
            
            for sample in tqdm(split_data, desc=f"Processing {split_name}"):
                context = sample['context']
                question = sample['question']
                answerA = sample['answerA']
                answerB = sample['answerB'] 
                answerC = sample['answerC']
                correct = sample['label']  # 1, 2, or 3
                
                formatted_sample = {
                    'instruction': f"Context: {context}\nQuestion: {question}\n\nSolution1: {answerA}\n\nSolution2: {answerB}\n\nSolution3: {answerC}\n\nAnswer format: solution1/solution2/solution3",
                    'output': f'solution{correct}',
                }
                formatted_data.append(formatted_sample)
            
            split_file = 'test' if split_name == 'validation' else split_name
            output_file = f'data_file/llm_adapt/social_i_qa/{split_file}.json'
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(formatted_data, f, ensure_ascii=False, indent=2)
            
            print(f"   {split_file}.json saved successfully ({len(formatted_data)} samples)")
        
        print("   Social IQa download completed!")
        return True
        
    except Exception as e:
        print(f"   Social IQa download failed: {e}")
        return False

def create_hellaswag_dataset():
    """Create HellaSwag dataset"""
    print("Downloading HellaSwag dataset...")
    
    try:
        dataset = load_dataset('hellaswag')
        os.makedirs('data_file/llm_adapt/hellaswag', exist_ok=True)
        
        for split_name in ['train', 'validation']:
            if split_name not in dataset:
                continue
                
            split_data = dataset[split_name]
            formatted_data = []
            
            print(f"   Processing {split_name} split ({len(split_data)} samples)")
            
            for sample in tqdm(split_data, desc=f"Processing {split_name}"):
                ctx = sample['ctx']
                endings = sample['endings']
                correct = int(sample['label']) if sample['label'] != '' else 0
                
                ending_instructions = [f"Solution{i+1}: {ending}" for i, ending in enumerate(endings)]
                
                formatted_sample = {
                    'instruction': f"Context: {ctx}\n\n" + '\n\n'.join(ending_instructions) + "\n\nAnswer format: solution1/solution2/solution3/solution4",
                    'output': f'solution{correct + 1}',
                }
                formatted_data.append(formatted_sample)
            
            split_file = 'test' if split_name == 'validation' else split_name
            output_file = f'data_file/llm_adapt/hellaswag/{split_file}.json'
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(formatted_data, f, ensure_ascii=False, indent=2)
            
            print(f"   {split_file}.json saved successfully ({len(formatted_data)} samples)")
        
        print("   HellaSwag download completed!")
        return True
        
    except Exception as e:
        print(f"   HellaSwag download failed: {e}")
        return False

def create_winogrande_dataset():
    """Create WinoGrande dataset"""
    print("Downloading WinoGrande dataset...")
    
    try:
        dataset = load_dataset('winogrande', 'winogrande_l')
        os.makedirs('data_file/llm_adapt/winogrande', exist_ok=True)
        
        for split_name in ['train', 'validation']:
            if split_name not in dataset:
                continue
                
            split_data = dataset[split_name]
            formatted_data = []
            
            print(f"   Processing {split_name} split ({len(split_data)} samples)")
            
            for sample in tqdm(split_data, desc=f"Processing {split_name}"):
                sentence = sample['sentence']
                option1 = sample['option1']
                option2 = sample['option2']
                correct = int(sample['answer']) if sample['answer'] != '' else 1
                
                formatted_sample = {
                    'instruction': f"Sentence: {sentence}\n\nSolution1: {option1}\n\nSolution2: {option2}\n\nAnswer format: solution1/solution2",
                    'output': f'solution{correct}',
                }
                formatted_data.append(formatted_sample)
            
            split_file = 'test' if split_name == 'validation' else split_name
            output_file = f'data_file/llm_adapt/winogrande/{split_file}.json'
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(formatted_data, f, ensure_ascii=False, indent=2)
            
            print(f"   {split_file}.json saved successfully ({len(formatted_data)} samples)")
        
        print("   WinoGrande download completed!")
        return True
        
    except Exception as e:
        print(f"   WinoGrande download failed: {e}")
        return False

def create_arc_easy_dataset():
    """Create ARC-Easy dataset"""
    print("Downloading ARC-Easy dataset...")
    
    try:
        dataset = load_dataset('ai2_arc', 'ARC-Easy')
        os.makedirs('data_file/llm_adapt/ARC-Easy', exist_ok=True)
        
        for split_name in ['train', 'validation', 'test']:
            if split_name not in dataset:
                continue
                
            split_data = dataset[split_name]
            formatted_data = []
            
            print(f"Processing {split_name} split ({len(split_data)} samples)")
            
            for sample in tqdm(split_data, desc=f"Processing {split_name}"):
                question = sample['question']
                choices = sample['choices']
                answerKey = sample['answerKey']
                
                options_text = []
                answer_idx = 0
                for i, choice in enumerate(choices['text']):
                    label = choices['label'][i]
                    options_text.append(f"Solution{i+1}: {choice}")
                    if label == answerKey:
                        answer_idx = i + 1
                
                formatted_sample = {
                    'instruction': f"{question}\n\n" + '\n\n'.join(options_text) + "\n\nAnswer format: solution1/solution2/solution3/solution4/solution5",
                    'output': f'solution{answer_idx}',
                }
                formatted_data.append(formatted_sample)
            
            output_file = f'data_file/llm_adapt/ARC-Easy/{split_name}.json'
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(formatted_data, f, ensure_ascii=False, indent=2)
            
            print(f"   {split_name}.json saved successfully ({len(formatted_data)} samples)")
        
        print("   ARC-Easy download completed!")
        return True
        
    except Exception as e:
        print(f"   ARC-Easy download failed: {e}")
        return False

def create_arc_challenge_dataset():
    """Create ARC-Challenge dataset"""
    print("Downloading ARC-Challenge dataset...")
    
    try:
        dataset = load_dataset('ai2_arc', 'ARC-Challenge')
        os.makedirs('data_file/llm_adapt/ARC-Challenge', exist_ok=True)
        
        for split_name in ['train', 'validation', 'test']:
            if split_name not in dataset:
                continue
                
            split_data = dataset[split_name]
            formatted_data = []
            
            print(f"   Processing {split_name} split ({len(split_data)} samples)")
            
            for sample in tqdm(split_data, desc=f"Processing {split_name}"):
                question = sample['question']
                choices = sample['choices']
                answerKey = sample['answerKey']
                
                options_text = []
                answer_idx = 0
                for i, choice in enumerate(choices['text']):
                    label = choices['label'][i]
                    options_text.append(f"Solution{i+1}: {choice}")
                    if label == answerKey:
                        answer_idx = i + 1
                
                formatted_sample = {
                    'instruction': f"{question}\n\n" + '\n\n'.join(options_text) + "\n\nAnswer format: solution1/solution2/solution3/solution4/solution5",
                    'output': f'solution{answer_idx}',
                }
                formatted_data.append(formatted_sample)
            
            output_file = f'data_file/llm_adapt/ARC-Challenge/{split_name}.json'
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(formatted_data, f, ensure_ascii=False, indent=2)
            
            print(f"   {split_name}.json saved successfully ({len(formatted_data)} samples)")
        
        print("   ARC-Challenge download completed!")
        return True
        
    except Exception as e:
        print(f"   ARC-Challenge download failed: {e}")
        return False

def create_openbookqa_dataset():
    """Create OpenBookQA dataset"""
    print("Downloading OpenBookQA dataset...")
    
    try:
        dataset = load_dataset('openbookqa', 'main')
        os.makedirs('data_file/llm_adapt/openbookqa', exist_ok=True)
        
        for split_name in ['train', 'validation', 'test']:
            if split_name not in dataset:
                continue
                
            split_data = dataset[split_name]
            formatted_data = []
            
            print(f"   Processing {split_name} split ({len(split_data)} samples)")
            
            for sample in tqdm(split_data, desc=f"Processing {split_name}"):
                question = sample['question_stem']
                choices = sample['choices']
                answerKey = sample['answerKey']
                
                options_text = []
                answer_idx = 0
                for i, choice in enumerate(choices['text']):
                    label = choices['label'][i]
                    options_text.append(f"Solution{i+1}: {choice}")
                    if label == answerKey:
                        answer_idx = i + 1
                
                formatted_sample = {
                    'instruction': f"{question}\n\n" + '\n\n'.join(options_text) + "\n\nAnswer format: solution1/solution2/solution3/solution4",
                    'output': f'solution{answer_idx}',
                }
                formatted_data.append(formatted_sample)
            
            output_file = f'data_file/llm_adapt/openbookqa/{split_name}.json'
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(formatted_data, f, ensure_ascii=False, indent=2)
            
            print(f"   {split_name}.json saved successfully ({len(formatted_data)} samples)")
        
        print("   OpenBookQA download completed!")
        return True
        
    except Exception as e:
        print(f"   OpenBookQA download failed: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description='Download all KnowHiRA evaluation datasets')
    parser.add_argument('--datasets', nargs='+',
                       choices=['boolq', 'piqa', 'siqa', 'hellas', 'winog', 'arce', 'arcc', 'obqa', 'all'],
                       default=['all'],
                       help='Datasets to download (default: all)')
    parser.add_argument('--force', action='store_true',
                       help='Force re-download existing datasets')
    
    args = parser.parse_args()
    
    print("Complete KnowHiRA evaluation dataset downloader")
    print("=" * 50)
    
    # Dataset configuration
    datasets_config = {
        'boolq': ("BoolQ", create_boolq_dataset),
        'piqa': ("PIQA", create_piqa_dataset),
        'siqa': ("Social IQa", create_siqa_dataset),
        'hellas': ("HellaSwag", create_hellaswag_dataset),
        'winog': ("WinoGrande", create_winogrande_dataset),
        'arce': ("ARC-Easy", create_arc_easy_dataset),
        'arcc': ("ARC-Challenge", create_arc_challenge_dataset),
        'obqa': ("OpenBookQA", create_openbookqa_dataset),
    }
    
    # Determine which datasets to download
    if 'all' in args.datasets:
        datasets_to_download = list(datasets_config.keys())
    else:
        datasets_to_download = args.datasets
    
    print(f"Plan to download datasets: {', '.join(datasets_to_download)}")
    print(f"Force re-download: {'Yes' if args.force else 'No'}")
    
    success_count = 0
    for dataset_key in datasets_to_download:
        if dataset_key in datasets_config:
            name, func = datasets_config[dataset_key]
            print(f"\nStarting download of {name}...")
            
            # Check if already exists (unless forced)
            path_map = {
                'boolq': 'data_file/llm_adapt/boolq',
                'piqa': 'data_file/llm_adapt/piqa',
                'siqa': 'data_file/llm_adapt/social_i_qa',
                'hellas': 'data_file/llm_adapt/hellaswag',
                'winog': 'data_file/llm_adapt/winogrande',
                'arce': 'data_file/llm_adapt/ARC-Easy',
                'arcc': 'data_file/llm_adapt/ARC-Challenge',
                'obqa': 'data_file/llm_adapt/openbookqa',
            }
            
            dataset_path = path_map[dataset_key]
            test_file = f'{dataset_path}/test.json'
            
            if not args.force and os.path.exists(test_file):
                print(f"{name} dataset already exists, skipping download")
                success_count += 1
                continue
            
            if func():
                success_count += 1
            else:
                print(f"{name} download failed")
    
    print("\n" + "=" * 50)
    print(f"Download completed! Success: {success_count}/{len(datasets_to_download)}")
    
    if success_count == len(datasets_to_download):
        print("All datasets downloaded successfully!")
        print("Verify dataset format:")
        
        # Verify generated file format
        for dataset_key in datasets_to_download:
            if dataset_key in ['boolq', 'piqa', 'siqa', 'hellas', 'winog', 'arce', 'arcc', 'obqa']:
                path_map = {
                    'boolq': 'data_file/llm_adapt/boolq',
                    'piqa': 'data_file/llm_adapt/piqa',
                    'siqa': 'data_file/llm_adapt/social_i_qa',
                    'hellas': 'data_file/llm_adapt/hellaswag',
                    'winog': 'data_file/llm_adapt/winogrande',
                    'arce': 'data_file/llm_adapt/ARC-Easy',
                    'arcc': 'data_file/llm_adapt/ARC-Challenge',
                    'obqa': 'data_file/llm_adapt/openbookqa',
                }
                
                test_file = f'{path_map[dataset_key]}/test.json'
                if os.path.exists(test_file):
                    with open(test_file, 'r') as f:
                        data = json.load(f)
                        print(f"   {dataset_key}: {len(data)} samples, example fields: {list(data[0].keys())}")
        
        print("\nNow you can run complete evaluation!")
    else:
        print("Some datasets download failed, check network connection and error information.")

if __name__ == "__main__":
    main()
