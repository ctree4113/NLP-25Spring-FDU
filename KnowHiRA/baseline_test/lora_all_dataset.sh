#!/bin/bash
python train_hira.py --dataset=common_170k --batch=16 --output_folder=temp --ckpt=./results_lora/gpt-neo-1.3B-common_170k-lora-lr=5.00e-05-seed=36--2025-05-24-03-00-53 --beam_size=8
python train_hira.py --dataset=boolq --batch=16 --output_folder=temp --ckpt=./results_lora/gpt-neo-1.3B-common_170k-lora-lr=5.00e-05-seed=36--2025-05-24-03-00-53 --beam_size=8
python train_hira.py --dataset=piqa --batch=16 --output_folder=temp --ckpt=./results_lora/gpt-neo-1.3B-common_170k-lora-lr=5.00e-05-seed=36--2025-05-24-03-00-53 --beam_size=8
python train_hira.py --dataset=siqa --batch=16 --output_folder=temp --ckpt=./results_lora/gpt-neo-1.3B-common_170k-lora-lr=5.00e-05-seed=36--2025-05-24-03-00-53 --beam_size=8
python train_hira.py --dataset=hellas --batch=16 --output_folder=temp --ckpt=./results_lora/gpt-neo-1.3B-common_170k-lora-lr=5.00e-05-seed=36--2025-05-24-03-00-53 --beam_size=8
python train_hira.py --dataset=winog --batch=16 --output_folder=temp --ckpt=./results_lora/gpt-neo-1.3B-common_170k-lora-lr=5.00e-05-seed=36--2025-05-24-03-00-53 --beam_size=8
python train_hira.py --dataset=arce --batch=16 --output_folder=temp --ckpt=./results_lora/gpt-neo-1.3B-common_170k-lora-lr=5.00e-05-seed=36--2025-05-24-03-00-53 --beam_size=8
python train_hira.py --dataset=arcc --batch=16 --output_folder=temp --ckpt=./results_lora/gpt-neo-1.3B-common_170k-lora-lr=5.00e-05-seed=36--2025-05-24-03-00-53 --beam_size=8
python train_hira.py --dataset=obqa --batch=16 --output_folder=temp --ckpt=./results_lora/gpt-neo-1.3B-common_170k-lora-lr=5.00e-05-seed=36--2025-05-24-03-00-53 --beam_size=8