#!/bin/bash
python train_hira.py --dataset=common_170k --batch=16 --output_folder=temp --ckpt=./results_ia3/gpt-neo-1.3B-common_170k-ia3-lr=1.00e-05--2025-05-24-06-58-08 --beam_size=8
python train_hira.py --dataset=boolq --batch=16 --output_folder=temp --ckpt=./results_ia3/gpt-neo-1.3B-common_170k-ia3-lr=1.00e-05--2025-05-24-06-58-08 --beam_size=8
python train_hira.py --dataset=piqa --batch=16 --output_folder=temp --ckpt=./results_ia3/gpt-neo-1.3B-common_170k-ia3-lr=1.00e-05--2025-05-24-06-58-08 --beam_size=8
python train_hira.py --dataset=siqa --batch=16 --output_folder=temp --ckpt=./results_ia3/gpt-neo-1.3B-common_170k-ia3-lr=1.00e-05--2025-05-24-06-58-08 --beam_size=8
python train_hira.py --dataset=hellas --batch=16 --output_folder=temp --ckpt=./results_ia3/gpt-neo-1.3B-common_170k-ia3-lr=1.00e-05--2025-05-24-06-58-08 --beam_size=8
python train_hira.py --dataset=winog --batch=16 --output_folder=temp --ckpt=./results_ia3/gpt-neo-1.3B-common_170k-ia3-lr=1.00e-05--2025-05-24-06-58-08 --beam_size=8
python train_hira.py --dataset=arce --batch=16 --output_folder=temp --ckpt=./results_ia3/gpt-neo-1.3B-common_170k-ia3-lr=1.00e-05--2025-05-24-06-58-08 --beam_size=8
python train_hira.py --dataset=arcc --batch=16 --output_folder=temp --ckpt=./results_ia3/gpt-neo-1.3B-common_170k-ia3-lr=1.00e-05--2025-05-24-06-58-08 --beam_size=8
python train_hira.py --dataset=obqa --batch=16 --output_folder=temp --ckpt=./results_ia3/gpt-neo-1.3B-common_170k-ia3-lr=1.00e-05--2025-05-24-06-58-08 --beam_size=8