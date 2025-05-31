# NLP-25Spring-FDU
Repository of the final project of Natural Language Processing 2025 Spring at Fudan University

train our model with 4 peft_types:
python train_hira.py --peft_type=hira --model_name=EleutherAI/gpt-neo-1.3B --r_ab=64 --target_modules="q_proj,v_proj" --batch=8 --grad_acc=32 --lr=5e-5 --epoch=1 --load_bit=16 --enable_grad_ckpt --dataset=common_170k --seed=36 --warmup=100 --eval_strategy=steps --eval_steps=80 --output_folder=results_hira

python train_hira.py --peft_type=fft --model_name=EleutherAI/gpt-neo-1.3B --batch=4 --grad_acc=64 --lr=1e-5 --epoch=1 --load_bit=16 --enable_grad_ckpt --dataset=common_170k --seed=36 --warmup=100 --eval_strategy=steps --eval_steps=80 --output_folder=results_fft

python train_hira.py --peft_type=lora --model_name=EleutherAI/gpt-neo-1.3B --r_ab=64 --target_modules="q_proj,v_proj" --batch=8 --grad_acc=32 --lr=5e-5 --epoch=1 --load_bit=16 --enable_grad_ckpt --dataset=common_170k --seed=36 --eval_strategy=steps --eval_steps=80 --output_folder=results_lora

python train_hira.py --peft_type=ia3 --model_name=EleutherAI/gpt-neo-1.3B --target_modules="q_proj,v_proj" --batch=32 --grad_acc=1 --lr=1e-5 --epoch=1 --load_bit=16 --enable_grad_ckpt --dataset=common_170k --eval_steps=80 --output_folder=results_ia3

test for 4 peft_types:
