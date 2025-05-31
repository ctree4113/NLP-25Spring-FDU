# NLP-25Spring-FDU
Repository of the final project of Natural Language Processing 2025 Spring at Fudan University

训练：
通过如下四个命令行操作：
python train_hira.py \
    --peft_type hira \
    --model_name EleutherAI/gpt-neo-1.3B \
    --r_ab 64 \
    --target_modules "q_proj,v_proj" \
    --batch 8 \
    --grad_acc 32 \
    --lr 5e-5 \
    --epoch 1 \
    --load_bit 16 \
    --enable_grad_ckpt \
    --dataset common_170k \
    --seed 36 \
    --warmup 100 \
    --eval_strategy steps \
    --eval_steps 80 \
    --output_folder results_hira


python train_hira.py \
    --peft_type fft \
    --model_name EleutherAI/gpt-neo-1.3B \
    --batch 4 \
    --grad_acc 64 \
    --lr 1e-5 \
    --epoch 1 \
    --load_bit 16 \
    --enable_grad_ckpt \
    --dataset common_170k \
    --seed 36 \
    --warmup 100 \
    --eval_strategy steps \
    --eval_steps 80 \
    --output_folder results_fft


python train_hira.py \
    --peft_type lora \
    --model_name EleutherAI/gpt-neo-1.3B \
    --r_ab 64 \
    --target_modules "q_proj,v_proj" \
    --batch 8 \
    --grad_acc 32 \
    --lr 5e-5 \
    --epoch 1 \
    --load_bit 16 \
    --enable_grad_ckpt \
    --dataset common_170k \
    --seed 36 \
    --eval_strategy steps \
    --eval_steps 80 \
    --output_folder results_lora


python train_hira.py \
    --peft_type ia3 \
    --model_name EleutherAI/gpt-neo-1.3B \
    --target_modules "q_proj,v_proj" \
    --batch 32 \
    --grad_acc 1 \
    --lr 1e-5 \
    --epoch 1 \
    --load_bit 16 \
    --enable_grad_ckpt \
    --dataset common_170k \
    --eval_steps 80 \
    --output_folder results_ia3

这四种微调方式去训练出4个微调模型



测试：
通过如下四种命令
test_all_dataset/fft_all_dataset.sh
test_all_dataset/hira_all_dataset.sh
test_all_dataset/lora_all_dataset.sh
test_all_dataset/ia3_all_dataset.sh
去得到每一种微调模型在9种数据集下的测试结果（是json格式的输出）

结果汇总：
通过test_all_dataset/analyze_all_methods.sh得到每种微调模型最后的结果汇总（按照准确率的形式输出）

