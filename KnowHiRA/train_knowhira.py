import os
import numpy as np
from dotenv import load_dotenv
load_dotenv('.env')
from tqdm import tqdm
from transformers import EarlyStoppingCallback

from dataset.dataset_hg import HGDataset
from dataset.format_inputs import format_causal_input, gen_max_new_token_map, token_length_map, dataset_map, task_map


from datetime import datetime
import jsonlines
import torch
import transformers
from pytictoc import TicToc
from models.get_models import print_trainable_parameters, get_tokenizer, get_prefix_tuning_models, get_hira_models, get_fft_models, get_knowhira_models
import argparse
from customized_trainer import customized_trainer

parser = argparse.ArgumentParser()
parser.add_argument('--peft_type', type=str,
                    choices=['prefix', 'hira', 'fft', 'knowhira'])
parser.add_argument('--enable_grad_ckpt', action='store_true')
parser.add_argument('--batch', type=int, default=32)
parser.add_argument('--grad_acc', type=int, default=1)
parser.add_argument('--num_workers', type=int, default=0)
parser.add_argument('--warmup', type=int, default=10)
parser.add_argument('--weight_decay', type=float, default=0.01)
parser.add_argument('--epoch', type=float, default=1)
parser.add_argument('--lr', type=float, default=1e-5)
parser.add_argument('--model_name', type=str, default="facebook/opt-125m")
parser.add_argument('--ckpt', type=str, default=None)
parser.add_argument('--dataset', type=str, default='boolq',
                    choices=['e2e_nlg', 'dailydialog', 'samsum', 'e2e_cleaned', 'boolq', 'mmlu', 'common_170k', 'gsm8k',
                             'piqa', 'siqa', 'hellas', 'winog', 'arce', 'arcc', 'obqa', 'convai2', 'meta_math',
                             'common_all'])
parser.add_argument('--dataset_analysis', action='store_true')

parser.add_argument('--dataset_ratio', type=float, default=1.0)
parser.add_argument('--local_rank', type=int, default=-1)
parser.add_argument('--ds_config', type=str, default=None)
parser.add_argument('--output_folder', type=str, default='results_knowhira')
parser.add_argument('--load_bit', type=int, default=16)
parser.add_argument('--r_ab', type=int, default=16)
parser.add_argument('--target_modules', type=str, default='q_proj, v_proj')
parser.add_argument('--eval_strategy', type=str, default='epoch', choices=['no', 'steps', 'epoch'])
parser.add_argument('--eval_steps', type=float, default=1.0)
parser.add_argument('--max_new_tokens', type=int, default=None)
parser.add_argument('--beam_size', type=int, default=None)
parser.add_argument('--virtual_tokens', type=int, default=8)
parser.add_argument('--compute_rank', action='store_true')
parser.add_argument('--compute_norm', action='store_true')
parser.add_argument('--load_order', type=int, default=-1)
parser.add_argument('--init_ab', type=str, default='kaiming,zero')
parser.add_argument('--train_ab', type=str, default='yy', help='y means yes, n means no')
parser.add_argument('--seed', default=None, type=int)
parser.add_argument('--do_sample', default='false', type=str)
parser.add_argument('--rand_R', action='store_true')
parser.add_argument('--exp_name', default='', type=str)
parser.add_argument('--decoding', type=str, default='default', choices=['default', 'greedy', 'fixed'])
parser.add_argument('--save_total_limit', type=int, default=1)
parser.add_argument('--early_stop_patience', type=int, default=0)

# KnowHiRA specific parameters
parser.add_argument('--knowledge_alpha', type=float, default=0.6, help='Weight for knowledge guidance')
parser.add_argument('--ortho_lambda', type=float, default=0.0001, help='Weight for orthogonal regularization')
parser.add_argument('--svd_rank_ratio', type=float, default=0.85, help='Ratio of SVD components to use')
parser.add_argument('--spectrum_init_scale', type=float, default=0.1, help='Scale for spectrum-aware initialization')
parser.add_argument('--adaptive_gating', action='store_true', help='Whether to use adaptive knowledge gating')

COMPUTE_DS_LENGTH = False
args = parser.parse_args()

# Detect if current process is main process for distributed training
def is_main_process():
    """Detect if current process is the main process"""
    if 'LOCAL_RANK' in os.environ:
        return int(os.environ['LOCAL_RANK']) == 0
    elif 'RANK' in os.environ:
        return int(os.environ['RANK']) == 0
    else:
        return True  # Default to main process for single GPU case

is_main_process = is_main_process()

# Output path settings for checkpoint evaluation mode
if args.compute_rank or args.compute_norm:
    assert args.ckpt is not None
if args.ckpt is not None:
    output_name = 'output_{}_{}'.format(args.load_order, args.dataset)
    if args.dataset == 'mmlu':
        output_name += '_prob'
    if args.max_new_tokens is not None:
        output_name += '_maxT={}'.format(args.max_new_tokens)
    if args.beam_size is not None:
        output_name += '_beam={}'.format(args.beam_size)
    output_path = '{}/{}_eval.jsonl'.format(args.ckpt, output_name)
    if os.path.exists(output_path) and not (args.compute_rank or args.compute_norm):
        print(f"File exists, skipped.")
        exit(0)
    print(f"Current ckpt args only supports inference!")
    output_jsonl = os.path.join(args.ckpt, 'output.jsonl')
    if os.path.exists(output_jsonl):
        with jsonlines.open(output_jsonl) as reader:
            dict_args = reader.read()
        print(f"dict_args: {dict_args}")
        print("Overriding args")
        args.peft_type = dict_args['peft_type']
        args.model_name = dict_args['model_name']
        args.r_ab = dict_args['r_ab']
        if 'rand_R' in dict_args.keys():
            args.rand_R = dict_args['rand_R']
        args.target_modules = dict_args['target_modules']
        # KnowHiRA specific parameters
        if 'knowledge_alpha' in dict_args.keys():
            args.knowledge_alpha = dict_args['knowledge_alpha']
        if 'ortho_lambda' in dict_args.keys():
            args.ortho_lambda = dict_args['ortho_lambda']
        if 'svd_rank_ratio' in dict_args.keys():
            args.svd_rank_ratio = dict_args['svd_rank_ratio']
        if 'spectrum_init_scale' in dict_args.keys():
            args.spectrum_init_scale = dict_args['spectrum_init_scale']
        if 'adaptive_gating' in dict_args.keys():
            args.adaptive_gating = dict_args['adaptive_gating']
    else:
        print(f'cannot find {output_jsonl}')
        exit(0)

dataset_name = args.dataset

MAX_NEW_TOKEN_LENGTH = gen_max_new_token_map[dataset_name]
MAX_TOKEN_LENGTH = token_length_map[dataset_name]

if args.max_new_tokens is not None:
    MAX_NEW_TOKEN_LENGTH = args.max_new_tokens

# Convert args to dict
args_dict = vars(args)
model_name = args.model_name
peft_type = args.peft_type
train_ab = args.train_ab

# Create directory based on timestamp
exp_name = f"{args.output_folder}/{model_name.split('/')[-1]}-{dataset_name}-{peft_type}-lr={format(args.lr, '.2e')}-"
if args.load_bit != 16:
    exp_name = exp_name + f'{args.load_bit}bit-'
if peft_type in ['hira', 'knowhira']:
    init_ab_ = ''.join([i[0] for i in args.init_ab.split(',')])
    exp_name = exp_name + f'r_ab={args.r_ab}-'
    exp_name = exp_name + f'init={init_ab_}-'
    exp_name = exp_name + f'train={train_ab}-'
    if peft_type == 'knowhira':
        exp_name = exp_name + f'kalpha={args.knowledge_alpha}-'
        exp_name = exp_name + f'ortho={args.ortho_lambda}-'

if args.seed is not None:
    def seed_everything(seed: int):
        import random, os
        import numpy as np
        import torch

        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True

    exp_name = exp_name + f'seed={args.seed}-'
    seed_everything(args.seed)

output_dir_by_time = exp_name + args.exp_name + '-' + datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

if args.ckpt is None:
    os.makedirs(output_dir_by_time, exist_ok=True)

train_dataset = HGDataset(dataset_map[dataset_name], 'train', task_map[dataset_name], training_ratio=args.dataset_ratio)
valid_dataset = HGDataset(dataset_map[dataset_name], 'validation', task_map[dataset_name],
                          training_ratio=args.dataset_ratio)
test_dataset = HGDataset(dataset_map[dataset_name], 'test', task_map[dataset_name], training_ratio=args.dataset_ratio)

if args.dataset_analysis:
    tokenizer_ds = get_tokenizer(model_name=model_name)
    train_dataset.length_analysis(tokenizer_ds)
    valid_dataset.length_analysis(tokenizer_ds)
    test_dataset.length_analysis(tokenizer_ds)
    exit(0)

if peft_type == 'hira':
    init_a, init_b = args.init_ab.split(',')
    model, tokenizer, model_config = get_hira_models(load_bit=args.load_bit,
                                                     model_name=model_name, enable_checkpoint=args.enable_grad_ckpt,
                                                     r_ab=args.r_ab,target_modules=args.target_modules,
                                                     train_ab=args.train_ab,
                                                     rand_R=args.rand_R)
elif peft_type == 'knowhira':
    init_a, init_b = args.init_ab.split(',')
    model, tokenizer, model_config = get_knowhira_models(
        load_bit=args.load_bit,
        model_name=model_name, 
        enable_checkpoint=args.enable_grad_ckpt,
        r_ab=args.r_ab,
        target_modules=args.target_modules,
        init_a=init_a,
        init_b=init_b,
        train_ab=args.train_ab,
        knowledge_alpha=args.knowledge_alpha,
        ortho_lambda=args.ortho_lambda,
        svd_rank_ratio=args.svd_rank_ratio,
        spectrum_init_scale=args.spectrum_init_scale,
        adaptive_gating=args.adaptive_gating
    )
elif peft_type == 'fft':
    model, tokenizer, model_config = get_fft_models(load_bit=args.load_bit,
                                                    model_name=model_name, enable_checkpoint=args.enable_grad_ckpt)
elif peft_type == 'prefix':
    model, tokenizer, model_config = get_prefix_tuning_models(load_bit=args.load_bit, model_name=model_name,
                                                              enable_checkpoint=args.enable_grad_ckpt,
                                                              virtual_tokens=args.virtual_tokens)
else:
    raise NotImplementedError('Not supported model!')

trainable_params = print_trainable_parameters(model)

def get_parameter_dict(model):
    return dict(model.named_parameters())

tokenizer_left = get_tokenizer(model_name=model_name)
tokenizer_left.padding_side = 'left'

tokenizer_right = get_tokenizer(model_name=model_name)
tokenizer_right.padding_side = 'right'

# Strengthen pad_token_id setting to ensure correct configuration for GPT-Neo and other models
if tokenizer_left.pad_token_id is None and 'llama-3' in model_name.lower():
    tokenizer_left.pad_token = tokenizer_left.bos_token
    tokenizer_right.pad_token = tokenizer_right.bos_token
    tokenizer.pad_token = tokenizer.bos_token
    model.config.pad_token_id = tokenizer.bos_token_id
elif tokenizer_left.pad_token_id is None:
    # For GPT-Neo and other models, use eos_token as pad_token
    if hasattr(tokenizer_left, 'eos_token') and tokenizer_left.eos_token is not None:
        tokenizer_left.pad_token = tokenizer_left.eos_token
        tokenizer_right.pad_token = tokenizer_right.eos_token
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.eos_token_id
    else:
        # Fallback: use unk_token
        tokenizer_left.pad_token = tokenizer_left.unk_token
        tokenizer_right.pad_token = tokenizer_right.unk_token
        tokenizer.pad_token = tokenizer.unk_token
        model.config.pad_token_id = tokenizer.unk_token_id

# Verify pad_token_id setting
print(f"Tokenizer configuration verification:")
print(f"  pad_token: {tokenizer.pad_token}")
print(f"  pad_token_id: {tokenizer.pad_token_id}")
print(f"  model.config.pad_token_id: {model.config.pad_token_id}")

# Ensure all tokenizer configurations are consistent
if tokenizer.pad_token_id != model.config.pad_token_id:
    print(f"Inconsistency detected, forcing synchronization...")
    model.config.pad_token_id = tokenizer.pad_token_id
    print(f"  Synchronized to: {model.config.pad_token_id}")

# Compute length for this dataset
if COMPUTE_DS_LENGTH:
    all_data = [train_dataset.__getitem__(idx) for idx in range(train_dataset.__len__())]
    all_data = [a['input'] + a['target'] for a in all_data]
    all_data_ids = tokenizer_right(all_data)
    all_data_len = [len(a) for a in all_data_ids['input_ids']]
    all_data_len = torch.tensor(all_data_len, dtype=torch.float)
    print(f"""
        AVG: {all_data_len.mean()}
        MAX: {all_data_len.max()}
        MIN: {all_data_len.min()}
    """)

test_steps = 0

def data_collator_e2e(features, return_tensors="pt"):
    batchfied_features = {}
    keys = features[0].keys()
    for key in keys:
        batchfied_features[key] = [f[key] for f in features]
    split = batchfied_features['split'][0]
    for_inference = (split == 'test')
    template_type = 0
    if dataset_name in ['boolq', 'gsm8k', 'common_170k', 'piqa', 'siqa', 'hellas', 'winog', 'arce', 'arcc', 'obqa',
                        'common_all']:
        template_type = 4
    if dataset_name in ['mmlu']:
        template_type = 7
    lm_input, lm_target = format_causal_input(batchfied_features, tokenizer_left, tokenizer_right,
                                              template_type=template_type, max_token_length=MAX_TOKEN_LENGTH,
                                              for_test=for_inference, shift_target=False,
                                              target_length=MAX_NEW_TOKEN_LENGTH)
    # Replace target pad to -100
    lm_target_ce = lm_target.clone()
    lm_target_ce[lm_target_ce == tokenizer_left.pad_token_id] = -100
    if peft_type in ['prefix']:
        lm_input['attention_mask'] = None
    batch = {**lm_input, 'labels': lm_target_ce}
    if for_inference:
        batch = lm_input
    return batch

generation_config = transformers.generation.GenerationConfig(
    max_length=MAX_TOKEN_LENGTH,
    num_beams=1,
)

callbacks = []
if args.early_stop_patience > 0:
    callbacks.append(EarlyStoppingCallback(early_stopping_patience=args.early_stop_patience))

eval_bz = max(1, int(args.batch / 4))
if dataset_name == 'common_170k':
    eval_bz = args.batch

# Add custom trainer with custom loss function for KnowHiRA
class KnowHiRASeq2SeqTrainer(customized_trainer.Seq2SeqTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.step_count = 0
        self.log_interval = 1000  # Log every 1000 steps to greatly reduce output frequency
    
    def compute_loss(self, model, inputs, return_outputs=False):
        """
        Custom loss function that adds orthogonal regularization loss for KnowHiRA
        """
        # Standard language model loss
        outputs = model(**inputs)
        lm_loss = outputs.get('loss')
        
        # If KnowHiRA model, add orthogonal regularization loss
        if peft_type == 'knowhira':
            ortho_loss = 0.0
            ortho_count = 0
            for module in model.modules():
                if hasattr(module, 'get_orthogonal_loss'):
                    module_ortho_loss = module.get_orthogonal_loss()
                    ortho_loss += module_ortho_loss
                    ortho_count += 1
            
            # Keep monitoring logic but greatly reduce print frequency (every 1000 steps)
            self.step_count += 1
            if self.step_count % self.log_interval == 0 and is_main_process:
                print(f"\nStep {self.step_count} Loss overview:")
                print(f"  LM Loss: {lm_loss.item():.4f} | Ortho Loss: {ortho_loss.item():.6f} | Total: {(lm_loss + ortho_loss).item():.4f}")
                if lm_loss.item() > 0:
                    print(f"  Ortho ratio: {(ortho_loss/lm_loss).item():.6f}")
            
            # Optional: If occasional loss monitoring is needed, uncomment below and adjust log_interval
            # Recommend setting to 500-1000 steps to avoid excessive frequency
            # self.step_count += 1
            # if self.step_count % 500 == 0 and is_main_process:  # Print every 500 steps
            #     print(f"\nStep {self.step_count} Loss details:")
            #     print(f"  Language model loss: {lm_loss.item():.6f}")
            #     print(f"  Orthogonal regularization loss: {ortho_loss.item():.6f}")
            #     print(f"  Total loss: {(lm_loss + ortho_loss).item():.6f}")
            #     print(f"  Orthogonal/Language model ratio: {(ortho_loss/lm_loss).item():.6f}")
            
            # Original detailed loss logging code (disabled)
            # self.step_count += 1
            # if self.step_count % self.log_interval == 0 and is_main_process:
            #     print(f"\nStep {self.step_count} Loss details:")
            #     print(f"  Language model loss: {lm_loss.item():.6f}")
            #     print(f"  Orthogonal regularization loss: {ortho_loss.item():.6f}")
            #     print(f"  Number of orthogonal modules: {ortho_count}")
            #     print(f"  Total loss: {(lm_loss + ortho_loss).item():.6f}")
            #     if lm_loss.item() > 0:
            #         print(f"  Orthogonal/Language model ratio: {(ortho_loss/lm_loss).item():.6f}")
            
            if ortho_loss > 0:
                total_loss = lm_loss + ortho_loss
            else:
                total_loss = lm_loss
        else:
            total_loss = lm_loss
                
        return (total_loss, outputs) if return_outputs else total_loss

trainer_class = KnowHiRASeq2SeqTrainer if peft_type == 'knowhira' else customized_trainer.Seq2SeqTrainer

trainer_args = transformers.Seq2SeqTrainingArguments(
    deepspeed=args.ds_config,
    local_rank=args.local_rank,
    dataloader_num_workers=args.num_workers,
    resume_from_checkpoint=args.ckpt,
    per_device_train_batch_size=args.batch,
    per_device_eval_batch_size=eval_bz,
    gradient_accumulation_steps=args.grad_acc,
    gradient_checkpointing=args.enable_grad_ckpt,
    warmup_steps=args.warmup,
    weight_decay=args.weight_decay,
    num_train_epochs=args.epoch,
    learning_rate=args.lr,
    bf16=True if torch.cuda.is_bf16_supported() and args.load_bit == 16 else False,
    fp16=True if not torch.cuda.is_bf16_supported() and args.load_bit == 16 else False,
    metric_for_best_model='eval_loss',
    greater_is_better=False,
    logging_steps=10,
    remove_unused_columns=False,
    save_on_each_node=False,
    save_safetensors=peft_type == 'fft',
    output_dir=output_dir_by_time,
    do_eval=True,
    evaluation_strategy=args.eval_strategy,
    save_strategy=args.eval_strategy,
    save_steps=args.eval_steps,
    logging_strategy='steps',
    save_total_limit=args.save_total_limit,
    report_to=['tensorboard'],
    eval_steps=args.eval_steps,
    eval_accumulation_steps=1,
    generation_config=generation_config,
    load_best_model_at_end=True,
    predict_with_generate=True,
    ddp_find_unused_parameters=False,
    lr_scheduler_type='cosine',
    warmup_ratio=0.1,
    dataloader_pin_memory=True,
)

trainer = trainer_class(
    model=model,
    train_dataset=train_dataset,
    eval_dataset=valid_dataset,
    callbacks=callbacks,
    args=trainer_args,
    data_collator=data_collator_e2e
)

if args.ckpt is not None:
    trainer._load_best_model(order=args.load_order)
    train_seconds = -1
else:
    train_tic = TicToc()
    train_tic.tic()
    trainer.train()
    train_seconds = train_tic.tocvalue()
    
    # Explicitly save final model state
    print("Saving final model checkpoint...")
    if args.local_rank in [-1, 0]:
        # Save model and configuration
        trainer.save_model()
        # Additionally save PEFT state
        if peft_type in ['hira', 'knowhira']:
            model.save_pretrained(output_dir_by_time)
        print(f"Model saved to: {output_dir_by_time}")

# Complete evaluation logic
kwgenargs = {}
if args.do_sample is not None:
    if args.do_sample.lower() in ['yes', 'true']:
        do_sample = True
    else:
        do_sample = False
    kwgenargs['do_sample'] = do_sample

if args.decoding == 'greedy':
    eval_result = trainer.predict(test_dataset,
                                  max_new_tokens=MAX_NEW_TOKEN_LENGTH,
                                  pad_token_id=tokenizer.pad_token_id,
                                  do_sample=False)
elif args.decoding == 'fixed':
    # Fixed decoding parameters to solve repetition generation problem
    # Create fixed parameters to avoid conflicts with kwgenargs
    fixed_kwargs = {
        'max_new_tokens': MAX_NEW_TOKEN_LENGTH,
        'pad_token_id': tokenizer.pad_token_id,
        'do_sample': True,  # Enable sampling
        'temperature': 0.3,  # Slightly increase temperature for diversity
        'top_p': 0.8,       # Stricter nucleus sampling
        'top_k': 10,        # Stricter top-k sampling
        'repetition_penalty': 1.5,  # Stronger repetition penalty
        'no_repeat_ngram_size': 2,  # Prevent 2-gram repetition (stricter)
        'length_penalty': 0.5,      # Stronger length penalty, encourage short answers
        'early_stopping': True,     # Early stopping
    }
    eval_result = trainer.predict(test_dataset, **fixed_kwargs)
elif dataset_name == 'common_170k' and args.decoding != 'fixed':
    # Only use original common_170k processing in non-fixed mode
    eval_result = trainer.predict(test_dataset,
                                  max_new_tokens=MAX_NEW_TOKEN_LENGTH,
                                  pad_token_id=tokenizer.pad_token_id,
                                  do_sample=False)
elif dataset_name == 'common_all':
    beam = 4
    eval_result = {}
    for _dataset_name in ['boolq', 'piqa', 'siqa', 'hellas', 'winog', 'arce', 'arcc', 'obqa']:
        print(f'decoding {_dataset_name}')
        test_dataset = HGDataset(dataset_map[_dataset_name], 'test', task_map[_dataset_name],
                                 training_ratio=args.dataset_ratio)
        _output_path = output_path.replace(dataset_name, _dataset_name)
        if os.path.exists(_output_path):
            print(f'{_output_path} exists, skipping')
            continue
        if args.beam_size is not None:
            beam = args.beam_size
        _eval_result = trainer.predict(test_dataset,
                                       max_new_tokens=MAX_NEW_TOKEN_LENGTH,
                                       pad_token_id=tokenizer.pad_token_id,
                                       temperature=0.1,
                                       top_p=0.75,
                                       top_k=40,
                                       num_beams=beam,
                                       **kwgenargs)
        eval_result[_output_path] = _eval_result
elif dataset_name in ['mmlu']:
    eval_results = []
    pbar = tqdm(test_dataset)
    id_a, id_b, id_c, id_d = tokenizer.convert_tokens_to_ids(['A', 'B', 'C', 'D'])
    options = ['A', 'B', 'C', 'D']
    context = []
    text_result = []
    ground_truth = []
    for row in pbar:
        model.eval()
        with torch.no_grad():
            keys = row.keys()
            batchfied_features = {}
            for key in keys:
                batchfied_features[key] = [row[key]]
            lm_input, lm_target = format_causal_input(batchfied_features, tokenizer_left, tokenizer_right,
                                                      template_type=7, max_token_length=MAX_TOKEN_LENGTH,
                                                      for_test=True, shift_target=False,
                                                      target_length=MAX_NEW_TOKEN_LENGTH)
            lm_input = lm_input.to('cuda')
            with torch.autocast('cuda'):
                prob = model(**lm_input).logits
            id_probs = prob[0][-1][[id_a, id_b, id_c, id_d]]
            prob_pred = options[id_probs.argmax().item()]
            answer = row['target']
            eval_results.append(prob_pred == answer)
            acc = np.asarray(eval_results).mean()
            pbar.set_postfix_str(f'Current ACC: {acc * 100}')
            context.append(row['input'])
            text_result.append(prob_pred)
            ground_truth.append(answer)
    print(f'ACC: {acc * 100}')
    with jsonlines.open(output_path, mode='w') as writer:
        writer.write(args_dict)
        writer.write(model_config.to_dict())
        writer.write({"acc": acc * 100})
        writer.write(trainable_params)
        for c, p, g in zip(context, text_result, ground_truth):
            writer.write({
                'context': c,
                'pred': p,
                'gt': g,
            })
    exit(0)
elif dataset_name in ['boolq', 'piqa', 'siqa', 'hellas', 'winog', 'arce', 'arcc', 'obqa', 'gsm8k']:
    beam = 4
    if args.beam_size is not None:
        beam = args.beam_size
    eval_result = trainer.predict(test_dataset,
                                  max_new_tokens=MAX_NEW_TOKEN_LENGTH,
                                  pad_token_id=tokenizer.pad_token_id,
                                  temperature=0.2,
                                  top_p=0.9,
                                  top_k=20,
                                  num_beams=beam,
                                  repetition_penalty=1.1,
                                  **kwgenargs)
else:
    # For other datasets (e2e/convai2 etc.)
    eval_result = trainer.predict(test_dataset,
                                  max_new_tokens=MAX_NEW_TOKEN_LENGTH,
                                  pad_token_id=tokenizer.pad_token_id,
                                  do_sample=False, num_beams=4,
                                  length_penalty=0.9, no_repeat_ngram_size=4)

# Fix: correctly handle eval_result path setting
if not isinstance(eval_result, dict):
    if args.ckpt is None:
        _final_output_path = '{}/output.jsonl'.format(output_dir_by_time)
    else:
        _final_output_path = output_path
    eval_result = {_final_output_path: eval_result}

for _output_path, _eval_result in eval_result.items():
    # convert logits into text
    if args.local_rank in [-1, 0]:
        logits = _eval_result.predictions
        logits[logits == -100] = tokenizer.pad_token_id
        raw_text_result = tokenizer.batch_decode(logits)
        text_result = []
        for tt in raw_text_result:
            tt = tt.replace(tokenizer.pad_token, '')
            keywords = [tokenizer.eos_token, 'Q:', 'R:']
            for keyword in keywords:
                if keyword in tt:
                    tt = tt[:tt.index(keyword)]
            text_result.append(tt)

        context = [test_dataset.__getitem__(i)['input'] for i in range(test_dataset.__len__())]
        ground_truth = [test_dataset.__getitem__(i)['target'] for i in range(test_dataset.__len__())]
        
        # Use baseline's simple cleaning approach
        # Only perform basic cleaning, no complex answer extraction
        cleaned_text_result = []
        for i, (original_pred, gt) in enumerate(zip(text_result, ground_truth)):
            # Basic cleaning: remove pad token and truncate
            cleaned_pred = original_pred.replace(tokenizer.pad_token, '').strip()
            
            # Remove content after common end symbols
            for keyword in [tokenizer.eos_token, 'Q:', 'R:']:
                if keyword in cleaned_pred:
                    cleaned_pred = cleaned_pred[:cleaned_pred.index(keyword)]
            
            cleaned_text_result.append({
                'pred': cleaned_pred.strip(),
                'gt': gt.strip()  # Keep original format
            })
        
        # Fix: consistently use _output_path (already set above)
        mem_used = torch.cuda.mem_get_info()[1] / 1024 / 1024 - torch.cuda.mem_get_info()[0] / 1024 / 1024

        with jsonlines.open(_output_path, mode='w') as writer:
            writer.write(args_dict)
            if peft_type != 'fft':
                writer.write(model_config.to_dict())
            else:
                writer.write('\n')
            writer.write({"mem_used": mem_used, "train_seconds": train_seconds})
            writer.write(trainable_params)
            for i, (c, cleaned_result) in enumerate(zip(context, cleaned_text_result)):
                writer.write({
                    'context': c,
                    'pred': cleaned_result['pred'],
                    'gt': cleaned_result['gt']
                })

print("Training/Evaluation script completed!")
