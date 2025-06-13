import os
import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer


def get_tokenizer(model_name="facebook/opt-1.3b"):
    return AutoTokenizer.from_pretrained(model_name, use_fast=False)


def get_models(model_name="facebook/opt-1.3b", enable_checkpoint=False):
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        # device_map='auto',
        torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16, low_cpu_mem_usage=True
    )

    if enable_checkpoint:
        model.gradient_checkpointing_enable()  # reduce number of stored activations

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    return model, tokenizer

def get_adapter_models(
    model_name="facebook/opt-1.3b", 
    enable_checkpoint=False, 
    load_bit=16,
    adapter_size=64
):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch.nn as nn

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        load_in_8bit=(load_bit == 8),
        torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    )
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)

    # Freeze main model parameters
    for param in model.parameters():
        param.requires_grad = False
        if param.ndim == 1:
            param.data = param.data.to(torch.float32)

    # Enable gradient checkpointing
    if enable_checkpoint:
        model.gradient_checkpointing_enable()
    model.enable_input_require_grads()

    # Force output to float32
    class CastOutputToFloat(nn.Sequential):
        def forward(self, x): 
            return super().forward(x).to(torch.float32)
    model.lm_head = CastOutputToFloat(model.lm_head)

    # Define adapter module
    class Adapter(nn.Module):
        def __init__(self, input_dim, bottleneck_dim=64):
            super().__init__()
            self.down_proj = nn.Linear(input_dim, bottleneck_dim)
            self.up_proj = nn.Linear(bottleneck_dim, input_dim)
            self.act = nn.GELU()

        def forward(self, x):
            return self.up_proj(self.act(self.down_proj(x)))  # No .weight reference

    # Insert adapters
    for name, module in model.named_modules():
        if "attn" in name and "c_attn" in name:
            out_features = module.weight.shape[0]
            module.adapter = Adapter(out_features, adapter_size)
            module.register_module("adapter", module.adapter)
            def forward_hook(module, input, output):
                return output + module.adapter(output)  # Correct usage
            module.register_forward_hook(forward_hook)
        elif "mlp" in name and "c_proj" in name:
            out_features = module.weight.shape[0]
            module.adapter = Adapter(out_features, adapter_size)
            module.register_module("adapter", module.adapter)
            def forward_hook(module, input, output):
                return output + module.adapter(output)
            module.register_forward_hook(forward_hook)

    # Enable only adapter parameters
    for name, param in model.named_parameters():
        if "adapter" in name:
            param.requires_grad = True

    print_trainable_parameters(model)
    return model, tokenizer, None

def get_ia3_models(
    model_name="facebook/opt-1.3b",
    enable_checkpoint=False,
    load_bit=16,
    target_modules=["k_proj", "v_proj", "down_proj"],  # Target modules for IA³ injection
    feedforward_modules=["down_proj"]  # Feedforward layer modules
):
    """
    Create model using built-in IA³ method from peft
    
    Args:
        model_name: Base model name
        enable_checkpoint: Whether to enable gradient checkpointing
        load_bit: Loading precision (16/8)
        target_modules: List of modules to inject IA³
        feedforward_modules: Feedforward layer modules (subset of target_modules)
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import IA3Config, get_peft_model
    import torch.nn as nn
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        load_in_8bit=(load_bit == 8),
        torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    )
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)

    # Freeze main model parameters
    for param in model.parameters():
        param.requires_grad = False
        if param.ndim == 1:
            param.data = param.data.to(torch.float32)

    # Enable gradient checkpointing
    if enable_checkpoint:
        model.gradient_checkpointing_enable()
    model.enable_input_require_grads()

    # Force output to float32
    class CastOutputToFloat(nn.Sequential):
        def forward(self, x): 
            return super().forward(x).to(torch.float32)
    model.lm_head = CastOutputToFloat(model.lm_head)

    # Create IA³ configuration
    ia3_config = IA3Config(
        task_type="CAUSAL_LM",  # Language model task
        target_modules=target_modules,  # Modules to inject IA³
        feedforward_modules=feedforward_modules  # Feedforward layer modules
    )

    # Apply IA³
    model = get_peft_model(model, ia3_config)
    print_trainable_parameters(model)
    return model, tokenizer, ia3_config
    
def get_layernorm_models(model_name="facebook/opt-1.3b", enable_checkpoint=False, load_bit=16,
                        target_modules=None, exclude_modules=None, modules_to_save=None):
    """
    Create model with LayerNorm tuning (LNTuning)
    
    Args:
        model_name: Base model name
        enable_checkpoint: Whether to enable gradient checkpointing
        load_bit: Loading precision (16/8)
        target_modules: Target module list or comma-separated string for LNTuning insertion
        exclude_modules: Module list or comma-separated string to exclude from LNTuning
        modules_to_save: Module list or comma-separated string to save separately
    """
    from peft import LNTuningConfig, get_peft_model
    import torch.nn as nn

    # Load parameter configuration
    load_params = {}
    if load_bit == 16:
        load_params['torch_dtype'] = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    # Load base model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        load_in_8bit=(load_bit == 8),
        **load_params,
    )

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)

    # Freeze base parameters
    for param in model.parameters():
        param.requires_grad = False
        if param.ndim == 1:  # Convert small parameters (like layernorm) to FP32 for stability
            param.data = param.data.to(torch.float32)

    # Enable gradient checkpointing
    if enable_checkpoint:
        model.gradient_checkpointing_enable()
    model.enable_input_require_grads()

    # Force output to float32
    class CastOutputToFloat(nn.Sequential):
        def forward(self, x): 
            return super().forward(x).to(torch.float32)
    model.lm_head = CastOutputToFloat(model.lm_head)

    # Module parameter processing function
    def process_modules(modules):
        if isinstance(modules, str):
            return modules.split(',')  # Convert comma-separated string to list
        return modules

    # Configure LNTuning parameters
    config = LNTuningConfig(
        task_type="CAUSAL_LM",  # Task type
        target_modules=process_modules(target_modules),  # Target modules
        exclude_modules=process_modules(exclude_modules),  # Exclude modules
        modules_to_save=process_modules(modules_to_save),  # Modules to save
    )

    # Apply LNTuning
    model = get_peft_model(model, config)
    print_trainable_parameters(model)  # Print trainable parameter statistics
    return model, tokenizer, config

def get_lora_models(model_name="facebook/opt-1.3b", enable_checkpoint=False, load_bit=16,
                   r=8, target_modules=None, lora_alpha=16, lora_dropout=0.1,
                   init_a='kaiming', init_b='zero', train_ab='yy'):

    from peft import LoraConfig, get_peft_model
    """
    Create model with LoRA adapters
    
    Args:
        model_name: Base model name
        enable_checkpoint: Whether to enable gradient checkpointing
        load_bit: Loading precision (16/8)
        r: LoRA rank
        target_modules: Modules to insert LoRA
        lora_alpha: LoRA scaling factor
        lora_dropout: LoRA dropout rate
        init_a: A matrix initialization method
        init_b: B matrix initialization method
        train_ab: Whether to train A/B matrices
    """
    load_params = {}
    if load_bit == 16:
        load_params['torch_dtype'] = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    
    # Load base model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        load_in_8bit=(load_bit == 8),
        **load_params,
    )
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    
    # Freeze base parameters
    for param in model.parameters():
        param.requires_grad = False
        if param.ndim == 1:
            param.data = param.data.to(torch.float32)
    
    # Enable gradient checkpointing
    if enable_checkpoint:
        model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    
    # Force output to float32
    class CastOutputToFloat(nn.Sequential):
        def forward(self, x): 
            return super().forward(x).to(torch.float32)
    model.lm_head = CastOutputToFloat(model.lm_head)
    
    # Process target modules
    if target_modules is None:
        target_modules = ["q_proj", "v_proj"]  # Default LoRA modules
    elif isinstance(target_modules, str):
        target_modules = target_modules.split(',')
    
    # Parse training parameters
    train_a, train_b = False, False
    if len(train_ab) >= 1:
        train_a = (train_ab[0].lower() == 'y')
    if len(train_ab) >= 2:
        train_b = (train_ab[1].lower() == 'y')
    
    # Create LoRA configuration
    config = LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        target_modules=target_modules,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        # init_a=init_a,
        # init_b=init_b,
        # train_a=train_a,
        # train_b=train_b,
    )
    
    # Apply LoRA
    model = get_peft_model(model, config)
    print_trainable_parameters(model)
    return model, tokenizer, config


def get_kasa_models(model_name="facebook/opt-1.3b", enable_checkpoint=False, load_bit=16,
                   rank=4, block_size=256, decompose_both=True, 
                   init_a='kaiming', init_b='zero', train_ab='yy'):

    from peft import KasaConfig, get_peft_model
    """
    Create model with KASA adapters (requires PEFT>=0.6.0)
    
    Args:
        model_name: Base model name
        enable_checkpoint: Whether to enable gradient checkpointing
        load_bit: Loading precision (16/8)
        rank: KASA rank
        block_size: Block size
        decompose_both: Whether to decompose QKV
        init_a: A matrix initialization method
        init_b: B matrix initialization method
        train_ab: Whether to train A/B matrices
    """
    try:
        from peft import KasaConfig  # PEFT 0.6+ supports KASA
    except ImportError:
        raise ImportError("KASA requires peft>=0.6.0. Please install with pip install -U peft")
    
    load_params = {}
    if load_bit == 16:
        load_params['torch_dtype'] = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    
    # Load base model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        load_in_8bit=(load_bit == 8),
        **load_params,
    )
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    
    # Freeze base parameters
    for param in model.parameters():
        param.requires_grad = False
        if param.ndim == 1:
            param.data = param.data.to(torch.float32)
    
    # Enable gradient checkpointing
    if enable_checkpoint:
        model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    
    # Force output to float32
    class CastOutputToFloat(nn.Sequential):
        def forward(self, x): 
            return super().forward(x).to(torch.float32)
    model.lm_head = CastOutputToFloat(model.lm_head)
    
    # Parse training parameters
    train_a, train_b = False, False
    if len(train_ab) >= 1:
        train_a = (train_ab[0].lower() == 'y')
    if len(train_ab) >= 2:
        train_b = (train_ab[1].lower() == 'y')
    
    # Create KASA configuration
    config = KasaConfig(
        r=rank,
        block_size=block_size,
        decompose_both=decompose_both,
        task_type="CAUSAL_LM",
        init_a=init_a,
        init_b=init_b,
        train_a=train_a,
        train_b=train_b,
    )
    
    # Apply KASA
    model = get_peft_model(model, config)
    print_trainable_parameters(model)
    return model, tokenizer, config




def get_fft_models(model_name="facebook/opt-1.3b", enable_checkpoint=False, load_bit=16):
    from hira import LoraConfig, get_peft_model
    load_params = {}
    if load_bit == 16:
        load_params = {'torch_dtype': torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16}
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        load_in_8bit=(load_bit == 8),
        #         device_map='auto',
        **load_params,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    if enable_checkpoint:
        model.gradient_checkpointing_enable()  # reduce number of stored activations
    model.enable_input_require_grads()

    print_trainable_parameters(model)
    return model, tokenizer, None


def get_hira_models(model_name="facebook/opt-1.3b", enable_checkpoint=False, load_bit=16,
                    r_ab=16, target_modules=None, init_a='kaiming',
                    init_b='zero', train_ab='yy',
                    rand_R=False):
    from hira import LoraConfig, get_peft_model
    from huggingface_hub import HfApi
    HfApi.endpoint = "https://hf-mirror.com"  # Set HuggingFace download endpoint
    os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'  # Double protection environment variable

    print(f"Current download endpoint: {HfApi.endpoint}")
    load_params = {}
    if load_bit == 16:
        load_params = {'torch_dtype': torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16}
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        load_in_8bit=(load_bit == 8),
        #         device_map='auto',
        **load_params,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)

    for param in model.parameters():
        param.requires_grad = False  # freeze the model - train adapters later
        if param.ndim == 1:
            # cast the small parameters (e.g. layernorm) to fp32 for stability
            param.data = param.data.to(torch.float32)
    if enable_checkpoint:
        model.gradient_checkpointing_enable()  # reduce number of stored activations
    model.enable_input_require_grads()

    class CastOutputToFloat(nn.Sequential):
        def forward(self, x): return super().forward(x).to(torch.float32)

    model.lm_head = CastOutputToFloat(model.lm_head)


    if target_modules is not None:
        target_modules = target_modules.split(',')
    else:
        target_modules = ["q_proj", "v_proj"]
    _train_ab = [True, True]
    for idx, char in enumerate(train_ab):
        _train_ab[idx] = (char == 'y')
    config = LoraConfig(
        rand_R=rand_R,
        scale_ab=1.0,
        r_ab=r_ab,
        target_modules=target_modules,
        bias="none",
        task_type="CAUSAL_LM",
        init_a=init_a,
        init_b=init_b,
        train_a=_train_ab[0],
        train_b=_train_ab[1],
    )

    model = get_peft_model(model, config)
    print_trainable_parameters(model)
    return model, tokenizer, config


# Using LoRA blocks (actually HiRA)
def get_hira_models_ch(model_name="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", enable_checkpoint=False, load_bit=16,
                    r_ab=16, target_modules=None, init_a='kaiming',
                    init_b='zero', train_ab='yy',
                    rand_R=False):
    """
    Force use of hf-mirror.com for model download, not dependent on environment variables
    """
    from hira import LoraConfig, get_peft_model
    # Force set mirror endpoint (key modification)
    from huggingface_hub import HfApi
    HfApi.endpoint = "https://hf-mirror.com"  # Set HuggingFace download endpoint
    os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'  # Double protection environment variable
    
    print(f"Current download endpoint: {HfApi.endpoint}")

    # Set loading parameters
    load_params = {
        'trust_remote_code': True,  # DeepSeek requires remote code support
    }
    if load_bit == 16:
        load_params['torch_dtype'] = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    elif load_bit == 8:
        load_params['load_in_8bit'] = True
    elif load_bit == 4:
        load_params['load_in_4bit'] = True

    print(f"Model identifier in use: {model_name}")

    # Load model and tokenizer
    try:
        model = AutoModelForCausalLM.from_pretrained(model_name, **load_params)
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, use_fast=False)
    except Exception as e:
        raise RuntimeError(
            f"Download from hf-mirror.com failed, please check:\n"
            f"1. Network access to https://hf-mirror.com\n"
            f"2. Model name is correct: {model_name}\n"
            f"Original error: {str(e)}"
        )

    # Set model parameters as non-trainable
    for param in model.parameters():
        param.requires_grad = False
        if param.ndim == 1:
            param.data = param.data.to(torch.float32)

    # Enable gradient checkpointing (optional)
    if enable_checkpoint:
        model.gradient_checkpointing_enable()
    model.enable_input_require_grads()

    # Force output to float32
    class CastOutputToFloat(nn.Sequential):
        def forward(self, x): return super().forward(x).to(torch.float32)
    model.lm_head = CastOutputToFloat(model.lm_head)

    # Set LoRA configuration
    if target_modules is not None:
        target_modules = target_modules.split(',')
    else:
        target_modules = ["q_proj", "k_proj", "v_proj"]  # Default LoRA insertion modules

    _train_ab = [True, True]
    for idx, char in enumerate(train_ab):
        _train_ab[idx] = (char == 'y')

    config = LoraConfig(
        rand_R=rand_R,
        scale_ab=1.0,
        r_ab=r_ab,
        target_modules=target_modules,
        bias="none",
        task_type="CAUSAL_LM",
        init_a=init_a,
        init_b=init_b,
        train_a=_train_ab[0],
        train_b=_train_ab[1],
    )

    # Apply LoRA
    model = get_peft_model(model, config)

    # Print trainable parameters
    print_trainable_parameters(model)

    return model, tokenizer, config




def get_prefix_tuning_models(model_name="facebook/opt-1.3b", enable_checkpoint=False, load_bit=8, virtual_tokens=8):
    load_params = {}
    if load_bit == 16:
        load_params = {'torch_dtype': torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16}
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        load_in_8bit=(load_bit == 8),
        **load_params,
    )
    from hira import LoraConfig, get_peft_model
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)

    for param in model.parameters():
        param.requires_grad = False  # freeze the model - train adapters later
        if param.ndim == 1:
            # cast the small parameters (e.g. layernorm) to fp32 for stability
            param.data = param.data.to(torch.float32)
    if enable_checkpoint:
        model.gradient_checkpointing_enable()  # reduce number of stored activations
    model.enable_input_require_grads()

    class CastOutputToFloat(nn.Sequential):
        def forward(self, x): return super().forward(x).to(torch.float32)

    model.lm_head = CastOutputToFloat(model.lm_head)

    from peft import PrefixTuningConfig, get_peft_model

    config = PrefixTuningConfig(
        num_virtual_tokens=virtual_tokens,
        task_type="CAUSAL_LM"
    )

    model = get_peft_model(model, config)
    print_trainable_parameters(model)
    return model, tokenizer, config


def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )
    return {"trainable": trainable_params, "all": all_param, "trainable%": 100 * trainable_params / all_param}
