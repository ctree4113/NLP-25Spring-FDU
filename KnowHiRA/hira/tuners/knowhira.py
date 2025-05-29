# coding=utf-8
# KnowHiRA: Knowledge-aware Hadamard-integrated Rank Adaptation
# 
# This implementation enhances HiRA with knowledge-aware mechanisms inspired by KaSA
# Main innovations:
# 1. Knowledge-guided Hadamard Updates
# 2. Adaptive Knowledge Gating
# 3. Orthogonal Parameter Regularization  
# 4. Spectrum-aware Initialization

import math
import re
import warnings
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import List, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..import_utils import is_bnb_available
from ..utils import (
    TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING,
    PeftConfig,
    PeftType,
    _freeze_adapter,
    _get_submodules,
    transpose,
)

if is_bnb_available():
    import bitsandbytes as bnb


@dataclass
class KnowHiRAConfig(PeftConfig):
    """
    Configuration class for KnowHiRA (Knowledge-aware Hadamard-integrated Rank Adaptation).
    
    Extends HiRA with knowledge-aware mechanisms:
    - SVD-based knowledge structure guidance
    - Adaptive knowledge gating
    - Orthogonal parameter regularization
    - Spectrum-aware initialization
    """
    # HiRA parameters
    r_ab: int = field(default=16, metadata={"help": "KnowHiRA rank dimension"})
    target_modules: Optional[Union[List[str], str]] = field(
        default=None,
        metadata={
            "help": "List of module names or regex expression of the module names to replace with KnowHiRA."
        },
    )
    lora_alpha: int = field(default=32, metadata={"help": "LoRA alpha for scaling"})
    lora_dropout: float = field(default=0.05, metadata={"help": "LoRA dropout"})
    fan_in_fan_out: bool = field(
        default=False,
        metadata={"help": "Set this to True if the layer to replace stores weight like (fan_in, fan_out)"},
    )
    bias: str = field(default="none", metadata={"help": "Bias type. Can be 'none', 'all' or 'lora_only'"})
    modules_to_save: Optional[List[str]] = field(
        default=None,
        metadata={
            "help": "List of modules apart from adaptation layers to be set as trainable and saved."
        },
    )
    
    # KnowHiRA specific parameters
    init_a: str = field(default='kaiming', metadata={"help": "Initialization for A matrix"})
    init_b: str = field(default='zero', metadata={"help": "Initialization for B matrix"})
    scale_ab: float = field(default=1.0, metadata={"help": "Scaling factor for AB product"})
    train_a: bool = field(default=True, metadata={"help": "Whether to train A matrix"})
    train_b: bool = field(default=True, metadata={"help": "Whether to train B matrix"})
    
    # Knowledge-aware parameters
    knowledge_alpha: float = field(default=0.5, metadata={"help": "Weight for knowledge guidance"})
    ortho_lambda: float = field(default=0.01, metadata={"help": "Weight for orthogonal regularization"})
    svd_rank_ratio: float = field(default=0.8, metadata={"help": "Ratio of SVD components to use for knowledge"})
    spectrum_init_scale: float = field(default=1.0, metadata={"help": "Scale factor for spectrum-aware initialization"})
    adaptive_gating: bool = field(default=True, metadata={"help": "Whether to use adaptive knowledge gating"})
    
    init_lora_weights: bool = field(default=True, metadata={"help": "Whether to initialize weights"})

    def __post_init__(self):
        self.peft_type = PeftType.KNOWHIRA


class KnowHiRAModel(torch.nn.Module):
    """
    KnowHiRA Model that enhances HiRA with knowledge-aware mechanisms.
    """
    
    def __init__(self, model, config, adapter_name):
        super().__init__()
        self.model = model
        self.forward = self.model.forward
        self.peft_config = config
        self.add_adapter(adapter_name, self.peft_config[adapter_name])

    def add_adapter(self, adapter_name, config=None):
        if config is not None:
            model_config = self.model.config.to_dict() if hasattr(self.model.config, "to_dict") else self.model.config
            config = self._prepare_knowhira_config(config, model_config)
            self.peft_config[adapter_name] = config
        self._find_and_replace(adapter_name)
        if len(self.peft_config) > 1 and self.peft_config[adapter_name].bias != "none":
            raise ValueError(
                "KnowHiRAModel supports only 1 adapter with bias. When using multiple adapters, set bias to 'none' for all adapters."
            )
        mark_only_knowhira_as_trainable(self.model, self.peft_config[adapter_name].bias, config)
        if self.peft_config[adapter_name].inference_mode:
            _freeze_adapter(self.model, adapter_name)

    def _find_and_replace(self, adapter_name):
        knowhira_config: KnowHiRAConfig = self.peft_config[adapter_name]
        loaded_in_8bit = getattr(self.model, "is_loaded_in_8bit", False)
        if loaded_in_8bit and not is_bnb_available():
            raise ImportError(
                "To use KnowHiRA with 8-bit quantization, please install the `bitsandbytes` package."
            )
        
        is_target_modules_in_base_model = False
        kwargs = {
            "r_ab": knowhira_config.r_ab,
            "lora_alpha": knowhira_config.lora_alpha,
            "lora_dropout": knowhira_config.lora_dropout,
            "fan_in_fan_out": knowhira_config.fan_in_fan_out,
            "init_lora_weights": knowhira_config.init_lora_weights,
            "scale_ab": knowhira_config.scale_ab,
            "init_a": knowhira_config.init_a,
            "init_b": knowhira_config.init_b,
            "train_a": knowhira_config.train_a,
            "train_b": knowhira_config.train_b,
            "knowledge_alpha": knowhira_config.knowledge_alpha,
            "ortho_lambda": knowhira_config.ortho_lambda,
            "svd_rank_ratio": knowhira_config.svd_rank_ratio,
            "spectrum_init_scale": knowhira_config.spectrum_init_scale,
            "adaptive_gating": knowhira_config.adaptive_gating,
        }
        
        key_list = [key for key, _ in self.model.named_modules()]
        for key in key_list:
            if isinstance(knowhira_config.target_modules, str):
                target_module_found = re.fullmatch(knowhira_config.target_modules, key)
            else:
                target_module_found = any(key.endswith(target_key) for target_key in knowhira_config.target_modules)
            
            if target_module_found:
                if not is_target_modules_in_base_model:
                    is_target_modules_in_base_model = True
                parent, target, target_name = _get_submodules(self.model, key)
                
                if hasattr(target, "bias"):
                    bias = target.bias is not None
                
                if isinstance(target, KnowHiRALayer):
                    target.update_layer(adapter_name, **kwargs)
                else:
                    if isinstance(target, torch.nn.Linear):
                        in_features, out_features = target.in_features, target.out_features
                        new_module = KnowHiRALinear(
                            adapter_name, in_features, out_features, bias=bias, **kwargs
                        )
                        # Pass pre-trained weights for SVD analysis
                        new_module.weight.data = target.weight.data.clone()
                        if bias:
                            new_module.bias.data = target.bias.data.clone()
                        self._replace_module(parent, target_name, new_module, target)
                    elif isinstance(target, torch.nn.Embedding):
                        # Not supported for Embedding layers
                        pass
                    elif loaded_in_8bit and isinstance(target, bnb.nn.Linear8bitLt):
                        # Not supported for 8-bit quantization
                        pass
        
        if not is_target_modules_in_base_model:
            raise ValueError(
                f"Target modules {knowhira_config.target_modules} not found in the base model."
            )

    def _replace_module(self, parent_module, child_name, new_module, old_module):
        setattr(parent_module, child_name, new_module)
        new_module.weight = old_module.weight
        if hasattr(old_module, "bias"):
            new_module.bias = old_module.bias

        if getattr(old_module, "state", None) is not None:
            new_module.state = old_module.state
            new_module.to(old_module.weight.device)

        # dispatch to correct device
        for name, module in new_module.named_modules():
            if "lora_" in name:
                module.to(old_module.weight.device)

    @staticmethod
    def _prepare_knowhira_config(peft_config, model_config):
        if peft_config.target_modules is None:
            if model_config["model_type"] not in TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING:
                raise ValueError("Please specify `target_modules` in `peft_config`")
            peft_config.target_modules = TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING[
                model_config["model_type"]
            ]
        return peft_config

    def __getattr__(self, name: str):
        """Forward missing attributes to the wrapped module."""
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.model, name)

    def get_peft_config_as_dict(self, inference: bool = False):
        config_dict = {}
        for key, value in self.peft_config.items():
            config = {k: v.value if isinstance(v, Enum) else v for k, v in asdict(value).items()}
            if inference:
                config["inference_mode"] = True
        config_dict[key] = config
        return config

    def set_adapter(self, adapter_name):
        for module in self.model.modules():
            if isinstance(module, KnowHiRALayer):
                if module.merged:
                    warnings.warn("Adapter cannot be set when the model is merged. Unmerging the model first.")
                    module.unmerge()
                module.active_adapter = adapter_name


def mark_only_knowhira_as_trainable(model: nn.Module, bias: str = "none", config: KnowHiRAConfig = None) -> None:
    """Mark only KnowHiRA parameters as trainable"""
    for n, p in model.named_parameters():
        if "lora_" not in n:
            p.requires_grad = False
    
    if bias == "none":
        pass
    elif bias == "all":
        for n, p in model.named_parameters():
            if "bias" in n:
                p.requires_grad = True
    elif bias == "lora_only":
        for m in model.modules():
            if isinstance(m, KnowHiRALayer) and hasattr(m, "bias") and m.bias is not None:
                m.bias.requires_grad = True
    else:
        raise NotImplementedError
    
    if config:
        for n, p in model.named_parameters():
            if "lora_" in n:
                p.requires_grad = False
        if config.train_a:
            for n, p in model.named_parameters():
                if "lora_A" in n:
                    p.requires_grad = True
        if config.train_b:
            for n, p in model.named_parameters():
                if "lora_B" in n:
                    p.requires_grad = True
        # Knowledge gate parameters are always trainable
        for n, p in model.named_parameters():
            if "knowledge_gate" in n:
                p.requires_grad = True


class KnowHiRALayer:
    """
    KnowHiRA Layer base class, containing knowledge-aware mechanisms
    """
    
    def __init__(self, in_features: int, out_features: int):
        self.r_ab = {}
        self.lora_alpha = {}
        self.scaling_ab = {}
        self.lora_dropout = nn.ModuleDict({})
        self.lora_A = nn.ParameterDict({})
        self.lora_B = nn.ParameterDict({})
        
        # Knowledge-aware parameters
        self.knowledge_gate = nn.ParameterDict({})  # Knowledge gate parameters
        self.knowledge_alpha = {}
        self.ortho_lambda = {}
        self.svd_rank_ratio = {}
        self.spectrum_init_scale = {}
        self.adaptive_gating = {}
        
        # SVD components cache
        self.U_cache = {}
        self.S_cache = {} 
        self.V_cache = {}
        
        self.merged = False
        self.disable_adapters = False
        self.in_features = in_features
        self.out_features = out_features

    def update_layer(self, adapter_name, r_ab, lora_alpha, lora_dropout, init_lora_weights, 
                    scale_ab, init_a, init_b, train_a, train_b, knowledge_alpha, ortho_lambda,
                    svd_rank_ratio, spectrum_init_scale, adaptive_gating):
        """Update layer parameters, including knowledge-aware mechanisms"""
        self.r_ab[adapter_name] = r_ab
        self.lora_alpha[adapter_name] = lora_alpha
        self.knowledge_alpha[adapter_name] = knowledge_alpha
        self.ortho_lambda[adapter_name] = ortho_lambda
        self.svd_rank_ratio[adapter_name] = svd_rank_ratio
        self.spectrum_init_scale[adapter_name] = spectrum_init_scale
        self.adaptive_gating[adapter_name] = adaptive_gating
        
        if lora_dropout > 0.0:
            lora_dropout_layer = nn.Dropout(p=lora_dropout)
        else:
            lora_dropout_layer = nn.Identity()
        
        self.lora_dropout.update(nn.ModuleDict({adapter_name: lora_dropout_layer}))
        
        # Create A and B matrices
        if r_ab > 0:
            self.lora_A.update(nn.ParameterDict({adapter_name: nn.Parameter(torch.randn(r_ab, self.in_features))}))
            self.lora_B.update(nn.ParameterDict({adapter_name: nn.Parameter(torch.randn(self.out_features, r_ab))}))
            self.scaling_ab[adapter_name] = scale_ab
        
        # Create knowledge gate parameters
        if adaptive_gating:
            self.knowledge_gate.update(nn.ParameterDict({adapter_name: nn.Parameter(torch.randn(r_ab))}))
        
        if init_lora_weights:
            self.reset_knowhira_parameters(adapter_name, init_a, init_b)
        
        self.to(self.weight.device)

    def compute_svd_knowledge(self, adapter_name):
        """Compute SVD decomposition of pre-trained weights, obtain knowledge structure"""
        if adapter_name not in self.U_cache:
            # Perform SVD decomposition on pre-trained weights
            device = self.weight.device
            U, S, V = torch.svd(self.weight.data.to(device))
            
            # Determine the number of singular values to use based on svd_rank_ratio
            rank = min(int(len(S) * self.svd_rank_ratio[adapter_name]), self.r_ab[adapter_name])
            
            self.U_cache[adapter_name] = U[:, :rank].to(device)
            self.S_cache[adapter_name] = S[:rank].to(device)
            self.V_cache[adapter_name] = V[:, :rank].to(device)
        
        return self.U_cache[adapter_name], self.S_cache[adapter_name], self.V_cache[adapter_name]

    def reset_knowhira_parameters(self, adapter_name, init_a, init_b):
        """Reset KnowHiRA parameters, including spectrum-aware initialization"""
        init_mapping = {'kaiming': nn.init.kaiming_uniform_, 'zero': nn.init.zeros_}
        init_kwargs = {'kaiming': {'a': math.sqrt(5)}, 'zero': {}}
        
        if adapter_name in self.lora_A.keys():
            if self.r_ab[adapter_name] > 0:
                # Standard initialization
                if init_a == 'kaiming':
                    nn.init.kaiming_uniform_(self.lora_A[adapter_name], a=math.sqrt(5))
                else:
                    init_mapping[init_a](self.lora_A[adapter_name], **init_kwargs[init_a])
                
                init_mapping[init_b](self.lora_B[adapter_name], **init_kwargs[init_b])
                
                # Improved spectrum-aware initialization - more conservative scaling
                if init_a == 'kaiming' and hasattr(self, 'spectrum_init_scale'):
                    try:
                        # Get SVD knowledge structure, but use more conservative scaling
                        U, S, V = self.compute_svd_knowledge(adapter_name)
                        
                        # Normalize singular values to [0.1, 1.0] range, avoid extreme scaling
                        S_norm = S / (S.max() + 1e-8)
                        S_norm = 0.1 + 0.9 * S_norm  # Map to [0.1, 1.0]
                        
                        # Use smaller scaling factor
                        conservative_scale = min(0.1, self.spectrum_init_scale[adapter_name])
                        
                        if len(S_norm) >= self.r_ab[adapter_name]:
                            # Only apply slight scaling to partial rows
                            scale_factor = S_norm[:self.r_ab[adapter_name]].unsqueeze(1) * conservative_scale
                            self.lora_A[adapter_name].data *= scale_factor
                    except Exception as e:
                        print(f"Warning: Spectrum initialization failed for {adapter_name}: {e}")
                        # If failed, use standard initialization
                        pass
                
                # Initialize knowledge gating parameters (values close to 0.5, balanced knowledge usage)
                if self.adaptive_gating[adapter_name]:
                    nn.init.constant_(self.knowledge_gate[adapter_name], 0.0)

    def get_knowledge_gating_matrix(self, adapter_name):
        """Get knowledge gating matrix G_Σ"""
        U, S, V = self.compute_svd_knowledge(adapter_name)
        
        # Ensure SVD results are on correct device
        device = self.weight.device
        S = S.to(device)
        
        # Normalize singular values
        S_norm = S / (S.max() + 1e-8)
        
        if self.adaptive_gating[adapter_name]:
            # Adaptive gating: G_Σ = Diag(σ_norm ⊙ sigmoid(g))
            gate_values = torch.sigmoid(self.knowledge_gate[adapter_name]).to(device)
            # Ensure dimension matching
            min_dim = min(len(S_norm), len(gate_values))
            gating_weights = S_norm[:min_dim] * gate_values[:min_dim]
        else:
            # Fixed gating: directly use normalized singular values
            gating_weights = S_norm[:self.r_ab[adapter_name]]
        
        return torch.diag(gating_weights).to(device)

    def compute_orthogonal_regularization(self, adapter_name):
        """Compute orthogonal regularization loss - improved version, avoid excessive loss"""
        if adapter_name not in self.lora_A or adapter_name not in self.lora_B:
            return torch.tensor(0.0, device=self.weight.device)
        
        A = self.lora_A[adapter_name]  # (r_ab, in_features)
        B = self.lora_B[adapter_name]  # (out_features, r_ab)
        
        # Improved orthogonal loss calculation
        # For A: encourage row vector orthogonality (A A^T close to I)
        AAT = torch.mm(A, A.t())  # (r_ab, r_ab)
        I_A = torch.eye(A.size(0), device=A.device, dtype=A.dtype)
        loss_A = torch.norm(AAT - I_A, p='fro') ** 2
        # Normalization: divide by matrix size
        loss_A = loss_A / (A.size(0) ** 2)
        
        # For B: since B is large (out_features >> r_ab), use different strategy
        # Only require B's column vectors not to be too large, not strictly orthogonal
        BTB = torch.mm(B.t(), B)  # (r_ab, r_ab)
        I_B = torch.eye(B.size(1), device=B.device, dtype=B.dtype)
        loss_B = torch.norm(BTB - I_B, p='fro') ** 2
        # Normalization: divide by matrix size
        loss_B = loss_B / (B.size(1) ** 2)
        
        # More balanced loss: reduce orthogonal loss impact, avoid over-regularization
        total_loss = (0.5 * loss_A + 0.1 * loss_B) * self.ortho_lambda[adapter_name]
        
        # Add gradient clipping to prevent loss explosion
        total_loss = torch.clamp(total_loss, max=1.0)
        
        return total_loss


class KnowHiRALinear(nn.Linear, KnowHiRALayer):
    """
    KnowHiRA implementation specifically applied to Linear layers
    """
    
    def __init__(
        self,
        adapter_name: str,
        in_features: int,
        out_features: int,
        r_ab: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        fan_in_fan_out: bool = False,
        scale_ab: float = 1.0,
        init_a: str = 'kaiming',
        init_b: str = 'zero',
        train_a: bool = True,
        train_b: bool = True,
        knowledge_alpha: float = 0.5,
        ortho_lambda: float = 0.01,
        svd_rank_ratio: float = 0.8,
        spectrum_init_scale: float = 1.0,
        adaptive_gating: bool = True,
        **kwargs,
    ):
        init_lora_weights = kwargs.pop("init_lora_weights", True)
        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        KnowHiRALayer.__init__(self, in_features=in_features, out_features=out_features)
        
        # Freeze pretrained weights
        self.weight.requires_grad = False
        self.train_a = train_a
        self.train_b = train_b
        self.fan_in_fan_out = fan_in_fan_out
        
        if fan_in_fan_out:
            self.weight.data = self.weight.data.T

        nn.Linear.reset_parameters(self)
        self.update_layer(
            adapter_name, r_ab, lora_alpha, lora_dropout, init_lora_weights, 
            scale_ab, init_a, init_b, train_a, train_b, knowledge_alpha, 
            ortho_lambda, svd_rank_ratio, spectrum_init_scale, adaptive_gating
        )
        self.active_adapter = adapter_name

    def forward(self, x: torch.Tensor):
        """
        KnowHiRA forward propagation: ΔW = W_0 ⊙ (A · G_Σ · B), where G_Σ is the knowledge-guided gate matrix
        """
        previous_dtype = x.dtype
        
        if self.active_adapter not in self.lora_A.keys():
            return F.linear(x, transpose(self.weight, self.fan_in_fan_out), bias=self.bias)
        
        if self.disable_adapters:
            result = F.linear(x, transpose(self.weight, self.fan_in_fan_out), bias=self.bias)
        elif self.r_ab[self.active_adapter] > 0:
            weight_dtype = self.weight.dtype
            
            # Get knowledge gate matrix
            G_sigma = self.get_knowledge_gating_matrix(self.active_adapter).to(weight_dtype)
            
            # Compute knowledge-guided update: A · G_Σ · B
            A = self.lora_A[self.active_adapter].to(weight_dtype)
            B = self.lora_B[self.active_adapter].to(weight_dtype)
            
            # Ensure matrix dimensions are compatible
            if G_sigma.size(0) < A.size(0):
                padding_size = A.size(0) - G_sigma.size(0)
                padding = torch.eye(padding_size, device=G_sigma.device, dtype=weight_dtype)
                G_sigma = torch.block_diag(G_sigma, padding)
            elif G_sigma.size(0) > A.size(0):
                G_sigma = G_sigma[:A.size(0), :A.size(0)]
            
            # Knowledge-guided Hadamard update: ΔW = W_0 ⊙ (A^T · G_Σ · B^T)
            AB_knowledge = torch.mm(torch.mm(A.t(), G_sigma), B.t())
            
            # Use knowledge weight alpha and scaling factor
            alpha = self.knowledge_alpha[self.active_adapter]
            scaled_AB = alpha * AB_knowledge * self.scaling_ab[self.active_adapter]
            
            # Hadamard product update
            updated_weight = (self.weight * (1 + scaled_AB)).to(weight_dtype)
            
            # Execute linear operation
            result = F.linear(x, transpose(updated_weight, self.fan_in_fan_out), bias=self.bias)
        else:
            result = F.linear(x, transpose(self.weight, self.fan_in_fan_out), bias=self.bias)

        result = result.to(previous_dtype)
        return result

    def get_orthogonal_loss(self):
        """Get orthogonal regularization loss for training loss function"""
        return self.compute_orthogonal_regularization(self.active_adapter)
