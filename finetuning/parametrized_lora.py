"""
A minimal implementaion of LoRA.
"""

import math
from functools import partial

import torch
import torch.nn.utils.parametrize as parametrize
from torch import nn


class LoRAParametrization(nn.Module):
    
    def __init__(self, fan_in, fan_out, fan_in_fan_out=False, rank=4, lora_dropout_p=0.0, lora_alpha=1):
        super().__init__()
        
        self.swap = (lambda x: (x[1], x[0])) if fan_in_fan_out else (lambda x: x)
        self.lora_A = nn.Parameter(torch.zeros(self.swap((rank, fan_in))))
        self.lora_B = nn.Parameter(torch.zeros(self.swap((fan_out, rank))))
        
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        
        self.lora_alpha, self.rank = lora_alpha, rank
        self.scaling = lora_alpha / rank
        self.lora_dropout = nn.Dropout(p=lora_dropout_p) if lora_dropout_p > 0 else lambda x: x
        self.dropout_fn = self._dropout if lora_dropout_p > 0 else lambda x: x
        self.register_buffer("lora_dropout_mask", torch.ones(self.swap((1, fan_in)), dtype=self.lora_A.dtype))
        self.forward_fn = self.lora_forward
        
        
    def _dropout(self, A):
        """Dropout from ones matrix and multiply by A matrix (A * dropout(ones)) @ x"""
        return A * self.lora_dropout(self.lora_dropout_mask)
        
    def lora_forward(self, X):
        """Perform the LoRA calculation and add it to the original weights."""
        return X + torch.matmul(*self.swap((self.lora_B, self.dropout_fn(self.lora_A)))).view(X.shape) * self.scaling
        
    def forward(self, X):
        return self.forward_fn(X)
        
    def disable_lora(self):
        """Basically, do nothing (LoRA is not enabled)"""
        self.forward_fn = lambda x: x
        
    @classmethod
    def from_linear(cls, layer, rank=4, lora_dropout_p=0.0, lora_alpha=1):
        fan_out, fan_in = layer.weight.shape
        return cls(
            fan_in, fan_out, fan_in_fan_out=False, rank=rank, lora_dropout_p=lora_dropout_p, lora_alpha=lora_alpha
        )
        
    @classmethod
    def from_conv2d(cls, layer, rank=4, lora_dropout_p=0.0, lora_alpha=1):
        fan_out, fan_in = layer.weight.view(layer.weight.shape[0], -1).shape
        return cls(
            fan_in, fan_out, fan_in_fan_out=False, rank=rank, lora_dropout_p=lora_dropout_p, lora_alpha=lora_alpha
        )
    
    @classmethod
    def from_embedding(cls, layer, rank=4, lora_dropout_p=0.0, lora_alpha=1):
        fan_out, fan_in = layer.weight.shape
        return cls(
            fan_in, fan_out, fan_in_fan_out=True, rank=rank, lora_dropout_p=lora_dropout_p, lora_alpha=lora_alpha
        )



# Default configuration initializes LoRA parametrization for linear layer model weights
default_lora_config = {
    nn.Linear: {
        "weight": partial(LoRAParametrization.from_linear, rank=4)
    },
}



        
def apply_lora(layer, register=True, merge=False, lora_config=default_lora_config):
    """
    Add LoRA parametrization to a layer (used with model.apply).
    
    We typically use register_parametrization to apply some constraints on module parameters.
    In the default LoRA config, the parametrization is partial(LoRAParametrization.from_linear, rank=4)
        and the constraint / operation is defined in the forward() method. 
    
    The arguments are the module, the tensor or weight to parametrize, and the parametrization function.        
        
    The partial(f, a) method allows us to create a "partial function," which fixes some arguments of
        an initial input function f with parameters a, b, c, etc. 
    """
    if register:
        if type(layer) in lora_config:
            for attr_name, parametrization in lora_config[type(layer)].items():
                parametrize.register_parametrization(layer, attr_name, parametrization(layer))
        else:
            if hasattr(layer, "parametrizations"):
                for attr_name in layer.parametrizations.keys():
                    parametrize.remove_parametrizations(layer, attr_name, leave_parametrized=merge)
    
    
def add_lora(model, lora_config=default_lora_config):
    """Add LoRA parametrization to all layers in a model."""
    model.apply(partial(apply_lora, lora_config=lora_config))
    
def add_lora_by_name(model):
    """Add LoRA parametrization to specific layers in a model by name."""
    for name, layer in model.named_modules():
        if any([m in name for m in target_module_names]):
            add_lora(layer, lora_config=lora_config)
    
def merge_lora(model):
    """Merge LoRA parametrization to all layers in a model. Removes parametrization."""
    model.apply(partial(apply_lora, register=False, merge=True))
    
def remove_lora(model):
    """Remove LoRA paramterization from all layers in a model."""
    model.apply(partial(apply_lora, register=False, merge=False))
    
    

