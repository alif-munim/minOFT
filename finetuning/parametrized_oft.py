"""
A minimal implementaion of oft.
"""

import math
from functools import partial

import torch
import torch.nn.utils.parametrize as parametrize
from torch import nn


def project(R, eps):
    I = torch.zeros((R.size(0), R.size(0)), dtype=R.dtype, device=R.device)
    diff = R - I
    norm_diff = torch.norm(diff)
    if norm_diff <= eps:
        return R
    else:
        return I + eps * (diff / norm_diff)

def project_batch(R, eps=1e-5):
    # scaling factor for each of the smaller block matrix
    eps = eps * 1 / torch.sqrt(torch.tensor(R.shape[0]))
    I = torch.zeros((R.size(1), R.size(1)), device=R.device, dtype=R.dtype).unsqueeze(0).expand_as(R)
    diff = R - I
    norm_diff = torch.norm(R - I, dim=(1, 2), keepdim=True)
    mask = (norm_diff <= eps).bool()
    out = torch.where(mask, R, I + eps * (diff / norm_diff))
    return out

class OFTParametrization(nn.Module):
    def __init__(
        self, in_features, out_features, bias=False, r=4, eps=1e-5, is_coft=True, block_share=False,
    ):
        super().__init__()

        assert in_features % r == 0, f"in_features {in_features} must be divisible by r {r}"

        # Get the number of available GPUs
        self.num_gpus = torch.cuda.device_count()
        # Set the device IDs for distributed training
        self.device_ids = list(range(self.num_gpus))

        self.in_features=in_features
        self.out_features=out_features

        # Define the fixed Linear layer: v
        self.OFT = torch.nn.Linear(in_features=in_features, out_features=out_features, bias=bias)

        # Define the reduction rate:
        self.r = r
        self.is_coft = is_coft

        self.fix_filt_shape = [in_features, out_features]

        # Define the trainable matrix parameter: R
        self.block_share = block_share
        if self.block_share:
            # Initialized as an identity matrix
            self.oft_R_shape = [in_features // self.r, in_features // self.r]
            self.oft_R = nn.Parameter(torch.zeros(self.oft_R_shape[0], self.oft_R_shape[0]), requires_grad=True)
  
            self.eps = eps * self.oft_R_shape[0] * self.oft_R_shape[0]
        else:
            # Initialized as an identity matrix
            self.oft_R_shape = [self.r, in_features // self.r, in_features // self.r]
            R = torch.zeros(self.oft_R_shape[1], self.oft_R_shape[1])
            R = torch.stack([R] * self.r)
            self.oft_R = nn.Parameter(R, requires_grad=True)
            self.eps = eps * self.oft_R_shape[1] * self.oft_R_shape[1]

    def forward(self, x):
        orig_dtype = x.dtype
        dtype = self.oft_R.dtype

        if self.block_share:
            if self.is_coft:
                with torch.no_grad():
                    self.oft_R.copy_(project(self.oft_R, eps=self.eps))
            orth_rotate = self.cayley(self.oft_R)
        else:
            if self.is_coft:
                with torch.no_grad():
                    self.oft_R.copy_(project_batch(self.oft_R, eps=self.eps))
            orth_rotate = self.cayley_batch(self.oft_R)

        # Block-diagonal parametrization
        block_diagonal_matrix = self.block_diagonal(orth_rotate)
        
        # fix filter
        # print(self.OFT)
        fix_filt = self.OFT.weight.data
        fix_filt = torch.transpose(fix_filt, 0, 1)
        filt = torch.mm(block_diagonal_matrix, fix_filt.to(dtype))
        filt = torch.transpose(filt, 0, 1)
 
        # Apply the trainable identity matrix
        bias_term = self.OFT.bias.data if self.OFT.bias is not None else None
        out = nn.functional.linear(input=x, weight=filt, bias=bias_term)

        return out #.to(orig_dtype)

    def cayley(self, data):
        r, c = list(data.shape)
        # Ensure the input matrix is skew-symmetric
        skew = 0.5 * (data - data.t())
        I = torch.eye(r, device=data.device)
        
        # Perform the Cayley parametrization
        Q = torch.mm(I + skew, torch.inverse(I - skew))
        return Q
    
    def cayley_batch(self, data):
        b, r, c = data.shape
        # Ensure the input matrix is skew-symmetric
        skew = 0.5 * (data - data.transpose(1, 2))
        I = torch.eye(r, device=data.device).unsqueeze(0).expand(b, r, c)

        # Perform the Cayley parametrization
        Q = torch.bmm(I - skew, torch.inverse(I + skew))

        return Q

    def block_diagonal(self, R):
        if self.block_share:
            # Create a list of R repeated block_count times
            blocks = [R] * self.r
        else:
            # Create a list of R slices along the third dimension
            blocks = [R[i, ...] for i in range(self.r)]

        # Use torch.block_diag to create the block diagonal matrix
        A = torch.block_diag(*blocks)

        return A

    def is_orthogonal(self, R, eps=1e-5):
        with torch.no_grad():
            RtR = torch.matmul(R.t(), R)
            diff = torch.abs(RtR - torch.eye(R.shape[1], dtype=R.dtype, device=R.device))
            return torch.all(diff < eps)

    def is_identity_matrix(self, tensor):
        if not torch.is_tensor(tensor):
            raise TypeError("Input must be a PyTorch tensor.")
        if tensor.ndim != 2 or tensor.shape[0] != tensor.shape[1]:
            return False
        identity = torch.eye(tensor.shape[0], device=tensor.device)
        return torch.all(torch.eq(tensor, identity))
    
    @classmethod
    def from_linear(cls, layer, bias=False, r=4, eps=1e-5, is_coft=True, block_share=False):
        in_features, out_features = layer.weight.shape
        return cls(
            in_features, out_features, bias=bias, r=r, eps=eps, is_coft=is_coft, block_share=block_share,
        )

    @classmethod
    def from_embedding(cls, layer, bias=False, r=4, eps=1e-5, is_coft=True, block_share=False):
        in_features, out_features = layer.weight.shape
        return cls(
            in_features, out_features, bias=bias, r=r, eps=eps, is_coft=is_coft, block_share=block_share,
        )

# Default configuration initializes oft parametrization for linear layer model weights
default_oft_config = {
    nn.Linear: {
        "weight": partial(OFTParametrization.from_linear, rank=4)
    },
}


        
def apply_oft(layer, register=True, merge=False, oft_config=default_oft_config):
    if register:
        if type(layer) in oft_config:
            for attr_name, parametrization in oft_config[type(layer)].items():
                parametrize.register_parametrization(layer, attr_name, parametrization(layer))
        else:
            if hasattr(layer, "parametrizations"):
                for attr_name in layer.parametrizations.keys():
                    parametrize.remove_parametrizations(layer, attr_name, leave_parametrized=merge)
    
    
def add_oft(model, oft_config=default_oft_config):
    """Add oft parametrization to all layers in a model."""
    model.apply(partial(apply_oft, oft_config=oft_config))
    
def add_oft_by_name(model):
    """Add oft parametrization to specific layers in a model by name."""
    for name, layer in model.named_modules():
        if any([m in name for m in target_module_names]):
            add_oft(layer, oft_config=oft_config)
    
def merge_oft(model):
    """Merge oft parametrization to all layers in a model. Removes parametrization."""
    model.apply(partial(apply_oft, register=False, merge=True))
    
def remove_oft(model):
    """Remove oft paramterization from all layers in a model."""
    model.apply(partial(apply_oft, register=False, merge=False))
    
    

