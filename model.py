import math
import inspect
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F

import os
import json
from transformers import WEIGHTS_NAME, CONFIG_NAME
from transformers.file_utils import PushToHubMixin
from huggingface_hub import HfApi, create_repo, Repository
from requests.exceptions import HTTPError



class LayerNorm(nn.Module):
    
    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None
        
    def forward(self, x):
        # Second parameter 'normalized_shape' is the shape of the input
        return F.layer_norm(x, self.weight.shape, self.weight, self.bias, 1e-5)
    
    
"""
Everything required for SA (single head and multi-head) in one class.
"""
class CausalSelfAttention(nn.Module):
    
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0, f"Embedding size {config.n_embd} is not divisible by number of heads {config.n_head}"
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        
    def forward(self, x):
        B, T, C = x.size()
        
        # The attn linear layer expands to 3x n_embd (embed dim)
        # When we split the output, we pass a chunk size of n_embd and split along dim 2 (channel dim)
        # As a result, we get 3 equal-sized chunks for q, k, and v
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        
        if self.flash:
            y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=True)
        
        # Concatenate outputs from all attenion heads
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))
        return y
    

"""
Basic MLP with a "fan-out" and "fan-in" transformation and GeLU activation.
"""
class MLP(nn.Module):
    
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)
        
    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x
    

"""
Transformer (decoder) blocks that will be stacked.
"""
class Block(nn.Module):
    
    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)
        
    def forward(self, x):
        
        # Calculations with residual connections
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x
    

@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50304
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = True
    
    def to_json_string(self):
        """
        Serializes this instance to a JSON string.
        """
        return json.dumps(self.__dict__, indent=2) + "\n"

class GPT(nn.Module):
    
    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config
        
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            dropout = nn.Dropout(config.dropout),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = LayerNorm(config.n_embd, bias=config.bias)
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        
        # Weight tying (?)
        self.transformer.wte.weight = self.lm_head.weight
        
        self.apply(self._init_weights)
        
    # Initialize weights: _ following a function name denotes in-place operation
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
     
    def forward(self, idx, targets=None):
        
        # The input idx denotes word (token) indices according to our dictionary
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, f"Cannot forward sequence length {t}, block size is only {self.config.block_size}"
        
        # The possible positions of a token given the time (t) sequence dim
        pos = torch.arange(0, t, dtype=torch.long, device=device)
        pos_emb = self.transformer.wpe(pos)
        
        # Token embeddings
        tok_emb = self.transformer.wte(idx)
        
        # Add embeddings and pass through all of the transformer blocks
        # Finally, apply layer normalization to output
        x = self.transformer.dropout(tok_emb + pos_emb)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        
        # Map the processed embeddings back to the vocabulary logits
        # During inference, set loss to none and forward LM head on last token embedding
        if targets is not None:
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            logits = self.lm_head(x[:, [-1], :]) # Forward the LM head on the last time (B, T, C) token
            loss = None
        
        return logits, loss
    
    
    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas)
        return optimizer
    
    
    def get_num_params(self, non_embedding=True):
        
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wpe.weight.numel()
        return n_params


    def estimate_mfu(self, fwdbwd_per_iter, dt):
        """
        Estimate model flops utilization (MFU) achieved:expected V100 peak FLOPS
        """
        N = self.get_num_params()
        cfg = self.config
        L, H, Q, T = cfg.n_layer, cfg.n_head, cfg.n_embd//cfg.n_head, cfg.block_size
        
        # Where does this calculation come from?
        flops_per_token = 6*N + 12*L*H*Q*T
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        
        flops_achieved = flops_per_iter * (1.0/dt)
        flops_promised = 112e12 # V100 GPU peak flops is 112
        mfu = flops_achieved / flops_promised
        return mfu
    
    # TODO: implement loading from pre-trained
    @classmethod 
    def from_pretrained(cls, model_name_or_path, override_args=None):
        print("model_name_or_path: ", model_name_or_path)
        override_args = override_args or {} # default to empty dict
        assert all(k == 'dropout' for k in override_args) # Only dropout can be overwritten
        from transformers import GPT2LMHeadModel
        
        custom = model_name_or_path in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        
        print(f'Loading weights from pre-trained GPT: {model_name_or_path}')
        model_hf = GPT2LMHeadModel.from_pretrained(model_name_or_path, resume_download=True)

        if custom:
            config_args = {
                'gpt2': dict(n_layer=12, n_head=12, n_embd=768), # 124M params
                'gpt2-medium': dict(n_layer=24, n_head=16, n_embd=1024), # 124M params
                'gpt2-large': dict(n_layer=36, n_head=20, n_embd=1280), # 124M params
                'gpt2-xl': dict(n_layer=48, n_head=25, n_embd=1600), # 124M params
            }[model_name_or_path]

            print("Forcing vocab_size=50257, block_size=1024, bias=True (GPT checkpoint pre-reqs)")
            config_args['vocab_size'] = 50257
            config_args['block_size'] = 1024
            config_args['bias'] = True

            if 'dropout' in override_args:
                print(f"Overriding dropout rate to {override_args['dropout']}")
                config_args['dropout'] = override_args['dropout']
        else:
            # model_hf = GPT2LMHeadModel.from_pretrained(model_name_or_path)

            # Update our GPTConfig with any override arguments provided
            override_args = override_args or {}
            config_args = {
                'vocab_size': model_hf.config.vocab_size,
                'block_size': model_hf.config.n_positions,
                'n_layer': model_hf.config.n_layer,
                'n_head': model_hf.config.n_head,
                'n_embd': model_hf.config.n_embd,
                'dropout': model_hf.config.summary_first_dropout,
                'bias': True
            }
            config_args.update(override_args)

        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()

        # Discard the .attn.bias mask/buffer (?)
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')]
        sd_hf = model_hf.state_dict()

        # Copy HF state dict
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')]
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')]

        # OpenAI checkpoints use Conv1D instead of Linear, so weights need to be transposed before import
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf) != {len(sd_keys)}}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model
    
    def save_pretrained(self, save_directory):
        
        sd = self.state_dict()
        sd_keys = sd.keys()

        # Discard the .attn.bias mask/buffer (?)
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')]

        # OpenAI checkpoints use Conv1D instead of Linear, so weights need to be transposed before import
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        for k in sd_keys:
            if any(k.endswith(w) for w in transposed):
                with torch.no_grad():
                    sd[k] = sd[k].t()
            else:
                with torch.no_grad():
                    sd[k] = sd[k]
                    
        torch.save(sd, os.path.join(save_directory, WEIGHTS_NAME))
        
        # Save the configuration file
        config_file = os.path.join(save_directory, CONFIG_NAME)
        with open(config_file, 'w', encoding='utf-8') as f:
            f.write(self.config.to_json_string())
        
    
#     def save_pretrained(self, save_directory):
#         """
#         Save the model weights and configuration file to a directory, so that
#         it can be re-loaded using the `from_pretrained` class method.
#         """
#         os.makedirs(save_directory, exist_ok=True)
#         # Save the weights
#         torch.save(self.state_dict(), os.path.join(save_directory, WEIGHTS_NAME))

#         # Save the configuration file
#         config_file = os.path.join(save_directory, CONFIG_NAME)
#         with open(config_file, 'w', encoding='utf-8') as f:
#             f.write(self.config.to_json_string())
    
    
    def push_to_hub(self, repo_name, organization=None, private=False, token=None, commit_message="Add model"):
        """
        Push the model to the Hugging Face Hub, creating the repository if it does not exist.
        """
        # If an organization is specified, prepend it to the repo_name
        full_repo_name = f"{organization}/{repo_name}" if organization else repo_name

        try:
            # Instantiate Hugging Face API instance
            api = HfApi()

            # Create a repository on the Hugging Face Hub
            create_repo(full_repo_name, private=private, token=token)
        
        except HTTPError as http_err:
            # If a HTTP error occurs, check if it's because the repo already exists
            if http_err.response.status_code != 409:
                raise

        # Construct the repository URL
        repo_url = f"https://huggingface.co/{full_repo_name}"

        # Clone the repository (or use an existing clone)
        repo_local_path = full_repo_name.split("/")[-1]
        repo = Repository(repo_local_path, clone_from=repo_url, use_auth_token=token)

        # Save model and configuration in the cloned repository directory
        self.save_pretrained(repo_local_path)

        # Push to the hub
        repo.git_pull()  # Ensure we are up-to-date with remote changes
        repo.push_to_hub(commit_message=commit_message)

    
    #TODO: Crop block size if needed
    def crop_block_size(self, block_size):
        assert block_size <= self.config.block_size
        self.config.block_size = block_size
        self.transformer.wpe.weight = nn.Parameter(self.transformer.wpe.weight[:block_size])
        for block in self.transformer.h:
            if hasattr(block.attn, 'bias'):
                block.attn.bias = block.attn.bias[:, :, :block_size, :block_size]
    
    
    #TODO: Implement generation function
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        
        for _ in range(max_new_tokens):
            
            # Crop sequence to block size 
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            
            # Crop logits to top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
                
            # Calculate probs and sample from distribution
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=-1)
        
        return idx
                
        