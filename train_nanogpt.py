"""
This training script can be run both on a single gpu in debug mode,
and also in a larger training run with distributed data parallel (ddp).

To run on a single GPU, example:
$ python train.py --batch_size=32 --compile=False

To run with DDP on 4 gpus on 1 node, example:
$ torchrun --standalone --nproc_per_node=4 train.py

To run with DDP on 4 gpus across 2 nodes, example:
- Run on the first (master) node with example IP 123.456.123.456:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=0 --master_addr=123.456.123.456 --master_port=1234 train.py
- Run on the worker node:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=1 --master_addr=123.456.123.456 --master_port=1234 train.py
(If your cluster does not have Infiniband interconnect prepend NCCL_IB_DISABLE=1)
"""

import os
import time
import math
import pickle
from contextlib import nullcontext
import inspect

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

from model import GPTConfig, GPT

from minoft.parametrized_lora import add_lora
from minoft.parametrized_oft import add_oft

from minoft.modular_oft import inject_trainable_oft
from minoft.modular_lora import inject_trainable_lora

from minoft.utils import (
    get_lora_params, 
    get_lora_state_dict, 
    get_oft_params, 
    get_oft_state_dict,
    get_poft_params,
    get_poft_state_dict,
    tie_weights, 
    tie_oft_weights
)

import tiktoken

debug = False

out_dir = 'out'
eval_interval = 2000
log_interval = 1
eval_iters = 200
eval_only = False 
always_save_checkpoint = True 
init_from = 'scratch' 

wandb_log = False 
wandb_project = 'owt'
wandb_run_name = 'gpt2' 

dataset = 'openwebtext'
gradient_accumulation_steps = 5 * 8 
batch_size = 12 
block_size = 1024

n_layer = 12
n_head = 12
n_embd = 768
dropout = 0.0 
bias = False 

learning_rate = 6e-4 
max_iters = 600000 
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0 

decay_lr = True 
warmup_iters = 2000 
lr_decay_iters = 600000 
min_lr = 6e-5 

backend = 'nccl' 
device = 'cuda' 
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' 
compile = True 

config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
exec(open('configurator.py').read()) 
config = {k: globals()[k] for k in config_keys} 

ft_method = "plora" if use_plora else "mlora" if use_mlora else "moft" if use_moft else ""

ddp = int(os.environ.get('RANK', -1)) != -1 
if ddp:
    init_process_group(backend=backend)
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0 
    seed_offset = ddp_rank 
    assert gradient_accumulation_steps % ddp_world_size == 0
    gradient_accumulation_steps //= ddp_world_size
else:
    master_process = True
    seed_offset = 0
    ddp_world_size = 1
tokens_per_iter = gradient_accumulation_steps * ddp_world_size * batch_size * block_size
print(f"tokens per iteration will be: {tokens_per_iter:,}")

if master_process:
    os.makedirs(out_dir, exist_ok=True)
torch.manual_seed(1337 + seed_offset)
torch.backends.cuda.matmul.allow_tf32 = True 
torch.backends.cudnn.allow_tf32 = True 
device_type = 'cuda' if 'cuda' in device else 'cpu' 

ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

data_dir = os.path.join('data', dataset)
train_data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
val_data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')
def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
    if device_type == 'cuda':

        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y

iter_num = 0
best_val_loss = 1e9

meta_path = os.path.join(data_dir, 'meta.pkl')
meta_vocab_size = None
if os.path.exists(meta_path):
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    meta_vocab_size = meta['vocab_size']
    print(f"found vocab_size = {meta_vocab_size} (inside {meta_path})")

model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=block_size,
                  bias=bias, vocab_size=None, dropout=dropout) 


if init_from == 'scratch':
    print("Initializing a new model from scratch")
    if meta_vocab_size is None:
        print("defaulting to vocab_size of GPT-2 to 50304 (50257 rounded up for efficiency)")
    model_args['vocab_size'] = meta_vocab_size if meta_vocab_size is not None else 50304
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
    
elif init_from.startswith('gpt2'):
    print(f"Initializing from OpenAI GPT-2 weights: {init_from}")
    override_args = dict(dropout=dropout)
    model = GPT.from_pretrained(init_from, override_args)
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
        model_args[k] = getattr(model.config, k)
        
elif init_from == 'resume':
    print(f"Resuming training from {out_dir}")
    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
    checkpoint = torch.load(ckpt_path, map_location=device)
    checkpoint_model_args = checkpoint['model_args']
    
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
        model_args[k] = checkpoint_model_args[k]
        
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
    state_dict = checkpoint['model']
    
    unwanted_prfeix = '_orig_mod.'
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    
    model.load_state_dict(state_dict)
    iter_num = checkpoint['iter_num']
    best_val_loss = checkpoint['best_val_loss']

        
if block_size < model.config.block_size:
    print(f"Cropping initial block size {model.config.block_size} to {block_size}")
    model.crop_block_size(block_size)
    model_args['block_size'] = block_size
    
if use_plora:
    add_lora(model, lora_config=lora_config)
    tie_weights(linear=model.lm_head, embedding=model.transformer.wte)
    
if use_poft:
    add_oft(model, oft_config=oft_config)
    tie_oft_weights(linear=model.lm_head, embedding=model.transformer.wte)
    

model.to(device)
scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))

def param_count(param_list):
    
    n = sum(p.numel() for p in param_list)
    if n < 1e6:
        return f"{n/1e3:.1f}k"
    else:
        return f"{n/1e6:.1f}M"

def configure_optimizers_ft(self, param_list, weight_decay, learning_rate, betas, device_type):
    
    optim_groups = [
        {
            "params": param_list,
            "weight_decay": weight_decay
        }
    ]
    
    use_fused = (device_type == "cuda") and ("fused" in inspect.signature(torch.optim.AdamW).parameters)
    print(f"using fused AdamW: {use_fused}")
    extra_args = dict(fused=True) if use_fused else dict()
    optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
    
    return optimizer


if use_plora:
    print(f'using parametrized lora fine-tuning...')
    
    lora_param_list = list(get_lora_params(model, print_shapes=False))
    optimizer = configure_optimizers_ft(model, lora_param_list, weight_decay, learning_rate, (beta1, beta2), device_type)
    
    print(f'freezing gpt2 model weights...')
    for name, param in model.named_parameters():
        if 'lora' not in name:
            param.requires_grad = False
    print(f"optimizing {param_count(lora_param_list)} parameters")
    
elif use_poft:
    print(f'using parametrized oft fine-tuning...')
    poft_param_list = list(get_poft_params(model, print_shapes=False))
    optimizer = configure_optimizers_ft(model, poft_param_list, weight_decay, learning_rate, (beta1, beta2), device_type)
    
    print(f'freezing gpt2 model weights...')
    for name, param in model.named_parameters():
        if 'oft' not in name:
            param.requires_grad = False
    print(f"optimizing {param_count(poft_param_list)} parameters")
    
elif use_moft:
    model.requires_grad_(False)
    print(f'using modular oft fine-tuning...')
    oft_params, train_names = inject_trainable_oft(model, target_replace_module=ft_modules, verbose=False, r=oft_r, eps=oft_eps, is_coft=oft_coft, block_share=oft_block_share)
    optimizer = configure_optimizers_ft(model, oft_params, weight_decay, learning_rate, (beta1, beta2), device_type)
    print(f"optimizing {param_count(oft_params)} parameters")
    
elif use_mlora:
    model.requires_grad_(False)
    print(f'using modular lora fine-tuning...')
    lora_params, train_names = inject_trainable_lora(model, target_replace_module=ft_modules, r=4, loras=None,verbose = False, dropout_p=0.0, scale=1.0,)
    optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)
    
    lora_param_list = list(get_lora_params(model, print_shapes=False))
    print(f"optimizing {param_count(lora_param_list)} parameters")
    
else:
    print(f'not using LoRA or OFT...')
    optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)
    
if debug: 
    for name, param in model.named_parameters():
        print(f"{name} // requires_grad: {param.requires_grad}")


if compile:
    print("compiling the model... (takes a ~minute)")
    unoptimized_model = model
    model = torch.compile(model) 

if ddp:
    print("initializing distributed training (DDP)")
    model = DDP(model, device_ids=[ddp_local_rank])

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            with ctx:
                logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

def get_lr(it):

    if it < warmup_iters:
        return learning_rate * it / warmup_iters

    if it > lr_decay_iters:
        return min_lr

    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) 
    return min_lr + coeff * (learning_rate - min_lr)

if wandb_log and master_process:
    import wandb
    wandb.init(project=wandb_project, name=wandb_run_name, config=config)

X, Y = get_batch('train') 
t0 = time.time()
local_iter_num = 0 
raw_model = model.module if ddp else model 
running_mfu = -1.0
enc = tiktoken.get_encoding("gpt2")

while True:

    lr = get_lr(iter_num) if decay_lr else learning_rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    if iter_num % eval_interval == 0 and master_process:
        losses = estimate_loss()
        print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        
        if wandb_log:
            wandb.log({
                "iter": iter_num,
                "train/loss": losses['train'],
                "val/loss": losses['val'],
                "lr": lr,
                "mfu": running_mfu*100, # convert to percentage
            })
        
        if losses['val'] < best_val_loss or always_save_checkpoint:
            best_val_loss = losses['val']
            if iter_num > 0:
                checkpoint = {
                    'model': raw_model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'model_args': model_args,
                    'iter_num': iter_num,
                    'best_val_loss': best_val_loss,
                    'config': config,
                }
                if use_plora:
                    checkpoint['lora'] = get_lora_state_dict(raw_model)
                elif use_poft:
                    checkpoint['oft'] = get_oft_state_dict(raw_model)
                
                print(f"saving checkpoint to {out_dir}")
                checkpoint_name = dataset + '_' +  init_from + '_' + ft_method + '_' + 'ckpt.pt'
                torch.save(checkpoint, os.path.join(out_dir, checkpoint_name))
    if iter_num == 0 and eval_only:
        break

    for micro_step in range(gradient_accumulation_steps):
        if ddp:
            model.require_backward_grad_sync = (micro_step == gradient_accumulation_steps - 1)
        with ctx:
            logits, loss = model(X, Y)
            loss = loss / gradient_accumulation_steps 

        X, Y = get_batch('train')
        scaler.scale(loss).backward()

    if grad_clip != 0.0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad(set_to_none=True)

    t1 = time.time()
    dt = t1 - t0
    t0 = t1
    if iter_num % log_interval == 0 and master_process:
        lossf = loss.item() * gradient_accumulation_steps
        if local_iter_num >= 5: 
            mfu = raw_model.estimate_mfu(batch_size * gradient_accumulation_steps, dt)
            running_mfu = mfu if running_mfu == -1.0 else 0.9*running_mfu + 0.1*mfu
        print(f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms, mfu {running_mfu*100:.2f}%")
    iter_num += 1
    local_iter_num += 1

    if iter_num > max_iters:
        break

if ddp:
    destroy_process_group()