import time
from functools import partial

import torch
from finetuning.parametrized_lora import LoRAParametrization
from finetuning.parametrized_oft import OFTParametrization

out_dir = 'out-guanaco'
eval_interval = 5
eval_iters = 40
max_iters = 100

# only save checkpoints if the validation loss improves
always_save_checkpoint = False
dataset = 'guanaco'

# the number of examples per iter:
# 1 batch_size * 32 grad_accum * 1024 tokens = 32,768 tokens/iter
# shakespeare has 301,966 tokens, so 1 epoch ~= 9.2 iters
batch_size = 1
gradient_accumulation_steps = 32
block_size = 1024

# finetune at constant LR
learning_rate = 3e-5
decay_lr = False

init_from = 'gpt2-large' # models are gpt2, gpt2-medium, gpt2-large, and gpt2-xl
use_plora = False
use_poft = False

use_mlora = False
use_moft = True

ft_method = "plora" if use_plora else "mlora" if use_mlora else "moft" if use_moft else ""
wandb_log = True # feel free to turn on
wandb_project = 'gpt2-guanaco'
wandb_run_name = 'ft-' + ft_method + '-' + str(time.time())


# if init_from == 'gpt2-xl':
#     # decrease grad accum from 32 to save memory
#     use_lora = True
#     gradient_accumulation_steps = 8
#     block_size = 128

if use_plora == True:
    learning_rate = 1e-3
    lora_dropout_p = 0.0
    rank = 4
    lora_alpha = 64
    lora_config = {
        torch.nn.Embedding: {
            "weight": partial(LoRAParametrization.from_embedding, rank=rank, lora_alpha=lora_alpha)
        },
        torch.nn.Linear: {
            "weight": partial(LoRAParametrization.from_linear, rank=rank, lora_alpha=lora_alpha)
        },
    }

elif use_poft == True:
    
    oft_bias=False
    oft_r=29 # Pre-trained GPT has a vocab size of 50257
    oft_eps=1e-3
    oft_coft=True
    oft_block_share=False
    
    oft_config = {
    torch.nn.Embedding: {
        "weight": partial(OFTParametrization.from_embedding, bias=oft_bias, r=oft_r, eps=oft_eps, is_coft=oft_coft, block_share=oft_block_share),
    },
    torch.nn.Linear: {
        "weight": partial(OFTParametrization.from_linear, bias=oft_bias, r=oft_r, eps=oft_eps, is_coft=oft_coft, block_share=oft_block_share),
    },
}
    
elif use_moft == True:
    compile = False
    oft_modules = ["CausalSelfAttention"]
    oft_r=4
    oft_eps=1e-3
    oft_coft=False
    oft_block_share=False
    
elif use_mlora == True:
    compile = False
    lora_modules = ["CausalSelfAttention"]