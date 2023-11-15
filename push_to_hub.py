import torch
import os

from model import GPTConfig, GPT
from finetuning.modular_lora import inject_trainable_lora, collapse_lora, monkeypatch_remove_lora
from finetuning.modular_oft import inject_trainable_oft, collapse_oft, monkeypatch_remove_oft

n_layer = 12
n_head = 12
n_embd = 768
dropout = 0.0 
bias = False 
batch_size = 12 
block_size = 1024
vocab_size = 50257

device = 'cuda'

config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
exec(open('configurator.py').read()) 
config = {k: globals()[k] for k in config_keys}


model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=block_size,
                  bias=bias, vocab_size=None, dropout=dropout) 

dataset = 'guanaco'
modular_ft = "" if ft_method == "" else ft_method[1:]
out_dir = f"out-{dataset}"
ckpt_name = f"{dataset}_{init_from}_{ft_method}_ckpt.pt"

ckpt_path = os.path.join(out_dir, ckpt_name)
checkpoint = torch.load(ckpt_path, map_location=device)
print(f"loaded checkpoint from {out_dir}/{ckpt_name}")

checkpoint_model_args = checkpoint['model_args']
for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
    model_args[k] = checkpoint_model_args[k]

    
    
gptconf = GPTConfig(**model_args)
model = GPT(gptconf)
state_dict = checkpoint['model']
print(f"initialized GPT2 from model args")

unwanted_prefix = '_orig_mod.'
for k, v in list(state_dict.items()):
    if k.startswith(unwanted_prefix):
        state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
print(f"copied checkpoint state dict")


if use_mlora or use_plora:
    model.requires_grad_(False)
    lora_params, train_names = inject_trainable_lora(model, target_replace_module=lora_modules, r=4, loras=None,verbose = False, dropout_p=0.0, scale=1.0,)
elif use_moft or use_poft:
    model.requires_grad_(False)
    oft_params, train_names = inject_trainable_oft(model, target_replace_module=oft_modules, verbose=False, r=oft_r, eps=oft_eps, is_coft=oft_coft, block_share=oft_block_share)


model.load_state_dict(state_dict)

if use_mlora or use_plora:
    print(f'removing LoRA from model...')
    collapse_lora(model)
    monkeypatch_remove_lora(model)
elif use_moft or use_poft:
    print(f'removing OFT from model...')
    collapse_oft(model)
    monkeypatch_remove_oft(model)

    

print(f"pushing model to huggingface hub")
# model.save_pretrained(out_dir)

if modular_ft == '':
    hf_repo = f"{init_from}-{dataset}"
else:
    hf_repo = f"{init_from}-{modular_ft}-{dataset}"
model.push_to_hub(organization=hf_org, repo_name=hf_repo)

print(f"testing model loading")
model.from_pretrained(f"{hf_org}/{hf_repo}")
print(f"successfully loaded model from {hf_org}/{hf_repo}")