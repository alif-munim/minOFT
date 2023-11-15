"""
Based on the Hugging Face "Causal Language Modeling" tutorial:
https://huggingface.co/docs/transformers/tasks/language_modeling

Adapted for OFT by Alif Munim
"""

from datasets import load_dataset
from transformers (
    AutoTokenizer, 
    AutoConfig,
    AutoModelForCausalLM, 
    RobertaForCausalLM, 
    DataCollatorForLanguageModeling,
    TrainingArguments, 
    Trainer
)

import math
import inspect
import torch

from finetuning.modular_oft import inject_trainable_oft 


# Define helper functions for data pre-processing and optimization

def preprocess_function(examples):
    return tokenizer([" ".join(x) for x in examples["answers.text"]])

def group_texts(examples):
    # Concatenate all texts.
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
    # customize this part to your needs.
    if total_length >= block_size:
        total_length = (total_length // block_size) * block_size
    # Split by chunks of block_size.
    result = {
        k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated_examples.items()
    }
    result["labels"] = result["input_ids"].copy()
    return result

def unique_modules(model):
    """
    Collect the different module types in your HF model so you can choose which one to target with OFT.
    """
    
    module_names = []
    for module in model.modules():
        name = module.__class__.__name__

        if name not in module_names:
            module_names.append(name)
            print(f'module: \n{module}\n\n')
        
    return module_names

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
    
    # use_fused = (device_type == "cuda") and ("fused" in inspect.signature(torch.optim.AdamW).parameters)
    use_fused = False
    print(f"using fused AdamW: {use_fused}")
    extra_args = dict(fused=True) if use_fused else dict()
    optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
    
    return optimizer

# Set OFT, training, and optimization parameters
oft_r=4
oft_eps=1e-3
oft_coft=False
oft_block_share=False

block_size = 128
learning_rate=2e-5
weight_decay=0.01
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0 
device = 'cuda'
device_type = 'cuda' if 'cuda' in device else 'cpu'
debug = False

# Load model, tokenizer, and data collator
tokenizer = AutoTokenizer.from_pretrained("roberta-base")
config = AutoConfig.from_pretrained("roberta-base")
config.is_decoder = True
model = RobertaForCausalLM.from_pretrained("roberta-base", config=config)

# Set pad token for GPT2 style models
# tokenizer.pad_token = tokenizer.eos_token
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

if debug:
    module_names = unique_modules(model)
    print(module_names)

# Hacky fix for RuntimeError: lazy wrapper should be called at most once
# https://github.com/pytorch/pytorch/issues/90613
torch.inverse(torch.ones((0, 0), device="cuda:0"))


ft_modules = ["RobertaSelfAttention"]
model.requires_grad_(False)
print(f'using modular oft fine-tuning...')
oft_params, train_names = inject_trainable_oft(model, target_replace_module=ft_modules, verbose=False, r=oft_r, eps=oft_eps, is_coft=oft_coft, block_share=oft_block_share)
optimizer = configure_optimizers_ft(model, oft_params, weight_decay, learning_rate, (beta1, beta2), device_type)
print(f'freezing gpt2 model weights...')

if debug:
    for name, param in model.named_parameters():
        print(f"{name} // requires_grad: {param.requires_grad}")
    module_names = unique_modules(model)
    print(module_names)
    
print(f"optimizing {param_count(oft_params)} parameters")


# Load and pre-process dataset
eli5 = load_dataset("eli5", split="train_asks[:5000]")
eli5 = eli5.train_test_split(test_size=0.2)
eli5 = eli5.flatten()

tokenized_eli5 = eli5.map(
    preprocess_function,
    batched=True,
    num_proc=4,
    remove_columns=eli5["train"].column_names,
)
lm_dataset = tokenized_eli5.map(group_texts, batched=True, num_proc=4)


# Begin training
training_args = TrainingArguments(
    output_dir="distilgpt2",
    evaluation_strategy="epoch",
    report_to="none",
    push_to_hub=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=lm_dataset["train"],
    eval_dataset=lm_dataset["test"],
    data_collator=data_collator,
    optimizers=(optimizer,None)
)

trainer.train()
eval_results = trainer.evaluate()
print(f"Perplexity: {math.exp(eval_results['eval_loss']):.2f}")

# Push model to hugging face hub
trainer.push_to_hub()