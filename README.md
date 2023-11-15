## MinOFT
A minimal GPT2 reimplementation of orthogonal fine-tuning (OFT), introduced in the paper [Controlling Text-to-Image Diffusion by Orthogonal Finetuning](https://arxiv.org/abs/2306.07280). LoRA updates the pretrained weight
matrix by adding a product of two low-rank matrices. OFT learns an orthogonal matrix to transform the neurons of the
same layer, and it achieves stronger generalization and consistently more stable training than LoRA.

![Comparison of LoRA and OFT](assets/oft-lora.png)

### Training on Guanaco
To prepare the guanaco dataset for GPT2 and train, run the following commands:
```
python data/guanaco/prepare.py
python train.py config/finetune_guanaco.py
```

To push your model to Hugging Face Hub after training, run the following command:
```
python push_to_hub.py config/finetune_guanaco.py
```

Below is a sample training run (100 iterations) comparing regular fine-tuning, LoRA, and OFT.
![Training run for gaunaco OASST dataset](assets/guanaco-training.png)


### Training on Tiny Shakespeare
To prepare the tiny shakespeare dataset for GPT2 and train, run the following commands:
```
python data/guanaco/prepare.py
python train.py config/finetune_shakespeare.py
```

To push your model to Hugging Face Hub after training, run the following command:
```
python push_to_hub.py config/finetune_shakespeare.py
```

Below is a sample training run on tiny shakespeare (100 iterations) comparing regular LoRA and OFT.
![Training run for tiny shakespeare dataset](assets/shakespeare-training.png)


### Special Thanks
This project is an extension of the [karpathy/nanoGPT](https://github.com/karpathy/nanoGPT) repository created by Andrej Karpathy.
It is also heavily inspired by [cccntu/minLoRA](https://github.com/cccntu/minLoRA/tree/main), although the much of the actual LoRA and OFT implementations are borrowed from [cloneofsimo/lora](https://github.com/cloneofsimo/lora) and [Zeju1997/oft](https://github.com/Zeju1997/oft) respectively.



### To-Do's
- [x] Add functionality to upload fine-tuned GPT2 models to huggingface hub
- [ ] Include scripts for LM evals
- [x] Merge and unload LoRA, OFT modules