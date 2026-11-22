---
library_name: peft
license: other
base_model: Qwen/Qwen2.5-1.5B-Instruct
tags:
- base_model:adapter:Qwen/Qwen2.5-1.5B-Instruct
- llama-factory
- lora
- transformers
pipeline_tag: text-generation
model-index:
- name: q15_f_notask
  results: []
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# q15_f_notask

This model is a fine-tuned version of [Qwen/Qwen2.5-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct) on the f_notask dataset.

## Model description

More information needed

## Intended uses & limitations

More information needed

## Training and evaluation data

More information needed

## Training procedure

### Training hyperparameters

The following hyperparameters were used during training:
- learning_rate: 0.0002
- train_batch_size: 2
- eval_batch_size: 8
- seed: 0
- distributed_type: multi-GPU
- num_devices: 2
- gradient_accumulation_steps: 4
- total_train_batch_size: 16
- total_eval_batch_size: 16
- optimizer: Use OptimizerNames.ADAMW_TORCH_FUSED with betas=(0.9,0.999) and epsilon=1e-08 and optimizer_args=No additional optimizer arguments
- lr_scheduler_type: cosine
- lr_scheduler_warmup_steps: 0.03
- num_epochs: 1.0

### Training results



### Framework versions

- PEFT 0.18.1
- Transformers 5.6.0
- Pytorch 2.8.0+cu128
- Datasets 4.0.0
- Tokenizers 0.22.2