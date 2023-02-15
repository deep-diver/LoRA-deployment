# LoRA-deployment

This repository demonstrates how to serve multiple [LoRA fine-tuned Stable Diffusions](https://huggingface.co/blog/lora) from 🧨 Diffusers library on Hugging Face Inference Endpoint. Since only fewㅇㄷ MB of checkpoint is produced after finetuning with LoRA, we can switch different checkpoint for different fine-tuned Stable Diffusion in super quick, memory efficient, and disk space efficient ways.

For demonstration purpose, I have tested the following Hugging Face Model repositories which has LoRA fine-tuned checkpoint(`pytorch_lora_weights.bin
`):
- [ethan_ai](https://huggingface.co/taesiri/ethan_ai_lora)
- [noto-emoji](https://huggingface.co/kuotient/noto-emoji-finetuned-lora)
- [pokemon](https://huggingface.co/pcuenq/pokemon-lora)

## Notebook

- [Pilot notebook](https://github.com/deep-diver/LoRA-deployment/blob/main/notebooks/pilot.ipynb): shows how to write and test a custom handler for Hugging Face Inference Endpoint in local or Colab environments
- [Inference notebook](https://github.com/deep-diver/LoRA-deployment/blob/main/notebooks/inference.ipynb): shows how to request inference to the custom handler deployed on Hugging Face Inference Endopint
- [Multi-workers inference notebook](https://github.com/deep-diver/LoRA-deployment/blob/main/notebooks/multiworker_inference.ipynb): shows how to run simultaneous requests to the custom handler deployed on Hugging Face Inference Endpoint in Colab environment

## Custom Handler

- [handler.py](https://github.com/deep-diver/LoRA-deployment/blob/main/custom_handler/handler.py): basic handler. This custom handler is proved to work with [this Hugging Face Model repo](https://huggingface.co/chansung/LoRA-deployment)
- [multiworker_handler.py](https://github.com/deep-diver/LoRA-deployment/blob/main/custom_handler/multiworker_handler.py): advanced handler with multiple worker(Stable Diffusion) pool. This custom handler is proved to work with [this Hugging Face Model repo](https://huggingface.co/chansung/LoRA-deployment-multiworkers)

## Script

- [inference.py](https://github.com/deep-diver/LoRA-deployment/blob/main/scripts/inference.py): standalone Python script to send requests to the custom handler deployed on Hugging Face Inference Endpoint

## Reference
- https://huggingface.co/blog/lora
