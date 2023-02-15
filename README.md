# LoRA-deployment

This repository demonstrates how to serve multiple [LoRA fine-tuned Stable Diffusions](https://huggingface.co/blog/lora) from 🧨 Diffusers library on Hugging Face Inference Endpoint. Since only few MB of checkpoint is produced after finetuning with LoRA, we can switch different checkpoint for different fine-tuned Stable Diffusion in super quick, memory efficient, and disk space efficient ways.

## Notebook

- [Pilot notebook](https://github.com/deep-diver/LoRA-deployment/blob/main/notebooks/pilot.ipynb): shows how to write and test a custom handler for Hugging Face Inference Endpoint in local or Colab environments.
- [Inference notebook](https://github.com/deep-diver/LoRA-deployment/blob/main/notebooks/inference.ipynb): shows how to run inference on the custom handler deployed to Hugging Face Inference Endopint.

## Reference
- https://huggingface.co/blog/lora
- https://huggingface.co/taesiri/ethan_ai_lora
- https://huggingface.co/kuotient/noto-emoji-finetuned-lora
- https://huggingface.co/junnyu/lora_dreambooth_dog_example
- https://huggingface.co/pcuenq/pokemon-lora
