from typing import Dict, List, Any
import sys
import base64
import logging
import copy

import numpy as np
import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler

class EndpointHandler():
  def __init__(self, path="", model_base="runwayml/stable-diffusion-v1-5"):
    self.pipe = StableDiffusionPipeline.from_pretrained(model_base, torch_dtype=torch.float16)
    self.pipe.scheduler = DPMSolverMultistepScheduler.from_config(self.pipe.scheduler.config)
    self.original_unet = copy.deepcopy(self.pipe.unet)

  def _generate_images(
    self,
    model_path,
    prompt,
    num_inference_steps=25, 
    guidance_scale=7.5, 
    num_images_per_prompt=1):

    if model_path == "base":
      self.pipe.unet = copy.deepcopy(self.original_unet)
    else:
      self.pipe.unet.load_attn_procs(model_path)
    self.pipe.to("cuda")

    pil_images = self.pipe(
      prompt=prompt,
      num_inference_steps=num_inference_steps,
      guidance_scale=guidance_scale,
      num_images_per_prompt=num_images_per_prompt).images

    np_images = []
    for i in range(len(pil_images)):
      np_images.append(np.asarray(pil_images[i]))

    return np.stack(np_images, axis=0)

  def __call__(self, data: Dict[str, Any]) -> str:
      prompt = data.pop("inputs", "test image")
      model_path = data.pop("model_path", "base")
      
      num_inference_steps = data.pop("num_inference_steps", 25)
      guidance_scale = data.pop("guidance_scale", 7.5)
      num_images_per_prompt = data.pop("num_images_per_prompt", 1)

      images = self._generate_images(
        model_path, prompt, 
        num_inference_steps, guidance_scale, num_images_per_prompt
      )

      return base64.b64encode(images.tobytes()).decode()
