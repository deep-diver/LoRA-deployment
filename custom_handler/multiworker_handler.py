from typing import Dict, List, Any
import sys
import base64
import logging
import copy

import numpy as np
import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler

class ReusablePipePool:
    def __init__(
      self, 
      size,
      model_base="runwayml/stable-diffusion-v1-5"
    ):
      self._reusablePipes = []
      for i in range(size):
        pipe = StableDiffusionPipeline.from_pretrained(
          model_base, torch_dtype=torch.float16
        )
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
        self._reusablePipes.append(pipe)

      if not self.empty():
        self.original_unet = copy.deepcopy(self._reusablePipes[0].unet)

    def acquire(self):
      return self._reusablePipes.pop()

    def release(self, reusablePipe):
      self._reusablePipes.append(reusablePipe)

    def empty(self):
      return len(self._reusablePipes) == 0

class EndpointHandler():
  def __init__(self, path=""):
    self.pool = ReusablePipePool(2)

  def _generate_images(
    self,
    model_path,
    prompt,
    num_inference_steps=25, 
    guidance_scale=7.5, 
    num_images_per_prompt=1):

    reusablePipe = None
    while not self.pool.empty():
      reusablePipe = self.pool.acquire()

    if model_path == "base":
      reusablePipe.unet = copy.deepcopy(self.pool.original_unet)
    else:
      reusablePipe.unet.load_attn_procs(model_path)
    reusablePipe.to("cuda")

    pil_images = reusablePipe(
      prompt=prompt,
      num_inference_steps=num_inference_steps,
      guidance_scale=guidance_scale,
      num_images_per_prompt=num_images_per_prompt).images

    self.pool.release(reusablePipe)

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
