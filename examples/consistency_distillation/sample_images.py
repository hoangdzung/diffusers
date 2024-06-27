from tqdm import tqdm
import torch
from diffusers import LatentConsistencyModelPipeline
from diffusers.models import AutoencoderKL, UNet2DConditionModel
from diffusers.schedulers import LCMScheduler
from transformers import CLIPImageProcessor, CLIPTextModel, CLIPTokenizer, CLIPVisionModelWithProjection, AutoTokenizer

import numpy as np
import sys
import json
import os

ckpt_path = sys.argv[1]
prompt_path = sys.argv[2]
save_dir = sys.argv[3]
batch_size = int(sys.argv[4])

model_name = "SimianLuo/LCM_Dreamshaper_v7"
vae = AutoencoderKL.from_pretrained(model_name, subfolder="vae")
text_encoder = CLIPTextModel.from_pretrained(model_name, subfolder="text_encoder")
tokenizer = AutoTokenizer.from_pretrained(model_name, subfolder="tokenizer", use_fast=False)
unet = UNet2DConditionModel.from_pretrained(ckpt_path)
scheduler = LCMScheduler.from_pretrained(model_name, subfolder="scheduler")
feature_extractor = CLIPImageProcessor.from_pretrained(model_name, subfolder="feature_extractor")

pipe = LatentConsistencyModelPipeline(
    vae=vae,
    text_encoder=text_encoder,
    tokenizer=tokenizer,
    unet=unet,
    scheduler=scheduler,
    safety_checker=None,
    feature_extractor=feature_extractor,
    requires_safety_checker=False
)
pipe.set_progress_bar_config(disable=True)
print("Move pipe to cuda")
pipe = pipe.to("cuda")
print("Done")

width = 512
height= 512
num_inference_steps = 1
guidance_scale = 8.0

prompts = json.load(open(prompt_path))
hashes = sorted(prompts)

os.makedirs(save_dir, exist_ok=True)
img_dir = os.path.join(save_dir, "images")
os.makedirs(img_dir, exist_ok=True)

all_images = []

for i in tqdm(range(0, len(hashes), batch_size)):
    prompt_batch = [prompts[hashes[hash_id]] for hash_id in range(i, min(i + batch_size, len(hashes)))]

    outputs = pipe(
        prompt=prompt_batch,
        width=width,
        height=height,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
        num_images_per_prompt=3,
        lcm_origin_steps=50,
        output_type="pil",
    )
    
    outputs.images[0].save(os.path.join(img_dir, f"{hashes[i]}.png"))

    images = [np.asarray(image) for image in outputs.images]
    all_images.extend(images[: batch_size//2])
all_images = np.stack(all_images)
np.savez(os.path.join(save_dir, 'stacked_images.npz'), arr_0=all_images)
