from diffusers import StableDiffusionImg2ImgPipeline, StableDiffusionPipeline, DDIMScheduler, ControlNetModel, StableDiffusionControlNetImg2ImgPipeline, StableDiffusionControlNetPipeline
from diffusers import DiffusionPipeline
import torch
from diffusers.utils import load_image
from PIL import Image
import torchvision.transforms as transforms
import numpy as np
from transformers import pipeline
import cv2
from io import BytesIO
# from src.image_generation.components.postprocessor import Postprocessor

'''
20230731
TODO: Init model, Readability, Captioning function(Clip, DeepDanBooru), Postprocessing(GFPGAN, ESRGAN, CODEFORMER), Generate model with different strength
TODO: Apply Multi lora(not supported YET), Train with different painting styles and artists with AnyLORA, Using fast Sampler(UniPCMultistepScheduler)
TODO: Merge Lora file in webui and use it in diffusers
'''

def image_grid(imgs, rows, cols):
    assert len(imgs) == rows*cols

    w, h = imgs[0].size
    grid = Image.new('RGB', size=(cols*w, rows*h))
    grid_w, grid_h = grid.size
    
    for i, img in enumerate(imgs):
        grid.paste(img, box=(i%cols*w, i//cols*h))
    return grid

# for Depth Estimation
controlnet = ControlNetModel.from_pretrained("lllyasviel/control_v11f1p_sd15_depth", torch_dtype=torch.float16)

# for Canny Edges
# controlnet = ControlNetModel.from_pretrained("lllyasviel/control_v11p_sd15_canny", torch_dtype=torch.float16)

# depth_estimator = pipeline('depth-estimation')
depth_estimator = pipeline(task="depth-estimation", model="Intel/dpt-large")

# buffer = load_image("./upper.jpg")
buffer = load_image("./man_upper.png")
# buffer = buffer.resize((512, 512))
# buffer = Image.open('/home/jinnnn/kohya_ss/images/img/cha.jpg').convert("RGB")

image_depth = depth_estimator(buffer)['depth']
image_depth = np.array(image_depth)
image_depth = image_depth[:, :, None]
image_depth = np.concatenate([image_depth, image_depth, image_depth], axis=2)
image_depth = Image.fromarray(image_depth)
# image_depth.save('depth1.png')

'''
Using BytesIO
'''
# buffer=open('./man_upper.png', 'rb')
# buffer.seek(0)
# image_bytes = buffer.read()
# images_bytes = Image.open(BytesIO(image_bytes))

'''Canny with controlnet'''
# buffer_numpy = np.array(buffer)
# low_threshold = 100
# high_threshold = 200
# image_canny = cv2.Canny(buffer_numpy, low_threshold, high_threshold)
# image_canny = image_canny[:, : ,None]
# image_canny = np.concatenate([image_canny, image_canny, image_canny], axis=2)
# image_canny = Image.fromarray(image_canny)

model_base = "/hdd/jinnnn/anylora.safetensors"

'''Using HuggingFace model: https://huggingface.co/emilianJR/AnyLORA/tree/main'''
# pipeline = StableDiffusionImg2ImgPipeline.from_pretrained("emilianJR/AnyLORA", 
#                                          torch_dtype=torch.float16, safety_checker=None,
#                                          requires_safety_checker=False).to('cuda')


pipe = StableDiffusionControlNetImg2ImgPipeline.from_single_file(model_base, controlnet=controlnet, torch_dtype=torch.float16, safety_checker=None, requires_safety_checker=False).to("cuda")

pipe.load_lora_weights(".", weight_name="/home/jinnnn/kohya_ss/lorafile/last.safetensors")
pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)

'''ClipSkip=2'''
clip_layers = pipe.text_encoder.text_model.encoder.layers
clip_skip = 1
if clip_skip > 0:
    pipe.text_encoder.text_model.encoder.layers = clip_layers[:-clip_skip]

'''Using xformers'''
pipe.enable_xformers_memory_efficient_attention()

prompt = "masterpiece, best quality, iom style, 1boy, cartoon"

negative_prompt = ("(low quality, worst quality:1.4), (bad anatomy), (inaccurate limb:1.2), "
                   "bad composition, inaccurate eyes, extra digit, fewer digits, (extra arms:1.2), large breasts")

pipe.safety_checker = None
pipe.requires_safety_checker = False

'''Image size is equal to 'image' variable size'''
images = pipe(control_image = image_depth,
                  prompt=prompt, image=buffer,
     negative_prompt=negative_prompt, 
     cross_attention_kwargs={"scale":0.5},
     num_inference_steps=30, 
     num_images_per_prompt=4,
     guidance_scale = 7,
     strength=0.6,
     generator=torch.Generator(device='cuda').manual_seed(0)).images
grid = image_grid(images, 1, 4)
grid.save('maneights.png', format='PNG')

