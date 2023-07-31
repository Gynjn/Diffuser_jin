import os
from PIL import Image

caption_model_name = 'blip-large'
clip_model_name = 'ViT-L-14/openai'

from clip_interrogator import Config, Interrogator

config = Config()
config.clip_model_name = clip_model_name
config.caption_model_name = caption_model_name
ci = Interrogator(config)

image = Image.open("./upper.jpg").convert('RGB')
print(ci.interrogate_fast(image))
