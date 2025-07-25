
import cv2
from insightface.app import FaceAnalysis
import torch

app = FaceAnalysis(name="buffalo_l", providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
app.prepare(ctx_id=0, det_size=(640, 640))

image = cv2.imread("/home/ryuichi/animins/slider_space_v2/dataset/image.png")
faces = app.get(image)
print(faces[0].normed_embedding)
print(faces[0].normed_embedding.shape)

# faceid_embeds = torch.from_numpy(faces[0].normed_embedding).unsqueeze(0)


# import torch
# from diffusers import StableDiffusionPipeline, DDIMScheduler, AutoencoderKL
# from PIL import Image

# from ip_adapter.ip_adapter_faceid import IPAdapterFaceIDPlus

# base_model_path = "SG161222/Realistic_Vision_V4.0_noVAE"
# vae_model_path = "stabilityai/sd-vae-ft-mse"
# image_encoder_path = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
# v2=False
# ip_ckpt = "/home/ryuichi/animins/slider_space_v2/dataset/ip-adapter-faceid-plusv2_sd15.bin"
# device = "cuda"
# noise_scheduler = DDIMScheduler(
#     num_train_timesteps=1000,
#     beta_start=0.00085,
#     beta_end=0.012,
#     beta_schedule="scaled_linear",
#     clip_sample=False,
#     set_alpha_to_one=False,
#     steps_offset=1,
# )
# vae = AutoencoderKL.from_pretrained(vae_model_path).to(dtype=torch.float16)
# pipe = StableDiffusionPipeline.from_pretrained(
#     base_model_path,
#     torch_dtype=torch.float16,
#     scheduler=noise_scheduler,
#     vae=vae,
#     feature_extractor=None,
#     safety_checker=None
# )

# # load ip-adapter
# ip_model = IPAdapterFaceIDPlus(pipe, image_encoder_path, ip_ckpt, device)

# # generate image
# prompt = "photo of a man his name is JJJ  in a garden,4k,"
# negative_prompt = "monochrome, lowres, bad anatomy, worst quality, low quality, blurry"

# images = ip_model.generate(
#      prompt=prompt, negative_prompt=negative_prompt, face_image=image, faceid_embeds=faceid_embeds, shortcut=v2, s_scale=1.0,
#      num_samples=4, width=512, height=768, num_inference_steps=30, seed=2023
#      )

# images[0].save("output.png")