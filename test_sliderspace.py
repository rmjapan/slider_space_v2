import random
from utils.model_util import load_model_illustrious_xl
import argparse
import torch
import json
import os

        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_id",
        type=str,
        default="John6666/any-illustrious-xl-for-lora-training-v01-sdxl",
        choices=[
            "stabilityai/stable-diffusion-xl-base-1.0", 
            "John6666/any-illustrious-xl-for-lora-training-v01-sdxl"
        ],
        help="Base model to use for testing"
    )
    parser.add_argument(
        "--distilled_ckpt",
        type=str,
        # default='dmd2',
        default='None',
        help="Path to DMD checkpoint"
    )
    parser.add_argument(
        "--device",
        type=str,
        default='cuda:0',
        help="Device to run training on"
    )
    # Prompt Configuration
    parser.add_argument(
        "--concept_prompts",
        type=str,
        nargs='+',
        # default=['1 girl,solo,masterpiece, "character looking straight ahead", "front view", "head shot", best quality, good quality'],
        default=['1 girl,solo,hatsune miku,masterpiece,best quality,character looking straight ahead,front view,head shot,simple background,general'],
        # default=['1 girl,solo,masterpiece,best quality,good quality,head shot,character looking straight ahead,front view,general,seductive smile:1.3,half-lidded eyes,full color'],
        help="List of concept prompts to use for training",
    )

    parser.add_argument(
        "--encoder",
        type=str,
        default='emotion_clip',
        choices=['clip', 'openvision','emotion_clip','emo_next'],
        help="Encoder to use for feature extraction"
    )
    parser.add_argument(
        "--dtype",
        type=torch.dtype,
        default=torch.bfloat16,
        help="Data type for model precision"
    )
    parser.add_argument(
        "--max_denoising_steps",
        type=int,
        # default=4,dmd用
        default=21,#illustrious用
        help="Maximum number of denoising steps"
    )








    # Parse arguments
    args = parser.parse_args()

    # Set up parameters from arguments
    test_params = {
        'model_id': args.model_id,
        'pretrained_model_name_or_path': args.model_id,
        'distilled': args.distilled_ckpt,
        'height': 1024,
        'width': 1024,
        'weight_dtype': args.dtype,
        'device': args.device,
        'encoder': args.encoder,
        'max_denoising_steps':args.max_denoising_steps,
    }
    seed=random.randint(0, 2**15)
    test_params['seed']=seed
    
    models,pipeline= load_model_illustrious_xl(test_params)
    concept_num=16898
    # Create main output directory
    output_dir = f"test_sliderspace_{concept_num}"
    version = 1
    while os.path.exists(output_dir):
        output_dir = f"test_sliderspace_{concept_num}_v{version}"
        version += 1
    os.makedirs(output_dir, exist_ok=True)
    
    # Save test parameters to JSON file
    test_params_save = {
        'model_id': args.model_id,
        'distilled_ckpt': args.distilled_ckpt,
        'device': args.device,
        'concept_prompts': args.concept_prompts,
        'encoder': args.encoder,
        'dtype': str(args.dtype),
        'max_denoising_steps': args.max_denoising_steps,
        'height': 1024,
        'width': 1024,
        'guidance_scale': 5.0,
        'negative_prompt': "lowres, (bad), text, error, fewer, extra, missing, worst quality, jpeg artifacts, low quality, watermark, unfinished, displeasing, oldest, early, chromatic aberration, signature, extra digits, artistic error, username, scan, abstract",
        'seed': seed,
        'concept_num': concept_num
    }
    
    with open(os.path.join(output_dir, 'test_params.json'), 'w') as f:
        json.dump(test_params_save, f, indent=2)
    
    seed=random.randint(0, 2**15)
    for sliider_num in range(0,64):
        pipeline.load_lora_weights(f"/home/ryuichi/animins/slider_space_v2/trained_sliders/sdxl/concept_{concept_num}",weight_name=f"slider_{sliider_num}.pt",adapter_name=f"slider_{sliider_num}")

        scale_list=[-2.0,-1.5,-1.0,-0.5,0.0,0.5,1.0,1.5,2.0]
        os.makedirs(f"{output_dir}/slider_{sliider_num}",exist_ok=True)
        for i in range(9):
            pipeline.set_adapters(f"slider_{sliider_num}",adapter_weights=scale_list[i])    
            image=pipeline(
                prompt=args.concept_prompts,
                guidance_scale=5.0,
                num_images_per_prompt=1,
                output_type="pil",
                negative_prompt="lowres, (bad), text, error, fewer, extra, missing, worst quality, jpeg artifacts, low quality, watermark, unfinished, displeasing, oldest, early, chromatic aberration, signature, extra digits, artistic error, username, scan, abstract",
                height=1024,
                width=1024,
                generator=torch.manual_seed(seed)
            ).images[0]
            image.save(f"{output_dir}/slider_{sliider_num}/test_image_{sliider_num}_{scale_list[i]:.1f}.png")
        pipeline.delete_adapters(f"slider_{sliider_num}")
            
