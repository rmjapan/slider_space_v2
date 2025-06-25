from diffusers import DiffusionPipeline
import os
import glob

def get_next_index(directory):
    # Get all illustrious_*.png files
    pattern = os.path.join(directory, "illustrious_*.png")
    existing_files = glob.glob(pattern)
    
    # Extract indices from filenames
    indices = []
    for file in existing_files:
        try:
            index = int(os.path.basename(file).split('_')[1].split('.')[0])
            indices.append(index)
        except (ValueError, IndexError):
            continue
    
    # Return next index (0 if no files exist)
    return max(indices) + 1 if indices else 0

def main():
    pipe = DiffusionPipeline.from_pretrained("John6666/a+to("cuda")

    prompt = "1boy,general,anime style face,8k"
    negative_prompt = "lowres, (bad), text, error, fewer, extra, missing, worst quality, jpeg artifacts, low quality, watermark, unfinished, displeasing, oldest, early, chromatic aberration, signature, extra digits, artistic error, username, scan, abstract"
    image = pipe(
                prompt,
                num_inference_steps=20,
                negative_prompt=negative_prompt,
                ).images[0]
    output_dir = "/home/ryuichi/animins/slider_space_v2/test_images_illustrious"
    os.makedirs(output_dir, exist_ok=True)

    # Get next available index and save image
    next_index = get_next_index(output_dir)
    output_path = os.path.join(output_dir, f"illustrious_{next_index}.png")
    image.save(output_path)

if __name__ == "__main__":
    for i in range(10):
        main()