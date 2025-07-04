from diffusers import DiffusionPipeline, UNet2DConditionModel
from transformers import CLIPTextModel
import torch
import os

# 結果保存用フォルダを作成
output_dir = "/home/ryuichi/animins/slider_space_v2/dataset/result_dreambooth"
os.makedirs(output_dir, exist_ok=True)
print(f"結果保存フォルダを作成しました: {output_dir}")

# 100ステップ目のチェックポイントを使用
checkpoint_dir = "/mnt/nas/rmjapan2000/animins/dreambooth-model_sdxl/checkpoint-1000"
# パイプラインを作成
print("パイプラインを作成中...")
pipeline = DiffusionPipeline.from_pretrained(
    "John6666/any-illustrious-xl-for-lora-training-v01-sdxl", 
    torch_dtype=torch.float32
).to("cuda")
pipeline.load_lora_weights(checkpoint_dir + "/pytorch_lora_weights.safetensors",
                           adapter_name="lora")
pipeline.set_adapters("lora",adapter_weights=1.0)
# ネガティブプロンプト定義  
negative_prompt = ["lowres", "(bad)", "text", "error", "fewer", "extra", "missing", "worst quality", "jpeg artifacts", "low quality", "watermark", "unfinished", "displeasing", "oldest", "early", "chromatic aberration", "signature", "extra digits", "artistic error", "username", "scan", "abstract", "side view"]


# 感情表現を含む様々なプロンプトでテスト
prompts = [
    "shs 1girl, neutral expression,masterpiece,high quality"
]

# 各プロンプトで画像を生成
for i, prompt in enumerate(prompts):
    print(f"生成中 ({i+1}/{len(prompts)}): {prompt}")
    
    # ネガティブプロンプト付きで生成
    image = pipeline(
        prompt=prompt, 
        negative_prompt=negative_prompt,
        num_inference_steps=25, 
        guidance_scale=7,
        width=1024,
        height=1024
    ).images[0]
    
    # ファイル名を感情に基づいて設定
    if "shs" in prompt:
        prefix = "shs"
    else:
        prefix = "comparison"
        
    if "neutral" in prompt:
        emotion = "neutral"
    elif "angry" in prompt:
        if "extremely" in prompt:
            emotion = "extremely_angry"
        else:
            emotion = "slightly_angry"
    elif "sad" in prompt:
        if "extremely" in prompt:
            emotion = "extremely_sad"
        else:
            emotion = "slightly_sad"
    elif "happy" in prompt:
        if "extremely" in prompt:
            emotion = "extremely_happy"
        else:
            emotion = "slightly_happy"
    elif "cheerful" in prompt:
        if "extremely" in prompt:
            emotion = "extremely_cheerful"
        else:
            emotion = "slightly_cheerful"
    else:
        emotion = "default"
    
    output_path = os.path.join(output_dir, f"{prefix}_{i:02d}_{emotion}.png")
    image.save(output_path)
    print(f"保存完了: {output_path}")

print("推論テスト完了！")
print(f"すべての画像が {output_dir} に保存されました。")
print(f"使用したネガティブプロンプト: {negative_prompt}")