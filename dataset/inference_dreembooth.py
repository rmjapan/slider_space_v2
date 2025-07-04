from diffusers import DiffusionPipeline, UNet2DConditionModel
from transformers import CLIPTextModel
import torch
import os

# 結果保存用フォルダを作成
output_dir = "/home/ryuichi/animins/slider_space_v2/dataset/result_dreambooth"
os.makedirs(output_dir, exist_ok=True)
print(f"結果保存フォルダを作成しました: {output_dir}")

# 100ステップ目のチェックポイントを使用
checkpoint_path = "/mnt/nas/rmjapan2000/animins/dreambooth-model/checkpoint-1000"

# UNetをロードして明示的にfloat32に変換
print("UNetをロード中...")
unet = UNet2DConditionModel.from_pretrained(f"{checkpoint_path}/unet", torch_dtype=torch.float32)
text_encoder = CLIPTextModel.from_pretrained(f"{checkpoint_path}/text_encoder", torch_dtype=torch.float32)

# パイプラインを作成
print("パイプラインを作成中...")
pipeline = DiffusionPipeline.from_pretrained(
    "hakurei/waifu-diffusion", 
    unet=unet,
    torch_dtype=torch.float32,
    text_encoder=text_encoder,
    safety_checker=None,
).to("cuda")

# ネガティブプロンプト定義
negative_prompt = (
    "worst quality,NSFW, low quality, bad quality, blur, blurry, "
    "bad anatomy, bad hands, extra fingers, missing fingers, "
    "deformed, distorted, disfigured, poorly drawn, "
    "artifacts, noise, jpeg artifacts, compression, "
    "signature, watermark, text, logo, username, "
    "duplicate, multiple faces, two heads, "
    "bad proportions, gross proportions, "
    "overexposed, underexposed, oversaturated"
)

# 感情表現を含む様々なプロンプトでテスト
prompts = [
    "shs 1girl, expressionless, general,4k,detailed face",
    "shs 1girl, relaxed, general",
    "shs 1girl, blank expression, general",
    "shs 1girl, stoic, general",
    "shs 1girl, angry, general",
    "shs 1girl, angry, general",
    "shs 1girl, rage, general",
    "shs 1girl, furious, general",
    "shs 1girl, annoyed, general",
    "shs 1girl, annoyed, general",
    "shs 1girl, rage, general",
    "shs 1girl, sad, general",
    "shs 1girl, sad, general",
    "shs 1girl, despair, general",
    "shs 1girl, crying, general",
    "shs 1girl, melancholic, general",
    "shs 1girl, heartbroken, general",
    "shs 1girl, depressed, general",
    "shs 1girl, smile, general",
    "shs 1girl, smile, general",
    "shs 1girl, joyful, general",
    "shs 1girl, joyful, general",
    "shs 1girl, cheerful, general",
    "shs 1girl, ecstatic, general",
    "shs 1girl, blissful, general",
    "shs 1girl, surprised, general",
    "shs 1girl, shocked, general",
    "shs 1girl, amazed, general",
    "shs 1girl, astonished, general",
    "shs 1girl, bewildered, general",
    "shs 1girl, confused, general",
    "shs 1girl, perplexed, general",
    "shs 1girl, puzzled, general",
    "shs 1girl, baffled, general",
    "shs 1girl, worried, general",
    "shs 1girl, fearful, general",
    "shs 1girl, terrified, general",
    "shs 1girl, nervous, general",
    "shs 1girl, anxious, general",
    "shs 1girl, disgust, general",
    "shs 1girl, repulsed, general",
    "shs 1girl, revolted, general",
    "shs 1girl, embarrassed, general",
    "shs 1girl, bashful, general",
    "shs 1girl, shy, general",
    "shs 1girl, flustered, general",
    "shs 1girl, sleepy, general",
    "shs 1girl, tired, general",
    "shs 1girl, exhausted, general",
    "shs 1girl, drowsy, general",
    "shs 1girl, determined, general",
    "shs 1girl, confident, general",
    "shs 1girl, resolute, general",
    "shs 1girl, fierce, general",
    "shs 1girl, mischievous, general",
    "shs 1girl, teasing, general",
    "shs 1girl, playful, general",
    "shs 1girl, sly, general",
    "shs 1girl, thoughtful, general",
    "shs 1girl, concentrated, general",
    "shs 1girl, pensive, general",
    "shs 1girl, meditative, general",
    "shs 1girl, bored, general",
    "shs 1girl, indifferent, general",
    "shs 1girl, listless, general",
    "shs 1girl, excited, general",
    "shs 1girl, energetic, general",
    "shs 1girl, thrilled, general",
    "shs 1girl, peaceful, general",
    "shs 1girl, tranquil, general",
    "shs 1girl, content, general",
    "shs 1girl, loving, general",
    "shs 1girl, affectionate, general",
    "shs 1girl, gentle, general",
    "shs 1girl, nostalgic, general",
    "shs 1girl, reminiscent, general",
    "shs 1girl, dreamy, general",
    "shs 1girl, curious, general",
    "shs 1girl, intrigued, general",
    "shs 1girl, investigative, general",
    "shs 1girl, bittersweet, general",
    "shs 1girl, conflicted, general",
    "shs 1girl, surprised, joyful, general",
    "shs 1girl, nervous, excited, general",
    "shs 1girl, masterpiece, best quality, detailed face, beautiful lighting, general",

    # "shs 1girl, neutral expression, general,4k,detailed face",
    # "shs 1girl, relaxed expression, general",
    # "shs 1girl, blank expression, general",
    # "shs 1girl, stoic expression, general",
    # "shs 1girl, slightly angry expression, general",
    # "shs 1girl, moderately angry expression, general",
    # "shs 1girl, extremely angry expression, general",
    # "shs 1girl, furious expression, general",
    # "shs 1girl, irritated expression, general",
    # "shs 1girl, annoyed expression, general",
    # "shs 1girl, rage expression, general",
    # "shs 1girl, slightly sad expression, general",
    # "shs 1girl, moderately sad expression, general",
    # "shs 1girl, extremely sad expression, general",
    # "shs 1girl, crying expression, general",
    # "shs 1girl, melancholic expression, general",
    # "shs 1girl, heartbroken expression, general",
    # "shs 1girl, depressed expression, general",
    # "shs 1girl, slightly happy expression, general",
    # "shs 1girl, moderately happy expression, general",
    # "shs 1girl, extremely happy expression, general",
    # "shs 1girl, joyful expression, general",
    # "shs 1girl, cheerful expression, general",
    # "shs 1girl, ecstatic expression, general",
    # "shs 1girl, blissful expression, general",
    # "shs 1girl, surprised expression, general",
    # "shs 1girl, shocked expression, general",
    # "shs 1girl, amazed expression, general",
    # "shs 1girl, astonished expression, general",
    # "shs 1girl, bewildered expression, general",
    # "shs 1girl, confused expression, general",
    # "shs 1girl, perplexed expression, general",
    # "shs 1girl, puzzled expression, general",
    # "shs 1girl, baffled expression, general",
    # "shs 1girl, worried expression, general",
    # "shs 1girl, fearful expression, general",
    # "shs 1girl, terrified expression, general",
    # "shs 1girl, nervous expression, general",
    # "shs 1girl, anxious expression, general",
    # "shs 1girl, disgusted expression, general",
    # "shs 1girl, repulsed expression, general",
    # "shs 1girl, revolted expression, general",
    # "shs 1girl, embarrassed expression, general",
    # "shs 1girl, bashful expression, general",
    # "shs 1girl, shy expression, general",
    # "shs 1girl, flustered expression, general",
    # "shs 1girl, sleepy expression, general",
    # "shs 1girl, tired expression, general",
    # "shs 1girl, exhausted expression, general",
    # "shs 1girl, drowsy expression, general",
    # "shs 1girl, determined expression, general",
    # "shs 1girl, confident expression, general",
    # "shs 1girl, resolute expression, general",
    # "shs 1girl, fierce expression, general",
    # "shs 1girl, mischievous expression, general",
    # "shs 1girl, teasing expression, general",
    # "shs 1girl, playful expression, general",
    # "shs 1girl, sly expression, general",
    # "shs 1girl, thoughtful expression, general",
    # "shs 1girl, concentrated expression, general",
    # "shs 1girl, pensive expression, general",
    # "shs 1girl, meditative expression, general",
    # "shs 1girl, bored expression, general",
    # "shs 1girl, indifferent expression, general",
    # "shs 1girl, listless expression, general",
    # "shs 1girl, excited expression, general",
    # "shs 1girl, energetic expression, general",
    # "shs 1girl, thrilled expression, general",
    # "shs 1girl, peaceful expression, general",
    # "shs 1girl, tranquil expression, general",
    # "shs 1girl, content expression, general",
    # "shs 1girl, loving expression, general",
    # "shs 1girl, affectionate expression, general",
    # "shs 1girl, gentle expression, general",
    # "shs 1girl, nostalgic expression, general",
    # "shs 1girl, reminiscent expression, general",
    # "shs 1girl, dreamy expression, general",
    # "shs 1girl, curious expression, general",
    # "shs 1girl, intrigued expression, general",
    # "shs 1girl, investigative expression, general",
    # "shs 1girl, bittersweet expression, general",
    # "shs 1girl, conflicted expression, general",
    # "shs 1girl, surprised joy expression, general",
    # "shs 1girl, nervous excitement expression, general",
    # "shs 1girl, masterpiece, best quality, detailed face, beautiful lighting, general",
    
    # # 比較用（shsなし）
    # "1girl, neutral expression, 4k, ultra high quality, detailed face, indoors, indoor lighting",
    # "1girl, slightly angry expression, furrowed brows, 4k, ultra high quality, detailed face, indoors, indoor lighting",
    # "1girl, extremely angry expression, glaring eyes, clenched jaw, 4k, ultra high quality, detailed face, indoors, indoor lighting", 
    # "1girl, slightly sad expression, downcast eyes, 4k, ultra high quality, detailed face, indoors, indoor lighting",
    # "1girl, extremely sad expression, tears in eyes, trembling lips, 4k, ultra high quality, detailed face, indoors, indoor lighting",
    # "1girl, slightly happy expression, gentle smile, 4k, ultra high quality, detailed face, indoors, indoor lighting",
    # "1girl, extremely happy expression, bright smile, sparkling eyes, 4k, ultra high quality, detailed face, indoors, indoor lighting",
    # "1girl, slightly cheerful expression, warm smile, 4k, ultra high quality, detailed face, indoors, indoor lighting",
    # "1girl, extremely cheerful expression, beaming smile, joyful eyes, 4k, ultra high quality, detailed face, indoors, indoor lighting",
    # "1girl, 4k, ultra high quality, masterpiece, best quality, detailed face, beautiful indoor lighting"
]

# 各プロンプトで画像を生成
# 出力ディレクトリ内の既存ファイル数を確認
existing_files = len([f for f in os.listdir(output_dir) if f.endswith('.png')])
print(f"既存ファイル数: {existing_files}")

for i, prompt in enumerate(prompts):
    print(f"生成中 ({i+1}/{len(prompts)}): {prompt}")
    
    # ネガティブプロンプト付きで生成
    image = pipeline(
        prompt=prompt, 
        negative_prompt=negative_prompt,
        num_inference_steps=50, 
        guidance_scale=7.5,
        width=1024,
        height=1024,
        seed=torch.Generator().manual_seed(42)
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

    output_path = os.path.join(output_dir, f"{prefix}_{i+existing_files}_{emotion}.png")
    image.save(output_path)
    print(f"保存完了: {output_path}")

print("推論テスト完了！")
print(f"すべての画像が {output_dir} に保存されました。")
print(f"使用したネガティブプロンプト: {negative_prompt}")