import open_clip
from PIL import Image
import torch

# OpenVisionモデルとトランスフォーマーを読み込む
clip_model, preprocess_train, preprocess_val = open_clip.create_model_and_transforms('hf-hub:UCSC-VLAA/openvision-vit-large-patch14-224')
tokenizer = open_clip.get_tokenizer('hf-hub:UCSC-VLAA/openvision-vit-large-patch14-224')
# 評価モードに設定
clip_model.eval()

# 画像を読み込んで前処理
画像 = preprocess_val(Image.open("/home/ryuichi/animins/slider_space_v2/test_images_illustrious/illustrious_25.png")).unsqueeze(0)
# テキストをトークン化
テキスト = tokenizer(["boy", "anime-style face", "anime-style boy", "anime-style girl", "anime-style stoic boy"])

# 勾配計算なしで推論を実行
with torch.no_grad():
    # 画像特徴量を抽出
    画像特徴量 = clip_model.encode_image(画像)
    # テキスト特徴量を抽出
    テキスト特徴量 = clip_model.encode_text(テキスト)
    # 特徴量を正規化
    画像特徴量 = 画像特徴量 / 画像特徴量.norm(dim=-1, keepdim=True)
    テキスト特徴量 = テキスト特徴量 / テキスト特徴量.norm(dim=-1, keepdim=True)
    # 類似度を計算して確率に変換
    テキスト確率 = (100.0 * 画像特徴量 @ テキスト特徴量.T).softmax(dim=-1)

# 結果を表示
print(f"テキスト確率: {テキスト確率}")
      
