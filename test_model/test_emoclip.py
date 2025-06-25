from EmotionCLIP.src.models.base import EmotionCLIP
import torch
from PIL import Image
import numpy as np
import torch.nn.functional as F
from transformers import CLIPTokenizer

# Initialize model
model = EmotionCLIP(
    video_len=8,  # デフォルト値
    backbone_checkpoint=None
)

# Load checkpoint
checkpoint = torch.load('/home/ryuichi/animins/slider_space_v2/EmotionCLIP/emotionclip_latest.pt')
model.load_state_dict(checkpoint['model'], strict=True)
model.eval()
model.to('cuda')

# Load and preprocess image
image = Image.open('/home/ryuichi/animins/slider_space_v2/training_images/sdxl/concept_13249/20.png')
image = image.resize((224, 224))
image = image.convert('RGB')
image = np.array(image)
image = image.transpose(2, 0, 1)  # HWC -> CHW
image = torch.from_numpy(image).float() / 255.0  # Normalize to [0, 1]
image = image.unsqueeze(0)  # Add batch dimension
image = image.to('cuda')

# Define emotion prompts
emotion_prompts = {
    "joy": "a photo depicting joy",
    "anger": "a photo depicting anger",
    "sadness": "a photo depicting sadness",
    "pleasure": "a photo depicting pleasure" # 「楽」はpleasureとしました
}
emotion_labels = list(emotion_prompts.keys())
text_prompts = list(emotion_prompts.values())

# Simple tokenizer (placeholder - ideally use CLIP's actual tokenizer)
# CLIP typically uses a context length of 77

# context_length = 77 # model.context_length should be used if available # Keep for reference if needed from model
# Determine context_length from the model's text configuration if possible, otherwise default
try:
    context_length = model.backbone.context_length
except AttributeError:
    print("Warning: model.backbone.context_length not found, defaulting to 77. Ensure this matches the model's training.")
    context_length = 77

# Load the correct CLIP tokenizer
# Based on ViT-B-32.json, "openai/clip-vit-base-patch32" is the corresponding Hugging Face model
tokenizer_name = "openai/clip-vit-base-patch32"
try:
    tokenizer = CLIPTokenizer.from_pretrained(tokenizer_name)
except Exception as e:
    print(f"Error loading tokenizer {tokenizer_name}: {e}")
    print("Please ensure you have an internet connection and the tokenizer name is correct.")
    print("Falling back to dummy tokenization, results will not be meaningful.")
    tokenizer = None # Fallback to indicate tokenizer loading failed

if tokenizer:
    tokenized_texts = tokenizer(
        text_prompts, 
        padding="max_length", 
        max_length=context_length, 
        truncation=True, 
        return_tensors="pt"
    )["input_ids"].to('cuda')
else:
    # Fallback to dummy tokenization if tokenizer loading failed
    print("Using dummy tokenization due to tokenizer loading failure.")
    tokenized_texts = torch.zeros((len(text_prompts), context_length), dtype=torch.long, device='cuda')


# Create dummy mask (全1のマスク)
# The mask for encode_image should match the image's spatial dimensions if it's per-pixel.
# However, the original EmotionCLIP VisualTransformer's forward method expects a mask that it can pool.
# Let's re-check the VisualTransformer's forward pass in base.py.
# VisualTransformer's forward in base.py:
# x1 = self.avg_pool(mask.unsqueeze(1).float()) # shape = [*, 1, grid, grid]
# This suggests the mask should be related to the input image dimensions before patch embedding.
# If the input image is (B, C, H, W), the mask could be (B, H, W).
# The original code used mask = torch.ones((1, 8), dtype=torch.bool, device='cuda')
# This was likely a placeholder or for a different input type (e.g., video frames).
# For a single image, a mask of (1, H, W) or (1, 1, H, W) might be more appropriate.
# Or, if the model handles it, a simpler mask related to sequence length after patch embedding might be used.
# Let's stick to a full attention mask for the image for now, similar to what a ViT might use for its patches + CLS token.
# The `encode_image` method in `EmotionCLIP` class directly calls `self.backbone.encode_image(image, image_mask)`.
# The `VisualTransformer.forward` takes `mask: torch.Tensor`.
# `x1 = self.avg_pool(mask.unsqueeze(1).float())`
# `x1 = x1.reshape(x1.shape[0], x1.shape[1], -1)`
# `x1 = x1.permute(0, 2, 1)`
# `x1 = x1 * self.positional_embedding[1:].to(x.dtype)`
# This implies the mask is used spatially. Given image is (1, 3, 224, 224), mask should be (1, 224, 224).
mask = torch.ones((1, 224, 224), dtype=torch.bool, device='cuda')


# Forward pass
with torch.no_grad():
    with torch.amp.autocast('cuda'):  # 非推奨の警告を修正
        print("Image shape:", image.shape)
        print("Image mask shape:", mask.shape)
        image_features = model.encode_image(image, mask)
        image_features = F.normalize(image_features, dim=-1)

        print("Tokenized texts shape:", tokenized_texts.shape)
        # Ensure tokenized_texts is on the same device as the model
        text_features = model.encode_text(tokenized_texts)
        text_features = F.normalize(text_features, dim=-1)

        # Calculate similarities
        # The logit_scale is usually applied before softmax if calculating probabilities
        # For just finding the "closest", raw similarity (dot product of normalized features) is enough.
        # similarity = (image_features @ text_features.T) * model.logit_scale.exp() # if you want scaled logits
        similarity = image_features @ text_features.T # Shape: (1, num_emotions)
        
        # Get probabilities (optional, but good for interpretation)
        probs = F.softmax(similarity * model.logit_scale.exp(), dim=-1)

print("Features shape:", image_features.shape)
# print("Features:", features) # This was from the old code, `features` is now `image_features`
print("Text features shape:", text_features.shape)
print("\n--- Emotion Analysis ---")
print(f"Image: /home/ryuichi/animins/slider_space_v2/test_images_illustrious/illustrious_6.png")
for i, label in enumerate(emotion_labels):
    print(f"Similarity with '{label}' ({text_prompts[i]}): {similarity[0, i].item():.4f} (Prob: {probs[0, i].item():.4f})")

most_similar_idx = torch.argmax(similarity, dim=-1).item()
most_similar_emotion = emotion_labels[most_similar_idx]
print(f"\nThe image is most similar to: {most_similar_emotion} (Similarity: {similarity[0, most_similar_idx].item():.4f}, Prob: {probs[0, most_similar_idx].item():.4f})")
