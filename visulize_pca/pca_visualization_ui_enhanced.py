#!/usr/bin/env python3
"""
拡張版ハイブリッドPCA可視化UI - Streamlit版（主成分選択機能付き）

機能:
1. インタラクティブなパラメータ調整
2. 散布図上に直接画像表示
3. 任意の主成分組み合わせ選択
4. リアルタイム可視化更新
5. 結果の保存・表示
6. 複数エンコーダー対応
7. 高解像度出力
8. 寄与率分析表示

UI版と直接版の良いところを組合わせたハイブリッドシステム + 主成分選択機能
"""

import streamlit as st
import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from transformers import CLIPModel, CLIPProcessor, AutoModel
import torch.nn.functional as F
from typing import List, Optional, Tuple
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import random
import io
import base64
import time
import seaborn as sns
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from io import BytesIO

# プロジェクトのルートディレクトリを検出
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)  # visulize_pcaの親ディレクトリ

# プロジェクトルートをPythonパスに追加
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# オプショナルなモジュールのインポート
OPENVISION_AVAILABLE = False
EMOTION_CLIP_AVAILABLE = False

try:
    from utils.model_util import load_openvision_model
    OPENVISION_AVAILABLE = True
except ImportError:
    pass

try:
    from EmotionCLIP.src.models.base import EmotionCLIP
    EMOTION_CLIP_AVAILABLE = True
except ImportError:
    pass

# Streamlitページ設定
st.set_page_config(
    page_title="🎨 拡張版ハイブリッドPCA可視化システム",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_data
def load_image_dataset(data_path: str, max_images: int) -> Tuple[List[Image.Image], List[str]]:
    """
    画像データセットを読み込む（キャッシュ対応）
    """
    data_path = Path(data_path)
    images = []
    image_paths = []
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
    
    if data_path.is_dir():
        for file_path in list(data_path.rglob('*'))[:max_images]:
            if file_path.suffix.lower() in image_extensions:
                try:
                    img = Image.open(file_path).convert('RGB')
                    images.append(img)
                    image_paths.append(str(file_path))
                except Exception:
                    continue
    
    return images, image_paths

@st.cache_resource
def load_encoder(encoder_type: str, device: str = 'cuda'):
    """
    エンコーダーを読み込む（リソースキャッシュ対応）
    """
    # セッション状態からチェックポイントパスを取得
    checkpoint_path = getattr(st.session_state, f'{encoder_type}_checkpoint', None)
    if not checkpoint_path:
        default_paths = get_default_checkpoint_paths()
        checkpoint_path = default_paths.get(encoder_type, "")
    
    if encoder_type == 'clip':
        try:
            model = CLIPModel.from_pretrained(
                checkpoint_path,
                torch_dtype=torch.float16
            )
            processor = CLIPProcessor.from_pretrained(checkpoint_path)
            st.info(f"✅ CLIP モデル読み込み完了: {checkpoint_path}")
        except Exception as e:
            st.error(f"❌ CLIP モデル読み込み失敗: {str(e)}")
            raise
        
    elif encoder_type == 'emotion_clip':
        if not EMOTION_CLIP_AVAILABLE:
            raise ImportError("EmotionCLIPのインポートに失敗しました")
        
        if not os.path.exists(checkpoint_path):
            candidates = find_checkpoint_candidates('emotion_clip')
            error_msg = f"EmotionCLIPのチェックポイントが見つかりません: {checkpoint_path}"
            if candidates:
                error_msg += f"\n\n利用可能なファイル:\n" + "\n".join(f"  • {p}" for p in candidates[:5])
                if len(candidates) > 5:
                    error_msg += f"\n  ... 他 {len(candidates) - 5} 個"
            raise FileNotFoundError(error_msg)
        
        try:
            checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
            model = EmotionCLIP(video_len=8, backbone_checkpoint=None)
            model.load_state_dict(checkpoint['model'], strict=True)
            processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
            st.info(f"✅ EmotionCLIP チェックポイント読み込み完了: {Path(checkpoint_path).name}")
        except Exception as e:
            raise RuntimeError(f"EmotionCLIPチェックポイントの読み込みに失敗: {str(e)}")
        
    elif encoder_type == 'dinov2-small':
        try:
            model = AutoModel.from_pretrained(
                checkpoint_path,
                torch_dtype=torch.float16
            )
            processor = None
            st.info(f"✅ DINOv2 モデル読み込み完了: {checkpoint_path}")
        except Exception as e:
            st.error(f"❌ DINOv2 モデル読み込み失敗: {str(e)}")
            raise
        
    elif encoder_type == 'openvision':
        if not OPENVISION_AVAILABLE:
            raise ImportError("OpenVisionのインポートに失敗しました")
        
        try:
            if os.path.exists(checkpoint_path):
                # ローカルファイルから読み込み
                model = load_openvision_model(checkpoint_path)
                st.info(f"✅ OpenVision チェックポイント読み込み完了: {Path(checkpoint_path).name}")
            else:
                # Hugging Face IDまたはデフォルトパスで読み込み
                model = load_openvision_model(checkpoint_path)
                st.info(f"✅ OpenVision モデル読み込み完了: {checkpoint_path}")
            processor = None
        except Exception as e:
            raise RuntimeError(f"OpenVisionモデルの読み込みに失敗: {str(e)}")
        
    else:
        raise ValueError(f"サポートされていないエンコーダータイプ: {encoder_type}")
    
    model.eval()
    model.to(device)
    if encoder_type == 'emotion_clip':
        model = model.to(torch.bfloat16)
    model.requires_grad_(False)
    
    return model, processor

@st.cache_data
def encode_images(images: List[Image.Image], encoder_type: str, batch_size: int = 8, device: str = 'cuda') -> np.ndarray:
    """
    画像をエンコードする（キャッシュ対応）
    """
    model, processor = load_encoder(encoder_type, device)
    
    all_features = []
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i in range(0, len(images), batch_size):
        batch_images = images[i:i+batch_size]
        progress = (i + batch_size) / len(images)
        progress_bar.progress(min(progress, 1.0))
        status_text.text(f"🔄 エンコード中: {i+1}-{min(i+batch_size, len(images))}/{len(images)}")
        
        with torch.no_grad():
            if encoder_type == 'clip':
                inputs = processor(images=batch_images, return_tensors="pt", padding=True)
                inputs = {k: v.to(device) for k, v in inputs.items()}
                outputs = model.get_image_features(**inputs)
                features = F.normalize(outputs, dim=-1)
                
            elif encoder_type == 'emotion_clip':
                clip_inputs = processor(images=batch_images, return_tensors="pt", padding=True)
                pixel_values = clip_inputs['pixel_values'].to(device)
                mask = torch.ones((len(batch_images), 224, 224), dtype=torch.bfloat16, device=device)
                features = model.encode_image(pixel_values.to(torch.bfloat16), mask)
                features = F.normalize(features, dim=-1)
                
            elif encoder_type == 'dinov2-small':
                from torchvision import transforms
                transform = transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])
                
                processed_images = []
                for img in batch_images:
                    processed_img = transform(img)
                    processed_images.append(processed_img)
                
                batch_tensor = torch.stack(processed_images).to(device)
                outputs = model(batch_tensor)
                features = outputs.last_hidden_state[:, 0, :]
                features = F.normalize(features, dim=-1)
            
            all_features.append(features.cpu().detach().float().numpy())
    
    progress_bar.empty()
    status_text.text("✅ エンコード完了!")
    
    return np.vstack(all_features)

@st.cache_data
def compute_pca(features: np.ndarray, n_components: int = 10) -> Tuple[np.ndarray, PCA]:
    """
    PCAを計算する（キャッシュ対応）
    """
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    pca = PCA(n_components=n_components)
    features_pca = pca.fit_transform(features_scaled)
    
    return features_pca, pca

def create_variance_plot(pca: PCA, max_components: int = 10) -> plt.Figure:
    """
    寄与率の可視化を作成
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    n_components = min(max_components, len(pca.explained_variance_ratio_))
    
    # 個別寄与率
    ax1.bar(range(1, n_components + 1), pca.explained_variance_ratio_[:n_components])
    ax1.set_xlabel('Principal Component')
    ax1.set_ylabel('Explained Variance Ratio')
    ax1.set_title('Individual Explained Variance Ratio')
    ax1.grid(True, alpha=0.3)
    
    # 累積寄与率
    cumsum_ratio = pca.explained_variance_ratio_[:n_components].cumsum()
    ax2.plot(range(1, n_components + 1), cumsum_ratio, 'bo-')
    ax2.axhline(y=0.8, color='r', linestyle='--', label='80%')
    ax2.axhline(y=0.9, color='g', linestyle='--', label='90%')
    ax2.axhline(y=0.95, color='orange', linestyle='--', label='95%')
    ax2.set_xlabel('Principal Component')
    ax2.set_ylabel('Cumulative Explained Variance Ratio')
    ax2.set_title('Cumulative Explained Variance Ratio')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def create_pca_visualization(features_pca: np.ndarray, images: List[Image.Image], 
                           image_paths: List[str], encoder_type: str, pca: PCA,
                           pc_x: int = 0, pc_y: int = 1, pc_z: int = None,
                           max_images_display: int = 25, image_size: int = 60, 
                           figsize: tuple = (16, 12), save_path: Optional[str] = None) -> plt.Figure:
    """
    選択された主成分でPCA可視化を作成する
    """
    # 表示する画像を選択
    if len(images) > max_images_display:
        display_indices = random.sample(range(len(images)), max_images_display)
        display_indices.sort()
    else:
        display_indices = list(range(len(images)))
    
    # matplotlib設定
    plt.rcParams['figure.dpi'] = 150
    plt.rcParams['savefig.dpi'] = 300
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['axes.unicode_minus'] = False
    
    # 図を作成
    if pc_z is None:  # 2D可視化
        fig, ax = plt.subplots(figsize=figsize)
        
        # 背景の散布図
        ax.scatter(features_pca[:, pc_x], features_pca[:, pc_y], 
                  alpha=0.3, s=15, c='lightgray', zorder=1, label='All Data Points')
        
        # 選択した点を強調
        selected_x = features_pca[display_indices, pc_x]
        selected_y = features_pca[display_indices, pc_y]
        ax.scatter(selected_x, selected_y, 
                  alpha=0.7, s=30, c='red', zorder=2, label='Image Display Points')
        
        # 画像配置
        for i, idx in enumerate(display_indices):
            try:
                img = images[idx].copy()
                img.thumbnail((image_size, image_size), Image.Resampling.LANCZOS)
                img_array = np.array(img)
                
                imagebox = OffsetImage(img_array, zoom=1.0)
                x_pos, y_pos = features_pca[idx, pc_x], features_pca[idx, pc_y]
                
                ab = AnnotationBbox(imagebox, (x_pos, y_pos),
                                  frameon=True, pad=0.2, zorder=3,
                                  boxcoords="data")
                ax.add_artist(ab)
                
            except Exception:
                ax.scatter(features_pca[idx, pc_x], features_pca[idx, pc_y], 
                         c='red', s=50, alpha=0.8, zorder=3, marker='x')
        
        ax.set_xlabel(f'PC{pc_x+1} (Variance Ratio: {pca.explained_variance_ratio_[pc_x]:.3f})', fontsize=14)
        ax.set_ylabel(f'PC{pc_y+1} (Variance Ratio: {pca.explained_variance_ratio_[pc_y]:.3f})', fontsize=14)
        
    else:  # 3D可視化
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')
        
        ax.scatter(features_pca[:, pc_x], features_pca[:, pc_y], features_pca[:, pc_z], 
                  alpha=0.6, s=50, c='lightblue')
        ax.scatter(features_pca[display_indices, pc_x], 
                  features_pca[display_indices, pc_y], 
                  features_pca[display_indices, pc_z],
                  alpha=0.9, s=100, c='red', label='Selected Points')
        
        ax.set_xlabel(f'PC{pc_x+1} (Variance Ratio: {pca.explained_variance_ratio_[pc_x]:.3f})')
        ax.set_ylabel(f'PC{pc_y+1} (Variance Ratio: {pca.explained_variance_ratio_[pc_y]:.3f})')
        ax.set_zlabel(f'PC{pc_z+1} (Variance Ratio: {pca.explained_variance_ratio_[pc_z]:.3f})')
    
    # タイトルと装飾
    if pc_z is None:
        title = f'{encoder_type.upper()} Encoder PCA Visualization: PC{pc_x+1} vs PC{pc_y+1}'
        title += f'\nImage Size: {image_size}px, Count: {len(display_indices)}'
    else:
        title = f'{encoder_type.upper()} Encoder PCA Visualization: PC{pc_x+1} vs PC{pc_y+1} vs PC{pc_z+1}'
    
    plt.title(title, fontsize=16, pad=20)
    ax.grid(True, alpha=0.3)
    if pc_z is None:
        ax.legend(loc='upper right', fontsize=10)
    
    plt.tight_layout()
    
    # 保存
    if save_path:
        save_dir = Path(save_path).parent
        save_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    
    return fig

def create_pca_matrix_plot(features_pca: np.ndarray, pca: PCA, max_components: int = 6) -> plt.Figure:
    """
    PCA散布図行列を作成
    """
    n_components = min(max_components, features_pca.shape[1])
    df = pd.DataFrame(features_pca[:, :n_components], 
                     columns=[f'PC{i+1}' for i in range(n_components)])
    
    # pairplotを作成
    g = sns.pairplot(df, diag_kind='hist', plot_kws={'alpha': 0.6, 's': 30})
    
    # 軸ラベルに寄与率を追加
    for i, ax in enumerate(g.axes[-1]):
        ax.set_xlabel(f'{ax.get_xlabel()} (Variance: {pca.explained_variance_ratio_[i]:.3f})')
    
    for i, ax in enumerate(g.axes):
        ax[0].set_ylabel(f'{ax[0].get_ylabel()} (Variance: {pca.explained_variance_ratio_[i]:.3f})')
    
    g.fig.suptitle('PCA Scatter Plot Matrix', fontsize=16, y=1.02)
    
    return g.fig

def create_interactive_pca_visualization(features_pca: np.ndarray, images: List[Image.Image], 
                                       image_paths: List[str], encoder_type: str, pca: PCA,
                                       pc_x: int = 0, pc_y: int = 1, pc_z: int = None,
                                       max_images_display: int = 50) -> go.Figure:
    """
    Plotlyを使ったインタラクティブPCA可視化を作成
    """
    # 表示する画像を選択
    if len(images) > max_images_display:
        display_indices = random.sample(range(len(images)), max_images_display)
        display_indices.sort()
    else:
        display_indices = list(range(len(images)))
    
    # 画像をbase64エンコードしてホバー用データを準備
    hover_data = []
    image_names = []
    
    for idx in display_indices:
        try:
            img = images[idx].copy()
            img.thumbnail((150, 150), Image.Resampling.LANCZOS)
            
            # base64エンコード
            buffer = BytesIO()
            img.save(buffer, format='PNG')
            img_str = base64.b64encode(buffer.getvalue()).decode()
            
            hover_data.append(f"<img src='data:image/png;base64,{img_str}' width='120'>")
            image_names.append(f"Image {idx}")
            
        except Exception as e:
            hover_data.append(f"画像読み込みエラー: {e}")
            image_names.append(f"Error {idx}")
    
    if pc_z is None:
        # 2D可視化
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=features_pca[display_indices, pc_x],
            y=features_pca[display_indices, pc_y],
            mode='markers',
            marker=dict(
                size=8,
                color='lightblue',
                line=dict(width=1, color='navy'),
                opacity=0.7
            ),
            text=[f"{image_names[i]}<br>PC{pc_x+1}: {features_pca[display_indices[i], pc_x]:.3f}<br>PC{pc_y+1}: {features_pca[display_indices[i], pc_y]:.3f}" 
                  for i in range(len(display_indices))],
            hovertemplate='%{text}<br>%{customdata}<extra></extra>',
            customdata=hover_data,
            name='Images'
        ))
        
        fig.update_layout(
            title=f'{encoder_type.upper()} エンコーダーのインタラクティブPCA可視化 (2D)',
            xaxis_title=f'PC{pc_x+1} (寄与率: {pca.explained_variance_ratio_[pc_x]:.3f})',
            yaxis_title=f'PC{pc_y+1} (寄与率: {pca.explained_variance_ratio_[pc_y]:.3f})',
            width=800,
            height=600,
            hovermode='closest'
        )
        
    else:
        # 3D可視化
        fig = go.Figure()
        
        fig.add_trace(go.Scatter3d(
            x=features_pca[display_indices, pc_x],
            y=features_pca[display_indices, pc_y],
            z=features_pca[display_indices, pc_z],
            mode='markers',
            marker=dict(
                size=6,
                color='lightblue',
                line=dict(width=1, color='navy'),
                opacity=0.7
            ),
            text=[f"{image_names[i]}<br>PC{pc_x+1}: {features_pca[display_indices[i], pc_x]:.3f}<br>PC{pc_y+1}: {features_pca[display_indices[i], pc_y]:.3f}<br>PC{pc_z+1}: {features_pca[display_indices[i], pc_z]:.3f}" 
                  for i in range(len(display_indices))],
            hovertemplate='%{text}<br>%{customdata}<extra></extra>',
            customdata=hover_data,
            name='Images'
        ))
        
        fig.update_layout(
            title=f'{encoder_type.upper()} エンコーダーのインタラクティブPCA可視化 (3D)',
            scene=dict(
                xaxis_title=f'PC{pc_x+1} (寄与率: {pca.explained_variance_ratio_[pc_x]:.3f})',
                yaxis_title=f'PC{pc_y+1} (寄与率: {pca.explained_variance_ratio_[pc_y]:.3f})',
                zaxis_title=f'PC{pc_z+1} (寄与率: {pca.explained_variance_ratio_[pc_z]:.3f})',
            ),
            width=800,
            height=600
        )
    
    return fig

def create_multi_dimension_comparison(features_pca: np.ndarray, pca: PCA, 
                                    max_components: int = 6) -> go.Figure:
    """
    複数次元の比較可視化を作成
    """
    n_components = min(max_components, features_pca.shape[1])
    
    # サブプロット作成
    dimension_pairs = [(i, j) for i in range(n_components) for j in range(i+1, n_components)][:6]
    
    fig = make_subplots(
        rows=2, cols=3,
        subplot_titles=[f'PC{pair[0]+1} vs PC{pair[1]+1}' for pair in dimension_pairs],
        specs=[[{'type': 'scatter'} for _ in range(3)] for _ in range(2)]
    )
    
    for idx, (pc_x, pc_y) in enumerate(dimension_pairs):
        row = idx // 3 + 1
        col = idx % 3 + 1
        
        fig.add_trace(
            go.Scatter(
                x=features_pca[:, pc_x],
                y=features_pca[:, pc_y],
                mode='markers',
                marker=dict(size=4, opacity=0.6),
                name=f'PC{pc_x+1} vs PC{pc_y+1}',
                showlegend=False
            ),
            row=row, col=col
        )
        
        fig.update_xaxes(title_text=f'PC{pc_x+1} ({pca.explained_variance_ratio_[pc_x]:.3f})', row=row, col=col)
        fig.update_yaxes(title_text=f'PC{pc_y+1} ({pca.explained_variance_ratio_[pc_y]:.3f})', row=row, col=col)
    
    fig.update_layout(
        title_text="複数次元PCA比較",
        height=600,
        width=900
    )
    
    return fig

def get_default_checkpoint_paths():
    """
    各エンコーダーのデフォルトチェックポイントパスを取得
    """
    return {
        'clip': "openai/clip-vit-large-patch14",  # Hugging Face ID
        'emotion_clip': "/home/ryuichi/animins/slider_space_v2/EmotionCLIP/emotionclip_latest.pt",
        'dinov2-small': "facebook/dinov2-small",  # Hugging Face ID
        'openvision': "/home/ryuichi/animins/slider_space_v2/models/openvision_checkpoint.pt"
    }

def find_checkpoint_candidates(encoder_type: str):
    """
    指定されたエンコーダーの候補チェックポイントを検索
    """
    candidates = []
    
    if encoder_type == 'emotion_clip':
        search_dirs = [
            "/home/ryuichi/animins/slider_space_v2/EmotionCLIP/",
            "/home/ryuichi/animins/EmotionCLIP/",
            "/home/ryuichi/EmotionCLIP/",
            str(Path.home() / "EmotionCLIP"),
            "./EmotionCLIP/",
            "../EmotionCLIP/"
        ]
        file_patterns = ["*.pt", "*.pth", "*.ckpt"]
        
    elif encoder_type == 'openvision':
        search_dirs = [
            "/home/ryuichi/animins/slider_space_v2/models/",
            "/home/ryuichi/animins/models/",
            str(Path.home() / "models"),
            "./models/",
            "../models/"
        ]
        file_patterns = ["*.pt", "*.pth", "*.ckpt"]
        
    elif encoder_type in ['clip', 'dinov2-small']:
        # Hugging Faceモデルの場合はローカルキャッシュも検索
        cache_dirs = [
            str(Path.home() / ".cache/huggingface/transformers/"),
            str(Path.home() / ".cache/huggingface/hub/"),
            "/tmp/huggingface_cache/"
        ]
        search_dirs = cache_dirs
        file_patterns = ["*.bin", "*.safetensors", "*.pt"]
        
    else:
        return []
    
    # ファイル検索
    for search_dir in search_dirs:
        if os.path.exists(search_dir):
            for pattern in file_patterns:
                for file_path in Path(search_dir).glob(pattern):
                    if file_path.is_file():
                        candidates.append(str(file_path))
    
    # 重複除去とソート
    return sorted(list(set(candidates)))

def create_checkpoint_selector(encoder_type: str, key_prefix: str = ""):
    """
    チェックポイント選択UIを作成
    """
    default_paths = get_default_checkpoint_paths()
    default_path = default_paths.get(encoder_type, "")
    
    st.sidebar.subheader(f"📁 {encoder_type.upper()} チェックポイント設定")
    
    # 選択方法
    selection_methods = ["デフォルト使用", "カスタムパス指定", "ファイルブラウザー"]
    
    # Hugging Faceモデルの場合は説明を追加
    if encoder_type in ['clip', 'dinov2-small']:
        selection_methods[0] = "デフォルト使用 (Hugging Face)"
    
    checkpoint_method = st.sidebar.radio(
        f"{encoder_type.upper()} 選択方法",
        selection_methods,
        index=0,
        key=f"{key_prefix}checkpoint_method_{encoder_type}"
    )
    
    if checkpoint_method.startswith("デフォルト使用"):
        checkpoint_path = default_path
        if encoder_type in ['clip', 'dinov2-small']:
            st.sidebar.info(f"🤗 Hugging Face ID: `{checkpoint_path}`")
        else:
            st.sidebar.info(f"📁 使用パス: `{checkpoint_path}`")
            
    elif checkpoint_method == "カスタムパス指定":
        help_text = "チェックポイントファイルのフルパスまたはHugging Face IDを入力"
        if encoder_type in ['clip', 'dinov2-small']:
            help_text += "\n例: openai/clip-vit-base-patch32 または /path/to/model"
        
        checkpoint_path = st.sidebar.text_input(
            f"{encoder_type.upper()} チェックポイントパス",
            value=default_path,
            help=help_text,
            key=f"{key_prefix}custom_path_{encoder_type}"
        )
        
    else:  # ファイルブラウザー
        candidates = find_checkpoint_candidates(encoder_type)
        
        if candidates:
            # 候補リストを作成（デフォルトとカスタムも含む）
            all_options = [
                f"デフォルト: {default_path}",
                "カスタムパス指定..."
            ] + [f"ローカル: {Path(p).name} ({p})" for p in candidates[:10]]  # 最大10個まで表示
            
            if len(candidates) > 10:
                all_options.append(f"... 他 {len(candidates) - 10} 個のファイル")
            
            selected_option = st.sidebar.selectbox(
                f"{encoder_type.upper()} ファイル選択",
                all_options,
                index=0,
                help="利用可能なチェックポイントファイルから選択",
                key=f"{key_prefix}file_browser_{encoder_type}"
            )
            
            if selected_option.startswith("デフォルト:"):
                checkpoint_path = default_path
            elif selected_option == "カスタムパス指定...":
                checkpoint_path = st.sidebar.text_input(
                    "カスタムパス",
                    value=default_path,
                    key=f"{key_prefix}custom_fallback_{encoder_type}"
                )
            elif selected_option.startswith("ローカル:"):
                # "ローカル: filename (full_path)" から full_path を抽出
                checkpoint_path = selected_option.split("(")[-1].rstrip(")")
            else:
                checkpoint_path = default_path
        else:
            st.sidebar.warning(f"⚠️ {encoder_type.upper()} のローカルファイルが見つかりません")
            checkpoint_path = st.sidebar.text_input(
                f"{encoder_type.upper()} パス（手動入力）",
                value=default_path,
                key=f"{key_prefix}manual_input_{encoder_type}"
            )
    
    # ファイル/ID の検証と情報表示
    if checkpoint_path:
        if checkpoint_path.startswith(("openai/", "facebook/", "microsoft/", "google/")):
            # Hugging Face ID の場合
            st.sidebar.success(f"🤗 Hugging Face モデル ID: `{checkpoint_path}`")
        elif os.path.exists(checkpoint_path):
            # ローカルファイルの場合
            file_size = os.path.getsize(checkpoint_path) / (1024 * 1024)  # MB
            st.sidebar.success(f"✅ ローカルファイル確認済み ({file_size:.1f} MB)")
            
            # ファイル詳細情報
            with st.sidebar.expander(f"📊 {encoder_type.upper()} ファイル詳細"):
                file_stat = os.stat(checkpoint_path)
                st.write(f"**パス**: `{checkpoint_path}`")
                st.write(f"**ファイル名**: `{Path(checkpoint_path).name}`")
                st.write(f"**サイズ**: {file_size:.2f} MB")
                st.write(f"**更新日時**: {time.ctime(file_stat.st_mtime)}")
        else:
            # 存在しないパスの場合
            if "/" in checkpoint_path and not checkpoint_path.startswith(("openai/", "facebook/")):
                st.sidebar.error(f"❌ ファイルが見つかりません: `{checkpoint_path}`")
            else:
                st.sidebar.warning(f"⚠️ Hugging Face IDまたはパスを確認: `{checkpoint_path}`")
    
    return checkpoint_path

def main():
    # タイトル
    st.title("🎨 拡張版ハイブリッドPCA可視化システム")
    st.markdown("### 主成分選択機能付きインタラクティブ可視化ツール")
    
    # 利用可能な機能の表示
    with st.expander("🔧 利用可能な機能とチェックポイント情報", expanded=False):
        col1, col2, col3 = st.columns(3)
        with col1:
            st.write("**✅ 基本エンコーダー**")
            
            # CLIP情報
            clip_checkpoint = getattr(st.session_state, 'clip_checkpoint', 'openai/clip-vit-large-patch14')
            st.write("• CLIP")
            if clip_checkpoint.startswith('openai/'):
                st.write(f"  └ 🤗 {Path(clip_checkpoint).name}")
            else:
                st.write(f"  └ 📁 {Path(clip_checkpoint).name}")
            
            # DINOv2情報
            dinov2_checkpoint = getattr(st.session_state, 'dinov2-small_checkpoint', 'facebook/dinov2-small')
            st.write("• DINOv2")
            if dinov2_checkpoint.startswith('facebook/'):
                st.write(f"  └ 🤗 {Path(dinov2_checkpoint).name}")
            else:
                st.write(f"  └ 📁 {Path(dinov2_checkpoint).name}")
        
        with col2:
            st.write("**🔄 オプション機能**")
            
            if EMOTION_CLIP_AVAILABLE:
                st.write("✅ EmotionCLIP")
                emotion_checkpoint = getattr(st.session_state, 'emotion_clip_checkpoint', '')
                if emotion_checkpoint:
                    st.write(f"  └ 📁 {Path(emotion_checkpoint).name}")
            else:
                st.write("❌ EmotionCLIP")
            
            if OPENVISION_AVAILABLE:
                st.write("✅ OpenVision")
                openvision_checkpoint = getattr(st.session_state, 'openvision_checkpoint', '')
                if openvision_checkpoint:
                    if openvision_checkpoint.startswith(('microsoft/', 'google/')):
                        st.write(f"  └ 🤗 {Path(openvision_checkpoint).name}")
                    else:
                        st.write(f"  └ 📁 {Path(openvision_checkpoint).name}")
            else:
                st.write("❌ OpenVision")
        
        with col3:
            st.write("**🎨 可視化機能**")
            st.write("✅ matplotlib可視化")
            st.write("✅ Plotly インタラクティブ")
            st.write("✅ 複数次元比較")
            st.write("✅ チェックポイント選択")

    # サイドバー - パラメータ設定
    st.sidebar.header("📋 パラメータ設定")
    
    # データ設定
    st.sidebar.subheader("📂 データ設定")
    data_path = st.sidebar.text_input(
        "画像データパス", 
        value="/home/ryuichi/animins/slider_space_v2/training_images/sdxl/concept_15109"
    )
    max_images = st.sidebar.slider("最大画像数", 5, 200, 50, 5)
    
    # エンコーダー設定
    st.sidebar.subheader("🤖 エンコーダー設定")
    available_encoders = ['clip', 'dinov2-small']
    encoder_descriptions = {
        'clip': 'CLIP (高品質・汎用)',
        'dinov2-small': 'DINOv2 (視覚特徴特化)'
    }
    
    if EMOTION_CLIP_AVAILABLE:
        available_encoders.insert(1, 'emotion_clip')
        encoder_descriptions['emotion_clip'] = 'EmotionCLIP (感情理解)'
    
    if OPENVISION_AVAILABLE:
        available_encoders.append('openvision')
        encoder_descriptions['openvision'] = 'OpenVision (カスタム)'
    
    encoder_type = st.sidebar.selectbox(
        "エンコーダー", 
        available_encoders, 
        format_func=lambda x: encoder_descriptions.get(x, x),
        index=1 if EMOTION_CLIP_AVAILABLE else 0
    )
    
    # 選択されたエンコーダーのチェックポイント設定
    checkpoint_path = create_checkpoint_selector(encoder_type, "main_")
    
    # セッション状態に保存
    st.session_state[f'{encoder_type}_checkpoint'] = checkpoint_path
    
    batch_size = st.sidebar.slider("バッチサイズ", 1, 16, 8, 1)
    
    # PCA設定
    st.sidebar.subheader("📊 PCA設定")
    max_pca_components = st.sidebar.slider("最大PCA成分数", 3, 20, 10, 1)
    
    # 可視化設定
    st.sidebar.subheader("🎨 可視化設定")
    max_images_display = st.sidebar.slider("表示画像数", 5, 50, 25, 1)
    image_size = st.sidebar.slider("画像サイズ (px)", 30, 120, 60, 10)
    figsize_width = st.sidebar.slider("図の幅", 10, 20, 16, 1)
    figsize_height = st.sidebar.slider("図の高さ", 8, 16, 12, 1)
    
    # 保存設定
    st.sidebar.subheader("💾 保存設定")
    save_results = st.sidebar.checkbox("結果を保存", value=True)
    save_path = st.sidebar.text_input(
        "保存パス", 
        value="/home/ryuichi/animins/slider_space_v2/pca_visualization/enhanced_ui_result.png"
    ) if save_results else None
    
    # 実行ボタン
    if st.sidebar.button("🚀 可視化実行", type="primary"):
        try:
            # データ読み込み
            with st.spinner("📂 画像データ読み込み中..."):
                images, image_paths = load_image_dataset(data_path, max_images)
            
            if len(images) == 0:
                st.error("❌ 画像が見つかりません")
                return
            
            st.success(f"✅ {len(images)}枚の画像を読み込みました")
            
            # 画像プレビュー
            st.subheader("📸 読み込み画像プレビュー")
            preview_cols = st.columns(6)
            for i, img in enumerate(images[:6]):
                with preview_cols[i % 6]:
                    st.image(img, caption=f"Image {i+1}", use_column_width=True)
            
            # エンコード
            with st.spinner(f"🔄 {encoder_type.upper()}エンコード中..."):
                features = encode_images(images, encoder_type, batch_size)
            
            st.success(f"✅ エンコード完了: 特徴量の形状 {features.shape}")
            
            # PCA計算
            with st.spinner("📊 PCA計算中..."):
                features_pca, pca = compute_pca(features, max_pca_components)
            
            # セッション状態に保存
            st.session_state.features_pca = features_pca
            st.session_state.pca = pca
            st.session_state.images = images
            st.session_state.image_paths = image_paths
            st.session_state.encoder_type = encoder_type
            st.session_state.max_images_display = max_images_display
            st.session_state.image_size = image_size
            st.session_state.figsize = (figsize_width, figsize_height)
            st.session_state.save_path = save_path
            
            st.success(f"✅ PCA計算完了: {features_pca.shape[1]}個の主成分を取得")
            
        except Exception as e:
            st.error(f"❌ エラーが発生しました: {str(e)}")
            st.exception(e)
    
    # PCAが計算されている場合の追加UI
    if hasattr(st.session_state, 'features_pca') and st.session_state.features_pca is not None:
        st.markdown("---")
        st.header("📊 PCA結果分析")
        
        # タブで機能を分割
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
            "🎯 主成分選択可視化", 
            "🚀 インタラクティブ可視化", 
            "📊 複数次元比較",
            "📈 寄与率分析", 
            "🔍 散布図行列", 
            "📋 統計情報"
        ])
        
        with tab1:
            st.subheader("🎯 主成分選択可視化")
            st.markdown("**任意の主成分の組み合わせで可視化できます**")
            
            # 主成分選択
            col1, col2, col3 = st.columns(3)
            
            n_components = st.session_state.features_pca.shape[1]
            component_options = list(range(1, n_components + 1))
            
            with col1:
                pc_x = st.selectbox("X軸 (PC)", component_options, index=0, key="pc_x") - 1
            
            with col2:
                pc_y = st.selectbox("Y軸 (PC)", component_options, index=1, key="pc_y") - 1
            
            with col3:
                use_3d = st.checkbox("3D表示", key="use_3d")
                if use_3d:
                    pc_z = st.selectbox("Z軸 (PC)", component_options, index=2, key="pc_z") - 1
                else:
                    pc_z = None
            
            # 可視化実行
            if st.button("🔄 可視化更新", key="update_viz"):
                with st.spinner("🎨 可視化作成中..."):
                    fig = create_pca_visualization(
                        st.session_state.features_pca, st.session_state.images, 
                        st.session_state.image_paths, st.session_state.encoder_type,
                        st.session_state.pca, pc_x, pc_y, pc_z,
                        st.session_state.max_images_display, st.session_state.image_size,
                        st.session_state.figsize, st.session_state.save_path
                    )
                
                st.pyplot(fig)
                
                # 選択された主成分の情報表示
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric(f"PC{pc_x+1} 寄与率", f"{st.session_state.pca.explained_variance_ratio_[pc_x]:.4f}")
                with col2:
                    st.metric(f"PC{pc_y+1} 寄与率", f"{st.session_state.pca.explained_variance_ratio_[pc_y]:.4f}")
                if pc_z is not None:
                    with col3:
                        st.metric(f"PC{pc_z+1} 寄与率", f"{st.session_state.pca.explained_variance_ratio_[pc_z]:.4f}")
                
                # ダウンロードボタン
                buf = io.BytesIO()
                fig.savefig(buf, format='png', dpi=300, bbox_inches='tight')
                buf.seek(0)
                
                filename_suffix = f"PC{pc_x+1}_vs_PC{pc_y+1}"
                if pc_z is not None:
                    filename_suffix += f"_vs_PC{pc_z+1}"
                
                st.download_button(
                    label="📥 PNG画像をダウンロード",
                    data=buf.getvalue(),
                    file_name=f"pca_visualization_{st.session_state.encoder_type}_{filename_suffix}_{int(time.time())}.png",
                    mime="image/png"
                )
                
                plt.close(fig)
        
        with tab2:
            st.subheader("🚀 インタラクティブPCA可視化")
            st.markdown("**Plotlyを使った高度なインタラクティブ可視化（ホバーで画像表示）**")
            
            # 主成分選択
            col1, col2, col3 = st.columns(3)
            
            n_components = st.session_state.features_pca.shape[1]
            component_options = list(range(1, n_components + 1))
            
            with col1:
                interactive_pc_x = st.selectbox("X軸 (PC)", component_options, index=0, key="interactive_pc_x") - 1
            
            with col2:
                interactive_pc_y = st.selectbox("Y軸 (PC)", component_options, index=1, key="interactive_pc_y") - 1
            
            with col3:
                interactive_use_3d = st.checkbox("3D表示", key="interactive_use_3d")
                if interactive_use_3d:
                    interactive_pc_z = st.selectbox("Z軸 (PC)", component_options, index=2, key="interactive_pc_z") - 1
                else:
                    interactive_pc_z = None
            
            # 表示設定
            interactive_max_display = st.slider(
                "表示画像数（インタラクティブ）", 
                min_value=10, 
                max_value=min(100, len(st.session_state.images)), 
                value=min(30, len(st.session_state.images)),
                key="interactive_max_display"
            )
            
            if st.button("🔄 インタラクティブ可視化生成", key="generate_interactive"):
                with st.spinner("🎨 インタラクティブ可視化生成中..."):
                    interactive_fig = create_interactive_pca_visualization(
                        st.session_state.features_pca, 
                        st.session_state.images, 
                        st.session_state.image_paths, 
                        st.session_state.encoder_type,
                        st.session_state.pca, 
                        interactive_pc_x, 
                        interactive_pc_y, 
                        interactive_pc_z,
                        interactive_max_display
                    )
                
                st.plotly_chart(interactive_fig, use_container_width=True)
                
                # HTMLダウンロード
                interactive_html = interactive_fig.to_html()
                st.download_button(
                    label="📥 インタラクティブHTMLをダウンロード",
                    data=interactive_html,
                    file_name=f"interactive_pca_{st.session_state.encoder_type}_{int(time.time())}.html",
                    mime="text/html"
                )
        
        with tab3:
            st.subheader("📊 複数次元比較")
            st.markdown("**複数の主成分ペアを同時に比較表示**")
            
            comparison_components = st.slider(
                "比較する主成分数", 
                min_value=3, 
                max_value=min(8, st.session_state.features_pca.shape[1]), 
                value=min(6, st.session_state.features_pca.shape[1]),
                key="comparison_components"
            )
            
            if st.button("📊 複数次元比較生成", key="generate_comparison"):
                with st.spinner("📊 複数次元比較生成中..."):
                    comparison_fig = create_multi_dimension_comparison(
                        st.session_state.features_pca, 
                        st.session_state.pca, 
                        comparison_components
                    )
                
                st.plotly_chart(comparison_fig, use_container_width=True)
                
                # HTMLダウンロード
                comparison_html = comparison_fig.to_html()
                st.download_button(
                    label="📥 比較図HTMLをダウンロード",
                    data=comparison_html,
                    file_name=f"multi_comparison_pca_{st.session_state.encoder_type}_{int(time.time())}.html",
                    mime="text/html"
                )
        
        with tab4:
            st.subheader("📈 寄与率分析")
            
            # 寄与率可視化
            variance_fig = create_variance_plot(st.session_state.pca, max_pca_components)
            st.pyplot(variance_fig)
            
            # 寄与率テーブル
            st.subheader("📋 主成分別寄与率")
            variance_data = []
            cumsum = 0
            for i, ratio in enumerate(st.session_state.pca.explained_variance_ratio_):
                cumsum += ratio
                variance_data.append({
                    "主成分": f"PC{i+1}",
                    "寄与率": f"{ratio:.4f}",
                    "寄与率(%)": f"{ratio*100:.2f}%",
                    "累積寄与率": f"{cumsum:.4f}",
                    "累積寄与率(%)": f"{cumsum*100:.2f}%"
                })
            
            st.dataframe(pd.DataFrame(variance_data), use_container_width=True)
            
            plt.close(variance_fig)
        
        with tab5:
            st.subheader("🔍 PCA散布図行列")
            
            matrix_components = st.slider("表示する主成分数", 3, min(8, n_components), min(6, n_components))
            
            if st.button("📊 散布図行列作成", key="create_matrix"):
                with st.spinner("🔍 散布図行列作成中..."):
                    matrix_fig = create_pca_matrix_plot(st.session_state.features_pca, 
                                                      st.session_state.pca, matrix_components)
                
                st.pyplot(matrix_fig)
                plt.close(matrix_fig)
        
        with tab6:
            st.subheader("📋 統計情報")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("エンコーダー", st.session_state.encoder_type.upper())
                st.metric("読み込み画像数", len(st.session_state.images))
                st.metric("特徴量次元", st.session_state.features_pca.shape[0])
                st.metric("PCA成分数", st.session_state.features_pca.shape[1])
            
            with col2:
                st.metric("PC1-PC2 累積寄与率", f"{st.session_state.pca.explained_variance_ratio_[:2].sum():.4f}")
                st.metric("PC1-PC3 累積寄与率", f"{st.session_state.pca.explained_variance_ratio_[:3].sum():.4f}")
                st.metric("全体累積寄与率", f"{st.session_state.pca.explained_variance_ratio_.sum():.4f}")
                st.metric("表示画像数", st.session_state.max_images_display)
    
    # 使用方法
    with st.expander("📖 使用方法"):
        st.markdown("""
        ### 🚀 拡張版ハイブリッドPCA可視化システムの使い方
        
        #### 1. 基本設定
        - **📂 データ設定**: 画像フォルダのパスと最大画像数を設定
        - **🤖 エンコーダー選択**: CLIP、EmotionCLIP、DINOv2、OpenVisionから選択
        - **📁 チェックポイント設定**: 各エンコーダーで3つの方法から選択
          - **デフォルト使用**: 推奨パス/Hugging Face IDを自動使用
          - **カスタムパス指定**: 任意のパスまたはHugging Face IDを手動入力
          - **ファイルブラウザー**: システムが自動検索したファイルから選択
        - **📊 PCA設定**: 最大PCA成分数を設定（3-20）
        - **🎨 可視化パラメータ**: 画像サイズ、表示数、図のサイズを調整
        - **🚀 実行**: 「可視化実行」ボタンをクリック
        
        #### 2. チェックポイント選択の詳細 📁
        
        **CLIP & DINOv2**:
        - デフォルト: Hugging Face公式モデル（自動ダウンロード）
        - カスタム: 独自の学習済みモデルまたは他のHugging Face ID
        - ブラウザー: ローカルキャッシュから選択
        
        **EmotionCLIP**:
        - デフォルト: プロジェクト内の標準チェックポイント
        - カスタム: 任意の.ptファイルパス
        - ブラウザー: 複数ディレクトリから.ptファイルを自動検索
        
        **OpenVision**:
        - デフォルト: プロジェクト内の標準チェックポイント
        - カスタム: 任意のモデルファイルパス
        - ブラウザー: modelsディレクトリから自動検索
        
        #### 3. 主成分選択可視化 🎯
        - **X軸・Y軸**: 任意の主成分を選択（PC1, PC2, PC3...）
        - **3D表示**: チェックでZ軸も選択可能
        - **リアルタイム更新**: パラメータ変更後「可視化更新」で即座に反映
        
        #### 4. 分析機能
        - **📈 寄与率分析**: 各主成分の重要度を確認
        - **🔍 散布図行列**: 複数主成分の関係を一度に表示
        - **📋 統計情報**: 詳細な数値データを確認
        
        ### ✨ 新機能
        - **🎯 任意主成分選択**: PC1 vs PC3、PC2 vs PC4など自由な組み合わせ
        - **📁 全エンコーダーチェックポイント選択**: 全てのエンコーダーでカスタマイズ可能
        - **🔍 自動ファイル検索**: ローカルファイルの自動検出とリスト表示
        - **📊 詳細分析**: 寄与率テーブル、散布図行列
        - **🔄 リアルタイム更新**: パラメータ変更で即座に再生成
        - **💾 高度な保存**: 主成分情報付きファイル名で自動保存
        
        ### 🛡️ 安全機能
        - **ファイル存在確認**: チェックポイントファイルの自動検証
        - **詳細情報表示**: ファイルサイズ、更新日時などの詳細情報
        - **エラーハンドリング**: わかりやすいエラーメッセージと解決策提示
        - **候補ファイル提案**: エラー時に利用可能な代替ファイルを自動提案
        """)
    
    # フッター
    st.markdown("---")
    st.markdown("🎨 **拡張版ハイブリッドPCA可視化システム** - 主成分選択機能付き")

if __name__ == "__main__":
    main()