# PCA Visualization Tool

統合されたPCA可視化ツールです。

## 📁 ファイル構成

### 🎨 統合版Streamlit UI
- **`pca_visualization_ui_enhanced.py`** - 全機能統合版Streamlit UI
  - 🎯 主成分選択機能（任意のPC組み合わせ）
  - 🚀 インタラクティブ可視化（Plotlyホバー画像表示）
  - 📊 複数次元比較（6軸同時表示）
  - 📈 寄与率分析（詳細テーブル付き）
  - 🔍 PCA散布図行列
  - 複数エンコーダー対応 (CLIP, EmotionCLIP, DinoV2)
  - matplotlib + plotly 両対応
  - 高解像度出力・HTMLエクスポート

### 🚀 ランチャー
- **`run_streamlit_ui.py`** - Streamlit起動用スクリプト
  - PyTorch + Streamlit 互換性対応
  - 自動環境設定

## 🎯 使用方法

### 起動方法
```bash
python run_streamlit_ui.py
```
または
```bash
streamlit run pca_visualization_ui_enhanced.py
```

### 主要機能

#### 🎯 主成分選択可視化
- 任意のPC組み合わせ選択（PC1 vs PC3、PC2 vs PC4など）
- 2D/3D可視化対応
- リアルタイム更新
- 散布図上に直接画像表示

#### 🚀 インタラクティブ可視化（NEW）
- Plotlyを使った高度なインタラクティブ可視化
- ホバーで画像表示
- ズーム・パン機能
- HTMLエクスポート

#### 📊 複数次元比較（NEW）
- 最大6つの主成分ペアを同時表示
- 複数次元の関係性を一目で把握
- カスタマイズ可能な軸数

#### 📈 寄与率分析
- 個別・累積寄与率グラフ
- 詳細な寄与率テーブル
- 主成分重要度の数値分析

#### 🔍 PCA散布図行列
- 複数主成分の相関関係表示
- カスタマイズ可能な軸数
- 高解像度出力対応

## 🔧 対応エンコーダー
- **CLIP**: `openai/clip-vit-large-patch14`
- **EmotionCLIP**: カスタムトレーニングモデル
- **DinoV2**: `facebook/dinov2-small`

## 💾 出力形式
- **PNG**: 高解像度静的画像
- **HTML**: インタラクティブファイル
- **データ**: 寄与率テーブル（CSV風）

## 🗑️ 整理履歴
以下のファイルは機能統合のため削除されました：
- ~~`enhanced_pca_ui.py`~~
- ~~`pca_visualization_enhanced.py`~~
- ~~`pca_visualization_ui.py`~~
- ~~`visualize_pca_unified.py`~~
- ~~`streamlit_pca_ui.py`~~
- ~~`visualize_clip_pca.py`~~ (コマンドライン版)
- ~~`visualize_clip_pca_unified.py`~~ (コマンドライン版)

**現在: 2ファイルに統合完了** 🎯 