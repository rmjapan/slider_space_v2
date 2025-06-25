#!/bin/bash

# ハイブリッドPCA可視化UIの起動スクリプト

echo "🚀 ハイブリッドPCA可視化システム起動中..."

# 必要なディレクトリを作成
mkdir -p /home/ryuichi/animins/slider_space_v2/pca_visualization

# StreamlitでUIを起動
cd /home/ryuichi/animins/slider_space_v2

echo "📋 起動情報:"
echo "  - ポート: 8501"
echo "  - ホスト: localhost"
echo "  - URL: http://localhost:8501"
echo ""
echo "🌐 ブラウザで上記URLにアクセスしてください"
echo "⚠️  終了するには Ctrl+C を押してください"
echo ""

# Streamlitアプリを起動
streamlit run pca_visualization_ui.py \
    --server.port 8501 \
    --server.address localhost \
    --server.headless true \
    --browser.serverAddress localhost \
    --browser.serverPort 8501 