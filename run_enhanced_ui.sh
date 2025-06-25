#!/bin/bash

# 拡張版ハイブリッドPCA可視化UI起動スクリプト
# 主成分選択機能付きの拡張版UIを起動します

echo "🎨 拡張版ハイブリッドPCA可視化システム起動中..."
echo "主成分選択機能付きインタラクティブ可視化ツール"
echo ""

# ディレクトリ移動
cd /home/ryuichi/animins/slider_space_v2

# 必要なディレクトリ作成
mkdir -p pca_visualization

# 既存のStreamlitプロセスを確認・停止
echo "🔍 既存のStreamlitプロセスを確認中..."
existing_processes=$(ps aux | grep "streamlit.*enhanced_pca_ui.py" | grep -v grep | awk '{print $2}')

if [ ! -z "$existing_processes" ]; then
    echo "🛑 既存のStreamlitプロセスを停止中..."
    echo "$existing_processes" | xargs kill -9
    sleep 2
    echo "✅ 既存プロセスを停止しました"
fi

echo ""
echo "🚀 拡張版ハイブリッドPCA可視化UIを起動しています..."
echo "📌 URL: http://localhost:8502"
echo ""
echo "✨ 新機能:"
echo "   🎯 任意の主成分選択（PC1 vs PC3、PC2 vs PC4など）"
echo "   📊 詳細な寄与率分析"
echo "   🔄 リアルタイム可視化更新"
echo "   💾 高度な保存機能"
echo ""
echo "❌ 停止するには Ctrl+C を押してください"
echo ""

# Streamlit実行（ポート8502を使用）
export STREAMLIT_SERVER_PORT=8502
streamlit run enhanced_pca_ui.py \
    --server.port=8502 \
    --server.address=0.0.0.0 \
    --server.headless=true \
    --server.runOnSave=true \
    --theme.base=light 