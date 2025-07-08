from flask import Flask, render_template, request, jsonify, redirect, url_for, flash
import os
import json
from PIL import Image
import base64
from io import BytesIO

app = Flask(__name__)
app.secret_key = 'your-secret-key-here'

# グローバル変数でセッション情報を管理
current_folder = None
output_folder = None
images = []
current_index = 0
captions = {}

# 表情の選択肢（日本語→英語）
EXPRESSIONS = {
    '笑顔': 'smile',
    '幸せ': 'happy', 
    '悲しい': 'sad',
    '怒り': 'angry',
    '驚き': 'surprised',
    '困惑': 'confused',
    '恥ずかしい': 'embarrassed',
    '赤面': 'blush',
    '泣いている': 'crying',
    '笑っている': 'laughing',
    '真剣': 'serious',
    '眠そう': 'sleepy',
    '心配': 'worried',
    '興奮': 'excited',
    '恥ずかしがり': 'shy',
    '自信': 'confident',
    '優しい': 'gentle',
    '落ち着いた': 'calm',
    '緊張': 'nervous',
    '陽気': 'cheerful'
}

# 強度の選択肢
INTENSITIES = {
    '弱い': 0.7,
    '普通': 1.0,
    'かなり': 1.3,
    'とても': 1.5
}

def load_existing_captions(output_folder):
    """既存のmetadata.jsonlファイルからキャプションを読み込む"""
    metadata_path = os.path.join(output_folder, "metadata.jsonl")
    existing_captions = {}
    
    if os.path.exists(metadata_path):
        try:
            with open(metadata_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        data = json.loads(line)
                        existing_captions[data['file_name']] = data['text']
        except Exception as e:
            print(f"既存キャプションの読み込みエラー: {e}")
    
    return existing_captions

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/select_folder', methods=['POST'])
def select_folder():
    global current_folder, output_folder, images, current_index, captions
    
    folder_path = request.form.get('folder_path')
    output_path = request.form.get('output_path')
    
    if not folder_path or not os.path.exists(folder_path):
        flash('画像フォルダが見つかりません')
        return redirect(url_for('index'))
    
    # 出力フォルダが指定されていない場合は画像フォルダと同じにする
    if not output_path:
        output_path = folder_path
    else:
        # 出力フォルダが存在しない場合は作成
        if not os.path.exists(output_path):
            try:
                os.makedirs(output_path)
            except Exception as e:
                flash(f'出力フォルダの作成に失敗しました: {str(e)}')
                return redirect(url_for('index'))
    
    # 画像ファイルを取得
    image_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.gif')
    images = [f for f in os.listdir(folder_path) 
              if f.lower().endswith(image_extensions)]
    
    if not images:
        flash('画像ファイルが見つかりません')
        return redirect(url_for('index'))
    
    current_folder = folder_path
    output_folder = output_path
    current_index = 0
    
    # 既存のキャプションを読み込み
    captions = load_existing_captions(output_folder)
    
    flash(f'{len(images)}個の画像が見つかりました。既存のキャプション{len(captions)}個を読み込みました。')
    
    return redirect(url_for('caption_page'))

@app.route('/caption')
def caption_page():
    global current_folder, output_folder, images, current_index
    
    if not current_folder or not images:
        flash('最初にフォルダを選択してください')
        return redirect(url_for('index'))
    
    current_image = images[current_index]
    image_path = os.path.join(current_folder, current_image)
    
    # 画像をBase64エンコードしてHTMLに埋め込み
    with open(image_path, 'rb') as img_file:
        img_data = base64.b64encode(img_file.read()).decode()
    
    # 既存のキャプションがあるかチェック
    current_caption = captions.get(current_image, '')
    caption_status = "既存" if current_caption else "新規"
    
    return render_template('caption.html', 
                         image_data=img_data,
                         filename=current_image,
                         current_index=current_index + 1,
                         total_images=len(images),
                         current_caption=current_caption,
                         caption_status=caption_status,
                         output_folder=output_folder,
                         expressions=EXPRESSIONS,
                         intensities=INTENSITIES)

@app.route('/generate_caption', methods=['POST'])
def generate_caption():
    """表情と強度から自動でキャプションを生成"""
    expression_jp = request.form.get('expression')
    intensity_name = request.form.get('intensity')
    
    if not expression_jp or not intensity_name:
        return jsonify({'success': False, 'message': '表情と強度を選択してください'})
    
    # 日本語から英語に変換
    expression_en = EXPRESSIONS.get(expression_jp)
    if not expression_en:
        return jsonify({'success': False, 'message': '不正な表情です'})
    
    intensity_value = INTENSITIES.get(intensity_name)
    if intensity_value is None:
        return jsonify({'success': False, 'message': '不正な強度です'})
    
    # キャプションを生成
    caption = f"shs,1girl,masterpiece,bestquality,portrait,({expression_en}:{intensity_value}),looking at viewer,general"
    
    return jsonify({
        'success': True,
        'caption': caption
    })

@app.route('/save_caption', methods=['POST'])
def save_caption():
    global current_index, captions
    
    caption = request.form.get('caption')
    filename = request.form.get('filename')
    
    if caption and filename:
        captions[filename] = caption
        # リアルタイムで保存
        save_single_caption_to_file(filename, caption)
    
    return jsonify({'success': True})

def save_single_caption_to_file(filename, caption):
    """単一のキャプションをファイルに即座に保存"""
    global output_folder, captions
    
    if not output_folder:
        return
    
    metadata_path = os.path.join(output_folder, "metadata.jsonl")
    
    try:
        # 既存のデータを読み込み
        existing_data = {}
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        data = json.loads(line)
                        existing_data[data['file_name']] = data['text']
        
        # 新しいキャプションを追加/更新
        existing_data[filename] = caption
        
        # ファイルに書き戻し
        with open(metadata_path, 'w', encoding='utf-8') as f:
            for file_name, text in existing_data.items():
                metadata = {
                    "file_name": file_name,
                    "text": text
                }
                f.write(json.dumps(metadata, ensure_ascii=False) + '\n')
    
    except Exception as e:
        print(f"リアルタイム保存エラー: {e}")

@app.route('/next_image')
def next_image():
    global current_index
    
    if current_index < len(images) - 1:
        current_index += 1
    
    return redirect(url_for('caption_page'))

@app.route('/prev_image')
def prev_image():
    global current_index
    
    if current_index > 0:
        current_index -= 1
    
    return redirect(url_for('caption_page'))

@app.route('/save_all', methods=['POST'])
def save_all():
    global current_folder, output_folder, captions
    
    if not current_folder or not captions:
        return jsonify({'success': False, 'message': 'データがありません'})
    
    # metadata.jsonlファイルに保存
    metadata_path = os.path.join(output_folder, "metadata.jsonl")
    
    try:
        with open(metadata_path, 'w', encoding='utf-8') as f:
            for filename, caption in captions.items():
                metadata = {
                    "file_name": filename,
                    "text": caption
                }
                f.write(json.dumps(metadata, ensure_ascii=False) + '\n')
        
        return jsonify({
            'success': True, 
            'message': f'キャプションが保存されました: {metadata_path}',
            'saved_count': len(captions)
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'保存に失敗しました: {str(e)}'
        })

@app.route('/reload_captions', methods=['POST'])
def reload_captions():
    """キャプションを再読み込み"""
    global captions, output_folder
    
    if output_folder:
        captions = load_existing_captions(output_folder)
        return jsonify({
            'success': True,
            'message': f'{len(captions)}個のキャプションを再読み込みしました'
        })
    else:
        return jsonify({
            'success': False,
            'message': '出力フォルダが設定されていません'
        })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)