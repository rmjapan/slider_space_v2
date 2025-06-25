import os
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

def slider_space_result_vis(num_sliders, save_path, slider_path):
    """
    スライダー結果を複数の図に分割して表示する関数
    
    Args:
        num_sliders (int): 一枚の図に収めるスライダーの数
        save_path (str): 結果画像の保存フォルダパス
        slider_path (str): スライダー画像が保存されているフォルダのパス
    """
    # 利用可能なスライダーフォルダを取得
    all_slider_folders = [f for f in os.listdir(slider_path) if os.path.isdir(os.path.join(slider_path, f))]
    all_slider_folders = sorted(all_slider_folders)
    
    if not all_slider_folders:
        print(f"No slider folders found in {slider_path}")
        return
    
    total_folders = len(all_slider_folders)
    num_images = (total_folders + num_sliders - 1) // num_sliders  # 切り上げ計算
    
    print(f"Total slider folders: {total_folders}")
    print(f"Creating {num_images} images with {num_sliders} sliders each")
    
    # 保存フォルダを作成
    os.makedirs(save_path, exist_ok=True)
    
    for image_idx in range(num_images):
        # 現在の画像で処理するスライダーフォルダを選択
        start_idx = image_idx * num_sliders
        end_idx = min(start_idx + num_sliders, total_folders)
        current_slider_folders = all_slider_folders[start_idx:end_idx]
        
        # 各スライダーフォルダの画像数を確認して最大値を取得
        max_images = 0
        slider_image_files = {}
        
        for slider_folder in current_slider_folders:
            slider_folder_path = os.path.join(slider_path, slider_folder)
            image_files = [f for f in os.listdir(slider_folder_path) if f.endswith(('.png', '.jpg', '.jpeg'))]
            image_files = sorted(image_files)
            slider_image_files[slider_folder] = image_files
            max_images = max(max_images, len(image_files))
        
        if max_images == 0:
            print(f"No images found in slider folders for image {image_idx}")
            continue
        
        current_num_sliders = len(current_slider_folders)
        print(f"Creating image {image_idx + 1}/{num_images} with {current_num_sliders} sliders and up to {max_images} images each")
        
        # 図のサイズを設定
        fig, axes = plt.subplots(current_num_sliders, max_images, figsize=(max_images * 3, current_num_sliders * 3))
        
        # 1つのスライダーまたは画像しかない場合の軸の処理
        if current_num_sliders == 1 and max_images == 1:
            axes = np.array([[axes]])
        elif current_num_sliders == 1:
            axes = axes.reshape(1, -1)
        elif max_images == 1:
            axes = axes.reshape(-1, 1)
        
        for slider_idx, slider_folder in enumerate(current_slider_folders):
            slider_folder_path = os.path.join(slider_path, slider_folder)
            image_files = slider_image_files[slider_folder]
            
            # 実際のスライダー番号を取得（フォルダ名から）
            actual_slider_num = start_idx + slider_idx
            
            # ファイル名からスケール値を抽出してソートする関数
            def extract_scale_from_filename(filename):
                try:
                    # ファイル拡張子を除去
                    name_without_ext = filename.replace('.png', '').replace('.jpg', '').replace('.jpeg', '')
                    # アンダースコアで分割
                    parts = name_without_ext.split('_')
                    # 最後の部分がスケール値として取得（例: test_image_0_-2.0.png -> -2.0）
                    if len(parts) > 0:
                        try:
                            scale_val = float(parts[-1])  # 最後の部分を取得
                            return scale_val
                        except ValueError:
                            pass
                    return 0.0  # デフォルト値
                except:
                    return 0.0
            
            # 画像ファイルをスケール値でソート
            sorted_image_files = sorted(image_files, key=extract_scale_from_filename)
            
            for img_idx in range(max_images):
                if img_idx < len(sorted_image_files):
                    image_file = sorted_image_files[img_idx]
                    image_path = os.path.join(slider_folder_path, image_file)
                    
                    try:
                        # 画像を読み込み
                        image = Image.open(image_path)
                        
                        # 画像を表示
                        axes[slider_idx, img_idx].imshow(image)
                        axes[slider_idx, img_idx].axis('off')
                        
                        # スケール値を抽出してタイトルに表示
                        scale_val = extract_scale_from_filename(image_file)
                        
                        # タイトルを設定（最初の列にスライダー名、最初の行にスケール値）
                        if img_idx == 0:
                            axes[slider_idx, img_idx].set_ylabel(f'Slider {actual_slider_num}', fontsize=10)
                        if slider_idx == 0:
                            axes[slider_idx, img_idx].set_title(f'Scale {scale_val}', fontsize=10)
                            
                    except Exception as e:
                        print(f"Error loading image {image_path}: {e}")
                        # エラーの場合は空のプロットを表示
                        axes[slider_idx, img_idx].text(0.5, 0.5, 'Error\nLoading\nImage', 
                                                      ha='center', va='center', transform=axes[slider_idx, img_idx].transAxes)
                        axes[slider_idx, img_idx].axis('off')
                else:
                    # 画像が足りない場合は空のプロットで埋める
                    axes[slider_idx, img_idx].axis('off')
        
        # レイアウトを調整
        plt.tight_layout()
        
        # 保存
        save_file_path = os.path.join(save_path, f"slider_visualization_{image_idx + 1:03d}.png")
        plt.savefig(save_file_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Saved: {save_file_path}")
    
    print(f"All {num_images} slider visualizations saved to: {save_path}")

def slider_space_result_vis_with_scales(num_sliders, save_path, slider_path, scales=None):
    """
    異なるスケールでのスライダー結果を複数画像に分割して可視化する関数
    
    Args:
        num_sliders (int): 一枚の図に収めるスライダーの数
        save_path (str): 結果画像の保存フォルダパス
        slider_path (str): スライダー画像が保存されているフォルダのパス
        scales (list, optional): 表示するスケール値のリスト。Noneの場合は自動検出
    """
    # 利用可能なスライダーフォルダを取得
    all_slider_folders = [f for f in os.listdir(slider_path) if os.path.isdir(os.path.join(slider_path, f))]
    all_slider_folders = sorted(all_slider_folders)
    
    if not all_slider_folders:
        print(f"No slider folders found in {slider_path}")
        return
    
    # スケールを自動検出する場合
    if scales is None:
        scales_set = set()
        for slider_folder in all_slider_folders:
            slider_folder_path = os.path.join(slider_path, slider_folder)
            image_files = [f for f in os.listdir(slider_folder_path) if f.endswith(('.png', '.jpg', '.jpeg'))]
            
            for image_file in image_files:
                # ファイル名からスケール値を抽出する試み
                try:
                    # 例: "test_image_0_1.5.png" -> 1.5, "test_image_0_-2.0.png" -> -2.0
                    name_without_ext = image_file.replace('.png', '').replace('.jpg', '').replace('.jpeg', '')
                    parts = name_without_ext.split('_')
                    if len(parts) > 0:
                        try:
                            scale_val = float(parts[-1])  # 最後の部分を取得
                            scales_set.add(scale_val)
                        except ValueError:
                            pass
                except:
                    continue
        
        scales = sorted(list(scales_set))
        if not scales:
            print("Could not detect scales from filenames. Using default scales.")
            scales = [-2.0, -1.0, 0.0, 1.0, 2.0]
    
    print(f"Using scales: {scales}")
    
    total_folders = len(all_slider_folders)
    num_images = (total_folders + num_sliders - 1) // num_sliders  # 切り上げ計算
    
    print(f"Total slider folders: {total_folders}")
    print(f"Creating {num_images} images with {num_sliders} sliders each")
    
    # 保存フォルダを作成
    os.makedirs(save_path, exist_ok=True)
    
    for image_idx in range(num_images):
        # 現在の画像で処理するスライダーフォルダを選択
        start_idx = image_idx * num_sliders
        end_idx = min(start_idx + num_sliders, total_folders)
        current_slider_folders = all_slider_folders[start_idx:end_idx]
        current_num_sliders = len(current_slider_folders)
        
        # 図のサイズを設定
        fig, axes = plt.subplots(current_num_sliders, len(scales), figsize=(len(scales) * 3, current_num_sliders * 3))
        
        # 軸の処理
        if current_num_sliders == 1 and len(scales) == 1:
            axes = np.array([[axes]])
        elif current_num_sliders == 1:
            axes = axes.reshape(1, -1)
        elif len(scales) == 1:
            axes = axes.reshape(-1, 1)
        
        for slider_idx, slider_folder in enumerate(current_slider_folders):
            slider_folder_path = os.path.join(slider_path, slider_folder)
            actual_slider_num = start_idx + slider_idx
            
            # スライダーフォルダ内の全画像ファイルを取得
            image_files = []
            try:
                for file in os.listdir(slider_folder_path):
                    if file.endswith(('.png', '.jpg', '.jpeg')):
                        # ファイル名からスケール値を抽出
                        file_without_ext = file.replace('.png', '').replace('.jpg', '').replace('.jpeg', '')
                        parts = file_without_ext.split('_')
                        
                        if len(parts) > 0:
                            try:
                                file_scale = float(parts[-1])  # 最後の部分を取得
                                image_files.append((file_scale, file))
                            except ValueError:
                                continue
            except Exception as e:
                print(f"Error reading directory {slider_folder_path}: {e}")
                continue
            
            # スケール値でソート
            image_files.sort(key=lambda x: x[0])
            
            # ソートされた画像を表示
            for scale_idx, (file_scale, image_file) in enumerate(image_files):
                if scale_idx >= len(scales):  # スケール数以上の画像は表示しない
                    break
                    
                image_path = os.path.join(slider_folder_path, image_file)
                
                if os.path.exists(image_path):
                    try:
                        # 画像を読み込み
                        image = Image.open(image_path)
                        
                        # 画像を表示
                        axes[slider_idx, scale_idx].imshow(image)
                        axes[slider_idx, scale_idx].axis('off')
                        
                        # タイトルを設定
                        if scale_idx == 0:
                            axes[slider_idx, scale_idx].set_ylabel(f'Slider {actual_slider_num}', fontsize=10)
                        if slider_idx == 0:
                            axes[slider_idx, scale_idx].set_title(f'Scale {file_scale:.1f}', fontsize=10)
                            
                    except Exception as e:
                        print(f"Error loading image {image_path}: {e}")
                        axes[slider_idx, scale_idx].text(0.5, 0.5, f'Error\nScale {file_scale:.1f}', 
                                                       ha='center', va='center', transform=axes[slider_idx, scale_idx].transAxes)
                        axes[slider_idx, scale_idx].axis('off')
                else:
                    # 画像が見つからない場合
                    axes[slider_idx, scale_idx].text(0.5, 0.5, f'No Image\nScale {file_scale:.1f}', 
                                                   ha='center', va='center', transform=axes[slider_idx, scale_idx].transAxes)
                    axes[slider_idx, scale_idx].axis('off')
        # レイアウトを調整
        plt.tight_layout()
        
        # 保存
        save_file_path = os.path.join(save_path, f"slider_scales_{image_idx + 1:03d}.png")
        plt.savefig(save_file_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Saved: {save_file_path}")
    
    print(f"All {num_images} slider scale visualizations saved to: {save_path}")

# 使用例
if __name__ == "__main__":
    # 基本的な使用例
    slider_path = "/home/ryuichi/animins/slider_space_v2/test_sliderspace/test_sliderspace_16898_v3"
    os.makedirs(slider_path + "/results", exist_ok=True)
    slider_space_result_vis(
        num_sliders=4,
        save_path=slider_path + "/results",
        slider_path=slider_path
    )
    
