import os
import cv2
from insightface.app import FaceAnalysis
from insightface.utils import face_align
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def debug_face_detection():
    print("=== InsightFace 顔検出デバッグ ===")
    
    # InsightFaceアプリケーションの初期化
    print("1. InsightFace初期化中...")
    app = FaceAnalysis(name="buffalo_l", providers=['CUDAExecutionProvider'])
    app.prepare(ctx_id=0, det_size=(1024, 1024))
    print("   初期化完了")
    
    # 画像の読み込み
    print("\n2. 画像読み込み...")
    image_path = "face.png"
    image = cv2.imread(image_path)
    if image is None:
        print(f"   エラー: {image_path}を読み込めませんでした")
        return
    
    print(f"   画像サイズ: {image.shape}")
    print(f"   画像データ型: {image.dtype}")
    print(f"   画像の値の範囲: {image.min()} - {image.max()}")
    
    # 顔検出の実行
    print("\n3. 顔検出実行...")
    faces = app.get(image)
    print(f"   検出された顔の数: {len(faces)}")
    
    if len(faces) == 0:
        print("   顔が検出されませんでした。原因を調査します...")
        
        # 異なるdet_sizeで試す前に、別のモデルも試してみる
        print("\n3.5. 別の検出モデルでテスト...")
        try:
            app_scrfd = FaceAnalysis(name="antelopev2", providers=['CUDAExecutionProvider'])
            app_scrfd.prepare(ctx_id=0, det_size=(640, 640))
            faces_scrfd = app_scrfd.get(image)
            print(f"   antelopev2モデル: {len(faces_scrfd)}個の顔")
            if len(faces_scrfd) > 0:
                faces = faces_scrfd  # 検出できた場合はこれを使用
        except Exception as e:
            print(f"   antelopev2モデルの読み込みに失敗: {e}")
        
        # 様々なdet_sizeで試す
        print("\n4. 異なるdet_sizeでの検出テスト...")
        test_sizes = [(640, 640), (512, 512), (320, 320), (256, 256)]
        
        for size in test_sizes:
            print(f"   det_size {size} でテスト中...")
            app.prepare(ctx_id=0, det_size=size)
            test_faces = app.get(image)
            print(f"     検出された顔: {len(test_faces)}")
            if len(test_faces) > 0:
                faces = test_faces
                break
        
        # 画像の前処理を試す
        print("\n5. 画像前処理後の検出テスト...")
        
        # グレースケール変換後にRGBに戻す
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        rgb_from_gray = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        faces_gray = app.get(rgb_from_gray)
        print(f"   グレースケール変換後: {len(faces_gray)}個の顔")
        if len(faces_gray) > 0:
            faces = faces_gray
            # グレースケール画像を保存
            plt.figure(figsize=(10, 10))
            plt.imshow(cv2.cvtColor(rgb_from_gray, cv2.COLOR_BGR2RGB))
            plt.title("Grayscale Converted")
            plt.axis('off')
            plt.savefig("debug_grayscale.png", dpi=150, bbox_inches='tight')
            plt.close()
        
        # ヒストグラム均等化
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        lab[:,:,0] = cv2.equalizeHist(lab[:,:,0])
        hist_eq = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        faces_hist = app.get(hist_eq)
        print(f"   ヒストグラム均等化後: {len(faces_hist)}個の顔")
        if len(faces_hist) > 0:
            faces = faces_hist
            # ヒストグラム均等化画像を保存
            plt.figure(figsize=(10, 10))
            plt.imshow(cv2.cvtColor(hist_eq, cv2.COLOR_BGR2RGB))
            plt.title("Histogram Equalized")
            plt.axis('off')
            plt.savefig("debug_hist_eq.png", dpi=150, bbox_inches='tight')
            plt.close()
        
        # 明度調整
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        hsv[:,:,2] = cv2.add(hsv[:,:,2], 30)  # 明度を上げる
        bright = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        faces_bright = app.get(bright)
        print(f"   明度調整後: {len(faces_bright)}個の顔")
        if len(faces_bright) > 0:
            faces = faces_bright
            # 明度調整画像を保存
            plt.figure(figsize=(10, 10))
            plt.imshow(cv2.cvtColor(bright, cv2.COLOR_BGR2RGB))
            plt.title("Brightness Adjusted")
            plt.axis('off')
            plt.savefig("debug_bright.png", dpi=150, bbox_inches='tight')
            plt.close()
        
        # コントラスト調整
        contrast_adj = cv2.convertScaleAbs(image, alpha=1.5, beta=0)
        faces_contrast = app.get(contrast_adj)
        print(f"   コントラスト調整後: {len(faces_contrast)}個の顔")
        if len(faces_contrast) > 0:
            faces = faces_contrast
            # コントラスト調整画像を保存
            plt.figure(figsize=(10, 10))
            plt.imshow(cv2.cvtColor(contrast_adj, cv2.COLOR_BGR2RGB))
            plt.title("Contrast Adjusted")
            plt.axis('off')
            plt.savefig("debug_contrast.png", dpi=150, bbox_inches='tight')
            plt.close()
        
        # 画像サイズを変更
        print("\n6. 画像サイズ変更後の検出テスト...")
        sizes_to_test = [512, 768, 1280]
        for size in sizes_to_test:
            resized = cv2.resize(image, (size, size))
            faces_resized = app.get(resized)
            print(f"   サイズ {size}x{size}: {len(faces_resized)}個の顔")
            if len(faces_resized) > 0:
                faces = faces_resized
                # リサイズ画像を保存
                plt.figure(figsize=(10, 10))
                plt.imshow(cv2.cvtColor(resized, cv2.COLOR_BGR2RGB))
                plt.title(f"Resized to {size}x{size}")
                plt.axis('off')
                plt.savefig(f"debug_resized_{size}.png", dpi=150, bbox_inches='tight')
                plt.close()
                break
        
        # 画像の統計情報
        print("\n7. 画像統計情報...")
        print(f"   平均値: B={np.mean(image[:,:,0]):.2f}, G={np.mean(image[:,:,1]):.2f}, R={np.mean(image[:,:,2]):.2f}")
        print(f"   標準偏差: B={np.std(image[:,:,0]):.2f}, G={np.std(image[:,:,1]):.2f}, R={np.std(image[:,:,2]):.2f}")
        
        # OpenCVのHaar Cascadeも試してみる
        print("\n8. OpenCV Haar Cascade テスト...")
        try:
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            opencv_faces = face_cascade.detectMultiScale(gray_img, 1.1, 4)
            print(f"   OpenCV Haar Cascade: {len(opencv_faces)}個の顔")
            
            if len(opencv_faces) > 0:
                # OpenCVで検出した顔を可視化
                result_img = image.copy()
                for (x, y, w, h) in opencv_faces:
                    cv2.rectangle(result_img, (x, y), (x+w, y+h), (0, 255, 0), 2)
                plt.figure(figsize=(10, 10))
                plt.imshow(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB))
                plt.title("OpenCV Haar Cascade Detection")
                plt.axis('off')
                plt.savefig("debug_opencv_detection.png", dpi=150, bbox_inches='tight')
                plt.close()
        except Exception as e:
            print(f"   OpenCV Haar Cascade エラー: {e}")
        
        # RGB変換を試す
        print("\n9. RGB変換テスト...")
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        bgr_from_rgb = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
        faces_rgb = app.get(bgr_from_rgb)
        print(f"   RGB変換後: {len(faces_rgb)}個の顔")
        if len(faces_rgb) > 0:
            faces = faces_rgb
        
    if len(faces) > 0:
        print("\n=== 顔が検出されました！ ===")
        result_img = image.copy()
        for i, face in enumerate(faces):
            print(f"   顔 {i+1}:")
            print(f"     境界ボックス: {face.bbox}")
            print(f"     信頼度: {face.det_score}")
            print(f"     ランドマーク: {face.kps.shape}")
            if hasattr(face, 'normed_embedding'):
                print(f"     埋め込み: {face.normed_embedding.shape}")
            
            # 境界ボックスを描画
            bbox = face.bbox.astype(int)
            cv2.rectangle(result_img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 3)
            
            # ランドマークを描画
            if hasattr(face, 'kps') and face.kps is not None:
                kps = face.kps.astype(int)
                for kp in kps:
                    cv2.circle(result_img, tuple(kp), 2, (255, 0, 0), -1)
        
        # 検出結果を保存
        plt.figure(figsize=(12, 12))
        plt.imshow(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB))
        plt.title(f"Face Detection Result - {len(faces)} faces found")
        plt.axis('off')
        plt.savefig("debug_face_detection_result.png", dpi=150, bbox_inches='tight')
        plt.close()
        print("   検出結果を debug_face_detection_result.png に保存しました")
    else:
        print("\n=== 顔は検出されませんでした ===")
        print("保存された画像ファイルを確認して、画像の内容を視覚的に確認してください:")
        print("- debug_original.png: 元画像")
        for file in os.listdir('.'):
            if file.startswith('debug_') and file.endswith('.png'):
                print(f"- {file}")

if __name__ == "__main__":
    debug_face_detection() 