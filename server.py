from flask import Flask, request, jsonify
import torch
import numpy as np
import cv2
import os
import time
import traceback
from io import BytesIO
from ultralytics import YOLO  # 使用通用YOLO類

app = Flask(__name__)

# 全局變量存放模型
model = None

def load_model():
    """載入YOLOv10模型到內存"""
    global model
    try:
        print("開始加載模型...")
        
        # 模型路徑
        model_path = "yolov10n.pt"
        
        # 如果本地沒有模型文件，嘗試下載
        if not os.path.exists(model_path):
            print(f"本地未找到模型文件 {model_path}，嘗試下載...")
            import urllib.request
            url = "https://github.com/THU-MIG/yolov10/releases/download/v1.1/yolov10n.pt"
            urllib.request.urlretrieve(url, model_path)
            print(f"模型已下載到 {model_path}")
        
        # 使用YOLO類加載模型
        model = YOLO(model_path)
        
        print("模型載入完成")
        
        # 模型預熱
        print("正在進行模型預熱...")
        dummy_img = np.zeros((640, 640, 3), dtype=np.uint8)
        for _ in range(2):
            _ = model(dummy_img)
        print("模型預熱完成")
        
        return True
    except Exception as e:
        print(f"模型載入失敗: {e}")
        traceback.print_exc()  # 打印詳細的錯誤堆疊
        return False

@app.route("/", methods=["GET"])
def hello():
    global model
    status = "loaded" if model is not None else "not loaded"
    gpu_info = f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'Not available'}"
    return f"Object Detection API is running. Model status: {status}. {gpu_info}"

@app.route('/predict', methods=['POST'])
def predict():
    global model
    if model is None:
        print("錯誤：模型未載入")
        return jsonify({'error': 'Model not loaded'}), 503
    
    try:
        # 檢查是否有檔案在請求中
        if 'image' not in request.files:
            return jsonify({'error': 'No image file in request'}), 400
        
        # 讀取圖片
        file = request.files['image']
        img_bytes = file.read()
        nparr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            return jsonify({'error': 'Invalid image'}), 400
        
        print(f"收到圖片，尺寸: {img.shape}")
        
        # 進行推論
        start_time = time.time()
        results = model(img)
        inference_time = time.time() - start_time
        
        # 正確解析Ultralytics YOLO模型的輸出
        detections = []
        
        # 檢查結果類型並適當處理
        print(f"結果類型: {type(results)}")
        
        # 新版Ultralytics YOLO輸出處理
        for result in results:
            boxes = result.boxes  # Boxes object for bbox outputs
            
            for i in range(len(boxes)):
                box = boxes[i]
                x1, y1, x2, y2 = box.xyxy[0].tolist()  # 邊界框坐標
                conf = float(box.conf[0])  # 置信度
                cls = int(box.cls[0])  # 類別ID
                name = result.names[cls]  # 類別名稱
                
                detections.append({
                    'xmin': float(x1),
                    'ymin': float(y1),
                    'xmax': float(x2),
                    'ymax': float(y2),
                    'confidence': float(conf),
                    'class': int(cls),
                    'name': name
                })
        
        fps = 1.0 / inference_time if inference_time > 0 else 0
        print(f"完成推論，檢測到 {len(detections)} 個物體，耗時: {inference_time:.4f}秒，FPS: {fps:.2f}")
        
        # 清理CUDA緩存，避免記憶體洩漏
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return jsonify({
            'detections': detections,
            'inference_time': inference_time,
            'fps': fps
        })
    
    except Exception as e:
        print(f"推論過程中發生錯誤: {e}")
        traceback.print_exc()  # 打印詳細的錯誤堆疊
        return jsonify({'error': str(e)}), 500

if __name__ == "__main__":
    # 先載入模型，再啟動API服務
    if load_model():
        print("API服務正在啟動...")
        # 啟動Flask應用
        app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
    else:
        print("模型載入失敗，API服務無法啟動")