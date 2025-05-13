import requests
import cv2
import time
import argparse
import numpy as np
from PIL import Image
import io
import os
import sys

def send_image(image_path, server_url="http://localhost:5000/predict", verbose=True):
    """發送圖片到服務器並獲取預測結果"""
    
    # 讀取圖片
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image at {image_path}")
    
    # turn image to jpeg format
    is_success, buffer = cv2.imencode(".jpg", img)
    if not is_success:
        raise ValueError("Failed to encode image")
    
    # HTTP request format
    files = {'image': ('image.jpg', buffer.tobytes(), 'image/jpeg')}
    
    # 發送請求並測量時間
    start_time = time.time()
    response = requests.post(server_url, files=files)
    end_time = time.time()
    
    # 計算響應時間
    response_time = end_time - start_time
    fps = 1.0 / response_time
    
    # 檢查響應
    if response.status_code == 200:
        detections = response.json()['detections']
        print(f"Response time: {response_time:.4f} seconds, FPS: {fps:.2f}")
        return detections, img
    else:
        print(f"Error: {response.status_code}, {response.text}")
        return None, img

def visualize_results(image, detections):
    """視覺化檢測結果"""
    for det in detections:
        x1, y1, x2, y2 = int(det['xmin']), int(det['ymin']), int(det['xmax']), int(det['ymax'])
        label = f"{det['name']} {det['confidence']:.2f}"
        
        # 畫框
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # 標籤
        cv2.putText(image, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    
    return image

def main():
    parser = argparse.ArgumentParser(description="Object Detection Client")
    parser.add_argument("--image", default="dog1.jpg", help="Path to image file")
    parser.add_argument("--server", default="http://localhost:5000/predict", help="Server URL")
    parser.add_argument("--loop", type=int, default=1, help="Number of times to repeat the request")
    parser.add_argument("--save", action="store_true", help="Save the result image")
    parser.add_argument("--silent", action="store_true", help="Suppress detailed output")
    parser.add_argument("--no-display", action="store_true", help="Do not display result image (headless mode)")
    parser.add_argument("--test", action="store_true", help="Run performance test")
    parser.add_argument("--test-runs", type=int, default=10, help="Number of runs for performance test")
    args = parser.parse_args()
    
    # 檢查圖片路徑
    if not os.path.exists(args.image):
        available_images = [f for f in os.listdir() if f.endswith(('.jpg', '.jpeg', '.png'))]
        print(f"錯誤: 找不到圖片 {args.image}")
        if available_images:
            print(f"可用的圖片: {', '.join(available_images)}")
            print(f"請嘗試: python client.py --image {available_images[0]}")
        sys.exit(1)
    
    # 運行性能測試
    if args.test:
        test_performance(args.image, args.server, args.test_runs, not args.silent)
        sys.exit(0)
    
    # 執行常規請求
    total_time = 0
    success_count = 0
    
    for i in range(args.loop):
        if not args.silent:
            print(f"\n請求 {i+1}/{args.loop}")
        
        # 發送圖片獲取結果
        start_time = time.time()
        detections, img = send_image(args.image, args.server, not args.silent)
        request_time = time.time() - start_time
        
        if detections is not None and img is not None:
            success_count += 1
            total_time += request_time
            
            # 視覺化結果
            result_img = visualize_results(img, detections)
            
            # 保存結果圖片
            if args.save:
                output_path = f"result_{os.path.basename(args.image)}"
                cv2.imwrite(output_path, result_img)
                if not args.silent:
                    print(f"結果已保存至 {output_path}")
            
            # 只在非無頭模式下顯示結果
            if not args.no_display:
                try:
                    cv2.imshow("Detection Results", result_img)
                    if args.loop > 1:
                        cv2.waitKey(1)  # 顯示1ms後繼續
                    else:
                        cv2.waitKey(0)  # 等待按鍵
                except Exception as e:
                    print(f"無法顯示圖像: {e}")
                    print("請使用 --no-display 參數在無頭環境中運行")
    
    # 計算平均處理時間和FPS
    if success_count > 0:
        avg_time = total_time / success_count
        avg_fps = success_count / total_time if total_time > 0 else 0
        print(f"\n總結: 成功 {success_count}/{args.loop} 次請求")
        print(f"平均處理時間: {avg_time:.4f}秒")
        print(f"平均FPS: {avg_fps:.2f}")
        
        if avg_fps < 10:
            print("\n⚠️ 警告: FPS低於目標 (10 FPS)，建議優化服務端性能")
    else:
        print("\n錯誤: 所有請求均失敗")
    
    # 等待用戶關閉視窗 (只在顯示模式和多次循環模式下)
    if success_count > 0 and args.loop > 1 and not args.no_display:
        try:
            print("\n按任意鍵關閉視窗...")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        except Exception:
            pass

if __name__ == "__main__":
    main()