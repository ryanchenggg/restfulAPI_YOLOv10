import requests
import cv2
import time
import argparse
import numpy as np
import os
import sys
import platform
from PIL import Image
import io

# 檢測是否在無頭環境中運行
def is_headless_environment():
    """檢測當前環境是否適合顯示圖形界面"""
    # 檢查環境變量
    if 'DISPLAY' not in os.environ or not os.environ['DISPLAY']:
        return True
    
    # 檢查平台
    if platform.system() == 'Linux':
        # 嘗試執行一個X11測試命令
        try:
            import subprocess
            result = subprocess.run(['xset', 'q'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            return result.returncode != 0
        except (FileNotFoundError, subprocess.SubprocessError):
            return True
    
    # 在Windows和MacOS上預設為有圖形界面
    return False

def send_image(image_path, server_url="http://localhost:5000/predict", verbose=True):
    """發送圖片到服務器並獲取預測結果"""
    
    # 檢查圖片是否存在
    if not os.path.exists(image_path):
        print(f"錯誤: 找不到圖片 {image_path}")
        return None, None
    
    try:
        # 讀取圖片
        img = cv2.imread(image_path)
        if img is None:
            print(f"錯誤: 無法讀取圖片 {image_path}")
            return None, None
        
        # 檢查服務器狀態
        try:
            status_resp = requests.get(server_url.replace("predict", ""))
            if status_resp.status_code != 200:
                print(f"警告: 服務器狀態異常: {status_resp.status_code}, {status_resp.text}")
            elif verbose:
                print(f"服務器狀態: {status_resp.text}")
        except Exception as e:
            print(f"警告: 無法檢查服務器狀態: {e}")
        
        # 編碼圖片為JPEG格式
        is_success, buffer = cv2.imencode(".jpg", img)
        if not is_success:
            print("錯誤: 圖片編碼失敗")
            return None, None
        
        # 準備HTTP請求
        files = {'image': ('image.jpg', buffer.tobytes(), 'image/jpeg')}
        
        if verbose:
            print(f"正在發送圖片 {image_path} 到 {server_url}...")
        
        # 發送請求並測量時間
        start_time = time.time()
        response = requests.post(server_url, files=files, timeout=30)
        end_time = time.time()
        
        # 計算響應時間
        response_time = end_time - start_time
        fps = 1.0 / response_time
        
        # 檢查響應
        if response.status_code == 200:
            result = response.json()
            detections = result.get('detections', [])
            inference_time = result.get('inference_time', 0)
            server_fps = result.get('fps', 0)
            
            if verbose:
                print(f"請求完成: 回應時間 {response_time:.4f}秒, 網路FPS: {fps:.2f}")
                print(f"推論時間: {inference_time:.4f}秒, 推論FPS: {server_fps:.2f}")
                print(f"檢測到 {len(detections)} 個物體")
            
            return detections, img
        else:
            print(f"錯誤: {response.status_code}, {response.text}")
            return None, img
    
    except requests.exceptions.ConnectionError:
        print(f"錯誤: 無法連接到服務器 {server_url}，請確認服務器是否啟動")
        return None, None
    except requests.exceptions.Timeout:
        print(f"錯誤: 請求超時")
        return None, None
    except Exception as e:
        print(f"錯誤: {e}")
        return None, None

def visualize_results(image, detections):
    """視覺化檢測結果"""
    img_copy = image.copy()
    for det in detections:
        # 適應不同輸出格式
        x1 = float(det.get('xmin', det.get('x1', 0)))
        y1 = float(det.get('ymin', det.get('y1', 0)))
        x2 = float(det.get('xmax', det.get('x2', 0)))
        y2 = float(det.get('ymax', det.get('y2', 0)))
        conf = float(det.get('confidence', det.get('conf', 0)))
        name = det.get('name', str(det.get('class', '')))
        
        # 轉換為整數
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        
        label = f"{name} {conf:.2f}"
        
        # 根據置信度調整顏色 (綠->黃->紅)
        color = (0, 255 * (1 - conf), 255 * conf)
        
        # 畫框
        cv2.rectangle(img_copy, (x1, y1), (x2, y2), color, 2)
        
        # 標籤背景
        text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
        cv2.rectangle(img_copy, (x1, y1 - text_size[1] - 10), (x1 + text_size[0], y1), color, -1)
        
        # 標籤文字
        cv2.putText(img_copy, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    
    return img_copy

def find_test_images():
    """在test目錄中查找圖片"""
    test_dir = "test"
    if not os.path.exists(test_dir):
        print(f"警告: test目錄不存在，創建中...")
        os.makedirs(test_dir)
        print(f"請在 {test_dir} 目錄中添加圖片文件")
        return []
    
    image_files = [f for f in os.listdir(test_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
    return [os.path.join(test_dir, f) for f in image_files]

def ensure_result_dir():
    """確保result目錄存在"""
    result_dir = "result"
    if not os.path.exists(result_dir):
        print(f"創建result目錄...")
        os.makedirs(result_dir)
    return result_dir

def main():
    parser = argparse.ArgumentParser(description="Object Detection Client")
    parser.add_argument("--image", help="Path to image file (default: all images in test directory)")
    parser.add_argument("--server", default="http://localhost:5000/predict", help="Server URL")
    parser.add_argument("--loop", type=int, default=1, help="Number of times to repeat the request")
    parser.add_argument("--silent", action="store_true", help="Suppress detailed output")
    parser.add_argument("--force-display", action="store_true", help="Force display even in headless environment")
    parser.add_argument("--test", action="store_true", help="Run performance test")
    parser.add_argument("--test-runs", type=int, default=10, help="Number of runs for performance test")
    args = parser.parse_args()
    
    # 自動檢測是否為無頭環境
    no_display = is_headless_environment() and not args.force_display
    if no_display:
        print("檢測到無頭環境，將不顯示圖像窗口")
    
    # 確保result目錄存在
    result_dir = ensure_result_dir()
    
    # 確定處理的圖片列表
    image_paths = []
    if args.image:
        # 如果指定了圖片，只處理該圖片
        if os.path.exists(args.image):
            image_paths = [args.image]
        else:
            print(f"錯誤: 找不到指定的圖片 {args.image}")
            sys.exit(1)
    else:
        # 否則處理test目錄中的所有圖片
        image_paths = find_test_images()
        if not image_paths:
            print("錯誤: 沒有找到可處理的圖片。請在test目錄添加圖片或使用--image參數指定圖片")
            sys.exit(1)
    
    # 顯示要處理的圖片
    if not args.silent:
        print(f"將處理以下 {len(image_paths)} 張圖片:")
        for i, path in enumerate(image_paths):
            print(f"{i+1}. {path}")
    
    # 執行處理
    total_time = 0
    success_count = 0
    all_fps = []
    
    for image_path in image_paths:
        image_name = os.path.basename(image_path)
        if not args.silent:
            print(f"\n處理圖片: {image_name}")
        
        for i in range(args.loop):
            if not args.silent and args.loop > 1:
                print(f"  請求 {i+1}/{args.loop}")
            
            # 發送圖片獲取結果
            start_time = time.time()
            detections, img = send_image(image_path, args.server, not args.silent)
            request_time = time.time() - start_time
            
            if detections is not None and img is not None:
                success_count += 1
                total_time += request_time
                fps = 1.0 / request_time
                all_fps.append(fps)
                
                # 視覺化結果
                result_img = visualize_results(img, detections)
                
                # 保存結果圖片到result目錄
                output_path = os.path.join(result_dir, f"result_{image_name}")
                cv2.imwrite(output_path, result_img)
                if not args.silent:
                    print(f"  結果已保存至 {output_path}")
                
                # 只在非無頭模式下顯示結果(Windows/MacOS)
                if not no_display:
                    try:
                        cv2.imshow(f"Detection Results - {image_name}", result_img)
                        if len(image_paths) * args.loop > 1:
                            cv2.waitKey(100)  # 顯示100ms後繼續
                        else:
                            cv2.waitKey(0)  # 等待按鍵
                    except Exception as e:
                        print(f"  無法顯示圖像: {e}")
                        no_display = True  # 如果顯示失敗，關閉後續顯示嘗試
    
    # 計算平均處理時間和FPS
    if success_count > 0:
        avg_time = total_time / success_count
        avg_fps = sum(all_fps) / len(all_fps)
        print(f"\n總結: 成功處理 {success_count}/{len(image_paths) * args.loop} 次請求")
        print(f"平均處理時間: {avg_time:.4f}秒")
        print(f"平均FPS: {avg_fps:.2f}")
        
        if avg_fps < 10:
            print("\n⚠️ 警告: FPS低於目標 (10 FPS)")
        else:
            print("\n✅ 性能達標: FPS ≥ 10")
    else:
        print("\n錯誤: 所有請求均失敗")
    
    # 等待用戶關閉視窗 (只在顯示模式下)
    if not no_display and success_count > 0:
        try:
            print("\n按任意鍵關閉所有視窗...")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        except Exception:
            pass

if __name__ == "__main__":
    main()