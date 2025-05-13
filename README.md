# YOLOv10 物件檢測 Restful API 服務

基於 YOLOv10 的物件檢測 API 服務，達到 ≧10 FPS 的推論速度。

## 安裝步驟

```bash
pip install -r requirements.txt

```

## 檔案說明

- `server.py`: 使用 Flask 開發的 Restful API 服務，自動下載並加載 YOLOv10n 模型
- `client.py`: 從 test 目錄讀取圖片，發送至服務端進行推論，結果保存至 result 目錄
- `test/`: 存放測試圖片的目錄
- `result/`: 存放輸出結果的目錄

## 測試方法

1. 啟動服務端:
```bash
python server.py
```

2. 運行客戶端:
```bash
python client.py
```

3. 檢查結果:
   - 確認 FPS ≧10
   - 查看 result 目錄中的檢測結果圖片

## 客戶端參數

```bash
python client.py --image test/dog.jpg  # 處理特定圖片
python client.py --loop 10              # 連續請求10次
python client.py --test                 # 性能測試
```

## 性能優化提示

- 確保使用 GPU 推論 (CUDA)
- 降低輸入圖片分辨率
