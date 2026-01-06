# Hướng Dẫn Sử Dụng - Comment Classification

## Tổng Quan

Dự án này phân loại comment tiếng Việt thành 2 lớp: **toxic** (độc hại) hoặc **non_toxic** (không độc hại) sử dụng Machine Learning.

---

## 📋 Cấu Trúc Project (Workflow)

### Mục Đích
Phân loại comment tiếng Việt thành 2 lớp: **toxic** (độc hại) hoặc **non_toxic** (không độc hại).

---

## 🔄 Workflow Chính (3 Giai Đoạn)

### **Giai Đoạn 1: Chuẩn Bị Dữ Liệu** (chỉ cần làm 1 lần)

Chạy 3 script theo thứ tự để có dữ liệu sẵn sàng:

```bash
cd src
python 01_download_vihsd.py      # Tải dataset từ HuggingFace
python 02_make_binary_labels.py  # Chuyển 3 lớp → 2 lớp
python 03_clean_text.py          # Làm sạch text
```

**Kết quả**: Dữ liệu sẵn sàng trong `data/processed/`

---

### **Giai Đoạn 2: Train Model** (chỉ cần làm 1 lần)

Có 2 lựa chọn:

#### **Option A: Model Đơn** (nhanh, đủ dùng)
```bash
python train_toxic.py
```
→ Tạo: `outputs/toxicity_pipeline.joblib` + `outputs/toxicity_meta.json`

#### **Option B: Ensemble Model** (chậm hơn, tốt hơn ~1-2%)
```bash
python train_ensemble.py
```
→ Tạo: `outputs/toxicity_ensemble.joblib` + metadata

---

### **Giai Đoạn 3: Sử Dụng Model** (predict)

Có 2 cách:

#### **Cách 1: Predict một text**
```bash
python predict_toxic.py --text "Bình luận cần kiểm tra"
```
→ Output JSON: `{"label": "toxic", "toxic_score": 0.85, ...}`

#### **Cách 2: Predict nhiều text** (batch)
```bash
python tools/predict_batch.py --input texts.csv --output results.json
```
→ Xử lý nhiều text cùng lúc, output JSON/CSV

---

## 🛠️ Các Tính Năng Bổ Sung (Tùy Chọn)

### **1. So Sánh Models**
```bash
python 04_train_ml_models.py
```
→ So sánh 4 models, lưu vào `outputs/model_comparison.csv`

### **2. Tối Ưu Hyperparameters**
```bash
python tools/hyperparameter_tuning.py --method random --n_iter 20
```
→ Tìm tham số tốt nhất, lưu vào `outputs/best_params.json`

### **3. Tối Ưu Threshold**
```bash
python tools/threshold_optimizer.py --metric f1
```
→ Tìm threshold tốt nhất, lưu vào `outputs/optimal_threshold.json`

### **4. Visualization**
```bash
python tools/visualize_results.py
```
→ Tạo các biểu đồ trong `outputs/plots/` (ROC curve, confusion matrix, ...)

---

## 📌 Những Điều Cần Biết

### **1. Input/Output**

- **Input**: Text tiếng Việt (tự động được làm sạch)
- **Output**: 
  - `label`: "toxic" hoặc "non_toxic"
  - `toxic_score`: Xác suất (0-1)
  - `threshold`: Ngưỡng phân loại (mặc định 0.7)

### **2. Model Files**

- **Model**: `outputs/toxicity_pipeline.joblib` (hoặc `toxicity_ensemble.joblib`)
- **Metadata**: `outputs/toxicity_meta.json` (chứa threshold, metrics, config)

### **3. Tùy Chỉnh**

- **Threshold**: `--threshold 0.6` khi predict
- **Model parameters**: `--C 1.5` khi train
- **Data path**: `--data_dir ../data/processed`

### **4. Performance**

- **Model đơn**: Macro F1 ~0.80, Accuracy ~0.89
- **Ensemble**: Macro F1 ~0.81+, tốt hơn 1-2%

---

## ⚡ Workflow Tối Thiểu (Nhanh Nhất)

Nếu đã có model, chỉ cần predict:

```bash
# Predict một text
python predict_toxic.py --text "Bình luận của bạn"

# Predict nhiều text
python tools/predict_batch.py --input texts.csv --output results.json
```

---

## 📝 Tóm Tắt

1. **Mục đích**: Phân loại toxic/non_toxic cho comment tiếng Việt
2. **3 giai đoạn**: Chuẩn bị dữ liệu → Train model → Predict
3. **2 loại model**: Đơn (nhanh) hoặc Ensemble (tốt hơn)
4. **2 cách predict**: Single text hoặc Batch
5. **Output**: JSON với label và toxic_score

---

## 💡 Lưu Ý Quan Trọng

1. **Thứ tự chạy**: Các file `01_`, `02_`, `03_` phải chạy theo thứ tự
2. **Dữ liệu**: Dataset ViHSD được tải tự động từ HuggingFace
3. **Model**: Mô hình được lưu dưới dạng `.joblib` và có thể load lại để sử dụng
4. **Text cleaning**: Text input sẽ tự động được làm sạch khi predict
5. **Tất cả outputs**: Đều được lưu trong folder `outputs/`

---

## 🚀 Quick Start

### Lần đầu sử dụng:

```bash
# 1. Cài đặt dependencies
pip install pandas scikit-learn datasets joblib matplotlib seaborn

# 2. Chuẩn bị dữ liệu
cd src
python 01_download_vihsd.py
python 02_make_binary_labels.py
python 03_clean_text.py

# 3. Train model
python train_toxic.py

# 4. Predict
python predict_toxic.py --text "Bình luận cần kiểm tra"
```

### Đã có model:

```bash
# Chỉ cần predict
python predict_toxic.py --text "Bình luận của bạn"
```

---

## 📚 Thêm Thông Tin

- Xem `README.md` để biết chi tiết về cấu trúc thư mục và các tính năng
- Xem `docs/IMPROVEMENTS.md` để biết các cải tiến đã thực hiện
- Xem `docs/NEW_FEATURES.md` để biết các tính năng mới

