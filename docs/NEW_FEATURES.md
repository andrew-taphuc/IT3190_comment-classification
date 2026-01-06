# 🆕 Các Tính Năng Mới

## Tổng Quan

Đã thêm các tính năng mới để cải thiện hiệu quả và tiện lợi khi sử dụng project:

1. **Hyperparameter Tuning** - Tự động tìm parameters tối ưu
2. **Threshold Optimization** - Tự động tìm threshold tốt nhất
3. **Visualization** - Tạo biểu đồ để phân tích kết quả
4. **Batch Prediction** - Predict nhiều text cùng lúc

---

## 1. 🔍 Hyperparameter Tuning

### File: `hyperparameter_tuning.py`

Tự động tìm hyperparameters tối ưu cho model sử dụng Grid Search hoặc Random Search.

### Cách sử dụng:

```bash
# Random search (nhanh hơn, khuyến nghị)
python hyperparameter_tuning.py --method random --n_iter 20

# Grid search (chậm hơn nhưng đầy đủ hơn)
python hyperparameter_tuning.py --method grid
```

### Tùy chọn:
- `--cv`: Số folds cho cross-validation (mặc định: 3)
- `--n_iter`: Số iterations cho random search (mặc định: 20)
- `--output`: File output (mặc định: `best_params.json`)

### Parameters được tune:
- `C`: Regularization parameter (0.5 - 3.0)
- `max_features`: Số features tối đa (30000, 40000, 50000)
- `ngram_range`: Word n-gram range ((1,1), (1,2), (1,3))

### Output:
File JSON chứa:
- `best_params`: Best parameters tìm được
- `best_cv_score`: Best cross-validation score
- `val_metrics`: Metrics trên validation set

---

## 2. 🎯 Threshold Optimization

### File: `threshold_optimizer.py`

Tự động tìm threshold tối ưu để phân loại toxic/non_toxic.

### Cách sử dụng:

```bash
# Tìm threshold tối ưu dựa trên F1 score
python threshold_optimizer.py --metric f1

# Optimize cho precision
python threshold_optimizer.py --metric precision

# Optimize cho recall
python threshold_optimizer.py --metric recall

# Balanced (F1 với recall >= 0.7)
python threshold_optimizer.py --metric balanced
```

### Metrics có thể optimize:
- `f1`: F1 score (mặc định)
- `precision`: Precision score
- `recall`: Recall score
- `balanced`: F1 score nhưng yêu cầu recall >= 0.7

### Output:
File JSON chứa:
- `optimal_threshold`: Threshold tối ưu
- `best_score`: Best score với threshold này
- `metrics_with_threshold`: Metrics khi sử dụng threshold mới

### Lợi ích:
- Tự động tìm threshold tốt nhất thay vì dùng giá trị mặc định
- Có thể optimize cho metric cụ thể (precision, recall, etc.)
- Cải thiện performance trên validation set

---

## 3. 📊 Visualization

### File: `visualize_results.py`

Tạo các biểu đồ để phân tích và so sánh kết quả model.

### Cách sử dụng:

```bash
python visualize_results.py
```

### Output:
Các file PNG trong `outputs/plots/`:
- `roc_curve.png`: ROC curve
- `pr_curve.png`: Precision-Recall curve
- `confusion_matrix.png`: Confusion matrix
- `model_comparison.png`: So sánh các models (nếu có model_comparison.csv)

### Tùy chọn:
- `--model`: Đường dẫn đến model (mặc định: `toxicity_pipeline.joblib`)
- `--val_csv`: Đường dẫn đến validation CSV
- `--output_dir`: Thư mục output (mặc định: `outputs/plots`)

### Lợi ích:
- Dễ dàng phân tích performance của model
- So sánh trực quan giữa các models
- Tạo báo cáo đẹp cho presentation

---

## 4. 📦 Batch Prediction

### File: `predict_batch.py`

Predict nhiều text cùng lúc, hiệu quả hơn khi xử lý nhiều text.

### Cách sử dụng:

#### Từ file CSV:
```bash
python predict_batch.py --input data.csv --text_col text --output results.json
```

#### Từ text file (một text mỗi dòng):
```bash
python predict_batch.py --input texts.txt --output results.csv --format csv
```

#### Từ stdin:
```bash
cat texts.txt | python predict_batch.py --output results.json
```

### Input formats:
- **CSV**: File CSV với cột text (chỉ định bằng `--text_col`)
- **Text file**: Một text mỗi dòng
- **stdin**: Đọc từ stdin (một text mỗi dòng)

### Output formats:
- **JSON**: Mảng các objects với predictions (mặc định)
- **CSV**: File CSV với các cột: text, label, toxic_score, threshold

### Tùy chọn:
- `--input`: File input (CSV hoặc text file)
- `--text_col`: Tên cột text nếu input là CSV
- `--output`: File output
- `--format`: Format output (`json` hoặc `csv`)
- `--threshold`: Threshold để phân loại

### Lợi ích:
- Xử lý nhiều text cùng lúc (nhanh hơn)
- Hỗ trợ nhiều format input/output
- Dễ tích hợp vào pipeline xử lý dữ liệu

---

## 📋 Workflow Khuyến Nghị

### 1. Train model cơ bản:
```bash
python train_toxic.py
```

### 2. Tìm hyperparameters tối ưu (tùy chọn):
```bash
python hyperparameter_tuning.py --method random --n_iter 20
# Sử dụng best_params.json để train lại với parameters tốt hơn
```

### 3. Tìm threshold tối ưu:
```bash
python threshold_optimizer.py --metric f1
# Cập nhật threshold trong toxicity_meta.json hoặc dùng khi predict
```

### 4. Visualize kết quả:
```bash
python visualize_results.py
```

### 5. Sử dụng model:
```bash
# Single prediction
python predict_toxic.py --text "Bình luận"

# Batch prediction
python predict_batch.py --input texts.csv --output results.json
```

---

## 🎯 Kết Quả Mong Đợi

Với các tính năng mới:
- **Hyperparameter tuning**: Có thể cải thiện 1-3% performance
- **Threshold optimization**: Cải thiện precision/recall theo nhu cầu
- **Visualization**: Dễ phân tích và trình bày kết quả
- **Batch prediction**: Xử lý nhanh hơn 10-100x khi có nhiều text

---

## 📝 Lưu Ý

1. **Hyperparameter tuning** có thể mất nhiều thời gian (vài giờ tùy vào data size)
2. **Threshold optimization** cần model đã train và validation set
3. **Visualization** cần matplotlib và seaborn (`pip install matplotlib seaborn`)
4. **Batch prediction** hiệu quả nhất khi xử lý > 100 texts

---

## 🔮 Tính Năng Có Thể Thêm

- [ ] Logging chi tiết cho training process
- [ ] SMOTE để xử lý class imbalance
- [ ] Model versioning và tracking
- [ ] API endpoint (Flask/FastAPI)
- [ ] Real-time monitoring dashboard
- [ ] A/B testing framework

