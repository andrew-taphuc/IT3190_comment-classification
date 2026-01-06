# Comment Classification - Phân loại bình luận độc hại

Dự án này thực hiện phân loại bình luận tiếng Việt thành hai lớp: **toxic** (độc hại) và **non_toxic** (không độc hại) sử dụng các mô hình Machine Learning truyền thống.

## 📁 Cấu trúc thư mục

```
comment-clf/
├── data/                    # Dữ liệu
│   ├── raw/                 # Dữ liệu thô từ ViHSD dataset
│   │   ├── train.csv        # Tập train gốc (3 lớp: CLEAN/OFFENSIVE/HATE)
│   │   ├── validation.csv   # Tập validation gốc
│   │   └── test.csv         # Tập test gốc
│   ├── interim/             # Dữ liệu trung gian (đã chuyển sang binary labels)
│   │   ├── train.csv        # Tập train (toxic/non_toxic)
│   │   ├── val.csv          # Tập validation (toxic/non_toxic)
│   │   └── test.csv         # Tập test (toxic/non_toxic)
│   └── processed/           # Dữ liệu đã được làm sạch (sẵn sàng để train)
│       ├── train.csv        # Tập train đã clean
│       ├── val.csv          # Tập validation đã clean
│       └── test.csv         # Tập test đã clean
│
├── src/                     # Source code
│   ├── 01_download_vihsd.py      # Tải dataset ViHSD từ HuggingFace
│   ├── 02_make_binary_labels.py  # Chuyển labels 3 lớp → 2 lớp
│   ├── 03_clean_text.py          # Làm sạch text
│   ├── 04_train_ml_models.py     # So sánh nhiều mô hình ML
│   ├── train_toxic.py             # Train mô hình thực tế để sử dụng
│   ├── train_ensemble.py         # Train ensemble model (nâng cao)
│   ├── predict_toxic.py           # Dự đoán text có toxic hay không
│   ├── core/                      # Core modules
│   │   ├── __init__.py            # Package init
│   │   ├── text_cleaner.py        # Module chứa hàm làm sạch text
│   │   ├── teencode_mapping.py    # Mapping teencode sang từ chuẩn
│   │   ├── config.py              # File cấu hình
│   │   ├── evaluation.py          # Module đánh giá model
│   │   ├── feature_extractor.py  # Module trích xuất features
│   │   └── utils.py               # Utility functions
│   └── tools/                     # Tools và scripts hỗ trợ
│       ├── __init__.py            # Package init
│       ├── predict_batch.py       # Dự đoán nhiều text cùng lúc (batch)
│       ├── hyperparameter_tuning.py # Tìm hyperparameters tối ưu
│       ├── threshold_optimizer.py # Tìm threshold tối ưu
│       └── visualize_results.py  # Visualize kết quả (ROC, PR curve, etc.)
│
├── notebooks/                # Jupyter notebooks
│   └── analysis.ipynb       # Notebook phân tích dữ liệu
│
├── outputs/                  # Kết quả (tất cả outputs)
│   ├── model_comparison.csv  # Bảng so sánh các mô hình
│   ├── plots/                # Các biểu đồ visualization
│   ├── toxicity_pipeline.joblib  # Mô hình đã train (để sử dụng)
│   ├── toxicity_meta.json    # Metadata của mô hình
│   ├── toxicity_ensemble.joblib  # Ensemble model (nếu có)
│   ├── toxicity_ensemble_meta.json  # Metadata của ensemble model
│   ├── best_params.json      # Best hyperparameters (nếu có)
│   └── optimal_threshold.json  # Optimal threshold (nếu có)
│
└── README.md                 # File này
```

## 📝 Giải thích các file

### 🔄 Pipeline xử lý dữ liệu (chạy tuần tự)

Các file này tạo thành pipeline xử lý dữ liệu từ raw → processed, cần chạy theo thứ tự:

#### `01_download_vihsd.py`
- **Mục đích**: Tải dataset ViHSD (Vietnamese Hate Speech Detection) từ HuggingFace
- **Input**: Không có (tải trực tiếp từ HuggingFace)
- **Output**: `data/raw/train.csv`, `data/raw/validation.csv`, `data/raw/test.csv`
- **Chức năng**: 
  - Tải dataset `uitnlp/vihsd` từ HuggingFace
  - Chuyển đổi sang định dạng CSV với 2 cột: `text` và `label`
  - Labels ban đầu là 3 lớp: `CLEAN`, `OFFENSIVE`, `HATE`

#### `02_make_binary_labels.py`
- **Mục đích**: Chuyển đổi labels từ 3 lớp sang 2 lớp (binary classification)
- **Input**: `data/raw/*.csv`
- **Output**: `data/interim/train.csv`, `data/interim/val.csv`, `data/interim/test.csv`
- **Chức năng**:
  - `CLEAN` → `non_toxic`
  - `OFFENSIVE` hoặc `HATE` → `toxic`
  - Lưu vào thư mục `interim/`

#### `03_clean_text.py`
- **Mục đích**: Làm sạch và chuẩn hóa text
- **Input**: `data/interim/*.csv`
- **Output**: `data/processed/train.csv`, `data/processed/val.csv`, `data/processed/test.csv`
- **Chức năng**:
  - Sử dụng hàm `clean_text()` từ module `text_cleaner.py`
  - Loại bỏ URLs, mentions, hashtags
  - Chuẩn hóa Unicode, teen code, ký tự lặp
  - Xóa các dòng trống và duplicate
  - Lưu vào thư mục `processed/` (sẵn sàng để train)

### 🔬 So sánh mô hình

#### `04_train_ml_models.py`
- **Mục đích**: So sánh hiệu suất của nhiều mô hình ML khác nhau
- **Input**: `data/processed/train.csv`, `data/processed/val.csv`, `data/processed/test.csv`
- **Output**: `outputs/model_comparison.csv` (bảng so sánh metrics)
- **Chức năng**:
  - Train và đánh giá 4 mô hình:
    - `MultinomialNB`: Naive Bayes
    - `LogisticRegression`: Hồi quy logistic
    - `LinearSVM`: Support Vector Machine tuyến tính
    - `RandomForest`: Random Forest
  - Sử dụng TF-IDF vectorization (1-2 grams)
  - Tính toán accuracy và macro F1-score trên validation và test set
  - Lưu kết quả so sánh vào CSV để phân tích

### 🚀 Train và sử dụng mô hình thực tế

#### `train_toxic.py`
- **Mục đích**: Train mô hình thực tế để sử dụng trong production
- **Input**: `data/processed/train.csv`, `data/processed/val.csv`, `data/processed/test.csv`
- **Output**: 
  - `toxicity_pipeline.joblib`: Mô hình đã train (có thể load và sử dụng)
  - `toxicity_meta.json`: Metadata của mô hình (threshold, metrics, config)
- **Chức năng**:
  - Sử dụng pipeline tối ưu: **word TF-IDF + char TF-IDF + LinearSVC + CalibratedClassifierCV**
  - Feature Union kết hợp word n-grams (1-2) và character n-grams (3-5)
  - CalibratedClassifierCV để có `predict_proba()` (xác suất)
  - **Cải tiến mới**: 
    - Text preprocessing cải thiện (xử lý emoji, punctuation)
    - Evaluation metrics đầy đủ (ROC-AUC, PR-AUC, confusion matrix)
    - Sử dụng config file để dễ tùy chỉnh
  - Lưu mô hình và metadata để sử dụng sau

#### `train_ensemble.py` ⭐ MỚI
- **Mục đích**: Train ensemble model với nhiều base models để cải thiện hiệu quả
- **Input**: `data/processed/train.csv`, `data/processed/val.csv`, `data/processed/test.csv`
- **Output**: 
  - `toxicity_ensemble.joblib`: Ensemble model đã train
  - `toxicity_ensemble_meta.json`: Metadata của ensemble model
- **Chức năng**:
  - Sử dụng **VotingClassifier** kết hợp 3 models:
    - LinearSVC (weight=2)
    - LogisticRegression (weight=1)
    - RandomForest (weight=1)
  - CalibratedClassifierCV để có probabilities tốt hơn
  - Thường cho kết quả tốt hơn model đơn lẻ

#### `predict_toxic.py`
- **Mục đích**: Dự đoán một text có toxic hay không
- **Input**: 
  - Model: `toxicity_pipeline.joblib`
  - Text: từ argument `--text` hoặc stdin
- **Output**: JSON với label, toxic_score, threshold
- **Chức năng**:
  - Load mô hình đã train
  - Dự đoán text và trả về:
    - `label`: "toxic" hoặc "non_toxic"
    - `toxic_score`: Xác suất toxic (0-1)
    - `threshold`: Ngưỡng để phân loại (mặc định 0.7)

#### `predict_batch.py` ⭐ MỚI
- **Mục đích**: Dự đoán nhiều text cùng lúc (batch prediction)
- **Input**: 
  - File CSV hoặc text file (một text mỗi dòng)
  - Hoặc stdin (một text mỗi dòng)
- **Output**: JSON hoặc CSV với predictions cho tất cả texts
- **Chức năng**:
  - Xử lý nhiều text cùng lúc (hiệu quả hơn)
  - Hỗ trợ input từ file CSV hoặc text file
  - Output có thể là JSON hoặc CSV

#### `hyperparameter_tuning.py` ⭐ MỚI
- **Mục đích**: Tìm hyperparameters tối ưu cho model
- **Input**: `data/processed/train.csv`, `data/processed/val.csv`
- **Output**: `best_params.json` với best parameters và metrics
- **Chức năng**:
  - Grid Search hoặc Random Search để tìm best parameters
  - Tune các tham số: C, max_features, ngram_range
  - Sử dụng cross-validation để đánh giá
  - Lưu kết quả để sử dụng khi train model

#### `threshold_optimizer.py` ⭐ MỚI
- **Mục đích**: Tìm threshold tối ưu cho classification
- **Input**: Model đã train, validation set
- **Output**: `optimal_threshold.json` với threshold và metrics
- **Chức năng**:
  - Tự động tìm threshold tốt nhất dựa trên F1, precision, recall
  - Có thể optimize cho metric cụ thể (f1, precision, recall, balanced)
  - Đánh giá metrics với threshold mới

#### `visualize_results.py` ⭐ MỚI
- **Mục đích**: Tạo các biểu đồ visualization cho kết quả model
- **Input**: Model đã train, validation set
- **Output**: Các file PNG trong `outputs/plots/`
- **Chức năng**:
  - Vẽ ROC curve
  - Vẽ Precision-Recall curve
  - Vẽ Confusion Matrix
  - So sánh các models (nếu có model_comparison.csv)

### 🛠️ Module hỗ trợ

#### `text_cleaner.py`
- **Mục đích**: Module chứa hàm `clean_text()` để làm sạch text
- **Chức năng**:
  - Chuẩn hóa Unicode (NFC)
  - Loại bỏ URLs, mentions (@user), hashtags (#tag)
  - **Cải tiến mới**: Xử lý emoji (thay thế bằng khoảng trắng)
  - Chuẩn hóa ký tự lặp (ví dụ: "đẹpppp" → "đẹpp")
  - **Cải tiến mới**: Chuẩn hóa dấu câu lặp (ví dụ: "!!!" → "!")
  - Map teen code sang từ chuẩn (ví dụ: "ko" → "không", "vcl" → "chửi")
  - Giữ lại chỉ ký tự tiếng Việt, số, và dấu câu cơ bản
  - Loại bỏ khoảng trắng thừa
- **Lưu ý**: Module này được thiết kế để có thể pickle được khi lưu mô hình với joblib

#### `core/` - Core Modules ⭐ MỚI
Các modules core được tổ chức trong folder `core/`:

- **`core/teencode_mapping.py`**: Module chứa dictionary mapping teencode sang từ chuẩn
- **`core/feature_extractor.py`**: Module trích xuất các features từ text (emoji, punctuation, etc.)
- **`core/evaluation.py`**: Module đánh giá model với nhiều metrics (ROC-AUC, PR-AUC, confusion matrix, etc.)
- **`core/config.py`**: File cấu hình tập trung cho model, data, và output paths
- **`core/utils.py`**: Utility functions (load data, phân tích label distribution, etc.)

#### `tools/` - Tools và Scripts ⭐ MỚI
Các tools và scripts hỗ trợ được tổ chức trong folder `tools/`:

- **`tools/predict_batch.py`**: Dự đoán nhiều text cùng lúc (batch prediction)
- **`tools/hyperparameter_tuning.py`**: Tìm hyperparameters tối ưu với Grid/Random Search
- **`tools/threshold_optimizer.py`**: Tìm threshold tối ưu cho classification
- **`tools/visualize_results.py`**: Tạo các biểu đồ visualization (ROC curve, PR curve, confusion matrix)

## 🗑️ File đã xóa

Các file sau đã được xóa vì không được sử dụng:
- `src/utils_text.py`: Chứa hàm `normalize_teencode()` nhưng đã có trong `text_cleaner.py`
- `data/raw/comment.csv`: File không được sử dụng trong pipeline

## 🚀 Hướng dẫn chạy

### Bước 1: Cài đặt dependencies

```bash
pip install pandas scikit-learn datasets joblib
```

### Bước 2: Xử lý dữ liệu (Pipeline)

Chạy các script theo thứ tự để xử lý dữ liệu từ raw → processed:

```bash
# Bước 1: Tải dataset ViHSD
cd src
python 01_download_vihsd.py

# Bước 2: Chuyển labels sang binary (toxic/non_toxic)
python 02_make_binary_labels.py

# Bước 3: Làm sạch text
python 03_clean_text.py
```

Sau khi chạy xong, bạn sẽ có dữ liệu đã xử lý trong `data/processed/`.

### Bước 3: So sánh mô hình (Tùy chọn)

Để so sánh hiệu suất của nhiều mô hình ML:

```bash
python 04_train_ml_models.py
```

Kết quả sẽ được lưu trong `outputs/model_comparison.csv`.

### Bước 4: Train mô hình thực tế

#### Option 1: Train model đơn (nhanh hơn)

Train mô hình để sử dụng trong production:

```bash
python train_toxic.py
```

Hoặc với các tùy chọn:

```bash
# Chỉ định thư mục dữ liệu
python train_toxic.py --data_dir ../data/processed

# Tùy chỉnh tham số C (regularization)
python train_toxic.py --C 1.5

# Tùy chỉnh threshold
python train_toxic.py --threshold 0.65
```

Sau khi train xong, bạn sẽ có:
- `outputs/toxicity_pipeline.joblib`: Mô hình đã train
- `outputs/toxicity_meta.json`: Metadata của mô hình (với metrics đầy đủ: ROC-AUC, PR-AUC, etc.)

#### Option 2: Train ensemble model (hiệu quả hơn) ⭐ MỚI

Train ensemble model với nhiều base models:

```bash
python train_ensemble.py
```

Hoặc với các tùy chọn:

```bash
python train_ensemble.py --C 2.0 --threshold 0.70
```

Sau khi train xong, bạn sẽ có:
- `outputs/toxicity_ensemble.joblib`: Ensemble model đã train
- `outputs/toxicity_ensemble_meta.json`: Metadata của ensemble model

**Lưu ý**: Ensemble model thường cho kết quả tốt hơn nhưng chậm hơn khi train và predict.

### Bước 5: Tìm hyperparameters tối ưu (Tùy chọn) ⭐ MỚI

```bash
# Random search (nhanh hơn, khuyến nghị)
python tools/hyperparameter_tuning.py --method random --n_iter 20

# Grid search (chậm hơn nhưng đầy đủ hơn)
python tools/hyperparameter_tuning.py --method grid
```

Kết quả sẽ được lưu trong `outputs/best_params.json`. Sau đó có thể sử dụng các parameters này khi train model.

### Bước 6: Tìm threshold tối ưu (Tùy chọn) ⭐ MỚI

```bash
# Tìm threshold tối ưu dựa trên F1 score
python tools/threshold_optimizer.py --metric f1

# Hoặc optimize cho precision/recall
python tools/threshold_optimizer.py --metric balanced
```

Kết quả sẽ được lưu trong `outputs/optimal_threshold.json`.

### Bước 7: Visualize kết quả (Tùy chọn) ⭐ MỚI

```bash
python tools/visualize_results.py
```

Sẽ tạo các biểu đồ trong `outputs/plots/`:
- `roc_curve.png`: ROC curve
- `pr_curve.png`: Precision-Recall curve
- `confusion_matrix.png`: Confusion matrix
- `model_comparison.png`: So sánh các models

### Bước 8: Sử dụng mô hình để dự đoán

#### Cách 1: Dự đoán một text

```bash
python predict_toxic.py --text "Bình luận cần kiểm tra ở đây"
```

#### Cách 2: Dự đoán từ stdin

```bash
echo "Bình luận cần kiểm tra" | python predict_toxic.py
```

#### Cách 3: Dự đoán với threshold tùy chỉnh

```bash
python predict_toxic.py --text "Bình luận" --threshold 0.6
```

#### Cách 4: Batch prediction (nhiều text) ⭐ MỚI

```bash
# Từ file CSV
python tools/predict_batch.py --input data.csv --text_col text --output outputs/results.json

# Từ text file (một text mỗi dòng)
python tools/predict_batch.py --input texts.txt --output outputs/results.csv --format csv

# Từ stdin
cat texts.txt | python tools/predict_batch.py --output outputs/results.json
```

#### Output mẫu:

```json
{
  "label": "toxic",
  "toxic_score": 0.85,
  "threshold": 0.7,
  "classes": ["non_toxic", "toxic"],
  "proba": [0.15, 0.85]
}
```

### Ví dụ workflow hoàn chỉnh

```bash
# 1. Xử lý dữ liệu
cd src
python 01_download_vihsd.py
python 02_make_binary_labels.py
python 03_clean_text.py

# 2. So sánh mô hình (tùy chọn)
python 04_train_ml_models.py

# 3. Train mô hình thực tế
python train_toxic.py

# 4. Test dự đoán
python predict_toxic.py --text "Đây là một bình luận độc hại"
```

## 📊 Kết quả mô hình

### Model đơn (train_toxic.py)
Pipeline: **word TF-IDF + char TF-IDF + LinearSVC + CalibratedClassifierCV**

- **Validation macro F1**: ~0.80
- **Test macro F1**: ~0.79
- **Validation accuracy**: ~0.89
- **Test accuracy**: ~0.89
- **ROC-AUC**: ~0.90+
- **PR-AUC**: ~0.85+

### Ensemble model (train_ensemble.py) ⭐ MỚI
Pipeline: **VotingClassifier(SVM + LR + RF) + CalibratedClassifierCV**

- Thường cho kết quả tốt hơn model đơn 1-2%
- Có thể đạt **macro F1 > 0.81** trên test set
- **Lưu ý**: Chậm hơn khi train và predict

## 🆕 Cải tiến mới

### 1. Text Preprocessing cải thiện
- ✅ Xử lý emoji tốt hơn
- ✅ Chuẩn hóa dấu câu lặp
- ✅ Tách teencode mapping sang file riêng

### 2. Evaluation Metrics đầy đủ
- ✅ ROC-AUC score
- ✅ PR-AUC score
- ✅ Confusion matrix visualization
- ✅ Per-class F1 scores

### 3. Ensemble Methods
- ✅ VotingClassifier với nhiều base models
- ✅ Cải thiện hiệu quả phát hiện toxic

### 4. Code Structure
- ✅ Config file tập trung
- ✅ Modules tách biệt rõ ràng
- ✅ Utility functions
- ✅ Feature extraction module (có thể mở rộng)

## 📌 Lưu ý

1. **Thứ tự chạy**: Các file `01_`, `02_`, `03_` phải chạy theo thứ tự
2. **Dữ liệu**: Dataset ViHSD được tải tự động từ HuggingFace, không cần tải thủ công
3. **Model**: Mô hình được lưu dưới dạng `.joblib` và có thể load lại để sử dụng
4. **Text cleaning**: Hàm `clean_text()` được tích hợp vào pipeline, nên text input sẽ tự động được làm sạch khi predict

## 🔧 Tùy chỉnh

- **Threshold**: Có thể điều chỉnh threshold trong `toxicity_meta.json` hoặc qua argument `--threshold` khi predict
- **Model parameters**: Có thể tùy chỉnh tham số C (regularization) trong `train_toxic.py` qua argument `--C`
- **Text cleaning**: Có thể chỉnh sửa hàm `clean_text()` trong `text_cleaner.py` để phù hợp với nhu cầu

