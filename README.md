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
│   ├── predict_toxic.py           # Dự đoán text có toxic hay không
│   └── text_cleaner.py            # Module chứa hàm làm sạch text
│
├── notebooks/                # Jupyter notebooks
│   └── analysis.ipynb       # Notebook phân tích dữ liệu
│
├── outputs/                  # Kết quả
│   └── model_comparison.csv  # Bảng so sánh các mô hình
│
├── toxicity_pipeline.joblib  # Mô hình đã train (để sử dụng)
├── toxicity_meta.json        # Metadata của mô hình
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
  - Lưu mô hình và metadata để sử dụng sau

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

### 🛠️ Module hỗ trợ

#### `text_cleaner.py`
- **Mục đích**: Module chứa hàm `clean_text()` để làm sạch text
- **Chức năng**:
  - Chuẩn hóa Unicode (NFC)
  - Loại bỏ URLs, mentions (@user), hashtags (#tag)
  - Chuẩn hóa ký tự lặp (ví dụ: "đẹpppp" → "đẹpp")
  - Map teen code sang từ chuẩn (ví dụ: "ko" → "không", "vcl" → "rất")
  - Giữ lại chỉ ký tự tiếng Việt, số, và dấu câu cơ bản
  - Loại bỏ khoảng trắng thừa
- **Lưu ý**: Module này được thiết kế để có thể pickle được khi lưu mô hình với joblib

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
- `toxicity_pipeline.joblib`: Mô hình đã train
- `toxicity_meta.json`: Metadata của mô hình

### Bước 5: Sử dụng mô hình để dự đoán

#### Cách 1: Dự đoán từ argument

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

Mô hình sử dụng pipeline: **word TF-IDF + char TF-IDF + LinearSVC + CalibratedClassifierCV**

- **Validation macro F1**: ~0.80
- **Test macro F1**: ~0.79
- **Validation accuracy**: ~0.89
- **Test accuracy**: ~0.89

## 📌 Lưu ý

1. **Thứ tự chạy**: Các file `01_`, `02_`, `03_` phải chạy theo thứ tự
2. **Dữ liệu**: Dataset ViHSD được tải tự động từ HuggingFace, không cần tải thủ công
3. **Model**: Mô hình được lưu dưới dạng `.joblib` và có thể load lại để sử dụng
4. **Text cleaning**: Hàm `clean_text()` được tích hợp vào pipeline, nên text input sẽ tự động được làm sạch khi predict

## 🔧 Tùy chỉnh

- **Threshold**: Có thể điều chỉnh threshold trong `toxicity_meta.json` hoặc qua argument `--threshold` khi predict
- **Model parameters**: Có thể tùy chỉnh tham số C (regularization) trong `train_toxic.py` qua argument `--C`
- **Text cleaning**: Có thể chỉnh sửa hàm `clean_text()` trong `text_cleaner.py` để phù hợp với nhu cầu

