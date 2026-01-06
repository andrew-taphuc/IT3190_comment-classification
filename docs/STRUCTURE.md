# 📁 Cấu Trúc Project

## Tổng Quan

Project đã được tổ chức lại với cấu trúc rõ ràng hơn, tách biệt giữa:
- **Core modules**: Các modules cốt lõi được sử dụng bởi nhiều scripts
- **Tools**: Các scripts hỗ trợ và utilities
- **Main scripts**: Các scripts chính để train và predict

---

## 📂 Cấu Trúc Thư Mục

```
src/
├── 01_download_vihsd.py      # Pipeline: Tải dataset
├── 02_make_binary_labels.py  # Pipeline: Chuyển labels
├── 03_clean_text.py          # Pipeline: Làm sạch text
├── 04_train_ml_models.py     # So sánh models
├── train_toxic.py             # Train model đơn
├── train_ensemble.py         # Train ensemble model
├── predict_toxic.py           # Predict single text
│
├── core/                     # Core modules
│   ├── __init__.py           # Package exports
│   ├── text_cleaner.py        # Text preprocessing
│   ├── teencode_mapping.py    # Teencode dictionary
│   ├── config.py              # Configuration
│   ├── evaluation.py          # Model evaluation
│   ├── feature_extractor.py  # Feature extraction
│   └── utils.py               # Utility functions
│
└── tools/                     # Tools và scripts hỗ trợ
    ├── __init__.py
    ├── predict_batch.py       # Batch prediction
    ├── hyperparameter_tuning.py # Hyperparameter tuning
    ├── threshold_optimizer.py  # Threshold optimization
    └── visualize_results.py    # Visualization
```

---

## 📦 Core Modules (`src/core/`)

Các modules cốt lõi được import thông qua package `core`:

```python
from core import clean_text, ModelConfig, evaluate_model, load_data_split
```

### Modules:

1. **`text_cleaner.py`**
   - Hàm `clean_text()`: Làm sạch và chuẩn hóa text
   - Các hàm normalize: Unicode, emoji, punctuation, teencode

2. **`teencode_mapping.py`**
   - Dictionary `TEENCODE`: Mapping teencode sang từ chuẩn

3. **`config.py`**
   - `ModelConfig`: Cấu hình cho model training
   - `DataConfig`: Cấu hình cho data paths
   - `OutputConfig`: Cấu hình cho output paths

4. **`evaluation.py`**
   - `evaluate_model()`: Đánh giá model với nhiều metrics
   - `print_evaluation_report()`: In báo cáo chi tiết

5. **`feature_extractor.py`**
   - `extract_text_features()`: Trích xuất features từ text
   - `combine_features()`: Kết hợp features

6. **`utils.py`**
   - `load_data_split()`: Load data từ CSV
   - `get_label_distribution()`: Phân tích phân bố labels
   - `ensure_dir()`: Tạo thư mục nếu chưa có

---

## 🛠️ Tools (`src/tools/`)

Các scripts hỗ trợ được tổ chức trong folder `tools/`:

### Scripts:

1. **`predict_batch.py`**
   - Batch prediction cho nhiều text
   - Hỗ trợ CSV, text file, hoặc stdin

2. **`hyperparameter_tuning.py`**
   - Tìm hyperparameters tối ưu
   - Grid Search hoặc Random Search

3. **`threshold_optimizer.py`**
   - Tìm threshold tối ưu cho classification
   - Optimize cho F1, precision, recall

4. **`visualize_results.py`**
   - Tạo các biểu đồ visualization
   - ROC curve, PR curve, confusion matrix

---

## 🔄 Cách Import

### Import từ core:

```python
# Cách 1: Import trực tiếp từ package
from core import clean_text, ModelConfig, evaluate_model

# Cách 2: Import từ module cụ thể (nếu cần)
from core.text_cleaner import clean_text
from core.config import ModelConfig
```

### Import trong tools:

Các file trong `tools/` cần thêm path để import từ `core`:

```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core import clean_text, ModelConfig
```

---

## 📝 Lưu Ý

1. **Tất cả imports đã được cập nhật**: Các file chính (`train_toxic.py`, `train_ensemble.py`, etc.) đã được cập nhật để import từ `core`

2. **Backward compatibility**: Các models đã train vẫn hoạt động vì `text_cleaner` vẫn có thể được import thông qua `core`

3. **Module paths**: Khi load model với joblib, cần đảm bảo `core` package có thể được import

---

## 🎯 Lợi Ích

1. **Tổ chức rõ ràng**: Tách biệt core modules và tools
2. **Dễ bảo trì**: Dễ tìm và sửa code
3. **Tái sử dụng**: Core modules có thể được import dễ dàng
4. **Mở rộng**: Dễ thêm modules mới vào `core/` hoặc `tools/`

---

## 🔧 Migration Guide

Nếu bạn có code cũ sử dụng imports trực tiếp:

### Trước:
```python
from text_cleaner import clean_text
from config import ModelConfig
```

### Sau:
```python
from core import clean_text, ModelConfig
```

Hoặc nếu chạy từ thư mục `src/`:
```python
from core import clean_text, ModelConfig
```

