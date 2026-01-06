import os
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier

# Import modules mới
from core import clean_text, print_evaluation_report, evaluate_model

os.makedirs("outputs", exist_ok=True)

TRAIN_PATH = "data/processed/train.csv"
VAL_PATH   = "data/processed/val.csv"
TEST_PATH  = "data/processed/test.csv"

def load_split(path: str):
    df = pd.read_csv(path).dropna(subset=["text", "label"]).copy()
    df["text"] = df["text"].astype(str)
    df["label"] = df["label"].astype(str)
    return df

def eval_model(name: str, model: Pipeline, X_train, y_train, X_val, y_val, X_test, y_test):
    """Đánh giá model với metrics đầy đủ."""
    print(f"\n{'='*60}")
    print(f"MODEL: {name}")
    print(f"{'='*60}")
    
    model.fit(X_train, y_train)

    # Validation
    val_pred = model.predict(X_val)
    val_proba = model.predict_proba(X_val) if hasattr(model, 'predict_proba') else None
    labels = sorted(list(set(y_val)))
    val_metrics = evaluate_model(y_val, val_pred, val_proba, labels)

    # Test
    test_pred = model.predict(X_test)
    test_proba = model.predict_proba(X_test) if hasattr(model, 'predict_proba') else None
    test_metrics = evaluate_model(y_test, test_pred, test_proba, labels)

    # In báo cáo chi tiết cho test set
    print_evaluation_report(y_test, test_pred, test_proba, labels, split_name=f"{name} - TEST")

    return {
        "model": name,
        "val_acc": val_metrics.get('accuracy', 0),
        "val_macro_f1": val_metrics.get('macro_f1', 0),
        "val_roc_auc": val_metrics.get('roc_auc', None),
        "test_acc": test_metrics.get('accuracy', 0),
        "test_macro_f1": test_metrics.get('macro_f1', 0),
        "test_roc_auc": test_metrics.get('roc_auc', None),
    }

def main():
    train_df = load_split(TRAIN_PATH)
    val_df   = load_split(VAL_PATH)
    test_df  = load_split(TEST_PATH)

    X_train, y_train = train_df["text"], train_df["label"]
    X_val, y_val     = val_df["text"], val_df["label"]
    X_test, y_test   = test_df["text"], test_df["label"]

    # TF-IDF settings (rất hay cho tiếng Việt + MXH)
    # Sử dụng clean_text để preprocess (nhất quán với train_toxic.py)
    tfidf = TfidfVectorizer(
        preprocessor=clean_text,  # ✅ Thêm text preprocessing
        ngram_range=(1, 2),
        max_features=50000,
        min_df=2
    )

    models = [
        ("MultinomialNB", MultinomialNB(alpha=0.5)),
        ("LogisticRegression", LogisticRegression(max_iter=3000, class_weight="balanced")),
        ("LinearSVM", LinearSVC(class_weight="balanced")),
        ("RandomForest", RandomForestClassifier(
            n_estimators=300,
            random_state=42,
            n_jobs=-1,
            class_weight="balanced_subsample"
        )),
    ]

    results = []
    for name, clf in models:
        pipe = Pipeline([
            ("tfidf", tfidf),
            ("clf", clf)
        ])
        res = eval_model(name, pipe, X_train, y_train, X_val, y_val, X_test, y_test)
        results.append(res)

    res_df = pd.DataFrame(results).sort_values(by="val_macro_f1", ascending=False)
    out_path = "outputs/model_comparison.csv"
    res_df.to_csv(out_path, index=False)

    print("\n===== Summary (sorted by val_macro_f1) =====")
    print(res_df.to_string(index=False))
    print(f"\nSaved comparison table -> {out_path}")

if __name__ == "__main__":
    main()
