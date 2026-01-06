# train_toxic.py
import json
import argparse
from dataclasses import dataclass

import pandas as pd
import joblib

from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import classification_report, f1_score

# Import hàm làm sạch từ module text_cleaner (có thể pickle được)
from text_cleaner import clean_text


# ----------------------------
# Config / utils
# ----------------------------
@dataclass
class Paths:
    train_csv: str
    val_csv: str
    test_csv: str
    model_out: str
    meta_out: str


def load_split(path: str, text_col: str = "text", label_col: str = "label"):
    df = pd.read_csv(path)
    if text_col not in df.columns or label_col not in df.columns:
        raise ValueError(f"{path} phải có cột '{text_col}' và '{label_col}'")

    df = df.dropna(subset=[text_col, label_col]).copy()
    X = df[text_col].astype(str).values
    y = df[label_col].astype(str).values
    return X, y


def build_pipeline(C: float = 2.0) -> Pipeline:
    """
    Pipeline:
      - word TF-IDF (1-2)
      - char TF-IDF (3-5)
      - FeatureUnion
      - LinearSVC (balanced)
      - CalibratedClassifierCV => predict_proba (score)
    """
    word_tfidf = TfidfVectorizer(
        preprocessor=clean_text,
        analyzer="word",
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
        sublinear_tf=True,
    )

    char_tfidf = TfidfVectorizer(
        preprocessor=clean_text,
        analyzer="char",
        ngram_range=(3, 5),
        min_df=2,
        max_df=0.95,
    )

    feats = FeatureUnion(
        [
            ("word_tfidf", word_tfidf),
            ("char_tfidf", char_tfidf),
        ]
    )

    base_svm = LinearSVC(C=C, class_weight="balanced", random_state=42)

    clf = CalibratedClassifierCV(
        estimator=base_svm,
        method="sigmoid",
        cv=3,
    )

    return Pipeline(
        [
            ("features", feats),
            ("clf", clf),
        ]
    )


def eval_split(name: str, pipe: Pipeline, X, y) -> float:
    pred = pipe.predict(X)
    macro = f1_score(y, pred, average="macro")
    print(f"\n=== {name} ===")
    print(f"macro_f1: {macro:.4f}")
    print(classification_report(y, pred, digits=4))
    return float(macro)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data/processed", help="Thư mục chứa train.csv/val.csv/test.csv")
    ap.add_argument("--train_csv", default=None, help="Override đường dẫn train.csv")
    ap.add_argument("--val_csv", default=None, help="Override đường dẫn val.csv")
    ap.add_argument("--test_csv", default=None, help="Override đường dẫn test.csv")

    ap.add_argument("--text_col", default="text")
    ap.add_argument("--label_col", default="label")

    ap.add_argument("--model_out", default="toxicity_pipeline.joblib")
    ap.add_argument("--meta_out", default="toxicity_meta.json")
    ap.add_argument("--C", type=float, default=2.0)
    ap.add_argument("--threshold", type=float, default=0.70, help="ngưỡng toxic mặc định lưu vào meta")

    args = ap.parse_args()

    train_csv = args.train_csv or f"{args.data_dir}/train.csv"
    val_csv = args.val_csv or f"{args.data_dir}/val.csv"
    test_csv = args.test_csv or f"{args.data_dir}/test.csv"

    paths = Paths(
        train_csv=train_csv,
        val_csv=val_csv,
        test_csv=test_csv,
        model_out=args.model_out,
        meta_out=args.meta_out,
    )

    print("Loading splits:")
    print(f"  train: {paths.train_csv}")
    print(f"  val  : {paths.val_csv}")
    print(f"  test : {paths.test_csv}")

    X_train, y_train = load_split(paths.train_csv, args.text_col, args.label_col)
    X_val, y_val = load_split(paths.val_csv, args.text_col, args.label_col)
    X_test, y_test = load_split(paths.test_csv, args.text_col, args.label_col)

    print(f"Sizes: train={len(X_train)}, val={len(X_val)}, test={len(X_test)}")

    pipe = build_pipeline(C=args.C)

    print("Training...")
    pipe.fit(X_train, y_train)

    val_macro = eval_split("VAL", pipe, X_val, y_val)
    test_macro = eval_split("TEST", pipe, X_test, y_test)

    # Save model
    joblib.dump(pipe, paths.model_out)

    # Save meta
    labels = sorted(list(set(list(y_train) + list(y_val) + list(y_test))))
    meta = {
        "labels": labels,
        "threshold_toxic": float(args.threshold),
        "model_out": paths.model_out,
        "train_csv": paths.train_csv,
        "val_csv": paths.val_csv,
        "test_csv": paths.test_csv,
        "text_col": args.text_col,
        "label_col": args.label_col,
        "C": float(args.C),
        "val_macro_f1": float(val_macro),
        "test_macro_f1": float(test_macro),
        "notes": "Pipeline: word+char TF-IDF + LinearSVC(class_weight=balanced) + CalibratedClassifierCV(sigmoid, cv=3)",
    }

    with open(paths.meta_out, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"\n✅ Saved model: {paths.model_out}")
    print(f"✅ Saved meta : {paths.meta_out}")


if __name__ == "__main__":
    main()
