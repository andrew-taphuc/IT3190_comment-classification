"""
Script để tìm hyperparameters tối ưu cho model.
"""
import json
import argparse
import joblib
import pandas as pd
import numpy as np

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import make_scorer, f1_score

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core import clean_text, DEFAULT_MODEL_CONFIG, ModelConfig, evaluate_model


def load_split(path: str, text_col: str = "text", label_col: str = "label"):
    df = pd.read_csv(path)
    if text_col not in df.columns or label_col not in df.columns:
        raise ValueError(f"{path} phải có cột '{text_col}' và '{label_col}'")
    
    df = df.dropna(subset=[text_col, label_col]).copy()
    X = df[text_col].astype(str).values
    y = df[label_col].astype(str).values
    return X, y


def build_pipeline_for_tuning():
    """Xây dựng pipeline để tuning (không có CalibratedClassifierCV để nhanh hơn)."""
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
    
    feats = FeatureUnion([
        ("word_tfidf", word_tfidf),
        ("char_tfidf", char_tfidf),
    ])
    
    base_svm = LinearSVC(class_weight="balanced", random_state=42, max_iter=3000)
    
    return Pipeline([
        ("features", feats),
        ("clf", base_svm),
    ])


def main():
    ap = argparse.ArgumentParser(description="Hyperparameter tuning")
    ap.add_argument("--data_dir", default="data/processed")
    ap.add_argument("--train_csv", default=None)
    ap.add_argument("--val_csv", default=None)
    ap.add_argument("--text_col", default="text")
    ap.add_argument("--label_col", default="label")
    ap.add_argument("--n_jobs", type=int, default=-1)
    ap.add_argument("--cv", type=int, default=3, help="Số folds cho cross-validation")
    ap.add_argument("--method", choices=["grid", "random"], default="random",
                   help="Grid search hoặc random search")
    ap.add_argument("--n_iter", type=int, default=20,
                   help="Số iterations cho random search")
    ap.add_argument("--output", default="outputs/best_params.json")
    
    args = ap.parse_args()
    
    train_csv = args.train_csv or f"{args.data_dir}/train.csv"
    val_csv = args.val_csv or f"{args.data_dir}/val.csv"
    
    print("Loading data...")
    X_train, y_train = load_split(train_csv, args.text_col, args.label_col)
    X_val, y_val = load_split(val_csv, args.text_col, args.label_col)
    
    # Combine train và val để có nhiều data hơn cho CV
    X_combined = np.concatenate([X_train, X_val])
    y_combined = np.concatenate([y_train, y_val])
    
    print(f"Training size: {len(X_combined)}")
    
    # Ensure output directory exists
    import os
    os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else "outputs", exist_ok=True)
    
    pipe = build_pipeline_for_tuning()
    
    # Parameter grid
    param_grid = {
        'clf__C': [0.5, 1.0, 1.5, 2.0, 2.5, 3.0],
        'features__word_tfidf__max_features': [30000, 40000, 50000],
        'features__word_tfidf__ngram_range': [(1, 1), (1, 2), (1, 3)],
    }
    
    # Scorer
    scorer = make_scorer(f1_score, average='macro')
    
    print(f"\n🔍 Tuning hyperparameters (method={args.method})...")
    
    if args.method == "grid":
        search = GridSearchCV(
            pipe,
            param_grid,
            cv=args.cv,
            scoring=scorer,
            n_jobs=args.n_jobs,
            verbose=2,
        )
    else:
        search = RandomizedSearchCV(
            pipe,
            param_grid,
            n_iter=args.n_iter,
            cv=args.cv,
            scoring=scorer,
            n_jobs=args.n_jobs,
            verbose=2,
            random_state=42,
        )
    
    search.fit(X_combined, y_combined)
    
    print(f"\n✅ Best parameters:")
    print(json.dumps(search.best_params_, indent=2))
    print(f"\n✅ Best CV score: {search.best_score_:.4f}")
    
    # Evaluate on validation set
    best_model = search.best_estimator_
    val_pred = best_model.predict(X_val)
    val_metrics = evaluate_model(y_val, val_pred, None, sorted(list(set(y_val))))
    
    print(f"\n📊 Validation metrics với best params:")
    print(f"  Accuracy: {val_metrics['accuracy']:.4f}")
    print(f"  Macro F1: {val_metrics['macro_f1']:.4f}")
    
    # Save results
    # Xử lý NaN: chuyển thành None để JSON hợp lệ
    best_cv_score = search.best_score_
    if pd.isna(best_cv_score) or np.isnan(best_cv_score):
        best_cv_score_json = None
    else:
        best_cv_score_json = float(best_cv_score)
    
    results = {
        "best_params": search.best_params_,
        "best_cv_score": best_cv_score_json,
        "val_metrics": {k: v for k, v in val_metrics.items() if k != 'confusion_matrix'},
        "cv_folds": args.cv,
        "method": args.method,
    }
    
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"\n💾 Saved results to: {args.output}")


if __name__ == "__main__":
    main()

