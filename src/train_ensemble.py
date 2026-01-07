"""
Script train model với ensemble methods để cải thiện hiệu quả.
"""
import json
import argparse
import joblib
import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import VotingClassifier, RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import cross_val_score

from core import clean_text, DEFAULT_MODEL_CONFIG, ModelConfig, print_evaluation_report


def load_split(path: str, text_col: str = "text", label_col: str = "label"):
    df = pd.read_csv(path)
    if text_col not in df.columns or label_col not in df.columns:
        raise ValueError(f"{path} phải có cột '{text_col}' và '{label_col}'")
    
    df = df.dropna(subset=[text_col, label_col]).copy()
    X = df[text_col].astype(str).values
    y = df[label_col].astype(str).values
    return X, y


def build_ensemble_pipeline(config: ModelConfig = None) -> Pipeline:
    """
    Xây dựng ensemble pipeline với nhiều base models.
    """
    if config is None:
        config = DEFAULT_MODEL_CONFIG
    
    # TF-IDF features
    word_tfidf = TfidfVectorizer(
        preprocessor=clean_text,
        analyzer="word",
        ngram_range=config.word_ngram_range,
        min_df=config.min_df,
        max_df=config.max_df,
        max_features=config.max_features,
        sublinear_tf=config.sublinear_tf,
    )
    
    char_tfidf = TfidfVectorizer(
        preprocessor=clean_text,
        analyzer="char",
        ngram_range=config.char_ngram_range,
        min_df=config.min_df,
        max_df=config.max_df,
    )
    
    feats = FeatureUnion([
        ("word_tfidf", word_tfidf),
        ("char_tfidf", char_tfidf),
    ])
    
    # Base models
    # LinearSVC không có predict_proba, cần wrap trong CalibratedClassifierCV
    svm_base = LinearSVC(
        C=config.svm_C,
        class_weight=config.svm_class_weight,
        random_state=config.random_state,
        max_iter=3000
    )
    svm = CalibratedClassifierCV(
        estimator=svm_base,
        method='sigmoid',
        cv=3
    )
    
    lr = LogisticRegression(
        C=config.svm_C * 0.5,
        class_weight=config.svm_class_weight,
        random_state=config.random_state,
        max_iter=3000
    )
    
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=20,
        class_weight=config.svm_class_weight,
        random_state=config.random_state,
        n_jobs=-1
    )
    
    # Ensemble với voting
    voting_clf = VotingClassifier(
        estimators=[
            ('svm', svm),
            ('lr', lr),
            ('rf', rf),
        ],
        voting='soft',  # Sử dụng predict_proba
        weights=[2, 1, 1]  # Ưu tiên SVM hơn
    )
    
    # Calibrate để có probabilities tốt hơn
    clf = CalibratedClassifierCV(
        estimator=voting_clf,
        method=config.calibration_method,
        cv=config.calibration_cv,
    )
    
    return Pipeline([
        ("features", feats),
        ("clf", clf),
    ])


def main():
    ap = argparse.ArgumentParser(description="Train ensemble model")
    ap.add_argument("--data_dir", default="data/processed")
    ap.add_argument("--train_csv", default=None)
    ap.add_argument("--val_csv", default=None)
    ap.add_argument("--test_csv", default=None)
    ap.add_argument("--text_col", default="text")
    ap.add_argument("--label_col", default="label")
    ap.add_argument("--model_out", default="outputs/toxicity_ensemble.joblib")
    ap.add_argument("--meta_out", default="outputs/toxicity_ensemble_meta.json")
    ap.add_argument("--C", type=float, default=2.0)
    ap.add_argument("--threshold", type=float, default=0.70)
    
    args = ap.parse_args()
    
    train_csv = args.train_csv or f"{args.data_dir}/train.csv"
    val_csv = args.val_csv or f"{args.data_dir}/val.csv"
    test_csv = args.test_csv or f"{args.data_dir}/test.csv"
    
    print("Loading splits...")
    X_train, y_train = load_split(train_csv, args.text_col, args.label_col)
    X_val, y_val = load_split(val_csv, args.text_col, args.label_col)
    X_test, y_test = load_split(test_csv, args.text_col, args.label_col)
    
    print(f"Sizes: train={len(X_train)}, val={len(X_val)}, test={len(X_test)}")
    
    # Ensure output directory exists
    import os
    os.makedirs(os.path.dirname(args.model_out) if os.path.dirname(args.model_out) else "outputs", exist_ok=True)
    
    # Build config
    model_config = ModelConfig(
        svm_C=args.C,
        default_threshold=args.threshold,
    )
    
    pipe = build_ensemble_pipeline(config=model_config)
    
    print("\n🚀 Training ensemble model...")
    pipe.fit(X_train, y_train)
    print("✅ Training completed!")
    
    # Evaluate
    val_metrics = print_evaluation_report(y_val, pipe.predict(X_val), 
                                         pipe.predict_proba(X_val), 
                                         sorted(list(set(y_val))), 
                                         "VALIDATION")
    test_metrics = print_evaluation_report(y_test, pipe.predict(X_test),
                                          pipe.predict_proba(X_test),
                                          sorted(list(set(y_test))),
                                          "TEST")
    
    # Save model
    joblib.dump(pipe, args.model_out)
    
    # Save meta
    labels = sorted(list(set(list(y_train) + list(y_val) + list(y_test))))
    meta = {
        "labels": labels,
        "threshold_toxic": float(args.threshold),
        "model_out": args.model_out,
        "model_type": "ensemble",
        "config": {
            "svm_C": float(model_config.svm_C),
            "word_ngram_range": list(model_config.word_ngram_range),
            "char_ngram_range": list(model_config.char_ngram_range),
        },
        "metrics": {
            "val": {k: v for k, v in val_metrics.items() if k != 'confusion_matrix'},
            "test": {k: v for k, v in test_metrics.items() if k != 'confusion_matrix'},
        },
        "notes": "Ensemble: VotingClassifier(SVM + LR + RF) + CalibratedClassifierCV",
    }
    
    with open(args.meta_out, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    
    print(f"\n✅ Saved model: {args.model_out}")
    print(f"✅ Saved meta : {args.meta_out}")


if __name__ == "__main__":
    main()

