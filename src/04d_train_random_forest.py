# 04d_train_random_forest.py
import os
import pandas as pd
import joblib
import json

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

from core import clean_text, print_evaluation_report, evaluate_model

os.makedirs("outputs", exist_ok=True)

TRAIN_PATH = "data/processed/train.csv"
VAL_PATH   = "data/processed/val.csv"
TEST_PATH  = "data/processed/test.csv"
MODEL_NAME = "RandomForest"

def load_split(path: str):
    df = pd.read_csv(path).dropna(subset=["text", "label"]).copy()
    df["text"] = df["text"].astype(str)
    df["label"] = df["label"].astype(str)
    return df

def main():
    print(f"\n{'='*60}")
    print(f"TRAINING MODEL: {MODEL_NAME}")
    print(f"{'='*60}")
    
    # Load data
    train_df = load_split(TRAIN_PATH)
    val_df   = load_split(VAL_PATH)
    test_df  = load_split(TEST_PATH)

    X_train, y_train = train_df["text"], train_df["label"]
    X_val, y_val     = val_df["text"], val_df["label"]
    X_test, y_test   = test_df["text"], test_df["label"]

    # TF-IDF settings
    tfidf = TfidfVectorizer(
        preprocessor=clean_text,
        ngram_range=(1, 2),
        max_features=50000,
        min_df=2
    )

    # Model
    clf = RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced_subsample"
    )
    
    # Pipeline
    pipe = Pipeline([
        ("tfidf", tfidf),
        ("clf", clf)
    ])

    # Train
    print("\n🚀 Training model...")
    pipe.fit(X_train, y_train)
    print("✅ Training completed!")

    # Evaluate
    labels = sorted(list(set(y_train)))
    
    # Validation
    val_pred = pipe.predict(X_val)
    val_proba = pipe.predict_proba(X_val) if hasattr(pipe, 'predict_proba') else None
    val_metrics = evaluate_model(y_val, val_pred, val_proba, labels)
    
    # Test
    test_pred = pipe.predict(X_test)
    test_proba = pipe.predict_proba(X_test) if hasattr(pipe, 'predict_proba') else None
    test_metrics = evaluate_model(y_test, test_pred, test_proba, labels)
    
    # Print reports
    print_evaluation_report(y_val, val_pred, val_proba, labels, split_name=f"{MODEL_NAME} - VALIDATION")
    print_evaluation_report(y_test, test_pred, test_proba, labels, split_name=f"{MODEL_NAME} - TEST")

    # Save model
    model_path = f"outputs/{MODEL_NAME.lower()}_pipeline.joblib"
    joblib.dump(pipe, model_path)
    print(f"\n💾 Saved model: {model_path}")

    # Save metrics
    metrics = {
        "model": MODEL_NAME,
        "val_metrics": {k: v for k, v in val_metrics.items() if k != 'confusion_matrix'},
        "test_metrics": {k: v for k, v in test_metrics.items() if k != 'confusion_matrix'},
    }
    
    metrics_path = f"outputs/{MODEL_NAME.lower()}_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print(f"💾 Saved metrics: {metrics_path}")

    # Return for comparison
    return {
        "model": MODEL_NAME,
        "val_acc": val_metrics.get('accuracy', 0),
        "val_macro_f1": val_metrics.get('macro_f1', 0),
        "val_roc_auc": val_metrics.get('roc_auc', None),
        "test_acc": test_metrics.get('accuracy', 0),
        "test_macro_f1": test_metrics.get('macro_f1', 0),
        "test_roc_auc": test_metrics.get('roc_auc', None),
    }

if __name__ == "__main__":
    main()

