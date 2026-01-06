from datasets import load_dataset
import pandas as pd
import os

os.makedirs("data/raw", exist_ok=True)

# Map label_id -> label text (theo ViHSD phổ biến)
LABEL_3 = {0: "CLEAN", 1: "OFFENSIVE", 2: "HATE"}

def to_csv(ds_split, out_path):
    df = pd.DataFrame(ds_split)

    # Detect text column
    if "text" in df.columns:
        text_col = "text"
    elif "free_text" in df.columns:
        text_col = "free_text"
    else:
        raise ValueError(f"Cannot find text column. Columns: {list(df.columns)}")

    # Detect label column
    if "label" in df.columns:
        label_col = "label"
    elif "label_id" in df.columns:
        label_col = "label_id"
    else:
        raise ValueError(f"Cannot find label column. Columns: {list(df.columns)}")

    out = df[[text_col, label_col]].rename(columns={text_col: "text", label_col: "label"})

    # If label is numeric, map to class names
    if pd.api.types.is_numeric_dtype(out["label"]):
        out["label"] = out["label"].map(LABEL_3)

    out.to_csv(out_path, index=False)
    print(f"Saved {out_path} | rows={len(out)} | cols={list(out.columns)}")

def main():
    ds = load_dataset("uitnlp/vihsd")
    print(ds)

    to_csv(ds["train"], "data/raw/train.csv")
    to_csv(ds["validation"], "data/raw/validation.csv")
    to_csv(ds["test"], "data/raw/test.csv")

if __name__ == "__main__":
    main()
