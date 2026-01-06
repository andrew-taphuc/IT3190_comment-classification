import pandas as pd
import os

# Import hàm làm sạch từ module text_cleaner
from text_cleaner import clean_text

IN_DIR = "data/interim"
OUT_DIR = "data/processed"
os.makedirs(OUT_DIR, exist_ok=True)

def process(split: str):
    in_path = f"{IN_DIR}/{split}.csv"
    out_path = f"{OUT_DIR}/{split}.csv"

    df = pd.read_csv(in_path).dropna(subset=["text", "label"]).copy()

    df["text"] = df["text"].astype(str).apply(clean_text)

    # drop empty + dedup
    df = df[df["text"].str.len() > 0].drop_duplicates(subset=["text"]).reset_index(drop=True)

    df.to_csv(out_path, index=False)
    print(f"Saved {out_path} | rows={len(df)}")
    print(df["label"].value_counts())

def main():
    for split in ["train", "val", "test"]:
        process(split)

if __name__ == "__main__":
    main()
