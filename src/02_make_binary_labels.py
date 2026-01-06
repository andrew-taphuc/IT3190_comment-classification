import pandas as pd
import os

RAW_DIR = "data/raw"
OUT_DIR = "data/interim"
os.makedirs(OUT_DIR, exist_ok=True)

def make_binary(in_path: str, out_path: str):
    df = pd.read_csv(in_path)

    # label hiện tại sau bước download đã là CLEAN/OFFENSIVE/HATE (string)
    # nếu vẫn là số thì bạn có thể map ở đây (phòng hờ)
    label_map_3 = {0: "CLEAN", 1: "OFFENSIVE", 2: "HATE"}
    if df["label"].dtype != object:
        df["label"] = df["label"].map(label_map_3)

    df["label"] = df["label"].apply(lambda x: "non_toxic" if x == "CLEAN" else "toxic")

    df.to_csv(out_path, index=False)
    print(f"Saved {out_path} | rows={len(df)}")
    print(df["label"].value_counts())

def main():
    make_binary(f"{RAW_DIR}/train.csv", f"{OUT_DIR}/train.csv")
    make_binary(f"{RAW_DIR}/validation.csv", f"{OUT_DIR}/val.csv")
    make_binary(f"{RAW_DIR}/test.csv", f"{OUT_DIR}/test.csv")

if __name__ == "__main__":
    main()
