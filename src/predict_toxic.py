# predict_toxic.py
import json
import argparse
import sys
import joblib

# Import text_cleaner để đảm bảo module có sẵn khi load model
# (model được lưu với hàm clean_text từ module này)
import text_cleaner


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="toxicity_pipeline.joblib")
    ap.add_argument("--meta", default="toxicity_meta.json")
    ap.add_argument("--text", default=None, help="Text cần predict. Nếu không có, đọc từ stdin.")
    ap.add_argument("--threshold", type=float, default=None, help="Override threshold toxic")
    args = ap.parse_args()

    pipe = joblib.load(args.model)

    meta = {}
    try:
        with open(args.meta, "r", encoding="utf-8") as f:
            meta = json.load(f)
    except FileNotFoundError:
        pass

    threshold = args.threshold
    if threshold is None:
        threshold = meta.get("threshold_toxic", 0.70)

    text = args.text
    if text is None:
        # đọc toàn bộ stdin
        text = sys.stdin.read().strip()

    if not text:
        print(json.dumps({"error": "empty_text"}, ensure_ascii=False))
        return

    # predict_proba: [p(class0), p(class1)] theo thứ tự pipe.classes_
    proba = pipe.predict_proba([text])[0]
    classes = list(pipe.classes_)

    # lấy p(toxic) theo đúng class name
    if "toxic" in classes:
        toxic_idx = classes.index("toxic")
    else:
        # fallback: nếu bạn dùng nhãn khác, lấy class cuối
        toxic_idx = -1

    toxic_score = float(proba[toxic_idx])
    label = "toxic" if toxic_score >= threshold else "non_toxic"

    out = {
        "label": label,
        "toxic_score": toxic_score,
        "threshold": threshold,
        "classes": classes,
        "proba": [float(x) for x in proba],
    }
    print(json.dumps(out, ensure_ascii=False))


if __name__ == "__main__":
    main()
