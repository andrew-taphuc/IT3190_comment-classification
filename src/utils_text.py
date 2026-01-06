"""
Các utility functions cho xử lý text.
Lưu ý: Các hàm làm sạch chính được định nghĩa trong 03_clean_text.py
File này giữ lại để tương thích ngược nếu có code cũ sử dụng.
"""

TEENCODE = {
    "ko": "không",
    "k": "không",
    "hok": "không",
    "khong": "không",
    "dc": "được",
    "đc": "được",
    "mn": "mọi người",
    "mik": "mình",
    "mk": "mình",
    "thik": "thích",
    "iu": "yêu",
    # Các từ chửi thề (để normalize)
    "vcl": "rất",
    "vl": "rất",
    "vãi": "rất",
    "đm": "chửi",
    "dm": "chửi",
    "dmm": "chửi",
    "cc": "chửi",
    "clm": "chửi",
    "cmm": "chửi",
    "địt": "chửi",
}

def normalize_teencode(text: str) -> str:
    """
    Chuẩn hóa teen code sang từ chuẩn.
    Lưu ý: Hàm này được đồng bộ với 03_clean_text.py
    """
    words = text.split()
    out = [TEENCODE.get(w, w) for w in words]
    return " ".join(out)
