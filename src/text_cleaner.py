"""
Module chứa các hàm làm sạch text.
Module này được tạo để có thể pickle được khi lưu model với joblib.
"""
import re
import unicodedata

# Compiled regex patterns (hiệu quả hơn)
URL_RE = re.compile(r"(https?://\S+|www\.\S+)", re.IGNORECASE)
MENTION_RE = re.compile(r"@\w+")
HASHTAG_RE = re.compile(r"#\w+")
MULTISPACE_RE = re.compile(r"\s+")
REPEAT_CHAR_RE = re.compile(r"(.)\1{2,}")  # ký tự lặp >=3 lần

# teen-code mapping (mở rộng với các từ chửi thề)
TEENCODE = {
    "ko": "không",
    "k": "không",
    "khong": "không",
    "hok": "không",
    "dc": "được",
    "đc": "được",
    "mik": "mình",
    "mk": "mình",
    "mn": "mọi người",
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


def normalize_unicode(text: str) -> str:
    """Chuẩn hóa Unicode về dạng NFC."""
    return unicodedata.normalize("NFC", text)


def normalize_repeated_chars(text: str) -> str:
    """Chuẩn hóa ký tự lặp: 'đẹpppp' -> 'đẹpp' (giữ 2 ký tự)."""
    def _repl(m):
        ch = m.group(1)
        return ch * 2
    return REPEAT_CHAR_RE.sub(_repl, text)


def normalize_teencode(text: str) -> str:
    """Chuẩn hóa teen code sang từ chuẩn."""
    tokens = text.split()
    tokens = [TEENCODE.get(t, t) for t in tokens]
    return " ".join(tokens)


def clean_text(s: str) -> str:
    """
    Làm sạch text với các bước:
    1. Normalize Unicode (NFC)
    2. Lowercase
    3. Remove URLs, mentions, hashtags
    4. Normalize repeated characters
    5. Apply teen code mapping
    6. Keep only Vietnamese chars + numbers + basic punctuation
    7. Collapse multiple spaces
    """
    if not isinstance(s, str):
        s = "" if s is None else str(s)

    # Normalize Unicode
    s = normalize_unicode(s)
    s = s.lower()

    # Remove URLs, mentions, hashtags
    s = URL_RE.sub(" ", s)
    s = MENTION_RE.sub(" ", s)
    s = HASHTAG_RE.sub(" ", s)

    # Normalize repeated characters
    s = normalize_repeated_chars(s)

    # Apply teen code mapping
    s = normalize_teencode(s)

    # Keep Vietnamese chars + numbers + basic punctuation + một số ký tự đặc biệt
    s = re.sub(
        r"[^0-9a-zàáảãạăắằẳẵặâấầẩẫậèéẻẽẹêếềểễệ"
        r"ìíỉĩịòóỏõọôốồổỗộơớờởỡợùúủũụưứừửữự"
        r"ỳýỷỹỵđ\s\.\,\!\?\-_/]",
        " ",
        s,
    )

    # Collapse multiple spaces
    s = MULTISPACE_RE.sub(" ", s).strip()
    return s

