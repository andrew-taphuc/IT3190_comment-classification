"""
Module chứa các hàm làm sạch text.
Module này được tạo để có thể pickle được khi lưu model với joblib.
"""
import re
import unicodedata
from .teencode_mapping import TEENCODE

# Compiled regex patterns (hiệu quả hơn)
URL_RE = re.compile(r"(https?://\S+|www\.\S+)", re.IGNORECASE)
MENTION_RE = re.compile(r"@\w+")
HASHTAG_RE = re.compile(r"#\w+")
MULTISPACE_RE = re.compile(r"\s+")
REPEAT_CHAR_RE = re.compile(r"(.)\1{2,}")  # ký tự lặp >=3 lần
EMOJI_RE = re.compile(
    "["
    "\U0001F600-\U0001F64F"  # emoticons
    "\U0001F300-\U0001F5FF"  # symbols & pictographs
    "\U0001F680-\U0001F6FF"  # transport & map symbols
    "\U0001F1E0-\U0001F1FF"  # flags (iOS)
    "\U00002702-\U000027B0"
    "\U000024C2-\U0001F251"
    "]+",
    flags=re.UNICODE
)
PUNCTUATION_RE = re.compile(r"[!?.]{2,}")  # Nhiều dấu câu liên tiếp


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


def normalize_emoji(text: str) -> str:
    """Chuẩn hóa emoji: thay thế bằng từ mô tả hoặc xóa."""
    # Thay emoji bằng khoảng trắng (có thể mở rộng để map sang từ)
    return EMOJI_RE.sub(" ", text)


def normalize_punctuation(text: str) -> str:
    """Chuẩn hóa dấu câu lặp: '!!!' -> '!'."""
    return PUNCTUATION_RE.sub(lambda m: m.group(0)[0], text)


def clean_text(s: str) -> str:
    """
    Làm sạch text với các bước:
    1. Normalize Unicode (NFC)
    2. Lowercase
    3. Remove URLs, mentions, hashtags
    4. Normalize emoji
    5. Normalize repeated characters
    6. Normalize punctuation
    7. Apply teen code mapping
    8. Keep only Vietnamese chars + numbers + basic punctuation
    9. Collapse multiple spaces
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

    # Normalize emoji
    s = normalize_emoji(s)

    # Normalize repeated characters
    s = normalize_repeated_chars(s)

    # Normalize punctuation
    s = normalize_punctuation(s)

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

