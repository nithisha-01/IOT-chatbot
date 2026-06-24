"""
language_handler.py
Fast Arabic/English detection: checks Unicode Arabic block ratio first
(instant, no API call), falls back to langdetect for edge cases.
"""
import re
from langdetect import detect, LangDetectException

ARABIC_RE = re.compile(r"[\u0600-\u06FF]")


def detect_language(text: str, override: str = None) -> str:
    """Returns 'ar' or 'en'. `override` ('ar'/'en') wins if provided
    (manual pill toggle in the UI)."""
    if override in ("ar", "en"):
        return override

    if not text or not text.strip():
        return "en"

    arabic_chars = len(ARABIC_RE.findall(text))
    if arabic_chars > 0 and arabic_chars / max(len(text), 1) > 0.15:
        return "ar"

    try:
        lang = detect(text)
        return "ar" if lang == "ar" else "en"
    except LangDetectException:
        return "en"
