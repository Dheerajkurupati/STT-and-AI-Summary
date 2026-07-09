from deep_translator import GoogleTranslator
from backend.utils import get_logger

logger = get_logger(__name__)

def translate_to_english(text: str) -> str:
    """Translate arbitrary text to English using deep-translator (Google Translate)."""
    if not text or not text.strip():
        return text

    try:
        # GoogleTranslator auto-detects source language by default and translates to target
        translated = GoogleTranslator(source='auto', target='en').translate(text)
        return translated if translated else text
    except Exception as exc:
        logger.error(f"Translation failed: {exc}")
        return text
