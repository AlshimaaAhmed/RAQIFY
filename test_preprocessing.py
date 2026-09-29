# ==============================
# IMPORTS
# ==============================
import re
import emoji

# ==============================
# 1. BASIC CLEANING
# ==============================
def basic_cleaning(text):
    if not text:
        return ""

    text = emoji.replace_emoji(text, replace='')
    text = re.sub(r"\[\d{2}/\d{2},.*?\]", "", text)
    text = re.sub(r"[!\"#$%&'()*+,\-./;<=>?@\[\\\]^_`{|}~]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    return text


# ==============================
# 2. REMOVE REPEATED CHARACTERS
# ==============================
def normalize_repeated_chars(text):
    return re.sub(r'(.)\1{2,}', r'\1\1', text)


# ==============================
# 3. ARABIC NORMALIZATION
# ==============================
def normalize_arabic(text):
    text = re.sub("[إأآا]", "ا", text)
    text = re.sub("ى", "ي", text)
    text = re.sub("ة", "ه", text)
    text = re.sub(r"[ًٌٍَُِّْـ]", "", text)
    return text


# ==============================
# 4. ENGLISH NORMALIZATION
# ==============================
def normalize_english(text):
    return re.sub(r"[A-Za-z]+", lambda match: match.group(0).lower(), text)


# ==============================
# 5. PROCESS SINGLE TEXT INPUT
# ==============================
def process_single_text(text):
    text = basic_cleaning(text)
    text = normalize_repeated_chars(text)
    text = normalize_arabic(text)
    text = normalize_english(text)
    return text


# ==============================
# 6. MAIN PIPELINE
# ==============================
def text_preprocessing_pipeline(text):
    if not text:
        return ""
    
    return process_single_text(text)