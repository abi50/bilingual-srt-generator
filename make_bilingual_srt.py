#!/usr/bin/env python3
#  coding: utf-8 
# Generate bilingual (English + Hebrew) subtitles in SRT format.
# - Transcribes English speech from an input media file using Whisper (open-source).
# - Translates each subtitle line to Hebrew using Argos Translate (offline-capable).
# - Writes UTF-8 SRT with two lines per cue: English on top, Hebrew below.
#
# USAGE (Windows PowerShell):
#   python -m venv .venv
#   .\.venv\Scripts\Activate.ps1
#   pip install --upgrade pip wheel
#   pip install faster-whisper argostranslate moviepy srt
#   # Install Argos Hebrew model (en->he) via:
#   #   python -m argostranslate.gui
#   # Then run:
#   python make_bilingual_srt.py "Alcohol is AMAZING.mp4" "Alcohol_is_AMAZING.srt"
#
# NOTES:
#   - First run downloads models (allowlist GitHub/HuggingFace if needed).
#   - After caching, it works offline.

import os
import sys
import srt
import datetime as dt
from typing import List
from faster_whisper import WhisperModel
import re
import requests


try:
    import argostranslate.package
    import argostranslate.translate
    HAVE_ARGOS = True
except Exception:
    HAVE_ARGOS = False

try:
    from symspellpy import SymSpell, Verbosity
    HAVE_SYMSPELL = True
except Exception:
    print("HAVE_SYMSPELL not imported")
    HAVE_SYMSPELL = False
    
SPELL_BACKEND = os.environ.get("SPELL_BACKEND", "lt").lower()
_HE_WORD_RE = re.compile(r"[\u0590-\u05FF]+(?:[׳'״-][\u0590-\u05FF]+)?")
_HE_LETTER_RE = re.compile(r"[\u0590-\u05FF]")
_HE_KEEP = {"AWS","API","GPU","CUDA","Docker","FFmpeg","Python","YouTube","S3","ID","URL"}

def _is_hebrew_word(tok: str) -> bool:
    return bool(_HE_LETTER_RE.search(tok)) and tok not in _HE_KEEP

_SYM = None

def _init_symspell(dict_path: str | None = None):
    global _SYM
    if not HAVE_SYMSPELL or _SYM is not None:

        return
    sym = SymSpell(max_dictionary_edit_distance=2)
    if dict_path and os.path.exists(dict_path):
        sym.load_dictionary(dict_path, term_index=0, count_index=1, encoding="utf-8")
    else:
        for w in ["שלום", "תודה", "מקצועי", "מחשב", "פרויקט", "וידאו", "כתוביות", "עברית", "אנגלית"]:
            sym.create_dictionary_entry(w, 10)
    _SYM = sym

def heb_spellfix_languagetool(text: str) -> str:
  
    try:
        # שרת ציבורי של LT: יש מגבלות קצב. אפשר גם להקים מקומית אם תרצי.
        url = "https://api.languagetool.org/v2/check"
        # כדי לשמור על פיסוק, נעבוד על כל הכתובית כיחידה אחת (קצרה בדרך כלל) 
        data = {
            "language": "he",
            "text": text
        }
        r = requests.post(url, data=data, timeout=8)
        if r.status_code != 200:
            print(f"[spell-lt] HTTP {r.status_code} – skipping")
            return text
        out = text
        matches = r.json().get("matches", [])
       
        for m in sorted(matches, key=lambda x: x.get("offset", 0), reverse=True):
            repls = m.get("replacements", [])
            if not repls:
                continue
            best = repls[0]["value"]
            off = m.get("offset", 0)
            length = m.get("length", 0)
            out = out[:off] + best + out[off+length:]
        return out

    except Exception as e:
        print(f"[spell-lt] fail: {e}")
        return text


def heb_spellfix_symspell(line: str) -> str:
    if not HAVE_SYMSPELL:
        return line
    _init_symspell()
    tokens = re.findall(r"\w+|[^\w\s]", line, flags=re.UNICODE)
    out = []
    for tok in tokens:
        if _is_hebrew_word(tok):
            sugg = _SYM.lookup(tok, Verbosity.CLOSEST, max_edit_distance=2)
            out.append(sugg[0].term if sugg else tok)
        else:
            out.append(tok)

    fixed = ""
    for i, t in enumerate(out):
        if i and (_HE_WORD_RE.match(t) and _HE_WORD_RE.match(out[i-1])):
            fixed += " " + t
        elif i and (_HE_WORD_RE.match(t) and out[i-1].isalnum()):
            fixed += " " + t
        else:
            fixed += t
    return fixed


def heb_spellfix(text: str) -> str:
    if not text or not text.strip():
        return text
    if SPELL_BACKEND == "lt":
        fixed = heb_spellfix_languagetool(text)
        if fixed != text:
            print("[spell] LT: applied")
        else:
            print("[spell] LT: no changes")
        return fixed
    elif SPELL_BACKEND == "symspell":
        if not HAVE_SYMSPELL:
            print("[spell] SymSpell not available → skip")
            return text
        fixed = heb_spellfix_symspell(text)
        if fixed != text:
            print("[spell] SymSpell: applied")
        else:
            print("[spell] SymSpell: no changes")
        return fixed
    else:
        print("[spell] disabled")
        return text


def _norm_lang(language: str | None) -> str | None:
 
    if not language:
        return None

    v = language.strip().lower()
    if v in {"", "auto", "none", "null", "undefined"}:
        return None

    aliases = {
        "en": {"en", "eng", "english", "angielski", "inglés"},
        "he": {"he", "iw", "heb", "hebrew", "עברית"},
        "fr": {"fr", "fra", "fre", "french", "français", "francais", "france", "צרפתית"},
        "ar": {"ar", "ara", "arabic", "العربية", "ערבית"},
        "ru": {"ru", "rus", "russian", "русский", "רוסית"},
        "es": {"es", "spa", "spanish", "español", "ספרדית"},
        "de": {"de", "ger", "deu", "german", "deutsch", "גרמנית"},
        "it": {"it", "ita", "italian", "italiano", "איטלקית"},
        "pt": {"pt", "por", "portuguese", "português", "פורטוגזית"},
        "tr": {"tr", "tur", "turkish", "türkçe", "טורקית"},
        "uk": {"uk", "ukr", "ukrainian", "українська", "אוקראינית"},
        "pl": {"pl", "pol", "polish", "polski", "פולנית"},
        "nl": {"nl", "dut", "nld", "dutch", "nederlands", "הולנדית"},
        "zh": {"zh", "zho", "chi", "chinese", "中文", "סינית"},
        "ja": {"ja", "jpn", "japanese", "日本語", "יפנית"},
        "ko": {"ko", "kor", "korean", "한국어", "קוריאנית"},
    }

    if v in aliases.keys():
        return v

    for key, names in aliases.items():
        if v in names:
            return key

    return None


def ensure_translation(text: str, src_lang: str | None, tgt_lang: str | None) -> tuple[str, str | None]:
    if not text.strip():
        return "",src_lang

    src = _norm_lang(src_lang)
    tgt = _norm_lang(tgt_lang) or "en"

    if src == tgt:
        return text, src

    if not HAVE_ARGOS:
        return text, src

    try:
        out= argostranslate.translate.translate(text, src or "" , tgt)
        # out= argot.translate(text, src or "", tgt)
        if out and out.strip() and out.strip() != text.strip():
            return out, tgt
        return text, src
    except Exception:
        return text, src




def transcribe_to_segments(media_path: str, trg_lang: str|None ,src_lang: str|None, model_size="medium",prefer_via_english=True):
    from faster_whisper import WhisperModel
    LOCAL_MODEL_DIR = r"D:\models\faster-whisper-small"


    
    src_lang = _norm_lang(src_lang)
    trg_lang = _norm_lang(trg_lang) or "en"
    try:
        model = WhisperModel(model_size, device="auto", compute_type="float32", local_files_only=False) 
        print("online model")
    except Exception:
        model = WhisperModel(LOCAL_MODEL_DIR, device="auto", compute_type="float32", local_files_only=True)
        print("locall model")


    same_lang = (src_lang is not None and trg_lang == src_lang)
    use_translate = (trg_lang == "en") or (prefer_via_english and trg_lang != "en" and not same_lang)


    task = "translate" if use_translate else "transcribe"

    initial_prompt = None
    if (task == "transcribe" and (src_lang == "he")):
        initial_prompt = ", מתקלקל היא אומרת שלום שלום וברוכים הבאים. היום נדבר על תהליך העבודה, בדיקות ופרקטיקות טובות."

    segments, info = model.transcribe(
        media_path,
        task=task,
        beam_size=9,
        language=src_lang,
        vad_filter=True,
        temperature=[0.0, 0.2, 0.4, 0.6],  
        compression_ratio_threshold=2.4, 
        vad_parameters=dict(
            threshold=0.5,                
            min_speech_duration_ms=900,   
            min_silence_duration_ms=1200,  
            speech_pad_ms=200,         
        ),
        condition_on_previous_text=True,
        initial_prompt=initial_prompt

    )

    segments_text_lang = "en" if task == "translate" else info.language
    MAX_DURATION = 3.0 
    out = []
    for i, seg in enumerate(segments, start=1):

      
        out.append({
            "index": i,
            "start": dt.timedelta(seconds=seg.start),
            "end": dt.timedelta(seconds=seg.end),
            "text": seg.text.strip()
        })
    return out, info.language, segments_text_lang


def make_bilingual_srt(media_path: str, out_path: str, trg_lang: str, src_lang: str):
    segments, detected_src, seg_text_lang = transcribe_to_segments(
        media_path, trg_lang, src_lang, model_size="medium", prefer_via_english=True
    )

    subs = []
    for i, seg in enumerate(segments, start=1):
        base_text = seg["text"]
        tgt_text, out_lang = ensure_translation(base_text, seg_text_lang, trg_lang)
        if (_norm_lang(out_lang) == "he"):
            tgt_text = heb_spellfix(tgt_text)

        content= _wrap_rtl_if_needed(tgt_text, out_lang)
        # if you want 2 lines-  source text and translated text: 
        # content = f"{_wrap_rtl_if_needed(base_text, seg_text_lang)}\n{_wrap_rtl_if_needed(tgt_text, out_lang)}"

        subs.append(srt.Subtitle(index=i, start=seg["start"], end=seg["end"], content=content))
   
    srt_text = srt.compose(subs)
    with open(out_path, "w", encoding="utf-8", newline="\n") as f:
        f.write(srt_text)

    print(f"Detected source language: {detected_src}")
    print(f"Segments language: {seg_text_lang} (depends on task)")
    print(f"Wrote SRT: {out_path}")


def _wrap_rtl_if_needed(text: str, lang: str | None) -> str:

    rtl = {"he", "ar", "fa", "ur"}
    if _norm_lang(lang) in rtl:
        return f"\u202B{text}\u202C"
    return text

def _segments_coverage_seconds(segments) -> float:
    last_end = 0.0
    for s in segments:
        try:
            last_end = float(getattr(s, "end", 0.0))  
        except Exception:
            pass
    return last_end

# אפשרות: קדם־עיבוד קל לאודיו
if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python make_bilingual_srt.py <input_media> <output_srt> [<target_lang>=en] [<source_lang>=auto]")
        sys.exit(1)
    inp = sys.argv[1]
    out = sys.argv[2]
    trg_lang = sys.argv[3] if len(sys.argv) >= 4 else "en"
    src_lang = sys.argv[4] if len(sys.argv) >= 5 and sys.argv[4].strip() else None
    make_bilingual_srt(inp, out, trg_lang, src_lang)
