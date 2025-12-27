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
from dotenv import load_dotenv
load_dotenv()  # טוען משתני סביבה מקובץ .env לתהליך
# טוען את .env גם מתקיית הEXE
base_path = getattr(sys, "_MEIPASS", os.path.dirname(os.path.abspath(__file__)))
env_path = os.path.join(base_path, ".env")
if os.path.exists(env_path):
    load_dotenv(env_path, override=True)

import datetime as dt
from typing import List
from faster_whisper import WhisperModel
import re
import requests

import xml.sax.saxutils as xml_escape
import math

try:
    import deepl
    HAVE_DEEPL = True
except Exception:
    HAVE_DEEPL = False


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

def _bilingual_content(src_text: str, tgt_text: str, tgt_lang: str | None) -> str:
    
    rtl = {"he", "ar", "fa", "ur"}
    is_rtl = (_norm_lang(tgt_lang) in rtl)
    if is_rtl:
        tgt_text = f"\u202B{tgt_text}\u202C"  # עטיפת RTL לשורה התחתונה
    return f"{src_text}\n{tgt_text}"

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


_DEEPL_LANG_MAP = {
    "en": "EN", "he": "HE", "iw": "HE", "fr": "FR", "ar": "AR",
    "ru": "RU", "es": "ES", "de": "DE", "it": "IT", "pt": "PT-PT",
    "tr": "TR", "uk": "UK", "pl": "PL", "nl": "NL", "zh": "ZH", "ja": "JA", "ko": "KO"
}
def _deepl_code(lang: str | None) -> str | None:
    if not lang: return None
    v = _norm_lang(lang)
    if not v: return None
    return _DEEPL_LANG_MAP.get(v)


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

def translate_segments_deepl_all(segments, src_lang: str | None, tgt_lang: str) -> list[str]:
    """
    מתרגם את *כל הטקסט* של הכתוביות בבת אחת עם הקשר,
    תוך שמירת שיוך מדויק לכל סגמנט ע"י תגיות XML <s i="...">...</s>.
    מפצל לבאצ'ים אם ארוך מדי.

    מחזיר: רשימת מחרוזות מתורגמות באותו סדר ובדיוק באורך len(segments).
    """
    if not HAVE_DEEPL:
        raise RuntimeError("deepl package not installed")

    auth = os.environ.get("DEEPL_API_KEY") or os.environ.get("DEEPL_AUTH_KEY")
    if not auth:
        raise RuntimeError("DEEPL_API_KEY env var not set")

    translator = deepl.Translator(auth)

    src = _deepl_code(src_lang)  # יכול להיות None (זיהוי אוטומטי)
    tgt = _deepl_code(tgt_lang) or "EN"

    # בונים XML: כל סגמנט עטוף בתגית <s i="index">text</s>
    # כדי להימנע מבעיות XML – עושים escape לתוכן.
    # נבצע פיצול לבאצ'ים ~20k תווים בטקסט כדי לא לחרוג ממגבלות.
    MAX_CHARS = 20000

    def build_xml(batch):
        parts = []
        parts.append("<doc>")
        for seg in batch:
            i = seg["index"]
            txt = seg["text"] or ""
            txt = xml_escape.escape(txt)
            parts.append(f'<s i="{i}">{txt}</s>')
        parts.append("</doc>")
        return "".join(parts)

    # נרוץ על הסגמנטים וניצור באצ'ים לפי ערך אורך מצטבר משוער של ה-XML
    batches = []
    cur, cur_len = [], 0
    for seg in segments:
        # הערכת תוספת אורך (תגיות + טקסט)
        add_len = len(seg["text"]) + 40
        if cur and (cur_len + add_len > MAX_CHARS):
            batches.append(cur)
            cur, cur_len = [], 0
        cur.append(seg)
        cur_len += add_len
    if cur:
        batches.append(cur)

    # נתרגם כל באץ' בבת אחת עם tag_handling=xml כדי לשמר את המבנה
    translated_by_index = {}
    for batch in batches:
        xml_text = build_xml(batch)
        result = translator.translate_text(
            xml_text,
            source_lang=src, target_lang=tgt,
            tag_handling="xml",
            split_sentences="nonewlines",
            preserve_formatting=True,
            outline_detection=False,  # אל תנסו לנחש מבנה כותרות וכו'
        )
        # כעת צריך "לחלץ" חזרה את הטקסט מתוך ה-XML המוחזר (התגיות נשמרות)
        # DeepL מחזיר טקסט עם אותן תגיות <s i="...">...</s>
        out = result.text

        # נחלץ באמצעות regex פשוט – נזהר שלא לשבור RTL:
        import re
        pattern = re.compile(r'<s i="(\d+)">(.*?)</s>', flags=re.DOTALL)
        for m in pattern.finditer(out):
            idx = int(m.group(1))
            inner = m.group(2)
            # להחזיר תווי XML לאחור:
            inner = inner.replace("&lt;","<").replace("&gt;",">").replace("&amp;","&")
            translated_by_index[idx] = inner.strip()

    # מייצרים רשימה מתורגמת בסדר נכון; אם משהו חסר – נחזיר ריק
    out_list = []
    for seg in segments:
        out_list.append(translated_by_index.get(seg["index"], ""))

    return out_list


MAX_DURATION = 3.0  

def _split_by_time(words, max_dur=MAX_DURATION):
    """מקבל words עם start/end ומחזיר רשימת תתי-סגמנטים קצרים"""
    chunks, cur_words = [], []
    seg_start = None
    for w in words:
        if seg_start is None:
            seg_start = w.start
        cur_words.append(w)
        if (w.end - seg_start) >= max_dur :
            chunks.append((seg_start, w.end, " ".join(x.word for x in cur_words).strip()))
            cur_words, seg_start = [], None
    if cur_words:
        chunks.append((seg_start, cur_words[-1].end, " ".join(x.word for x in cur_words).strip()))
    return chunks


def transcribe_to_segments(media_path: str, trg_lang: str|None ,src_lang: str|None, model_size="medium",prefer_via_english=True):
    from faster_whisper import WhisperModel
    base_path = getattr(sys, '_MEIPASS', os.path.dirname(os.path.abspath(__file__)))
    LOCAL_MODEL_DIR = os.path.join(base_path, 'models', 'faster-whisper-medium')


    
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
        word_timestamps=True,
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

    out = []
    
    for i, seg in enumerate(segments, start=1):
        if hasattr(seg, "words") and seg.words:
            for (st, en, txt) in _split_by_time(seg.words, MAX_DURATION):
                out.append({
                    "index": len(out)+1,
                    "start": dt.timedelta(seconds=st),
                    "end": dt.timedelta(seconds=en),
                    "text": txt,
                })
        else:
            out.append({
                "index": len(out)+1,
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
        # === תרגום כל הקובץ בבת אחת (עם הקשר) באמצעות DeepL, עם נפילה ל-Argos אם אין API ===
    try:
        translated_texts = translate_segments_deepl_all(segments, seg_text_lang, trg_lang)
        out_lang = trg_lang  # יעד בפועל
        use_deepl = True
        print("[deepl] used for whole-file translation")
    except Exception as e:
        print(f"[deepl] fallback to per-line: {e}")
        translated_texts = []
        use_deepl = False

    subs = []
    for i, seg in enumerate(segments, start=1):
        base_text = seg["text"]
        if use_deepl:
            tgt_text = translated_texts[i-1] if i-1 < len(translated_texts) else ""
            out_lang = trg_lang
        else:
            # נפילה: תרגום שורה-שורה כבעבר
            tgt_text, out_lang = ensure_translation(base_text, seg_text_lang, trg_lang)

        # תיקון שגיאות כתיב בעברית (אם יעד עברית)
        if (_norm_lang(out_lang) == "he"):
            tgt_text = heb_spellfix(tgt_text)

        content = _bilingual_content(base_text, tgt_text, out_lang)
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
