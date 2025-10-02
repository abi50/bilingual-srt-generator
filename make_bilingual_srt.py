#!/usr/bin/env python3
# -*- coding: utf-8 -*-
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

try:
    import argostranslate.package
    import argostranslate.translate
    HAVE_ARGOS = True
except Exception:
    HAVE_ARGOS = False


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




def transcribe_to_segments(media_path: str, trg_lang: str|None ,src_lang: str|None, model_size="small",prefer_via_english=True):
    from faster_whisper import WhisperModel
    LOCAL_MODEL_DIR = r"D:\models\faster-whisper-small"

    src_lang = _norm_lang(src_lang)
    trg_lang = _norm_lang(trg_lang) or "en"
    try:
        model = WhisperModel(LOCAL_MODEL_DIR, compute_type="float32", local_files_only=True)
    except Exception:
        model = WhisperModel(model_size, device="auto") 
    same_lang = (src_lang is not None and trg_lang == src_lang)
    use_translate = (trg_lang == "en") or (prefer_via_english and trg_lang != "en" and not same_lang)


    task = "translate" if use_translate else "transcribe"

    segments, info = model.transcribe(
        media_path,
        task=task,
        beam_size=7,
        language=src_lang,
        vad_filter=True,
        temperature=[0.0, 0.2, 0.4, 0.6],  
        compression_ratio_threshold=2.4, 
        vad_parameters=dict(
            threshold=0.5,                
            min_speech_duration_ms=250,   
            min_silence_duration_ms=500,  
            speech_pad_ms=200,         
        ),
        condition_on_previous_text=True
    )

    segments_text_lang = "en" if task == "translate" else info.language

    out = []
    for i, seg in enumerate(segments, start=1):
        out.append({
            "index": i,
            "start": dt.timedelta(seconds=float(seg.start)),
            "end": dt.timedelta(seconds=float(seg.end)),
            "text": seg.text.strip()
        })
    return out, info.language, segments_text_lang


def make_bilingual_srt(media_path: str, out_path: str,trg_lang: str ,src_lang: str):
    segments, detected_src, seg_text_lang = transcribe_to_segments(
        media_path, trg_lang, src_lang, model_size="small", prefer_via_english=True
    )

    subs = []
    for i, seg in enumerate(segments, start=1):
        base_text = seg["text"]
        tgt_text, out_lang = ensure_translation(base_text, seg_text_lang, trg_lang)
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
