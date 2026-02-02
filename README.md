# make_bilingual_srt

A command-line tool for generating SRT subtitles from video files, with support for transcription and optional translation into multiple languages.

The tool is based on OpenAI Whisper for transcription and can optionally use DeepL for higher-quality translations.  
It is designed to work reliably even when external translation services are unavailable.

---

## Purpose

- Generate SRT subtitle files from video/audio files
- Transcribe speech in the source language
- Optionally translate subtitles into another language
- Remain stable and usable even without internet access or external translation APIs

---

## Usage (Windows executable)

```powershell
.\make_bilingual_srt.exe <video_file> <output_srt> <target_language> [source_language]
```

### Examples

Transcription only (Hebrew → Hebrew):
```powershell
.\make_bilingual_srt.exe video.mp4 subtitles.srt he he
```

Translate to another language:
```powershell
.\make_bilingual_srt.exe video.mp4 subtitles.srt en
```

```powershell
.\make_bilingual_srt.exe video.mp4 subtitles.srt fr
```

---

## Language Support

The tool supports multiple languages, depending on the capabilities of:

- **Whisper** – for transcription and translation to English
- **DeepL** – for high-quality translation to supported target languages

Actual supported languages depend on the underlying engines.

---

## DeepL (Optional)

For higher-quality translations, the tool can use DeepL.

To enable DeepL translation, define a system environment variable:

```powershell
setx DEEPL_API_KEY "YOUR_API_KEY"
```

After setting the variable, open a new PowerShell window before running the tool again.

If DeepL is unavailable (e.g., due to network or SSL issues), the tool automatically falls back to an internal translation mechanism and continues running without crashing.

---

## Notes

- No API keys are embedded in the code or executable.
- Each user is responsible for providing their own DeepL API key.
- The executable file (EXE) is distributed separately via GitHub Releases.

---

## License

Free to use and adapt as needed.
