#!/usr/bin/env python3
"""
transcriber.py - Speech-to-Text & Hinglish Transliteration Engine.

Unifies Whisper audio transcription with Gemini LLM Hinglish transliteration,
high-performance local rule-based fallback, rate limiting, and persistent disk caching.

Key Invariants:
1. Exact Whisper word-level timestamps ('start', 'end') preserved without drift.
2. Standardized google-genai SDK using gemini-3.5-flash-lite and structured schema list[Item(i, r)].
3. Never passes thinking_config(thinking_budget=0) to prevent HTTP 400 errors.
4. Robust fallback key loading: os.getenv("GEMINI_API_KEY") -> graceful local rule-based fallback.
5. Strict Latin-only guarantee: Devanagari regex check and ASCII sanitization prevent tofu boxes (□□□).
6. Rate limiting (default 8.0 RPM) and disk caching in ~/.cache/hinglish_captions/.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import string
import time
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

# ----------------- Strict Latin & Alphabet Constants -----------------
DEV = re.compile(r"[\u0900-\u097F]")
PUNCT = " .,?!।॥\"'“”‘’-–—…:;()[]{}"
ALLOWED_CHARS = set(string.ascii_letters + string.digits + "'- ")
DEVANAGARI_DIGITS = str.maketrans("०१२३४५६७८९", "0123456789")

CACHE_DIR = Path.home() / ".cache" / "hinglish_captions"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_WHISPER_INITIAL_PROMPT = (
    "Ye ek Hindi/Hinglish lecture video hai jisme teacher exam, padhai, concepts aur motivation ke baare mein baat kar rahe hain."
)

# ----------------- Comprehensive Hinglish Dictionary -----------------
# Merged from high-frequency conversational particles, pronouns, auxiliaries,
# and common modern loanwords (Whisper often outputs English loanwords in Devanagari).
_RAW_DICT = {
    # Pronouns & Particles
    "वो": "wo", "वह": "woh", "ये": "ye", "यह": "yeh", "है": "hai", "हैं": "hain", "था": "tha",
    "थे": "the", "थी": "thi", "हो": "ho", "होगा": "hoga", "होगी": "hogi", "होंगे": "honge",
    "का": "ka", "की": "ki", "के": "ke", "को": "ko", "ने": "ne", "से": "se", "में": "mein",
    "पर": "par", "पे": "pe", "और": "aur", "भी": "bhi", "तो": "toh", "कि": "ki", "कुछ": "kuch",
    "नहीं": "nahi", "ना": "na", "मैं": "main", "मैंने": "maine", "हम": "hum", "आप": "aap",
    "आपके": "aapke", "आपका": "aapka", "आपकी": "aapki", "अपने": "apne", "अपना": "apna",
    "अपनी": "apni", "मुझे": "mujhe", "मेरा": "mera", "मेरी": "meri", "मेरे": "mere",
    "तुम": "tum", "तुम्हारे": "tumhare", "तुम्हारी": "tumhari", "तुम्हारा": "tumhara",
    "तुझे": "tujhe", "तू": "tu", "क्या": "kya", "क्यों": "kyun", "क्योंकि": "kyunki",
    "कैसे": "kaise", "कौन": "kaun", "कब": "kab", "कहाँ": "kahan", "कहां": "kahan",
    "कितना": "kitna", "कितनी": "kitni", "कितने": "kitne", "कोई": "koi", "किसी": "kisi",
    "लोगों": "logon", "लोग": "log", "साथ": "saath", "जब": "jab", "तब": "tab", "अब": "ab",
    "जो": "jo", "जैसे": "jaise", "ऐसे": "aise", "वैसे": "waise", "इस": "is", "उस": "us",
    "इन": "in", "उन": "un", "इसमें": "ismein", "उसमें": "usmein", "इसलिए": "isliye",
    "लेकिन": "lekin", "मगर": "magar", "अगर": "agar", "बहुत": "bahut", "बस": "bas", "सब": "sab",
    "सबको": "sabko", "यहाँ": "yahan", "यहां": "yahan", "वहाँ": "wahan", "वहां": "wahan",
    "अभी": "abhi", "फिर": "phir", "ही": "hi", "तक": "tak", "लिए": "liye", "बाद": "baad",
    "पहले": "pehle", "हाँ": "haan", "हां": "haan", "जी": "ji", "ठीक": "theek", "अच्छा": "achha",
    "सही": "sahi", "गलत": "galat", "चाहिए": "chahiye", "सकते": "sakte", "सकता": "sakta",
    "सकती": "sakti", "रहा": "raha", "रहे": "rahe", "रही": "rahi", "करना": "karna",
    "करो": "karo", "करें": "karen", "करते": "karte", "करता": "karta", "करती": "karti",
    "कर": "kar", "करके": "karke", "किया": "kiya", "गया": "gaya", "गए": "gaye", "गई": "gayi",
    "दो": "do", "एक": "ek", "तीन": "teen", "चार": "chaar", "पांच": "paanch", "दोनों": "dono",
    "देखो": "dekho", "देखते": "dekhte", "देखा": "dekha", "सुनो": "suno", "समझ": "samajh",
    "समझना": "samajhna", "समझे": "samjhe", "बोलो": "bolo", "बताना": "batana", "बताओ": "batao",
    "चलो": "chalo", "आओ": "aao", "साल": "saal", "दिन": "din", "समय": "samay", "वक्त": "waqt",
    "वक़्त": "waqt", "बात": "baat", "काम": "kaam", "मतलब": "matlab", "एकदम": "ekdam",
    "सर": "sir", "मेहनत": "mehnat", "सपना": "sapna", "बारी": "baari", "सच": "sach",
    "दिखाना": "dikhana", "ईमानदारी": "imandari", "इमानदारी": "imandari", "अगले": "agle",

    # Common English Loanwords transcribed in Devanagari
    "इवेंट": "event", "मोमेंट": "moment", "फंक्शन": "function", "एग्जाम": "exam",
    "सेलेक्ट": "select", "पेरेंट्स": "parents", "स्टेज": "stage", "क्लास": "class",
    "क्वेश्चन": "question", "प्रॉब्लम": "problem", "सॉल्यूशन": "solution",
    "वीडियो": "video", "स्टूडेंट": "student", "टीचर": "teacher", "स्कूल": "school",
    "कॉलेज": "college", "टॉपिक": "topic", "कांसेप्ट": "concept", "प्लान": "plan",
    "चैप्टर": "chapter", "नंबर": "number", "रिजल्ट": "result", "मार्क": "mark"
}

LOCAL_DICT: Dict[str, str] = {unicodedata.normalize("NFC", k): v for k, v in _RAW_DICT.items()}

# ----------------- Modern Hindi Phonetic Consonants & Matras -----------------
CONSONANTS: Dict[str, str] = dict(zip(
    "कखगघङचछजझञटठडढणतथदधनपफबभमयरलवशषसहळ",
    ["k", "kh", "g", "gh", "n", "ch", "chh", "j", "jh", "n",
     "t", "th", "d", "dh", "n", "t", "th", "d", "dh", "n",
     "p", "ph", "b", "bh", "m", "y", "r", "l", "v", "sh", "sh", "s", "h", "l"]
))

NUKTA_MAP: Dict[str, str] = {
    "क": "q", "ख": "kh", "ग": "g", "ज": "z", "ड": "r", "ढ": "rh", "फ": "f", "य": "y"
}

MATRAS: Dict[str, str] = {
    "ा": "a", "ि": "i", "ी": "ee", "ु": "u", "ू": "oo", "ृ": "ri",
    "े": "e", "ै": "ai", "ो": "o", "ौ": "au", "ॉ": "o", "ॅ": "e"
}

INDEPENDENT_VOWELS: Dict[str, str] = {
    "अ": "a", "आ": "aa", "इ": "i", "ई": "ee", "उ": "u", "ऊ": "oo", "ऋ": "ri",
    "ए": "e", "ऐ": "ai", "ओ": "o", "औ": "au", "ऑ": "o"
}


def sanitize_latin(text: str) -> str:
    """
    Hard sanitizer ensuring the output contains ONLY Latin characters, numbers,
    apostrophes, hyphens, and spaces. Permanently prevents tofu boxes (□□□).
    """
    if not text:
        return ""
    # Normalize unicode to decomposed form and replace typographer quotes/dashes
    s = unicodedata.normalize("NFKD", text)
    s = s.replace("’", "'").replace("‘", "'").replace("`", "'")
    s = s.replace("–", "-").replace("—", "-")
    # Retain strictly allowed characters
    s = "".join(ch for ch in s if ch in ALLOWED_CHARS)
    # Collapse multiple whitespaces and trim
    return re.sub(r"\s+", " ", s).strip()


def devanagari_to_roman(word: str) -> str:
    """
    Local rule-based phonetic romanization for modern Hindi tokens,
    incorporating schwa deletion, nuktas, matras, and nasalization.
    """
    if word in LOCAL_DICT:
        return LOCAL_DICT[word]

    units: List[Dict[str, Any]] = []
    i = 0
    word_len = len(word)

    while i < word_len:
        ch = word[i]
        if ch in CONSONANTS:
            c = CONSONANTS[ch]
            if i + 1 < word_len and word[i + 1] == "\u093c":  # Nukta mark
                c = NUKTA_MAP.get(ch, c)
                i += 1
            units.append({"c": c, "v": None, "nas": False, "vis": False})
        elif ch in MATRAS and units:
            units[-1]["v"] = MATRAS[ch]
        elif ch == "\u094d" and units:  # Halant (virama)
            units[-1]["v"] = ""
        elif ch in "\u0902\u0901" and units:  # Anusvara / Chandrabindu
            units[-1]["nas"] = True
        elif ch == "\u0903" and units:  # Visarga
            units[-1]["vis"] = True
        elif ch in INDEPENDENT_VOWELS:
            units.append({"c": "", "v": INDEPENDENT_VOWELS[ch], "nas": False, "vis": False})
        i += 1

    n = len(units)
    if n == 0:
        return ""

    # Schwa deletion rule at the word ending
    if n > 1 and units[-1]["v"] is None:
        units[-1]["v"] = ""

    # Medial schwa deletion for unstressed syllables
    for idx in range(n - 2, 0, -1):
        if units[idx]["v"] is None and units[idx - 1]["v"] != "" and units[idx + 1]["v"] != "":
            units[idx]["v"] = ""

    out_parts: List[str] = []
    for idx, u in enumerate(units):
        part = u["c"] + ("a" if u["v"] is None else u["v"])
        if u["nas"]:
            nxt = units[idx + 1]["c"] if idx + 1 < n else ""
            part += "m" if nxt[:1] in ("p", "b", "m") else "n"
        if u["vis"]:
            part += "h"
        out_parts.append(part)

    return "".join(out_parts)


def local_romanize(token: str) -> str:
    """
    Transliterate a single token using the local dictionary and phonetic engine.
    Always returns a sanitized Latin string.
    """
    cleaned = unicodedata.normalize("NFC", token).translate(DEVANAGARI_DIGITS).strip(PUNCT)
    if not cleaned:
        return ""
    if cleaned in LOCAL_DICT:
        return LOCAL_DICT[cleaned]

    # Split token into Devanagari and non-Devanagari segments
    segments = re.findall(r"[\u0900-\u097F]+|[^\u0900-\u097F]+", cleaned)
    romanized_segments = [
        devanagari_to_roman(seg) if DEV.match(seg) else seg
        for seg in segments
    ]
    result = "".join(romanized_segments)
    return sanitize_latin(result)


# ----------------- Gemini LLM Transliteration Engine -----------------
SYSTEM_INSTRUCTION = """You romanize Whisper speech tokens from a Hindi/Hinglish video (any subject: school, college, coding, finance, motivation, cooking, etc.) into natural texting-style Hinglish for Reel captions.

INPUT: JSON with 'topic_context' (opening words of the video), 'nearby_context' (tokens around this batch), optional 'reference_script' and 'glossary', and 'items': [{i, t}].
OUTPUT: for EVERY item return {i, r}, where r is the romanization of that single token t.

RULES:
- Roman (Latin) letters only. Never output Devanagari or any other script.
- One output per input item. Never merge, split, drop, add or reorder items. Keep each 'i' unchanged.
- Spell the way Indians type on WhatsApp: wo, ye, hai, hain, nahi, kya, kyun, mein, aap, logon, waqt, dekha, tha, baari. Never Sanskrit/ITRANS style.
- Do not translate to English. Romanize what was said. But English words that the speaker used (Whisper wrote them in Devanagari, e.g. इवेंट, मोमेंट, फंक्शन, एग्जाम) must return in normal English spelling (event, moment, function, exam).
- Infer the subject from the context and spell domain terms in their standard English form (e.g. maths: function, domain, range; coding: variable, loop).
- If a token is already Latin, keep it, only fixing obvious case.
- No punctuation. Lowercase, except acronyms (JEE, NEET, IIT, AI).
- If a token looks like a Whisper mishearing, use the context to pick the most plausible word, but still output exactly one item for it."""


@dataclass
class WordCaption:
    """Word-level caption entry with exact start/end timestamps."""
    word: str
    start: float
    end: float
    raw_token: str


class HinglishTranscriber:
    """
    Speech transcription and Hinglish transliteration pipeline.
    Combines Whisper word timestamps with Gemini LLM Hinglish transliteration
    and robust local rule-based fallback.
    """

    def __init__(
        self,
        whisper_model: str = "large-v3",
        llm_model: Optional[str] = None,
        api_key: Optional[str] = None,
        rpm: float = 8.0,
        batch_size: int = 50,
        no_llm: bool = False,
        uppercase: bool = True
    ) -> None:
        self.whisper_model_name = whisper_model
        self.llm_model_name = llm_model or os.getenv("HINGLISH_MODEL", "gemini-3.5-flash-lite")
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        self.rpm = max(rpm, 1.0)
        self.batch_size = batch_size
        self.no_llm = no_llm
        self.uppercase = uppercase
        self._last_call_time = 0.0

    def _call_gemini_batch(
        self,
        client: Any,
        cfg: Any,
        batch_tokens: List[Tuple[int, str]],
        topic_context: str,
        nearby_context: str,
        glossary: str,
        script_hint: str
    ) -> Dict[int, str]:
        """Call Gemini API for a single token batch with structured JSON schema."""
        todo = batch_tokens
        payload: Dict[str, Any] = {
            "topic_context": topic_context,
            "nearby_context": nearby_context,
            "items": [{"i": i, "t": t} for i, t in todo]
        }
        if glossary:
            payload["glossary"] = glossary
        if script_hint:
            payload["reference_script"] = script_hint[:4000]

        gap = 60.0 / self.rpm
        elapsed = time.time() - self._last_call_time
        if elapsed < gap:
            time.sleep(gap - elapsed)
        self._last_call_time = time.time()

        response = client.models.generate_content(
            model=self.llm_model_name,
            contents=json.dumps(payload, ensure_ascii=False),
            config=cfg
        )

        results: Dict[int, str] = {}
        raw_items = json.loads(response.text)
        todo_indices = {i for i, _ in todo}

        for item in raw_items:
            i = item.get("i")
            r = item.get("r")
            if i in todo_indices and isinstance(r, str) and not DEV.search(r):
                sanitized = sanitize_latin(r)
                if sanitized:
                    results[i] = sanitized
        return results

    def romanize_tokens(
        self,
        tokens: List[str],
        glossary: str = "",
        script_hint: str = ""
    ) -> Dict[int, str]:
        """
        Romanize an array of speech tokens.
        Checks persistent cache first, calls Gemini with rate limiting and retries,
        and falls back gracefully to local rule-based transliteration.
        """
        romanized_map: Dict[int, str] = {}
        if not tokens:
            return romanized_map

        use_gemini = not self.no_llm and bool(self.api_key)
        client = None
        cfg = None

        if not self.api_key and not self.no_llm:
            print("[romanize] GEMINI_API_KEY not found in environment -> proceeding with local phonetic romanizer.")
            use_gemini = False

        if use_gemini:
            try:
                from google import genai
                from google.genai import types
                from pydantic import BaseModel

                class Item(BaseModel):
                    i: int
                    r: str

                client = genai.Client(api_key=self.api_key, http_options=types.HttpOptions(timeout=60000))
                # Never pass thinking_config(thinking_budget=0) to prevent HTTP 400 errors
                cfg = types.GenerateContentConfig(
                    system_instruction=SYSTEM_INSTRUCTION,
                    temperature=0.0,
                    response_mime_type="application/json",
                    response_schema=list[Item]
                )
            except Exception as e:
                print(f"[romanize] Failed to initialize google-genai client ({e}) -> local fallback only.")
                use_gemini = False

        topic_context = " ".join(tokens[:150])
        total_tokens = len(tokens)

        for start_idx in range(0, total_tokens, self.batch_size):
            end_idx = min(start_idx + self.batch_size, total_tokens)
            batch_indices = list(range(start_idx, end_idx))
            batch_tokens = [tokens[i] for i in batch_indices]
            nearby_context = " ".join(tokens[max(0, start_idx - 200): min(total_tokens, end_idx + 200)])

            # Persistent Cache Lookup
            cache_key = hashlib.sha256(
                json.dumps(
                    ["v5", self.llm_model_name, batch_tokens, glossary, script_hint[:2000]],
                    ensure_ascii=False
                ).encode("utf-8")
            ).hexdigest()[:24]
            cache_file = CACHE_DIR / f"{cache_key}.json"

            if cache_file.exists():
                try:
                    cached_words = json.loads(cache_file.read_text(encoding="utf-8"))
                    if len(cached_words) == len(batch_indices):
                        romanized_map.update(dict(zip(batch_indices, cached_words)))
                        continue
                except Exception:
                    pass

            batch_results: Dict[int, str] = {}
            if use_gemini and client and cfg:
                for attempt in range(4):
                    todo = [(i, tokens[i]) for i in batch_indices if i not in batch_results]
                    if not todo:
                        break
                    try:
                        got = self._call_gemini_batch(
                            client=client,
                            cfg=cfg,
                            batch_tokens=todo,
                            topic_context=topic_context,
                            nearby_context=nearby_context,
                            glossary=glossary,
                            script_hint=script_hint
                        )
                        batch_results.update(got)
                    except Exception as exc:
                        err_str = str(exc)
                        print(f"[romanize] Batch [{start_idx}:{end_idx}] attempt {attempt + 1} failed: {type(exc).__name__}: {err_str[:120]}")
                        if any(token in err_str.upper() for token in ["INVALID_ARGUMENT", "NOT_FOUND", "API_KEY", "PERMISSION"]):
                            print("[romanize] Permanent API error detected -> disabling LLM, switching to local fallback.")
                            use_gemini = False
                            break
                        if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str:
                            backoff = 20.0 * (attempt + 1)
                            print(f"[romanize] Rate limit encountered. Backing off for {backoff:.0f}s...")
                            time.sleep(backoff)
                        elif "503" in err_str or "UNAVAILABLE" in err_str:
                            backoff = 2.0 * (attempt + 1)
                            print(f"[romanize] Service temporarily unavailable (503). Retrying in {backoff:.0f}s...")
                            time.sleep(backoff)

            # Check if all items in batch succeeded and write to cache
            if len(batch_results) == len(batch_indices):
                cache_file.write_text(
                    json.dumps([batch_results[i] for i in batch_indices], ensure_ascii=False),
                    encoding="utf-8"
                )

            romanized_map.update(batch_results)

        return romanized_map

    def transcribe_and_romanize(
        self,
        video_path: Path | str,
        glossary: str = "",
        script_hint: str = "",
        whisper_prompt: Optional[str] = None
    ) -> List[WordCaption]:
        """
        Transcribe video audio using Whisper with word timestamps,
        then romanize Devanagari tokens to Hinglish.
        Guarantees 100% Latin ASCII words.
        """
        import whisper

        path_obj = Path(video_path).expanduser().resolve()
        if not path_obj.is_file():
            raise FileNotFoundError(f"Video file not found: {path_obj}")

        print(f"[transcribe] Loading Whisper model '{self.whisper_model_name}'...")
        wm = whisper.load_model(self.whisper_model_name)

        prompt = whisper_prompt.strip() if whisper_prompt else DEFAULT_WHISPER_INITIAL_PROMPT
        if glossary and glossary.strip():
            prompt = f"{prompt} Key terms: {glossary.strip()}"

        transcribe_kwargs: Dict[str, Any] = {
            "language": "hi",
            "task": "transcribe",
            "word_timestamps": True,
            "condition_on_previous_text": False,
            "no_speech_threshold": 0.3,
            "initial_prompt": prompt,
            "fp16": False  # Stability invariant on Apple Silicon
        }

        print(f"[transcribe] Transcribing audio with word-level timestamps from: {path_obj.name} (no_speech_thr=0.3, prompt_len={len(prompt)})...")
        whisper_result = wm.transcribe(str(path_obj), **transcribe_kwargs)

        # Extract words from segments without timestamp drift
        raw_words: List[Dict[str, Any]] = []
        for segment in whisper_result.get("segments", []):
            for word_info in segment.get("words", []):
                cleaned_word = word_info.get("word", "").strip()
                if cleaned_word:
                    raw_words.append({
                        "word": cleaned_word,
                        "start": float(word_info.get("start", 0.0)),
                        "end": float(word_info.get("end", 0.0))
                    })

        tokens = [w["word"] for w in raw_words]
        total_tokens = len(tokens)
        print(f"[romanize] Transcribed {total_tokens} words. Processing Hinglish transliteration...")

        llm_map = self.romanize_tokens(tokens, glossary=glossary, script_hint=script_hint)

        output_words: List[WordCaption] = []
        llm_count = 0
        fallback_count = 0

        for i, w_info in enumerate(raw_words):
            raw_token = tokens[i]
            roman_word = llm_map.get(i)
            if roman_word:
                llm_count += 1
            else:
                roman_word = local_romanize(raw_token)
                fallback_count += 1

            # Strict Latin sanitization guarantee
            roman_word = sanitize_latin(roman_word)
            if not roman_word:
                roman_word = sanitize_latin(local_romanize(raw_token))

            if self.uppercase:
                roman_word = roman_word.upper()

            # Stability Invariant: Permanently block Devanagari from reaching the video renderer
            assert not DEV.search(roman_word), f"Devanagari character leaked: {roman_word!r}"
            assert roman_word.isascii(), f"Non-ASCII character leaked: {roman_word!r}"

            output_words.append(WordCaption(
                word=roman_word,
                start=w_info["start"],
                end=w_info["end"],
                raw_token=raw_token
            ))

        print(f"[romanize] Completed transliteration: {llm_count} via Gemini, {fallback_count} via local engine.")
        return output_words
