"""
speech_emotion.py - M2 speech emotion perception via Zhizengzeng Qwen-Omni API.

Project contract:
    predict(audio_path) -> {
        "emotion": str,
        "confidence": float,
        "reasoning": str
    }

This implementation is laptop-friendly. It does not run a local audio model.
It calls the Zhizengzeng OpenAI-compatible API with qwen3-omni-flash,
then normalizes the model output into the shared emotion labels defined in config.py.

Required .env file in project root:
    ZHIZENGZENG_API_KEY=your_zhizengzeng_api_key
    ZHIZENGZENG_BASE_URL=https://api.zhizengzeng.com/v1
    AUDIO_MODEL=qwen3-omni-flash

Optional:
    SPEECH_MAX_AUDIO_BYTES=8388608
    TEST_AUDIO_PATH=test.wav

Install dependencies:
    pip install openai python-dotenv
"""

import base64
import json
import os
from pathlib import Path

from config import EMOTION_LABELS, MOCK_EMOTION
from dotenv import load_dotenv


# ============================================================
# API configuration
# ============================================================

ZHIZENGZENG_BASE_URL = os.getenv(
    "ZHIZENGZENG_BASE_URL",
    "https://api.zhizengzeng.com/v1",
)

AUDIO_MODEL = os.getenv(
    "AUDIO_MODEL",
    "qwen3-omni-flash",
)

MAX_AUDIO_BYTES = int(
    os.getenv("SPEECH_MAX_AUDIO_BYTES", str(8 * 1024 * 1024))
)


# ============================================================
# Emotion label mapping
# ============================================================

RAW_TO_CONFIG = {
    "neutral": "neutral",
    "calm": "neutral",
    "normal": "neutral",
    "unclear": "neutral",
    "ambiguous": "neutral",
    "confused": "neutral",
    "whisper": "neutral",
    "whispered": "neutral",

    "happy": "happy",
    "happiness": "happy",
    "joy": "happy",
    "joyful": "happy",
    "excited": "happy",
    "positive": "happy",
    "cheerful": "happy",

    "sad": "sad",
    "sadness": "sad",
    "depressed": "sad",
    "down": "sad",
    "low": "sad",

    "angry": "angry",
    "anger": "angry",
    "mad": "angry",
    "irritated": "angry",
    "annoyed": "angry",

    "fear": "fear",
    "fearful": "fear",
    "afraid": "fear",
    "anxious": "fear",
    "anxiety": "fear",
    "nervous": "fear",
    "worried": "fear",
    "panic": "fear",
    "panicked": "fear",

    "surprise": "surprise",
    "surprised": "surprise",
    "astonished": "surprise",

    "disgust": "disgust",
    "disgusted": "disgust",
    "contempt": "disgust",
}


# ============================================================
# Safe fallback
# ============================================================

def _fallback(reason):
    """Return a safe neutral result so app.py will not crash."""
    return {
        "emotion": MOCK_EMOTION,
        "confidence": 0.0,
        "reasoning": f"fallback: {reason}",
    }


# ============================================================
# Utility functions
# ============================================================

def _audio_format(path):
    """Infer audio format from file suffix."""
    suffix = Path(path).suffix.lower().lstrip(".")
    if suffix in {"wav", "mp3", "m4a", "flac", "ogg", "webm"}:
        return suffix
    return "wav"


def _audio_mime(fmt):
    """Map audio format to MIME type for data URL."""
    return {
        "wav": "audio/wav",
        "mp3": "audio/mpeg",
        "m4a": "audio/mp4",
        "flac": "audio/flac",
        "ogg": "audio/ogg",
        "webm": "audio/webm",
    }.get(fmt, "audio/wav")


def _audio_to_data_url(path):
    """
    Convert audio file to data URL.

    Some Qwen/DashScope-compatible gateways expect audio input to look like:
        data:audio/wav;base64,xxxxx
    instead of raw base64 only.
    """
    fmt = _audio_format(path)
    encoded_audio = base64.b64encode(Path(path).read_bytes()).decode("utf-8")
    return fmt, f"data:{_audio_mime(fmt)};base64,{encoded_audio}"


def _normalize_emotion(value):
    """Map raw model emotion label into config.EMOTION_LABELS."""
    value = str(value or "").strip().lower()
    mapped = RAW_TO_CONFIG.get(value, value)
    return mapped if mapped in EMOTION_LABELS else MOCK_EMOTION


def _safe_confidence(value):
    """Convert confidence into a safe float in [0, 1]."""
    try:
        value = float(value)
    except (TypeError, ValueError):
        return 0.0

    return round(min(max(value, 0.0), 1.0), 2)

def _calibrate_confidence(emotion, confidence, reasoning):
    """
    Calibrate over-confident neutral predictions.

    Some omni models tend to output neutral with very high confidence for
    acted or semantically ordinary speech. This is risky for multimodal fusion,
    because speech may dominate face emotion with an over-confident neutral.
    """
    reasoning_l = str(reasoning or "").lower()
    confidence = float(confidence)

    neutral_template_phrases = [
        "calm and matter-of-fact",
        "flat and matter-of-fact",
        "no emotional inflection",
        "emotionally neutral",
        "ordinary sentence",
        "ordinary content",
    ]

    if emotion == "neutral":
        if any(p in reasoning_l for p in neutral_template_phrases):
            confidence = min(confidence, 0.60)

    return round(min(max(confidence, 0.0), 1.0), 2)


def _extract_json(text):
    """
    Extract JSON from model output.

    The prompt asks the model to return JSON only, but some models may still add
    markdown or extra text. This function first tries direct json.loads, then
    extracts the first {...} block.
    """
    text = (text or "").strip()
    if not text:
        return {}

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        try:
            return json.loads(text[start:end + 1])
        except json.JSONDecodeError:
            return {}

    return {}


def _get_message_content(message):
    """
    Robustly read completion message content.

    Most OpenAI-compatible APIs return content as a string.
    Some multimodal APIs may return a list of content parts.
    """
    content = getattr(message, "content", None)

    if isinstance(content, str):
        return content

    if isinstance(content, list):
        pieces = []
        for item in content:
            if isinstance(item, dict):
                if item.get("type") == "text":
                    pieces.append(str(item.get("text", "")))
                elif "text" in item:
                    pieces.append(str(item.get("text", "")))
                elif "content" in item:
                    pieces.append(str(item.get("content", "")))
            else:
                pieces.append(str(item))
        return "\n".join(pieces).strip()

    return str(content or "")


# ============================================================
# API call
# ============================================================

def _call_qwen_omni_audio(audio_path):
    """
    Call Zhizengzeng OpenAI-compatible API with qwen3-omni-flash.

    This version first sends audio as OpenAI-style input_audio with a data URL.
    If the gateway does not accept that format, it tries a second common Qwen-style
    audio_url payload.
    """
    from dotenv import load_dotenv
    from openai import OpenAI

    load_dotenv()

    api_key = (
        os.getenv("ZHIZENGZENG_API_KEY")
        or os.getenv("OPENAI_API_KEY")
    )

    base_url = os.getenv(
        "ZHIZENGZENG_BASE_URL",
        ZHIZENGZENG_BASE_URL,
    )

    model = os.getenv(
        "AUDIO_MODEL",
        AUDIO_MODEL,
    )

    if not api_key:
        return _fallback("ZHIZENGZENG_API_KEY is not set")

    path = Path(audio_path)
    if not path.exists():
        return _fallback(f"audio file does not exist: {audio_path}")

    size = path.stat().st_size
    if size <= 0:
        return _fallback("audio file is empty")

    if size > MAX_AUDIO_BYTES:
        return _fallback(
            f"audio file is too large ({size} bytes); "
            f"limit is {MAX_AUDIO_BYTES} bytes"
        )

    fmt, audio_data_url = _audio_to_data_url(path)

    prompt = f"""
    You are a multimodal speech emotion recognition system for a real-world conversational AI.

    You will receive an audio clip of spoken dialogue.

    You must analyze emotion using BOTH:

    (1) Vocal cues:
    - pitch
    - energy
    - rhythm
    - pauses
    - tone
    - speaking speed
    - intensity
    - voice stability

    (2) Semantic cues (if speech is understandable):
    - implied meaning
    - conversational intent
    - emotional content of the sentence

    IMPORTANT:
    - Do NOT rely only on tone
    - Do NOT rely only on text meaning
    - Combine both sources

    Emotion labels:
    {", ".join(EMOTION_LABELS)}

    Decision rules:
    - happy → laughter, excitement, positive tone, enthusiasm
    - sad → low energy, slow speech, downward tone
    - angry → high energy, sharp tone, stress, forceful speech
    - fear → shaky voice, anxiety, uncertainty, nervousness
    - surprise → sudden pitch change, exclamation, abrupt reaction
    - disgust → rejection tone, aversion, negative reaction
    - neutral → only if BOTH tone and meaning are emotionally flat

    Confidence rules:
    - 0.8–0.95: very clear emotion
    - 0.5–0.8: moderate certainty
    - 0.2–0.5: ambiguous
    - NEVER default neutral with 0.95 unless extremely certain

    Return ONLY JSON:
    {{
    "emotion": "...",
    "confidence": 0.0,
    "reasoning": "short explanation using both audio and meaning cues"
    }}
    """

    print(
        f"[speech] calling model={model}, base_url={base_url}, "
        f"audio={path.name}, format={fmt}, size={size} bytes"
    )

    client = OpenAI(
        api_key=api_key,
        base_url=base_url,
        timeout=90.0,
    )

    # ------------------------------------------------------------
    # Attempt 1: OpenAI-style input_audio, but data is data URL
    # ------------------------------------------------------------
    try:
        completion = client.chat.completions.create(
            model=model,
            modalities=["text"],
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt,
                        },
                        {
                            "type": "input_audio",
                            "input_audio": {
                                "data": audio_data_url,
                                "format": fmt,
                            },
                        },
                    ],
                }
            ],
        )

        content = _get_message_content(completion.choices[0].message)
        data = _extract_json(content)

        emotion = _normalize_emotion(data.get("emotion"))
        confidence = _safe_confidence(data.get("confidence"))
        reasoning = str(
            data.get("reasoning")
            or content
            or "Qwen-Omni returned no reasoning."
        ).strip()
        confidence = _calibrate_confidence(emotion, confidence, reasoning)

        return {
            "emotion": emotion,
            "confidence": confidence,
            "reasoning": reasoning,
        }

    except Exception as first_exc:
        first_error = str(first_exc)

    # ------------------------------------------------------------
    # Attempt 2: Qwen/DashScope-style audio_url payload
    # ------------------------------------------------------------
    try:
        completion = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt,
                        },
                        {
                            "type": "audio_url",
                            "audio_url": {
                                "url": audio_data_url,
                            },
                        },
                    ],
                }
            ],
        )

        content = _get_message_content(completion.choices[0].message)
        data = _extract_json(content)

        emotion = _normalize_emotion(data.get("emotion"))
        confidence = _safe_confidence(data.get("confidence"))
        reasoning = str(
            data.get("reasoning")
            or content
            or "Qwen-Omni returned no reasoning."
        ).strip()
        confidence = _calibrate_confidence(emotion, confidence, reasoning)

        return {
            "emotion": emotion,
            "confidence": confidence,
            "reasoning": reasoning,
        }

    except Exception as second_exc:
        return _fallback(
            "speech API unavailable. "
            f"Attempt 1 failed: {first_error}. "
            f"Attempt 2 failed: {second_exc}"
        )


# ============================================================
# Public project interface
# ============================================================

def predict(audio_path):
    """
    Return speech emotion in the schema expected by app.py.

    Parameters:
        audio_path: path to an audio file, usually generated by app.py from Gradio mic input

    Returns:
        {
            "emotion": one of config.EMOTION_LABELS,
            "confidence": float in [0, 1],
            "reasoning": str
        }
    """
    if audio_path is None:
        return _fallback("no audio input")

    try:
        return _call_qwen_omni_audio(audio_path)
    except Exception as exc:
        return _fallback(f"speech API unavailable: {exc}")


def predict_text(text):
    """
    MELD 测试专用：文本情绪识别（不走音频）
    """
    from openai import OpenAI
    import os
    import json

    load_dotenv()

    api_key = (
        os.getenv("ZHIZENGZENG_API_KEY")
        or os.getenv("OPENAI_API_KEY")
    )

    if not api_key:
        raise RuntimeError(
            "Missing API key: set ZHIZENGZENG_API_KEY or OPENAI_API_KEY in .env or environment."
        )

    client = OpenAI(
        api_key=api_key,
        base_url=os.getenv("ZHIZENGZENG_BASE_URL", "https://api.zhizengzeng.com/v1"),
    )

    prompt = f"""
You are a conversational emotion recognition system.

Analyze the emotion of the following utterance:

"{text}"

You MUST choose one label from:
neutral, happy, sad, angry, fear, surprise, disgust

Return ONLY valid JSON:
{{
  "emotion": "...",
  "confidence": 0.0,
  "reasoning": "short explanation"
}}
"""

    resp = client.chat.completions.create(
        model="qwen3-omni-flash",
        messages=[{"role": "user", "content": prompt}]
    )

    content = resp.choices[0].message.content

    try:
        return json.loads(content)
    except:
        return {
            "emotion": "neutral",
            "confidence": 0.0,
            "reasoning": "parse error"
        }

# ============================================================
# Local test
# ============================================================

if __name__ == "__main__":
    test_audio = os.getenv("TEST_AUDIO_PATH")

    if test_audio:
        print(json.dumps(predict(test_audio), ensure_ascii=False, indent=2))
    else:
        print(
            "No TEST_AUDIO_PATH set. Example:\n"
            "Windows PowerShell:\n"
            "  $env:TEST_AUDIO_PATH='test.wav'; python speech_emotion.py\n\n"
            "macOS/Linux:\n"
            "  TEST_AUDIO_PATH=test.wav python speech_emotion.py\n\n"
            "Fallback test:"
        )
        print(json.dumps(predict(None), ensure_ascii=False, indent=2))