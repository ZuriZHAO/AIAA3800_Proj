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
# 本地 SER 后端（wav2vec2）—— 用于 RAVDESS 消融
# ============================================================
# 背景：通用 qwen-omni-flash API 读不好情绪语调（见 experiment_plan §8.6），
# RAVDESS 消融改用专用 SER 模型：直接读韵律、给 softmax 置信度（正好喂 H2）。
# 通过 SPEECH_BACKEND=ser 启用（默认 api，保留部署时的语义能力）。
# ⚠️ ehcalabres 在 RAVDESS 上微调过 → 对 RAVDESS 是 in-domain，准确率是乐观上界，
#    与人脸①（跨域）不对称，报告须如实标注（见 experiment_plan §8.6）。

SER_MODEL_NAME = os.getenv(
    "SER_MODEL", "ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition")

# ehcalabres 8 类 → 全队 7 类契约词表
SER_TO_CONFIG = {
    "angry": "angry", "calm": "neutral", "disgust": "disgust", "fearful": "fear",
    "happy": "happy", "neutral": "neutral", "sad": "sad", "surprised": "surprise",
}

_SER_MODEL = None
_SER_FE = None
_SER_DEVICE = None


def _build_ser_class():
    """复刻 ehcalabres 的自定义分类头（stock Wav2Vec2ForSequenceClassification 的
    projector/classifier 结构与其权重不匹配，会得到随机头）。结构：
    encoder → 时间维 mean-pool → dense → tanh → output。"""
    import torch
    import torch.nn as nn
    from transformers import Wav2Vec2Model, AutoConfig

    class Wav2Vec2SER(nn.Module):
        def __init__(self, name):
            super().__init__()
            self.config = AutoConfig.from_pretrained(name, local_files_only=True)
            self.wav2vec2 = Wav2Vec2Model.from_pretrained(name, local_files_only=True)
            h, n = self.config.hidden_size, self.config.num_labels
            self.dense = nn.Linear(h, h)
            self.output = nn.Linear(h, n)

        def forward(self, input_values, attention_mask=None):
            hs = self.wav2vec2(input_values, attention_mask=attention_mask).last_hidden_state
            x = torch.tanh(self.dense(hs.mean(dim=1)))
            return self.output(x)

    return Wav2Vec2SER


def _lazy_ser():
    """首次调用时构建 SER 模型并从本地缓存载入真实分类头权重（懒加载单例）。"""
    global _SER_MODEL, _SER_FE, _SER_DEVICE
    if _SER_MODEL is not None:
        return
    import torch
    from transformers import AutoFeatureExtractor
    from huggingface_hub import hf_hub_download

    _SER_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    _SER_FE = AutoFeatureExtractor.from_pretrained(SER_MODEL_NAME, local_files_only=True)

    model = _build_ser_class()(SER_MODEL_NAME)

    # 从缓存 checkpoint 载入自定义头（classifier.dense.* / classifier.output.*）
    try:
        wf = hf_hub_download(SER_MODEL_NAME, "model.safetensors", local_files_only=True)
        from safetensors.torch import load_file
        sd = load_file(wf)
    except Exception:
        wf = hf_hub_download(SER_MODEL_NAME, "pytorch_model.bin", local_files_only=True)
        sd = torch.load(wf, map_location="cpu", weights_only=True)
    model.dense.weight.data = sd["classifier.dense.weight"]
    model.dense.bias.data = sd["classifier.dense.bias"]
    model.output.weight.data = sd["classifier.output.weight"]
    model.output.bias.data = sd["classifier.output.bias"]

    _SER_MODEL = model.to(_SER_DEVICE).eval()


def _read_wav_16k(path):
    """读 wav → 单声道 float32 @16kHz（wav2vec2 要求 16k）。"""
    import numpy as np
    import soundfile as sf
    data, sr = sf.read(path)
    if getattr(data, "ndim", 1) > 1:
        data = data.mean(axis=1)
    data = np.asarray(data, dtype="float32")
    if sr != 16000 and data.size:
        import scipy.signal as sps
        data = sps.resample(data, int(len(data) * 16000 / sr)).astype("float32")
    return data


def _predict_ser(audio_path):
    """本地 SER：wav2vec2 前向 → softmax → 映射进 7 类词表，带 softmax 置信度。"""
    import torch
    _lazy_ser()
    wav = _read_wav_16k(audio_path)
    if wav.size == 0:
        return _fallback("empty audio for SER")
    inputs = _SER_FE(wav, sampling_rate=16000, return_tensors="pt", padding=True)
    input_values = inputs["input_values"].to(_SER_DEVICE)
    attn = inputs.get("attention_mask")
    attn = attn.to(_SER_DEVICE) if attn is not None else None
    with torch.no_grad():
        logits = _SER_MODEL(input_values, attention_mask=attn)[0]
    probs = torch.softmax(logits, dim=-1).cpu().numpy()
    idx = int(probs.argmax())
    raw = _SER_MODEL.config.id2label[idx].lower()
    return {
        "emotion": SER_TO_CONFIG.get(raw, "neutral"),
        "confidence": round(float(probs[idx]), 4),
        "reasoning": f"(local SER {SER_MODEL_NAME.split('/')[-1]}) raw_label={raw}",
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
    You are a speech emotion recognition system for a real-world conversational AI companion.
    You will receive an audio clip of spoken dialogue. Judge the speaker's emotion.

    PRIMARY signal — vocal prosody (HOW it is said):
    pitch, energy, rhythm, pauses, tone, speaking speed, intensity, voice stability.

    SECONDARY signal — spoken words (WHAT is said), used ADAPTIVELY:
    - Use the words ONLY IF they explicitly express an emotion
      (e.g. "I feel so sad", "this is terrifying").
    - If the words are emotionally NEUTRAL or merely descriptive/observational
      (e.g. "Kids are talking by the door"), do NOT let the neutral wording pull the
      prediction toward neutral — judge from vocal prosody instead.
    - The emotion is often carried ONLY by the voice; never assume neutral just because
      the sentence content is ordinary.

    Emotion labels:
    {", ".join(EMOTION_LABELS)}

    Decision rules (prosody-first):
    - happy → laughter, bright/energetic tone, enthusiasm
    - sad → low energy, slow speech, downward/heavy tone
    - angry → high energy, sharp/harsh tone, tension, forceful speech
    - fear → shaky/tense voice, tremor, uncertainty, nervousness
    - surprise → sudden pitch jump, exclamation, abrupt reaction
    - disgust → aversive/rejecting tone
    - neutral → ONLY when the voice is prosodically flat AND the words express no emotion

    Confidence rules:
    - 0.8–0.95: clear emotional prosody
    - 0.5–0.8: moderate
    - 0.2–0.5: ambiguous
    - Do NOT output high-confidence neutral just because the sentence is an ordinary
      statement — that is exactly when you must rely on vocal tone.

    Return ONLY JSON:
    {{
    "emotion": "...",
    "confidence": 0.0,
    "reasoning": "short explanation, citing vocal prosody first and words only if emotionally expressive"
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

    # SPEECH_BACKEND=ser → 本地 wav2vec2（RAVDESS 消融用）；默认 api（部署保留语义）
    backend = os.getenv("SPEECH_BACKEND", "api").strip().lower()
    if backend in {"ser", "local", "wav2vec2"}:
        try:
            return _predict_ser(audio_path)
        except Exception as exc:
            return _fallback(f"local SER unavailable: {exc}")

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