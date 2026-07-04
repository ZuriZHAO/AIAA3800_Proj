"""
music_gen.py —— ⑥ 音乐生成（M3）
============================================================
EmotiCompanion · AIAA 3800 · HKUST(GZ)

接收 llm_reason.infer() 输出的 music_spec 字符串，生成可播放音频。
优先使用 Hugging Face Transformers MusicGen（与 music_generation_demo.ipynb 一致），
不可用时自动 fallback 到简单 mock wav。

接口契约（app.py 自动加载）：
    generate(music_spec) -> str | (int, np.ndarray) | None
"""

from __future__ import annotations

import random
import wave
from datetime import datetime
from pathlib import Path
from typing import Literal, Optional, Union

import numpy as np

AudioReturn = Union[str, tuple[int, np.ndarray], None]

MODEL_NAME = "facebook/musicgen-small"
DEFAULT_DURATION_SEC = 8
OUTPUT_DIR = Path("outputs")

# Lazy-loaded MusicGen singletons (Transformers API, per course demo)
_MODEL = None
_PROCESSOR = None
_DEVICE: str | None = None
_SAMPLE_RATE: int | None = None

PromptStyle = Literal["calm", "upbeat", "focus", "default"]


def _get_output_stem() -> str:
    """Unique filename stem: emoti_YYYYMMDD_HHMMSS_mmm_rrr."""
    now = datetime.now()
    ms = now.microsecond // 1000
    suffix = random.randint(0, 999)
    return now.strftime(f"emoti_%Y%m%d_%H%M%S_{ms:03d}_{suffix:03d}")


def _classify_prompt_style(music_spec: str) -> PromptStyle:
    """Map prompt keywords to a coarse fallback audio style."""
    text = music_spec.lower()

    calm_keys = ("slow", "calm", "ambient", "soft", "warm", "soothing", "gentle", "low energy")
    upbeat_keys = ("upbeat", "bright", "energetic", "positive", "uplifting", "cheerful", "playful")
    focus_keys = ("focus", "neutral", "balanced", "steady", "minimal", "unobtrusive")

    if any(k in text for k in calm_keys):
        return "calm"
    if any(k in text for k in upbeat_keys):
        return "upbeat"
    if any(k in text for k in focus_keys):
        return "focus"
    return "default"


def _get_model():
    """Lazy-load MusicGen via Hugging Face Transformers (course demo API)."""
    global _MODEL, _PROCESSOR, _DEVICE, _SAMPLE_RATE
    if _MODEL is not None:
        return _MODEL, _PROCESSOR, _DEVICE, _SAMPLE_RATE

    import torch
    from transformers import AutoProcessor, MusicgenForConditionalGeneration

    _DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    _PROCESSOR = AutoProcessor.from_pretrained(MODEL_NAME)
    _MODEL = MusicgenForConditionalGeneration.from_pretrained(MODEL_NAME).to(_DEVICE)
    _MODEL.eval()
    _SAMPLE_RATE = int(_MODEL.config.audio_encoder.sampling_rate)
    return _MODEL, _PROCESSOR, _DEVICE, _SAMPLE_RATE


def _generate_with_musicgen(music_spec: str, output_stem: str) -> Optional[str]:
    """
    Generate audio with MusicGen (Transformers).
    Returns path to .wav on success, None on any failure.
    """
    try:
        import torch
        import scipy.io.wavfile

        model, processor, device, sample_rate = _get_model()

        inputs = processor(
            text=[music_spec],
            padding=True,
            return_tensors="pt",
        ).to(device)

        max_new_tokens = int(DEFAULT_DURATION_SEC * 50)  # 50 tokens ≈ 1 second

        with torch.no_grad():
            audio_values = model.generate(
                **inputs,
                do_sample=True,
                guidance_scale=3.0,
                temperature=1.0,
                max_new_tokens=max_new_tokens,
            )

        wav = audio_values[0, 0].cpu().float().numpy()
        out_path = OUTPUT_DIR / f"{output_stem}.wav"
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        scipy.io.wavfile.write(str(out_path), rate=sample_rate, data=wav)
        return str(out_path)
    except Exception:
        # Reset lazy singleton so a later retry can reload after env fix
        global _MODEL, _PROCESSOR, _DEVICE, _SAMPLE_RATE
        _MODEL = _PROCESSOR = _DEVICE = _SAMPLE_RATE = None
        return None


def _generate_fallback_wav(music_spec: str, output_path: str) -> Optional[str]:
    """
    Simple synthetic wav for UI / pipeline testing when MusicGen is unavailable.
    Uses numpy + stdlib wave only.
    """
    try:
        style = _classify_prompt_style(music_spec)
        sr = 22050
        duration = 4.0
        n = int(sr * duration)
        t = np.linspace(0.0, duration, n, endpoint=False)

        if style == "calm":
            freq, mod_freq, amp = 196.0, 0.25, 0.14
        elif style == "upbeat":
            freq, mod_freq, amp = 440.0, 1.8, 0.18
        elif style == "focus":
            freq, mod_freq, amp = 330.0, 0.6, 0.12
        else:
            freq, mod_freq, amp = 261.63, 0.5, 0.15

        envelope = 0.55 + 0.45 * np.sin(2.0 * np.pi * mod_freq * t)
        fade = max(1, int(0.08 * sr))
        env = np.ones(n, dtype=np.float64)
        env[:fade] = np.linspace(0.0, 1.0, fade)
        env[-fade:] = np.linspace(1.0, 0.0, fade)

        signal = amp * envelope * env * np.sin(2.0 * np.pi * freq * t)
        signal += 0.04 * amp * np.sin(2.0 * np.pi * freq * 1.5 * t)

        pcm = (np.clip(signal, -1.0, 1.0) * 32767.0).astype(np.int16)

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with wave.open(output_path, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sr)
            wf.writeframes(pcm.tobytes())
        return output_path
    except Exception:
        return None


def generate(music_spec: str) -> AudioReturn:
    """
    Generate music from an English text prompt.

    Args:
        music_spec: Music description string from llm_reason.infer()["music_spec"].

    Returns:
        str path to .wav (preferred), or None on total failure.
    """
    try:
        prompt = str(music_spec or "").strip()
        if not prompt:
            return None

        stem = _get_output_stem()
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        out_path = str(OUTPUT_DIR / f"{stem}.wav")

        result = _generate_with_musicgen(prompt, stem)
        if result and Path(result).is_file():
            return result

        print("[M3] MusicGen unavailable, using fallback wav.")
        return _generate_fallback_wav(prompt, out_path)
    except Exception:
        return None


if __name__ == "__main__":
    prompt = (
        "A slow, warm ambient instrumental piece with soft piano, gentle pads, "
        "low energy, and a comforting atmosphere, no vocals."
    )
    path = generate(prompt)
    print("Generated audio:", path)
