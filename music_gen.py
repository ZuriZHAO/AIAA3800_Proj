from __future__ import annotations

import os
import random
import time
import wave
from datetime import datetime
from pathlib import Path
from typing import Literal, Optional, Union

import numpy as np
from dotenv import load_dotenv

load_dotenv(override=True)

AudioReturn = Union[str, tuple[int, np.ndarray], None]

MODEL_NAME = "facebook/musicgen-small"
MODEL_DIR = Path(os.getenv("MUSICGEN_MODEL_DIR", "/home/user/models/musicgen-small"))

DEFAULT_DURATION_SEC = int(os.getenv("MUSICGEN_DURATION_SEC", "15"))
MAX_DURATION_SEC = int(os.getenv("MUSICGEN_MAX_DURATION_SEC", "120"))

# 关键：单次 MusicGen 生成最多 30 秒，超过就分段；20 秒已在当前环境实测稳定
SINGLE_GENERATE_MAX_SEC = int(os.getenv("MUSICGEN_SINGLE_MAX_SEC", "30"))

TOKENS_PER_SEC = float(os.getenv("MUSICGEN_TOKENS_PER_SEC", "50"))
OUTPUT_DIR = Path(os.getenv("MUSICGEN_OUTPUT_DIR", "outputs"))

_MODEL = None
_PROCESSOR = None
_DEVICE: str | None = None
_SAMPLE_RATE: int | None = None

PromptStyle = Literal["calm", "upbeat", "focus", "default"]


def _get_output_stem() -> str:
    now = datetime.now()
    ms = now.microsecond // 1000
    suffix = random.randint(0, 999)
    return now.strftime(f"emoti_%Y%m%d_%H%M%S_{ms:03d}_{suffix:03d}")


def _sanitize_duration(duration_sec=None) -> int:
    if duration_sec is None:
        duration_sec = DEFAULT_DURATION_SEC
    try:
        d = int(round(float(duration_sec)))
    except Exception:
        d = DEFAULT_DURATION_SEC
    if d <= 0:
        return 0
    return max(1, min(MAX_DURATION_SEC, d))


def _split_duration(total_sec: int) -> list[int]:
    """把总时长切成每段最多 30 秒。例：75 -> [20, 20, 20, 15]"""
    single_max = max(1, min(30, SINGLE_GENERATE_MAX_SEC))
    parts = []
    remain = int(total_sec)
    while remain > 0:
        d = min(single_max, remain)
        parts.append(d)
        remain -= d
    return parts


def _fix_musicgen_special_tokens(model):
    """修复 MusicGen config 里 2048 special token 导致的 CUDA 越界问题。"""
    def patch_obj(obj):
        if obj is None:
            return
        for attr in ("pad_token_id", "bos_token_id", "decoder_start_token_id"):
            if hasattr(obj, attr):
                val = getattr(obj, attr, None)
                if val is None or val >= 2048:
                    setattr(obj, attr, 0)
        if hasattr(obj, "eos_token_id"):
            val = getattr(obj, "eos_token_id", None)
            if val is not None and val >= 2048:
                setattr(obj, "eos_token_id", None)

    for obj in [
        getattr(model, "config", None),
        getattr(model, "generation_config", None),
        getattr(getattr(model, "config", None), "decoder", None),
        getattr(getattr(model, "decoder", None), "config", None),
        getattr(getattr(model, "decoder", None), "generation_config", None),
    ]:
        patch_obj(obj)

    gc = getattr(model, "generation_config", None)
    dc = getattr(getattr(model, "decoder", None), "config", None)
    print(
        "[M3] fixed special tokens:",
        "gen.pad=", getattr(gc, "pad_token_id", None),
        "gen.bos=", getattr(gc, "bos_token_id", None),
        "decoder.pad=", getattr(dc, "pad_token_id", None),
        "decoder.bos=", getattr(dc, "bos_token_id", None),
    )


def _classify_prompt_style(music_spec: str) -> PromptStyle:
    text = str(music_spec or "").lower()
    if any(k in text for k in ("slow", "calm", "ambient", "soft", "warm", "soothing", "gentle", "low energy")):
        return "calm"
    if any(k in text for k in ("upbeat", "bright", "energetic", "positive", "uplifting", "cheerful", "playful")):
        return "upbeat"
    if any(k in text for k in ("focus", "neutral", "balanced", "steady", "minimal", "unobtrusive")):
        return "focus"
    return "default"


def _get_model():
    global _MODEL, _PROCESSOR, _DEVICE, _SAMPLE_RATE

    if _MODEL is not None:
        return _MODEL, _PROCESSOR, _DEVICE, _SAMPLE_RATE

    import torch
    from transformers import AutoProcessor, MusicgenForConditionalGeneration

    requested_device = os.getenv("MUSICGEN_DEVICE", "").strip().lower()
    if requested_device in {"cpu", "cuda"}:
        _DEVICE = requested_device
    else:
        _DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    use_local = MODEL_DIR.exists() and any(MODEL_DIR.iterdir())
    model_source = str(MODEL_DIR) if use_local else MODEL_NAME

    print(f"[M3] Loading MusicGen from: {model_source}")
    print(f"[M3] Device: {_DEVICE}")

    _PROCESSOR = AutoProcessor.from_pretrained(model_source, local_files_only=use_local)
    _MODEL = MusicgenForConditionalGeneration.from_pretrained(
        model_source,
        local_files_only=use_local,
    ).to(_DEVICE)

    _MODEL.eval()
    _SAMPLE_RATE = int(_MODEL.config.audio_encoder.sampling_rate)
    return _MODEL, _PROCESSOR, _DEVICE, _SAMPLE_RATE


def _generate_one_chunk(prompt: str, duration_sec: int, idx: int, total: int) -> tuple[int, np.ndarray] | None:
    """单次 MusicGen 生成，duration_sec 一定 <= 30。"""
    global _MODEL, _PROCESSOR, _DEVICE, _SAMPLE_RATE

    try:
        import torch

        model, processor, device, sample_rate = _get_model()

        # 30s * 50 = 1500 tokens，当前环境实测稳定；30s/1500 tokens 会触发 codebook reshape 问题
        max_new_tokens = int(duration_sec * TOKENS_PER_SEC)
        max_new_tokens = max(1, min(max_new_tokens, 1500))

        chunk_prompt = (
            f"{prompt} "
            "Continuous instrumental background music with no vocals; keep the mood, tempo, "
            "energy, key centre, and instrumentation consistent; use gradual development and "
            "smooth transitions; avoid abrupt intros, breakdowns, final cadences, or dramatic endings."
        )

        print(
            f"[M3] Generating chunk {idx + 1}/{total}: "
            f"duration={duration_sec}s, max_new_tokens={max_new_tokens}"
        )
        print(f"[M3] Prompt: {chunk_prompt[:500]}")

        inputs = processor(text=[chunk_prompt], padding=True, return_tensors="pt")
        inputs = {
            k: (v.to(device) if hasattr(v, "to") else v)
            for k, v in inputs.items()
        }

        t0 = time.perf_counter()
        with torch.inference_mode():
            audio_values = model.generate(
                **inputs,
                do_sample=True,
                guidance_scale=3.0,
                temperature=1.0,
                max_new_tokens=max_new_tokens,
            )

        print(f"[M3] Chunk {idx + 1}/{total} finished in {time.perf_counter() - t0:.2f}s")

        wav = audio_values[0, 0].detach().cpu().float().numpy()
        wav = np.nan_to_num(wav, nan=0.0, posinf=0.0, neginf=0.0).astype("float32")

        target_n = int(sample_rate * duration_sec)
        if wav.shape[0] > target_n:
            wav = wav[:target_n]
        elif wav.shape[0] < target_n:
            wav = np.pad(wav, (0, target_n - wav.shape[0]))

        return int(sample_rate), wav

    except Exception as e:
        print("[M3] MusicGen chunk generation failed:", repr(e))
        if "device-side assert" in str(e):
            print("[M3] CUDA device-side assert detected. Stop python app.py and restart before retrying.")
        _MODEL = _PROCESSOR = _DEVICE = _SAMPLE_RATE = None
        return None


def _concat_with_crossfade(chunks: list[np.ndarray], sample_rate: int, crossfade_sec: float = 0.12) -> np.ndarray:
    """分段拼接，加一个很短的 crossfade，减少接缝突兀。"""
    if not chunks:
        return np.zeros(1, dtype=np.float32)

    out = chunks[0].astype("float32")
    fade_n = int(sample_rate * crossfade_sec)

    for chunk in chunks[1:]:
        chunk = chunk.astype("float32")
        if fade_n > 0 and len(out) > fade_n and len(chunk) > fade_n:
            fade_out = np.linspace(1.0, 0.0, fade_n, dtype=np.float32)
            fade_in = np.linspace(0.0, 1.0, fade_n, dtype=np.float32)
            mixed = out[-fade_n:] * fade_out + chunk[:fade_n] * fade_in
            out = np.concatenate([out[:-fade_n], mixed, chunk[fade_n:]])
        else:
            out = np.concatenate([out, chunk])

    return out


def _write_int16_wav(path: str, sample_rate: int, audio: np.ndarray):
    """写成 int16 wav，兼容 Gradio / 浏览器 / wave 模块。"""
    audio = np.asarray(audio, dtype=np.float32)
    audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)

    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    if peak > 1.0:
        audio = audio / peak

    pcm = (np.clip(audio, -1.0, 1.0) * 32767.0).astype(np.int16)

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(int(sample_rate))
        wf.writeframes(pcm.tobytes())


def _generate_with_musicgen(music_spec: str, output_stem: str, duration_sec=None) -> Optional[str]:
    """总生成入口：<=30s 单段，>30s 自动多段拼接。"""
    try:
        prompt = str(music_spec or "").strip()
        if not prompt:
            return None

        total_duration = _sanitize_duration(duration_sec)
        if total_duration <= 0:
            print("[M3] duration=0, skip MusicGen.")
            return None

        parts = _split_duration(total_duration)

        print(
            f"[M3] Generating MusicGen audio: total={total_duration}s, "
            f"single_max={min(30, SINGLE_GENERATE_MAX_SEC)}s, chunks={parts}"
        )

        arrays = []
        sample_rate = None

        for i, d in enumerate(parts):
            result = _generate_one_chunk(prompt, d, i, len(parts))
            if result is None:
                return None
            sr, wav = result
            sample_rate = sr
            arrays.append(wav)

        final_wav = _concat_with_crossfade(arrays, int(sample_rate))

        # crossfade 会轻微缩短，总长度最后对齐到用户滑条选择的秒数
        target_n = int(int(sample_rate) * total_duration)
        if final_wav.shape[0] > target_n:
            final_wav = final_wav[:target_n]
        elif final_wav.shape[0] < target_n:
            final_wav = np.pad(final_wav, (0, target_n - final_wav.shape[0]))

        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        out_path = OUTPUT_DIR / f"{output_stem}.wav"
        _write_int16_wav(str(out_path), int(sample_rate), final_wav)

        print(f"[M3] Wrote MusicGen wav: {out_path}")
        return str(out_path)

    except Exception as e:
        print("[M3] MusicGen generation failed:", repr(e))
        return None


def _generate_fallback_wav(music_spec: str, output_path: str, duration_sec=None) -> Optional[str]:
    """兜底简单 wav，只在 MusicGen 出错时使用。"""
    try:
        style = _classify_prompt_style(music_spec)
        duration = _sanitize_duration(duration_sec)
        if duration <= 0:
            return None

        sr = 22050
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

        _write_int16_wav(output_path, sr, signal)
        return output_path

    except Exception as e:
        print("[M3] fallback wav failed:", repr(e))
        return None


def generate(music_spec: str, duration_sec=None) -> AudioReturn:
    """外部接口：app.py 会调用 generate(music_spec, duration_sec)。"""
    try:
        prompt = str(music_spec or "").strip()
        if not prompt:
            return None

        duration = _sanitize_duration(duration_sec)
        if duration <= 0:
            return None

        stem = _get_output_stem()
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        out_path = str(OUTPUT_DIR / f"{stem}.wav")

        result = _generate_with_musicgen(prompt, stem, duration)
        if result and Path(result).is_file():
            return result

        print("[M3] MusicGen unavailable, using fallback wav.")
        return _generate_fallback_wav(prompt, out_path, duration)

    except Exception as e:
        print("[M3] generate failed:", repr(e))
        return None


if __name__ == "__main__":
    prompt = (
        "A slow, warm ambient instrumental piece with soft piano, gentle pads, "
        "low energy, and a comforting atmosphere, no vocals."
    )
    for d in [20, 25, 45, 75]:
        print("=" * 60)
        print(f"Test duration={d}")
        path = generate(prompt, d)
        print("Generated audio:", path)
