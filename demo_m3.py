"""
demo_m3.py —— M3 独立 demo（LLM 推理 + 音乐生成对比）
============================================================
不依赖 Gradio / app.py，对比 standard baseline 与 tom_cot 两种推理模式，
并可选保存结果到 outputs/m3_demo_results.json。

运行：
    python demo_m3.py
"""

from __future__ import annotations

import json
from pathlib import Path

from llm_reason import infer_with_mode
from music_gen import generate

TEST_STATES = [
    {
        "name": "fear_high_fatigue",
        "state": {
            "dominant_emotion": "fear",
            "confidence": 0.74,
            "fatigue": "high",
            "face_conf": 0.78,
            "speech_conf": 0.61,
            "fatigue_conf": 0.85,
        },
    },
    {
        "name": "happy_low_fatigue",
        "state": {
            "dominant_emotion": "happy",
            "confidence": 0.82,
            "fatigue": "low",
            "face_conf": 0.86,
            "speech_conf": 0.75,
            "fatigue_conf": 0.70,
        },
    },
    {
        "name": "sad_high_fatigue",
        "state": {
            "dominant_emotion": "sad",
            "confidence": 0.69,
            "fatigue": "high",
            "face_conf": 0.72,
            "speech_conf": 0.66,
            "fatigue_conf": 0.81,
        },
    },
    {
        "name": "neutral_medium_fatigue",
        "state": {
            "dominant_emotion": "neutral",
            "confidence": 0.58,
            "fatigue": "medium",
            "face_conf": 0.55,
            "speech_conf": 0.60,
            "fatigue_conf": 0.63,
        },
    },
]

OUTPUT_JSON = Path("outputs/m3_demo_results.json")


def _path_exists(path: str | None) -> bool:
    return bool(path) and Path(path).is_file()


def _run_mode(state: dict, mode: str) -> dict:
    result = infer_with_mode(state, mode=mode)
    audio_path = generate(result["music_spec"])
    return {
        "need": result["need"],
        "reasoning": result["reasoning"],
        "music_spec": result["music_spec"],
        "audio_path": audio_path,
        "audio_exists": _path_exists(audio_path),
    }


def main() -> None:
    all_results: list[dict] = []

    print("=" * 60)
    print("EmotiCompanion M3 Demo — standard vs tom_cot")
    print("=" * 60)

    for case in TEST_STATES:
        name = case["name"]
        state = case["state"]

        print(f"\n{'─' * 60}")
        print(f"Case: {name}")
        print(f"State: {json.dumps(state, ensure_ascii=False)}")

        standard = _run_mode(state, "standard")
        tom = _run_mode(state, "tom_cot")

        print(f"\n[standard] need: {standard['need']}")
        print(f"[standard] music_spec: {standard['music_spec']}")
        print(f"[standard] audio: {standard['audio_path']} "
              f"(exists={standard['audio_exists']})")

        print(f"\n[tom_cot] need: {tom['need']}")
        print(f"[tom_cot] reasoning:\n{tom['reasoning']}")
        print(f"[tom_cot] music_spec: {tom['music_spec']}")
        print(f"[tom_cot] audio: {tom['audio_path']} (exists={tom['audio_exists']})")

        all_results.append({
            "case": name,
            "state": state,
            "standard": standard,
            "tom_cot": tom,
        })

    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_JSON.open("w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

    print(f"\n{'=' * 60}")
    print(f"Saved results -> {OUTPUT_JSON}")
    print("=" * 60)


if __name__ == "__main__":
    main()
