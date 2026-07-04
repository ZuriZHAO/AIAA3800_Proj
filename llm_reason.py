"""
llm_reason.py —— ⑤ LLM 需求推理（M3）
============================================================
EmotiCompanion · AIAA 3800 · HKUST(GZ)

接收融合层 ④ 的统一状态 JSON，推断用户可能的心理需求并生成 MusicGen 描述。

课程方法参考（当前为规则化 / 模板模拟，不调用外部 LLM）：
  · ToM  — llm_based_theory_of_mind_reasoning_demo.ipynb (L10P1)
           figure4_style：先标注 mental state，再显式 ToM 推理 beliefs/needs
  · CoT  — chain_of_thought_prompting_demo.ipynb (L10P2)
           Kojima 2022 two-stage：Stage 1 推理链 → Stage 2 用 trigger 提取最终输出

伦理说明：本模块仅提供 affective music companionship 支持，不做医疗诊断或治疗。

接口契约（app.py 自动加载）：
    infer(state) -> {"need": str, "reasoning": str, "music_spec": str}

实验对比（demo / 消融用）：
    infer_with_mode(state, mode="tom_cot"|"standard") -> 同上

可选真实 LLM 后端（环境变量 EMOTI_LLM_BACKEND=openai）：
    默认 rule（规则版 ToM+CoT）；OpenAI 失败时自动回退 rule。
"""

from __future__ import annotations

import json
import os
import re
import textwrap

# ---------------------------------------------------------------------------
# Constants & schema
# ---------------------------------------------------------------------------

VALID_EMOTIONS = frozenset({
    "neutral", "happy", "sad", "angry", "fear", "surprise", "disgust",
})
VALID_FATIGUE = frozenset({"low", "medium", "high"})

NEGATIVE_EMOTIONS = frozenset({"sad", "angry", "fear", "disgust"})
LOW_CONFIDENCE_THRESHOLD = 0.45

_DEFAULT_STATE = {
    "dominant_emotion": "neutral",
    "confidence": 0.5,
    "fatigue": "medium",
    "face_conf": 0.5,
    "speech_conf": 0.5,
    "fatigue_conf": 0.5,
}

# Kojima 2022 Stage-2 trigger adapted for music generation (see CoT demo §7)
_COT_STAGE2_TRIGGER = "Therefore, the final music prompt is:"

_STANDARD_BASELINE_NOTE = (
    "Standard baseline:\n"
    "This version maps the detected dominant emotion directly to a music "
    "description without explicit Theory-of-Mind need inference or multi-step reasoning."
)

# emotion -> (need label, music_spec) for standard baseline (Figure 1 style)
_STANDARD_EMOTION_MAP: dict[str, tuple[str, str]] = {
    "happy": (
        "upbeat positive instrumental",
        "A bright, upbeat positive instrumental with light piano and soft bells, "
        "tempo around 100 BPM, medium energy, cheerful atmosphere, no vocals, "
        "suitable for a short 5-10 second clip.",
    ),
    "sad": (
        "slow gentle instrumental",
        "A slow, gentle instrumental with soft piano and warm strings, "
        "tempo around 60 BPM, low energy, melancholic but comforting atmosphere, "
        "no vocals, suitable for a short 5-10 second clip.",
    ),
    "angry": (
        "calming low-energy instrumental",
        "A calming, low-energy instrumental with warm pads and soft electric piano, "
        "tempo around 65 BPM, very low energy, tension-reducing atmosphere, "
        "no vocals, suitable for a short 5-10 second clip.",
    ),
    "fear": (
        "soft calming instrumental",
        "A soft, calming instrumental with gentle ambient pads and subtle piano, "
        "tempo around 58 BPM, very low energy, reassuring atmosphere, "
        "no vocals, suitable for a short 5-10 second clip.",
    ),
    "neutral": (
        "balanced gentle instrumental",
        "A balanced, gentle instrumental with soft Rhodes and light texture, "
        "tempo around 75 BPM, low-to-medium energy, neutral-warm atmosphere, "
        "no vocals, suitable for a short 5-10 second clip.",
    ),
    "surprise": (
        "light curious instrumental",
        "A light, curious instrumental with marimba and soft plucks, "
        "tempo around 90 BPM, medium-low energy, gently playful atmosphere, "
        "no vocals, suitable for a short 5-10 second clip.",
    ),
    "disgust": (
        "calming low-energy instrumental",
        "A smooth, calming low-energy instrumental with warm pads and soft piano, "
        "tempo around 62 BPM, low energy, stabilizing atmosphere, "
        "no vocals, suitable for a short 5-10 second clip.",
    ),
}

_FALLBACK_RESULT = {
    "need": "safe, neutral companionship",
    "reasoning": (
        "The input state could not be interpreted reliably, so the system chooses "
        "a conservative neutral music response."
    ),
    "music_spec": (
        "A gentle, balanced instrumental piece with soft piano, light ambient texture, "
        "medium-slow tempo, low-to-moderate energy, and no vocals, "
        "suitable for a short 5 to 10 second clip."
    ),
}

# OpenAI backend: light cache keyed by (emotion, fatigue, confidence)
_LLM_CACHE: dict[tuple, dict[str, str]] = {}

# Field length caps (truncate without breaking app)
_MAX_NEED_LEN = 200
_MAX_REASONING_LEN = 2500
_MAX_MUSIC_SPEC_LEN = 600

# Unsafe / diagnostic phrases → trigger rule fallback (affective support only)
_UNSAFE_PATTERNS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(p, re.IGNORECASE)
    for p in (
        r"\bdiagnos",
        r"\bthe user has anxiety\b",
        r"\bthe user is depressed\b",
        r"\buser has depression\b",
        r"\buser has anxiety\b",
        r"\bmental illness\b",
        r"\btreatment for\b",
        r"\bwill cure\b",
        r"\bcure the user\b",
        r"\btherapy for mental\b",
        r"\bclinical condition\b",
        r"\bpsychiatric\b",
    )
)

_OPENAI_SYSTEM_PROMPT = textwrap.dedent("""
    You are EmotiCompanion, an affective music companion system (NOT a medical device).
    You provide music companionship and affective support only — never medical diagnosis,
    clinical labels, or treatment claims.

    Given multimodal affective state labels, infer what short instrumental music the user
    may benefit from using Theory-of-Mind and two-stage reasoning.

    OUTPUT FORMAT — return ONLY a JSON object with exactly these keys:
    {
      "need": "short phrase describing inferred music companionship need",
      "reasoning": "concise two-stage explanation",
      "music_spec": "one English text prompt suitable for MusicGen"
    }

    Rules:
    - Output JSON only. No Markdown, no code fences, no extra text.
    - "need" must be a short phrase (not a paragraph).
    - "reasoning" MUST contain both sections:
        Stage 1 - Theory-of-Mind need inference
        Stage 2 - Music response planning
    - "music_spec" must be a single English string (NOT a nested object).
    - music_spec should describe mood, tempo, energy, instrumentation, atmosphere.
    - Prefer instrumental, no vocals, suitable for a 5-10 second clip.
    - Use cautious language: may benefit from, might need, could support.
    - NEVER use: diagnose, anxiety disorder, depressed, treatment, cure, therapy for
      mental illness, or any clinical diagnosis of the user.
""").strip()


# ---------------------------------------------------------------------------
# Input normalization
# ---------------------------------------------------------------------------

def _safe_float(value: object, default: float = 0.5) -> float:
    try:
        x = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default
    if x != x:
        return default
    return max(0.0, min(1.0, x))


def _normalize_state(state: dict | None) -> dict:
    raw = state if isinstance(state, dict) else {}
    emotion = str(raw.get("dominant_emotion", _DEFAULT_STATE["dominant_emotion"])).lower()
    if emotion not in VALID_EMOTIONS:
        emotion = "neutral"
    fatigue = str(raw.get("fatigue", _DEFAULT_STATE["fatigue"])).lower()
    if fatigue not in VALID_FATIGUE:
        fatigue = "medium"
    return {
        "dominant_emotion": emotion,
        "confidence": _safe_float(raw.get("confidence"), _DEFAULT_STATE["confidence"]),
        "fatigue": fatigue,
        "face_conf": _safe_float(raw.get("face_conf"), _DEFAULT_STATE["face_conf"]),
        "speech_conf": _safe_float(raw.get("speech_conf"), _DEFAULT_STATE["speech_conf"]),
        "fatigue_conf": _safe_float(raw.get("fatigue_conf"), _DEFAULT_STATE["fatigue_conf"]),
    }


def _confidence_label(confidence: float) -> str:
    if confidence >= 0.7:
        return "high"
    if confidence >= 0.45:
        return "moderate"
    return "low"


def _format_observed_state(normalized: dict) -> str:
    """Observed state block (external multimodal perception)."""
    emo = normalized["dominant_emotion"]
    conf = normalized["confidence"]
    fatigue = normalized["fatigue"]
    return (
        f"Dominant emotion: {emo}; fatigue level: {fatigue}; "
        f"fusion confidence: {conf:.2f} ({_confidence_label(conf)}). "
        f"Modality confidence — face: {normalized['face_conf']:.2f}, "
        f"speech: {normalized['speech_conf']:.2f}, fatigue: {normalized['fatigue_conf']:.2f}."
    )


# ---------------------------------------------------------------------------
# Prompt builders (course-style; for future real LLM API)
# ---------------------------------------------------------------------------

def build_tom_prompt(state: dict) -> str:
    """
    ToM-enhanced + two-stage CoT prompt (figure4_style + Kojima 2022).

    Inspired by figure4_style_cmv in llm_based_theory_of_mind_reasoning_demo.ipynb:
    inject mental-state labels, then ask for explicit ToM reasoning before the answer.

    Inspired by zero_shot_cot_two_stage in chain_of_thought_prompting_demo.ipynb:
    Stage 1 = reasoning extraction; Stage 2 = answer extraction via a trigger phrase.
    """
    n = _normalize_state(state)
    emo, fatigue, conf = n["dominant_emotion"], n["fatigue"], n["confidence"]
    return textwrap.dedent(f"""
        You are EmotiCompanion, an affective music companion (NOT a medical system).
        We analyzed the user's observed affective state from multimodal sensors:
        - Dominant emotion label: {emo}
        - Fatigue level label: {fatigue}
        - Fusion confidence: {conf:.2f}
        - Face modality confidence: {n['face_conf']:.2f}
        - Speech modality confidence: {n['speech_conf']:.2f}
        - Fatigue modality confidence: {n['fatigue_conf']:.2f}

        Task: infer what music companionship the user may need right now, then plan
        a short instrumental music prompt for MusicGen.

        Use theory-of-mind: reason about the user's possible internal state and need
        from the OBSERVED labels only. Use cautious language (may / might / could benefit).
        Do NOT diagnose medical or clinical conditions.

        Stage 1 — Theory-of-Mind need inference:
        Let's think step by step.
        - Observed state: summarize the labels above.
        - Possible inner state: what the user may be experiencing internally.
        - Possible need / intention: calming, emotional support, grounding, focus, etc.

        Stage 2 — Music response planning:
        Based on Stage 1, choose a music response strategy (reduce arousal, provide warmth,
        avoid overstimulation, maintain rhythm, etc.), list music attributes (mood, tempo,
        energy, instrumentation), then output ONE English music prompt string.

        {_COT_STAGE2_TRIGGER}
    """).strip()


def build_standard_prompt(state: dict) -> str:
    """
    Standard baseline prompt (figure1_style — direct mapping, no ToM / no CoT).

    Inspired by figure1_style_cmv in llm_based_theory_of_mind_reasoning_demo.ipynb:
    task instruction only, no mental-state reasoning scaffold.
    """
    n = _normalize_state(state)
    return textwrap.dedent(f"""
        You are EmotiCompanion. Map the detected dominant emotion directly to a
        short English instrumental music prompt for MusicGen.

        Do NOT use theory-of-mind reasoning or multi-step chain-of-thought.

        Dominant emotion label: {n['dominant_emotion']}

        Output one music description string (mood, tempo, energy, instrumentation, no vocals).
    """).strip()


# ---------------------------------------------------------------------------
# Stage 1 — ToM need inference (rule-based simulation)
# ---------------------------------------------------------------------------

def _infer_possible_inner_state(normalized: dict) -> str:
    """Possible inner state — cautious ToM inference, not diagnosis."""
    emotion = normalized["dominant_emotion"]
    fatigue = normalized["fatigue"]
    confidence = normalized["confidence"]
    conf_label = _confidence_label(confidence)

    if confidence < LOW_CONFIDENCE_THRESHOLD:
        return (
            "The system cautiously infers that the user's internal state is uncertain "
            "given low fusion confidence; the user may be in a mixed or transitional "
            "affective state that should not be over-interpreted."
        )

    inner_map = {
        ("fear", "high"): (
            "The user may be experiencing elevated tension combined with mental exhaustion; "
            "they might feel on edge yet in need of rest."
        ),
        ("fear", "low"): (
            "The user may be experiencing alertness or unease while still relatively energetic; "
            "they might need stabilization rather than sleep-inducing calm."
        ),
        ("sad", "high"): (
            "The user may be experiencing low mood with depleted energy; "
            "they might benefit from gentle emotional warmth rather than stimulation."
        ),
        ("sad", "medium"): (
            "The user may be experiencing subdued mood with moderate energy; "
            "they might appreciate quiet companionship."
        ),
        ("angry", "high"): (
            "The user may be experiencing frustration or irritability alongside fatigue; "
            "they might need de-escalation and recovery rather than confrontation."
        ),
        ("angry", "low"): (
            "The user may be experiencing elevated tension with remaining alertness; "
            "they might need grounding and emotional regulation."
        ),
        ("happy", "low"): (
            "The user may be in a positive mood with available energy; "
            "they might want to sustain uplift without becoming overstimulated."
        ),
        ("neutral", "medium"): (
            "The user may be in a steady, non-dramatic state with moderate tiredness; "
            "they might prefer unobtrusive background support."
        ),
        ("neutral", "high"): (
            "The user may feel mentally flat or worn down despite a neutral emotional label; "
            "they might need restorative low-arousal companionship."
        ),
        ("surprise", "low"): (
            "The user may be in a moment of mild arousal or novelty; "
            "they might appreciate lightly playful but non-overwhelming stimulation."
        ),
        ("disgust", "medium"): (
            "The user may be experiencing aversive tension; "
            "they might need a stabilizing, non-intrusive atmosphere."
        ),
    }
    key = (emotion, fatigue)
    if key in inner_map:
        return inner_map[key]

    if emotion in NEGATIVE_EMOTIONS:
        return (
            f"The user may be experiencing negative affect ({emotion}) at {fatigue} fatigue; "
            f"the system cautiously infers a need for emotional regulation support."
        )
    if emotion == "happy":
        return (
            "The user may be experiencing positive affect; "
            "they might want to maintain a pleasant mood."
        )
    return (
        f"Given {emotion} emotion and {fatigue} fatigue ({conf_label} confidence), "
        f"the system cautiously infers a balanced internal state."
    )


def _infer_possible_need(normalized: dict) -> tuple[str, str]:
    """Return (need label, brief ToM rationale)."""
    emotion = normalized["dominant_emotion"]
    fatigue = normalized["fatigue"]
    confidence = normalized["confidence"]

    if confidence < LOW_CONFIDENCE_THRESHOLD:
        need = "safe, neutral, non-intrusive companionship"
        rationale = (
            "Because fusion confidence is low, the system avoids a strong need claim "
            "and selects a conservative, non-intrusive companionship goal."
        )
        return need, rationale

    rules: list[tuple[callable, str, str]] = [
        (
            lambda: emotion in {"fear", "sad", "angry"} and fatigue == "high",
            "calming, emotional support, and low arousal",
            "High fatigue combined with negative emotion suggests the user may benefit "
            "from restful support that lowers arousal.",
        ),
        (
            lambda: emotion in {"fear", "angry"} and fatigue == "low",
            "grounding, de-escalation, and stable rhythm",
            "Negative emotion with low fatigue suggests grounding rather than sedating calm.",
        ),
        (
            lambda: emotion == "sad" and fatigue in {"medium", "high"},
            "warm companionship and gentle comfort",
            "Sadness with elevated fatigue points toward warm, gentle comfort.",
        ),
        (
            lambda: emotion == "happy" and fatigue == "low",
            "maintaining positive energy and light momentum",
            "Positive mood with available energy suggests sustaining uplift gently.",
        ),
        (
            lambda: emotion == "neutral" and fatigue == "medium",
            "stable focus and light companionship",
            "Neutral emotion with medium fatigue suggests focus-supporting companionship.",
        ),
        (
            lambda: emotion == "surprise" and fatigue in {"low", "medium"},
            "gentle curiosity and playful but non-overwhelming stimulation",
            "Surprise may call for lightly playful music that avoids overstimulation.",
        ),
        (
            lambda: emotion in {"disgust", "angry"},
            "tension reduction and emotional regulation",
            "Elevated tension may require music that supports emotional regulation.",
        ),
        (
            lambda: emotion == "happy",
            "uplifting companionship and balanced positive mood",
            "Positive affect may be supported with balanced uplifting music.",
        ),
        (
            lambda: emotion == "neutral" and fatigue == "high",
            "calming restoration and low-arousal comfort",
            "High fatigue with neutral emotion suggests restorative low-arousal support.",
        ),
        (
            lambda: emotion == "neutral" and fatigue == "low",
            "light engagement and steady ambient support",
            "Neutral-low-fatigue suggests steady ambient support.",
        ),
        (
            lambda: emotion == "sad",
            "gentle comfort and emotional warmth",
            "Sadness suggests gentle comfort and emotional warmth.",
        ),
        (
            lambda: emotion == "fear",
            "calming reassurance and reduced arousal",
            "Fear suggests calming reassurance and reduced arousal.",
        ),
    ]

    for condition, need, rationale in rules:
        if condition():
            return need, rationale

    return (
        "balanced, adaptive companionship",
        f"For {emotion} with {fatigue} fatigue, a balanced companionship strategy is chosen.",
    )


def _tom_stage1(normalized: dict) -> tuple[str, str, str, str]:
    """
    Stage 1 — Theory-of-Mind need inference.
    Returns (observed, inner_state, need, stage1_reasoning_block).
    """
    observed = _format_observed_state(normalized)
    inner_state = _infer_possible_inner_state(normalized)
    need, rationale = _infer_possible_need(normalized)
    block = (
        f"Observed state: {observed}\n"
        f"Possible inner state: {inner_state}\n"
        f"Possible need / intention: {need}. {rationale}"
    )
    return observed, inner_state, need, block


# ---------------------------------------------------------------------------
# Stage 2 — Music response planning (CoT answer extraction stage)
# ---------------------------------------------------------------------------

def _music_strategy_for_need(need: str, normalized: dict) -> str:
    """Music response strategy from inferred need."""
    if "non-intrusive" in need or "safe" in need:
        return "avoid overstimulation; provide safe, neutral ambient support"
    if "low arousal" in need or "calming" in need or "restoration" in need:
        return "reduce arousal; provide warmth and emotional support"
    if "grounding" in need or "de-escalation" in need or "stable rhythm" in need:
        return "maintain stable rhythm; support grounding and de-escalation"
    if "warm companionship" in need or "gentle comfort" in need or "emotional warmth" in need:
        return "provide warmth; gentle comfort without pushing energy up"
    if "positive energy" in need or "uplifting" in need:
        return "encourage gentle energy; maintain positive mood without overstimulation"
    if "stable focus" in need or "light companionship" in need or "light engagement" in need:
        return "support focus; keep accompaniment unobtrusive and steady"
    if "curiosity" in need or "playful" in need:
        return "allow gentle curiosity; avoid overwhelming stimulation"
    if "tension reduction" in need or "emotional regulation" in need:
        return "reduce tension; support emotional regulation"
    if "reassurance" in need or "reduced arousal" in need:
        return "provide reassurance; gently lower arousal"
    return "balanced music companionship adapted to observed affect"


def _music_attributes_for_spec(music_spec: str) -> str:
    """Extract a short attribute summary from the final spec string."""
    lower = music_spec.lower()
    parts: list[str] = []
    if "slow" in lower or "55" in lower or "58" in lower or "60" in lower or "65" in lower:
        parts.append("slow tempo")
    elif "mid-tempo" in lower or "80 bpm" in lower:
        parts.append("mid tempo")
    elif "95" in lower or "100" in lower or "90" in lower:
        parts.append("moderate-to-upbeat tempo")
    else:
        parts.append("moderate tempo")

    if "very low energy" in lower:
        parts.append("very low energy")
    elif "low energy" in lower or "low-to-medium" in lower:
        parts.append("low energy")
    elif "medium energy" in lower:
        parts.append("medium energy")
    else:
        parts.append("balanced energy")

    for kw in ("piano", "pads", "guitar", "strings", "marimba", "rhodes", "bells"):
        if kw in lower:
            parts.append(kw)
            break

    parts.append("no vocals")
    return ", ".join(parts)


def _build_music_spec(need: str, normalized: dict) -> str:
    """Map need + state to MusicGen-friendly English prompt string."""
    emotion = normalized["dominant_emotion"]
    fatigue = normalized["fatigue"]
    confidence = normalized["confidence"]
    low_conf = confidence < LOW_CONFIDENCE_THRESHOLD

    if low_conf:
        return (
            "A soft, neutral ambient instrumental with gentle pads and subtle piano, "
            "slow tempo around 60 BPM, very low energy, calm and non-intrusive atmosphere, "
            "no vocals, suitable for a short 5-10 second clip."
        )

    spec_rules: list[tuple[callable, str]] = [
        (
            lambda: "low arousal" in need or "calming" in need or "restoration" in need,
            "A slow, warm ambient instrumental piece with soft piano and gentle pads, "
            "tempo around 55-65 BPM, very low energy, soothing and comforting atmosphere, "
            "no vocals, suitable for a short 5-10 second clip.",
        ),
        (
            lambda: "grounding" in need or "de-escalation" in need or "stable rhythm" in need,
            "A steady mid-tempo instrumental with muted percussion and warm bass, "
            "tempo around 80 BPM, moderate-low energy, grounded and stable atmosphere, "
            "no vocals, suitable for a short 5-10 second clip.",
        ),
        (
            lambda: "warm companionship" in need or "gentle comfort" in need or "emotional warmth" in need,
            "A gentle, warm instrumental with acoustic guitar and soft strings, "
            "tempo around 70 BPM, low-to-medium energy, intimate comforting atmosphere, "
            "no vocals, suitable for a short 5-10 second clip.",
        ),
        (
            lambda: "positive energy" in need or "uplifting" in need,
            "A bright, uplifting instrumental with light piano and soft bells, "
            "tempo around 95 BPM, medium energy, cheerful but not overwhelming atmosphere, "
            "no vocals, suitable for a short 5-10 second clip.",
        ),
        (
            lambda: "stable focus" in need or "light companionship" in need or "light engagement" in need,
            "A clean, minimal lo-fi instrumental with soft Rhodes and subtle texture, "
            "tempo around 75 BPM, low energy, focused and unobtrusive atmosphere, "
            "no vocals, suitable for a short 5-10 second clip.",
        ),
        (
            lambda: "curiosity" in need or "playful" in need,
            "A light, playful instrumental with marimba and soft synth plucks, "
            "tempo around 100 BPM, medium-low energy, gently curious atmosphere, "
            "no vocals, suitable for a short 5-10 second clip.",
        ),
        (
            lambda: "tension reduction" in need or "emotional regulation" in need,
            "A smooth, calming instrumental with warm pads and soft electric piano, "
            "tempo around 65 BPM, low energy, tension-reducing atmosphere, "
            "no vocals, suitable for a short 5-10 second clip.",
        ),
        (
            lambda: "reassurance" in need or "reduced arousal" in need,
            "A quiet, reassuring ambient instrumental with soft piano and airy pads, "
            "tempo around 60 BPM, very low energy, safe and calming atmosphere, "
            "no vocals, suitable for a short 5-10 second clip.",
        ),
    ]

    for condition, spec in spec_rules:
        if condition():
            return spec

    if emotion in NEGATIVE_EMOTIONS:
        return (
            "A calm, supportive ambient instrumental with soft piano and gentle pads, "
            "tempo around 65 BPM, low energy, soothing atmosphere, "
            "no vocals, suitable for a short 5-10 second clip."
        )
    if emotion == "happy":
        return (
            "A warm, pleasant instrumental with light piano and soft strings, "
            "tempo around 90 BPM, medium energy, positive atmosphere, "
            "no vocals, suitable for a short 5-10 second clip."
        )
    if fatigue == "high":
        return (
            "A restful ambient instrumental with soft pads and subtle piano, "
            "tempo around 58 BPM, very low energy, drowsy-comfort atmosphere, "
            "no vocals, suitable for a short 5-10 second clip."
        )
    return (
        "A balanced ambient instrumental with soft piano and gentle texture, "
        "tempo around 72 BPM, low-to-medium energy, neutral-warm atmosphere, "
        "no vocals, suitable for a short 5-10 second clip."
    )


def _cot_stage2(need: str, normalized: dict) -> tuple[str, str, str, str]:
    """
    Stage 2 — Music response planning (CoT answer-extraction analogue).
    Returns (strategy, attributes, music_spec, stage2_block).
    """
    music_spec = _build_music_spec(need, normalized)
    strategy = _music_strategy_for_need(need, normalized)
    attributes = _music_attributes_for_spec(music_spec)
    block = (
        f"Music strategy: {strategy}.\n"
        f"Music attributes: {attributes}.\n"
        f"Final music prompt: {music_spec}"
    )
    return strategy, attributes, music_spec, block


# ---------------------------------------------------------------------------
# Mode implementations
# ---------------------------------------------------------------------------

def _infer_tom_cot(normalized: dict) -> dict[str, str]:
    """
    ToM + two-stage CoT (default production mode).

    Stage 1 mirrors figure4_style ToM scaffolding (mental state → need).
    Stage 2 mirrors Kojima CoT answer extraction (strategy → final prompt via trigger).
    """
    _, _, need, stage1_block = _tom_stage1(normalized)
    _, _, music_spec, stage2_block = _cot_stage2(need, normalized)

    reasoning = (
        "Stage 1 - Theory-of-Mind need inference:\n"
        f"{stage1_block}\n\n"
        "Stage 2 - Music response planning:\n"
        f"{stage2_block}\n"
        f"{_COT_STAGE2_TRIGGER} {music_spec}"
    )
    return {"need": need, "reasoning": reasoning, "music_spec": music_spec}


def _infer_standard(normalized: dict) -> dict[str, str]:
    """Standard baseline — direct emotion-to-music (figure1_style, no ToM / no CoT)."""
    emotion = normalized["dominant_emotion"]
    need, music_spec = _STANDARD_EMOTION_MAP.get(
        emotion, _STANDARD_EMOTION_MAP["neutral"]
    )
    reasoning = (
        f"{_STANDARD_BASELINE_NOTE}\n\n"
        f"Detected dominant emotion: {emotion}.\n"
        f"Mapped directly to need: {need}.\n"
        f"Final music prompt: {music_spec}"
    )
    return {"need": need, "reasoning": reasoning, "music_spec": music_spec}


def infer_with_mode(state: dict, mode: str = "tom_cot") -> dict[str, str]:
    """
    实验用推理入口：standard baseline vs tom_cot（始终使用规则版，不受 OpenAI 影响）。

    Args:
        state: 融合层统一状态 JSON。
        mode: "tom_cot"（默认）或 "standard"。

    Returns:
        {"need": str, "reasoning": str, "music_spec": str}
    """
    try:
        normalized = _normalize_state(state)
        if str(mode).lower().strip() == "standard":
            return _infer_standard(normalized)
        return _infer_tom_cot(normalized)
    except Exception:
        return dict(_FALLBACK_RESULT)


# ---------------------------------------------------------------------------
# Optional LLM API backend (EMOTI_LLM_BACKEND=openai | deepseek)
# DeepSeek uses OpenAI-compatible API: https://api.deepseek.com
# ---------------------------------------------------------------------------

_LLM_BACKENDS = frozenset({"openai", "deepseek"})

# Default endpoints / models per provider
_LLM_DEFAULTS = {
    "openai": {"base_url": None, "model": "gpt-4o-mini"},
    "deepseek": {"base_url": "https://api.deepseek.com", "model": "deepseek-chat"},
}


def _get_backend() -> str:
    """Return 'rule', 'openai', or 'deepseek'; unknown values fall back to 'rule'."""
    backend = os.getenv("EMOTI_LLM_BACKEND", "rule").lower().strip()
    return backend if backend in _LLM_BACKENDS else "rule"


def _uses_llm_api() -> bool:
    return _get_backend() in _LLM_BACKENDS


def _get_llm_api_key() -> str:
    """API key lookup: EMOTI_LLM_API_KEY > provider-specific > OPENAI_API_KEY."""
    for name in ("EMOTI_LLM_API_KEY", "DEEPSEEK_API_KEY", "OPENAI_API_KEY"):
        key = os.getenv(name, "").strip()
        if key:
            return key
    return ""


def _get_llm_base_url() -> str | None:
    explicit = os.getenv("EMOTI_LLM_BASE_URL", "").strip()
    if explicit:
        return explicit
    backend = _get_backend()
    if backend in _LLM_DEFAULTS:
        return _LLM_DEFAULTS[backend]["base_url"]
    return None


def _get_llm_model() -> str:
    if os.getenv("EMOTI_LLM_MODEL", "").strip():
        return os.getenv("EMOTI_LLM_MODEL", "").strip()
    backend = _get_backend()
    if backend in _LLM_DEFAULTS:
        return _LLM_DEFAULTS[backend]["model"]
    return "gpt-4o-mini"


def _make_cache_key(state: dict) -> tuple:
    n = _normalize_state(state)
    return (n["dominant_emotion"], n["fatigue"], round(n["confidence"], 2))


def _truncate_field(text: str, max_len: int) -> str:
    text = str(text).strip()
    if len(text) <= max_len:
        return text
    cut = text[: max_len - 3].rsplit(" ", 1)[0]
    return (cut or text[: max_len - 3]) + "..."


def _contains_unsafe_language(*texts: str) -> bool:
    combined = " ".join(t for t in texts if t)
    return any(p.search(combined) for p in _UNSAFE_PATTERNS)


def _strip_json_fence(text: str) -> str:
    text = text.strip()
    m = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL | re.IGNORECASE)
    return m.group(1).strip() if m else text


def _parse_llm_json(text: str) -> dict[str, str]:
    """Parse LLM JSON output; raise ValueError on failure."""
    raw = _strip_json_fence(text)
    # If extra prose wraps JSON, try to extract the outermost object
    if not raw.startswith("{"):
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        if m:
            raw = m.group(0)
    data = json.loads(raw)
    if not isinstance(data, dict):
        raise ValueError("LLM output is not a JSON object")
    return {str(k): v for k, v in data.items()}


def _validate_llm_result(data: dict) -> dict[str, str]:
    """Validate and normalize LLM JSON; raise ValueError if unsafe or invalid."""
    for key in ("need", "reasoning", "music_spec"):
        if key not in data:
            raise ValueError(f"Missing key: {key}")
        val = data[key]
        if isinstance(val, dict):
            raise ValueError(f"{key} must be a string, not dict")
        if not isinstance(val, str) or not val.strip():
            raise ValueError(f"{key} must be a non-empty string")

    need = _truncate_field(data["need"], _MAX_NEED_LEN)
    reasoning = _truncate_field(data["reasoning"], _MAX_REASONING_LEN)
    music_spec = _truncate_field(data["music_spec"], _MAX_MUSIC_SPEC_LEN)

    if _contains_unsafe_language(need, reasoning, music_spec):
        raise ValueError("Unsafe diagnostic language detected in LLM output")

    if "Stage 1" not in reasoning or "Stage 2" not in reasoning:
        # Soft fix: prepend structure if model omitted headers
        reasoning = (
            "Stage 1 - Theory-of-Mind need inference:\n"
            + reasoning
            + "\n\nStage 2 - Music response planning:\n"
            + f"Final music prompt: {music_spec}"
        )
        reasoning = _truncate_field(reasoning, _MAX_REASONING_LEN)

    return {"need": need, "reasoning": reasoning, "music_spec": music_spec}


def _call_openai_llm(prompt: str) -> dict[str, str]:
    """Call OpenAI-compatible Chat Completions API; return validated result."""
    api_key = _get_llm_api_key()
    if not api_key:
        raise RuntimeError("LLM API key not set (EMOTI_LLM_API_KEY / DEEPSEEK_API_KEY / OPENAI_API_KEY)")

    model = _get_llm_model()
    base_url = _get_llm_base_url()

    from openai import OpenAI

    client = OpenAI(api_key=api_key, base_url=base_url) if base_url else OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": _OPENAI_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"{prompt}\n\n"
                    "Respond with JSON only containing keys: need, reasoning, music_spec."
                ),
            },
        ],
        temperature=0.3,
        max_tokens=800,
        response_format={"type": "json_object"},
    )
    content = response.choices[0].message.content or ""
    parsed = _parse_llm_json(content)
    return _validate_llm_result(parsed)


def _infer_with_openai_or_fallback(state: dict) -> dict[str, str]:
    """Try OpenAI ToM+CoT; on any failure fall back to rule-based tom_cot."""
    try:
        cache_key = _make_cache_key(state)
        if cache_key in _LLM_CACHE:
            return dict(_LLM_CACHE[cache_key])

        prompt = build_tom_prompt(state)
        result = _call_openai_llm(prompt)
        _LLM_CACHE[cache_key] = result
        return dict(result)
    except Exception as exc:
        print(f"[M3] OpenAI LLM unavailable ({type(exc).__name__}), using rule fallback.")
        return infer_with_mode(state, mode="tom_cot")


def infer(state: dict) -> dict[str, str]:
    """
    app.py 调用入口。

    EMOTI_LLM_BACKEND=rule (default): 规则版 ToM+CoT，无网络依赖。
    EMOTI_LLM_BACKEND=openai|deepseek: 真实 LLM（OpenAI 兼容 API）；失败时自动回退 rule。
    """
    try:
        if _uses_llm_api():
            return _infer_with_openai_or_fallback(state)
        return infer_with_mode(state, mode="tom_cot")
    except Exception:
        return dict(_FALLBACK_RESULT)


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    test_state = {
        "dominant_emotion": "fear",
        "confidence": 0.74,
        "fatigue": "high",
        "face_conf": 0.78,
        "speech_conf": 0.61,
        "fatigue_conf": 0.85,
    }

    backend = _get_backend()
    model = _get_llm_model()
    base_url = _get_llm_base_url()
    has_key = bool(_get_llm_api_key())

    print("=== M3 LLM backend config ===")
    print(f"EMOTI_LLM_BACKEND  = {backend!r} (env default: rule)")
    print(f"EMOTI_LLM_MODEL    = {model!r}")
    print(f"EMOTI_LLM_BASE_URL = {base_url!r}")
    print(f"LLM API key set    = {has_key}")
    print()

    print("=== infer(state) [uses backend above] ===")
    try:
        print(infer(test_state))
    except Exception as exc:
        print(f"[error] infer failed: {type(exc).__name__}: {exc}")
    print()

    print("=== infer_with_mode standard (always rule) ===")
    print(infer_with_mode(test_state, mode="standard"))
    print()

    print("=== infer_with_mode tom_cot (always rule) ===")
    print(infer_with_mode(test_state, mode="tom_cot"))
    print()

    if _uses_llm_api() and not has_key:
        print("[note] LLM backend enabled but API key is missing → infer() falls back to rule.")
