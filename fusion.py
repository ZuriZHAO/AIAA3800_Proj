"""
fusion.py - M2/M4 multimodal emotion fusion for EmotiCompanion.

Project contract used by app.py:
    fuse(face, speech, fatigue) -> {
        "dominant_emotion": str,
        "confidence": float,
        "fatigue": str,
        "face_conf": float,
        "speech_conf": float,
        "fatigue_conf": float
    }

Fusion modes:
    1. naive        : equal-vote / simple voting baseline (arm C)
    2. weighted     : confidence-weighted fusion, DEFAULT for the real app (arm D)
    3. weighted_cam : weighted + GradCAM face-reliability gating (arm E)
    4. bayes        : per-class-reliability Bayesian late fusion (arm F). Uses offline-
                      learned confusion likelihoods (fusion_bayes_priors.json). This is the
                      only mode that beat the best single modality on a balanced regime
                      (see docs/experiment_plan.md §3.7). NOT default: gain is small and
                      depends on the priors matching the deployment distribution; the
                      output class set is limited to the priors' training classes.

Usage in app.py:
    state = fuse(face, speech, fatigue)          # defaults to weighted
Usage in experiments:
    state_c = fuse(face, speech, fatigue, mode="naive")
    state_d = fuse(face, speech, fatigue, mode="weighted")
    state_f = fuse(face, speech, fatigue, mode="bayes")
"""

from __future__ import annotations

import json
import math
import os
from typing import Dict, Mapping, Optional, Tuple

try:
    from config import EMOTION_LABELS, FATIGUE_LEVELS, MOCK_EMOTION, MOCK_FATIGUE
except Exception:
    # Fallback makes this file runnable by itself during local debugging.
    EMOTION_LABELS = ["neutral", "happy", "sad", "angry", "fear", "surprise", "disgust"]
    FATIGUE_LEVELS = ["low", "medium", "high"]
    MOCK_EMOTION = "neutral"
    MOCK_FATIGUE = "low"

# Static reliability weights for the weighted arm D.
# These match the M2 plan: speech is slightly more stable than a single face frame.
FACE_WEIGHT = 0.45
SPEECH_WEIGHT = 0.55

# In naive two-modality voting, disagreements are ties. The tie rule must be fixed
# and must not use confidence, otherwise the naive arm becomes another weighted arm.
NAIVE_TIE_PRIORITY = ("speech", "face")


def _clamp01(value, default: float = 0.0) -> float:
    """Convert value to float and clamp to [0, 1]."""
    try:
        x = float(value)
    except (TypeError, ValueError):
        x = default
    return max(0.0, min(1.0, x))


def _norm_emotion(value) -> str:
    """Normalize emotion into the project emotion label set."""
    emo = str(value or "").strip().lower()
    return emo if emo in EMOTION_LABELS else MOCK_EMOTION


def _norm_fatigue(value) -> str:
    """Normalize fatigue level into the project fatigue label set."""
    fatigue = str(value or "").strip().lower()
    return fatigue if fatigue in FATIGUE_LEVELS else MOCK_FATIGUE


def _read_inputs(
    face: Optional[Mapping],
    speech: Optional[Mapping],
    fatigue: Optional[Mapping],
) -> Tuple[str, float, str, float, str, float]:
    """Read and sanitize face/speech/fatigue dictionaries."""
    face = face or {}
    speech = speech or {}
    fatigue = fatigue or {}

    face_emo = _norm_emotion(face.get("emotion"))
    speech_emo = _norm_emotion(speech.get("emotion"))
    fatigue_level = _norm_fatigue(fatigue.get("fatigue_level") or fatigue.get("fatigue"))

    face_conf = _clamp01(face.get("confidence", 0.0))
    speech_conf = _clamp01(speech.get("confidence", 0.0))
    fatigue_conf = _clamp01(fatigue.get("confidence", 0.0))

    return face_emo, face_conf, speech_emo, speech_conf, fatigue_level, fatigue_conf


def _result(
    dominant: str,
    confidence: float,
    fatigue_level: str,
    face_conf: float,
    speech_conf: float,
    fatigue_conf: float,
    mode: str,
    face_emo: str,
    speech_emo: str,
) -> Dict:
    """Build the unified state JSON expected by app.py and later LLM reasoning."""
    return {
        "dominant_emotion": _norm_emotion(dominant),
        "confidence": round(_clamp01(confidence), 2),
        "fatigue": _norm_fatigue(fatigue_level),
        "face_conf": round(_clamp01(face_conf), 2),
        "speech_conf": round(_clamp01(speech_conf), 2),
        "fatigue_conf": round(_clamp01(fatigue_conf), 2),
        # Extra fields are useful for experiment logging; downstream modules can ignore them.
        "fusion_mode": mode,
        "face_emotion": face_emo,
        "speech_emotion": speech_emo,
        "modal_agreement": bool(face_emo == speech_emo),
    }


def fuse_naive(face, speech, fatigue=None) -> Dict:
    """
    Naive fusion arm C: equal-vote baseline.

    Each available modality contributes one vote to its predicted emotion.
    With only face and speech, disagreement creates a tie, so we use a fixed
    tie-break priority speech > face. Confidence is reported but not used to
    choose the emotion, keeping this arm distinct from confidence-weighted fusion.
    """
    face_emo, face_conf, speech_emo, speech_conf, fatigue_level, fatigue_conf = _read_inputs(
        face, speech, fatigue
    )

    # If both modalities are effectively unavailable, return safe neutral.
    if face_conf == 0.0 and speech_conf == 0.0:
        return _result(
            MOCK_EMOTION, 0.0, fatigue_level, face_conf, speech_conf, fatigue_conf,
            "naive", face_emo, speech_emo
        )

    votes = {}
    sources = {"face": face_emo, "speech": speech_emo}
    for source, emo in sources.items():
        # If a modality has zero confidence, treat it as missing.
        conf = face_conf if source == "face" else speech_conf
        if conf > 0.0:
            votes[emo] = votes.get(emo, 0) + 1

    max_vote = max(votes.values())
    candidates = [emo for emo, count in votes.items() if count == max_vote]

    if len(candidates) == 1:
        dominant = candidates[0]
    else:
        # Fixed, non-confidence tie rule for equal-vote baseline.
        dominant = speech_emo if "speech" in NAIVE_TIE_PRIORITY and speech_conf > 0 else face_emo

    # Naive confidence is a conservative agreement/reliability score for reporting.
    if face_emo == speech_emo and face_conf > 0.0 and speech_conf > 0.0:
        fusion_conf = (face_conf + speech_conf) / 2.0
    else:
        # Conflict should lower the confidence even if one source is confident.
        active_confs = [c for c in (face_conf, speech_conf) if c > 0.0]
        fusion_conf = 0.5 * (sum(active_confs) / len(active_confs)) if active_confs else 0.0

    return _result(
        dominant, fusion_conf, fatigue_level, face_conf, speech_conf, fatigue_conf,
        "naive", face_emo, speech_emo
    )


def fuse_weighted(face, speech, fatigue=None, use_reliability=False) -> Dict:
    """
    Weighted fusion: confidence-weighted voting.

    Score(emotion) = sum(static_modality_weight * modality_confidence)
    The modality weights are normalized over active modalities. Fatigue is not used
    for emotion voting; it is passed through as an independent arousal/energy signal.

    use_reliability=False -> arm D（原样）。
    use_reliability=True  -> arm E（路线B · CAM 门控）：把人脸权重乘以 GradCAM 派生的
      人脸可靠性 face["reliability"]∈[0,1]（缺省 1.0=不门控）。注意力跑到脸外→可靠性低
      →人脸权重被压低、语音占比升高。可靠性怎么算见 face_emotion.cam_reliability()。
    """
    face_emo, face_conf, speech_emo, speech_conf, fatigue_level, fatigue_conf = _read_inputs(
        face, speech, fatigue
    )

    mode_str = "weighted_cam" if use_reliability else "weighted"

    # 人脸可靠性：只在 arm E 生效；设 0.1 下限，避免 r=0 且语音缺失时 weight_sum 为 0。
    r_face = 1.0
    if use_reliability:
        r_face = max(0.1, _clamp01(face.get("reliability", 1.0)))

    active = []
    if face_conf > 0.0:
        active.append(("face", face_emo, FACE_WEIGHT * r_face, face_conf))
    if speech_conf > 0.0:
        active.append(("speech", speech_emo, SPEECH_WEIGHT, speech_conf))

    if not active:
        return _result(
            MOCK_EMOTION, 0.0, fatigue_level, face_conf, speech_conf, fatigue_conf,
            mode_str, face_emo, speech_emo
        )

    # Normalize weights over available modalities, so if one modality is missing,
    # the other can still produce a meaningful confidence.
    weight_sum = sum(w for _, _, w, _ in active)
    scores = {}
    for _, emo, weight, conf in active:
        normalized_weight = weight / weight_sum
        scores[emo] = scores.get(emo, 0.0) + normalized_weight * conf

    # Deterministic tie-break: highest score; if exactly tied, prefer speech label, then face label.
    max_score = max(scores.values())
    candidates = [emo for emo, score in scores.items() if abs(score - max_score) < 1e-12]
    if len(candidates) == 1:
        dominant = candidates[0]
    elif speech_emo in candidates:
        dominant = speech_emo
    elif face_emo in candidates:
        dominant = face_emo
    else:
        dominant = sorted(candidates)[0]

    fusion_conf = scores.get(dominant, 0.0)

    return _result(
        dominant, fusion_conf, fatigue_level, face_conf, speech_conf, fatigue_conf,
        mode_str, face_emo, speech_emo
    )


# Offline-learned per-class reliability priors for the bayes arm (arm F).
# Small committed artifact (like models/face_landmarker.task), NOT runtime output.
# Regenerate: python scripts/fusion_bayes.py --data <d> --cache <c> --export models/fusion_bayes_priors.json
_BAYES_PRIORS_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "models", "fusion_bayes_priors.json")
_BAYES_PRIORS: Optional[Dict] = None
_BAYES_LOAD_TRIED = False


def _load_bayes_priors() -> Optional[Dict]:
    """Lazy-load the Bayesian priors JSON once; return None if unavailable."""
    global _BAYES_PRIORS, _BAYES_LOAD_TRIED
    if _BAYES_LOAD_TRIED:
        return _BAYES_PRIORS
    _BAYES_LOAD_TRIED = True
    try:
        with open(_BAYES_PRIORS_PATH, encoding="utf-8") as f:
            _BAYES_PRIORS = json.load(f)
    except Exception:
        _BAYES_PRIORS = None
    return _BAYES_PRIORS


def fuse_bayes(face, speech, fatigue=None) -> Dict:
    """
    Bayesian per-class-reliability fusion (arm F).

    Combines the two modalities by their *offline-learned per-class reliability*
    (confusion likelihoods) rather than self-reported confidence:

        P(true=c | face=f, speech=s) ∝ P(c) · P(face=f | true=c) · P(speech=s | true=c)

    The priors P(c) and likelihoods P(pred|true) are learned offline on a labeled
    validation set (fusion_bayes_priors.json). A missing modality (confidence 0) simply
    drops its likelihood term. Output emotions are restricted to the priors' training
    classes. If the priors file is absent, this falls back to weighted fusion so the app
    never breaks.
    """
    priors = _load_bayes_priors()
    face_emo, face_conf, speech_emo, speech_conf, fatigue_level, fatigue_conf = _read_inputs(
        face, speech, fatigue
    )

    if priors is None:
        # Graceful fallback: behave like weighted, but tag the mode so logs are honest.
        res = fuse_weighted(face, speech, fatigue, use_reliability=False)
        res["fusion_mode"] = "bayes(priors-missing->weighted)"
        return res

    if face_conf == 0.0 and speech_conf == 0.0:
        return _result(
            MOCK_EMOTION, 0.0, fatigue_level, face_conf, speech_conf, fatigue_conf,
            "bayes", face_emo, speech_emo
        )

    classes = priors["classes"]
    face_lik = priors["face_lik"]
    speech_lik = priors["speech_lik"]
    prior = priors["prior"]

    log_scores = {}
    for c in classes:
        s = math.log(prior[c])
        if face_conf > 0.0:  # only count a modality that actually reported
            lc = face_lik[c]
            s += math.log(lc.get(face_emo, lc.get("__floor__", 1e-6)))
        if speech_conf > 0.0:
            lc = speech_lik[c]
            s += math.log(lc.get(speech_emo, lc.get("__floor__", 1e-6)))
        log_scores[c] = s

    # argmax with a deterministic tie-break consistent with the other arms.
    max_log = max(log_scores.values())
    candidates = [c for c, v in log_scores.items() if abs(v - max_log) < 1e-12]
    if len(candidates) == 1:
        dominant = candidates[0]
    elif speech_emo in candidates:
        dominant = speech_emo
    elif face_emo in candidates:
        dominant = face_emo
    else:
        dominant = sorted(candidates)[0]

    # Confidence = softmax posterior of the winning class (numerically stable).
    exps = {c: math.exp(v - max_log) for c, v in log_scores.items()}
    total = sum(exps.values()) or 1.0
    fusion_conf = exps[dominant] / total

    return _result(
        dominant, fusion_conf, fatigue_level, face_conf, speech_conf, fatigue_conf,
        "bayes", face_emo, speech_emo
    )


def fuse(face, speech, fatigue=None, mode: Optional[str] = None) -> Dict:
    """
    Public interface used by app.py and experiment scripts.

    Parameters:
        face:    {"emotion": str, "confidence": float, ...}
        speech:  {"emotion": str, "confidence": float, "reasoning": str, ...}
        fatigue: {"fatigue_level": str, "confidence": float, ...}
        mode:    "weighted" (default) / "naive" / "weighted_cam" / "bayes".
                 If omitted, uses env FUSION_MODE or weighted.

    Returns:
        unified state JSON.
    """
    mode = (mode or os.getenv("FUSION_MODE", "weighted")).strip().lower()
    if mode in {"naive", "simple", "vote", "voting", "equal"}:
        return fuse_naive(face, speech, fatigue)
    if mode in {"weighted", "confidence", "confidence_weighted", "cw"}:
        return fuse_weighted(face, speech, fatigue, use_reliability=False)
    if mode in {"weighted_cam", "cam", "cam_gated", "e"}:
        return fuse_weighted(face, speech, fatigue, use_reliability=True)
    if mode in {"bayes", "bayesian", "per_class", "perclass", "f"}:
        return fuse_bayes(face, speech, fatigue)
    raise ValueError(
        f"Unknown fusion mode: {mode}. Use 'naive' / 'weighted' / 'weighted_cam' / 'bayes'.")


if __name__ == "__main__":
    tests = [
        (
            {"emotion": "neutral", "confidence": 0.50},
            {"emotion": "happy", "confidence": 0.95, "reasoning": "bright tone"},
            {"fatigue_level": "low", "confidence": 0.80},
        ),
        (
            {"emotion": "sad", "confidence": 0.90},
            {"emotion": "neutral", "confidence": 0.40, "reasoning": "unclear"},
            {"fatigue_level": "medium", "confidence": 0.70},
        ),
        (
            {"emotion": "angry", "confidence": 0.60},
            {"emotion": "angry", "confidence": 0.80, "reasoning": "sharp tone"},
            {"fatigue_level": "high", "confidence": 0.60},
        ),
        (
            {"emotion": "neutral", "confidence": 0.0},
            {"emotion": "neutral", "confidence": 0.0, "reasoning": "fallback"},
            {"fatigue_level": "low", "confidence": 0.0},
        ),
    ]

    for i, (face, speech, fatigue) in enumerate(tests, 1):
        print("=" * 80)
        print(f"Test {i}")
        print("naive   :", fuse(face, speech, fatigue, mode="naive"))
        print("weighted:", fuse(face, speech, fatigue, mode="weighted"))
