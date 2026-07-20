"""
safety.py —— ⑨ 心理健康风险分流（Safety Router，M?）
================================================================
EmotiCompanion · AIAA 3800 · HKUST(GZ)

在「感知融合④ → LLM 推理⑤」之间加一层风险分流。识别策略与 llm_reason.py 一致：
  · 优先真实 LLM：结合「用户文本/对话 + 情绪 + 疲劳 + 置信度」综合判断心理风险
  · LLM 未配置 / 调用失败 → 自动回退规则版（关键词 + 情绪疲劳启发式）
  · 危机词命中 → 无论 LLM 怎么判，强制 high（安全硬兜底）

伦理说明：仅做「困扰信号」提示与求助引导，不做医疗诊断/治疗。
求助热线/咨询方式为固定模板（不由 LLM 生成），避免编造错误的危机资源。

接口契约（app.py 自动加载，缺文件时回退 mock）：
    screen(state, text=None, lang="en") -> {
        "risk_level": "low"|"medium"|"high",
        "score": float,          # 0~1
        "signals": [str],        # 命中的信号 / LLM 原因，打印在 ⑤ 推理区
        "banner": str,           # 一行横幅，前置到 ⑤ 推理文本
        "care_message": str,     # 关怀+求助语（high 时显示在「给你的说明」）
        "pause_music": bool,     # 是否暂停自动推送音乐
        "source": str,           # "llm" / "rule" / "rule(crisis-override)"，调试用
    }
"""
from __future__ import annotations
import re

# ---- 复用 llm_reason 的后端配置（单一真源）；导入失败则退化为纯规则 ----
try:
    from llm_reason import (
        _uses_llm_api, _get_llm_api_key, _get_llm_base_url, _get_llm_model,
        _contains_unsafe_language, _parse_llm_json,
    )
    _LLM_IMPORTS_OK = True
except Exception:
    _LLM_IMPORTS_OK = False

    def _uses_llm_api():
        return False


# ======================= 规则版（回退 + 硬兜底） =======================

_HIGH = tuple(re.compile(p, re.IGNORECASE) for p in (
    r"suicide", r"kill myself", r"end my life", r"want to die",
    r"self[-\s]?harm", r"hurt myself", r"no reason to live", r"can'?t go on",
    r"自杀", r"想死", r"不想活", r"活不下去", r"结束生命", r"自残", r"伤害自己", r"撑不下去",
))
_MED = tuple(re.compile(p, re.IGNORECASE) for p in (
    r"hopeless", r"worthless", r"exhausted", r"can'?t cope", r"overwhelmed",
    r"so alone", r"give up", r"burn(ed)?\s?out", r"depress",
    r"绝望", r"没希望", r"崩溃", r"扛不住", r"好累", r"孤独", r"放弃", r"压力好大", r"喘不过气",
))
_NEG = frozenset({"sad", "fear", "angry", "disgust"})
_RANK = {"low": 0, "medium": 1, "high": 2}


def _rule_text_risk(text):
    t = str(text or "").strip()
    if not t:
        return "low", []
    if any(p.search(t) for p in _HIGH):
        return "high", ["text(rule): 命中高风险/危机词"]
    if any(p.search(t) for p in _MED):
        return "medium", ["text(rule): 命中强烈痛苦词"]
    return "low", []


def _affect_risk(state):
    """仅基于 ④ 状态的辅助信号；单帧情绪最多升到 medium，不单独判 high。"""
    state = state or {}
    emo = str(state.get("dominant_emotion", "neutral")).lower()
    fatigue = str(state.get("fatigue", "medium")).lower()
    conf = float(state.get("confidence", 0.0) or 0.0)
    if emo in _NEG and conf >= 0.6:
        if fatigue == "high":
            return "medium", [f"affect: 负面情绪({emo})+高疲劳, conf={conf:.2f}"]
        return "low", [f"affect: 负面情绪({emo}), conf={conf:.2f}"]
    return "low", []


def _rule_risk(state, text):
    t_lv, t_sig = _rule_text_risk(text)
    a_lv, a_sig = _affect_risk(state)
    level = t_lv if _RANK[t_lv] >= _RANK[a_lv] else a_lv
    return level, (t_sig + a_sig)


# ============================= LLM 版 =============================

_SAFETY_SYSTEM = (
    "You are the Safety Router of EmotiCompanion, an affective MUSIC companion (NOT a medical "
    "device). Assess the user's current psychological-distress risk to decide whether to keep "
    "playing music or to pause and show a care/help-seeking prompt. This is affective triage, "
    "NOT diagnosis.\n"
    "Consider BOTH the user's own words AND the observed multimodal state (emotion, fatigue, "
    "confidence). The user's explicit words weigh most.\n"
    "Risk levels:\n"
    "- high: explicit or strongly implied self-harm/suicidal intent, or severe acute crisis.\n"
    "- medium: notable distress/hopelessness/overwhelm but no self-harm intent.\n"
    "- low: no meaningful distress signal.\n"
    "Be cautious; when genuinely unsure between two levels, pick the LOWER one, EXCEPT never "
    "downgrade explicit self-harm/suicidal intent (always high).\n"
    "OUTPUT JSON ONLY, exactly these keys:\n"
    '{ "risk_level": "low|medium|high", "score": 0.0, "reasons": ["short reason", "..."] }\n'
    "Rules: JSON only, no markdown/fences. score in [0,1]. reasons = 1-3 short phrases. "
    "Do NOT output diagnoses, clinical labels, treatment, or any phone numbers."
)

_VALID_LEVELS = {"low", "medium", "high"}
_SAFETY_CACHE: dict[tuple, tuple] = {}


def _call_safety_llm(user_content):
    """调 OpenAI 兼容 API，返回 (level, signals)；任何问题抛异常交给上层回退。"""
    api_key = _get_llm_api_key()
    if not api_key:
        raise RuntimeError("no api key")
    from openai import OpenAI
    base_url = _get_llm_base_url()
    client = OpenAI(api_key=api_key, base_url=base_url) if base_url else OpenAI(api_key=api_key)
    resp = client.chat.completions.create(
        model=_get_llm_model(),
        messages=[{"role": "system", "content": _SAFETY_SYSTEM},
                  {"role": "user", "content": user_content}],
        temperature=0.0,
        max_tokens=300,
        response_format={"type": "json_object"},
    )
    data = _parse_llm_json(resp.choices[0].message.content or "")
    level = str(data.get("risk_level", "")).lower().strip()
    if level not in _VALID_LEVELS:
        raise ValueError(f"bad risk_level: {level!r}")
    reasons = data.get("reasons") or []
    if isinstance(reasons, str):
        reasons = [reasons]
    reasons = [f"llm: {str(r).strip()}" for r in reasons if str(r).strip()][:3]
    # 拒绝含诊断性措辞的输出（复用 llm_reason 的拦截）
    if _contains_unsafe_language(" ".join(reasons)):
        raise ValueError("unsafe language in reasons")
    return level, (reasons or ["llm: (no reason given)"])


def _llm_risk(state, text, lang):
    """优先 LLM；不可用/失败 → None（交给规则回退）。带轻量缓存。"""
    if not (_LLM_IMPORTS_OK and _uses_llm_api()):
        return None
    state = state or {}
    emo = str(state.get("dominant_emotion", "neutral")).lower()
    fatigue = str(state.get("fatigue", "medium")).lower()
    conf = float(state.get("confidence", 0.0) or 0.0)
    text = str(text or "").strip()

    key = (emo, fatigue, round(conf, 1), text)
    if key in _SAFETY_CACHE:
        return _SAFETY_CACHE[key]

    user_content = (
        f"Observed state — emotion: {emo}, fatigue: {fatigue}, confidence: {conf:.2f}.\n"
        f"User text/dialogue: {text or '(none provided)'}\n"
        "Assess psychological-distress risk. Respond with JSON only."
    )
    try:
        result = _call_safety_llm(user_content)
        _SAFETY_CACHE[key] = result
        return result
    except Exception as exc:
        print(f"[M?/safety] LLM screen unavailable ({type(exc).__name__}), rule fallback.")
        return None


# ===================== 关怀 / 求助资源（固定模板，不由 LLM 生成） =====================

# 心理咨询预约方式（HKUST(GZ)）+ 国家心理援助热线。号码/链接为安全关键内容，写死不外传给模型。
_CARE_ZH = (
    "看起来你现在可能正经历很难熬的时刻。你并不孤单——愿意的话，可以和信任的人聊聊，"
    "或通过下面的方式联系专业支持：\n"
    "· 心理咨询预约方式（香港科技大学（广州））：\n"
    "  - 线上预约：http://counsel.hkust-gz.edu.cn/ （用校园用户名和密码登录）\n"
    "  - 邮件预约：counseling@hkust-gz.edu.cn\n"
    "  - 现场预约：C2 校园活动中心二楼 202 房间 · 心理咨询中心（工作时间）\n"
    "· 国家心理援助热线：12356\n"
    "如果你有伤害自己的念头，请立即拨打 12356 或前往最近的急诊。我会先暂停自动推送音乐，把这个空间留给你。"
)
_CARE_EN = (
    "It looks like you may be going through a really hard moment. You're not alone — if you're "
    "willing, reach out to someone you trust, or to professional support:\n"
    "· HKUST(GZ) counseling appointments:\n"
    "  - Online: http://counsel.hkust-gz.edu.cn/ (log in with your campus username & password)\n"
    "  - Email: counseling@hkust-gz.edu.cn\n"
    "  - In person: Room 202, 2/F, C2 Campus Activity Center · Counseling Center (working hours)\n"
    "· National psychological assistance hotline: 12356\n"
    "If you have thoughts of hurting yourself, please call 12356 now or go to the nearest "
    "emergency service. I'll pause auto music push and keep this space for you."
)


# ===================== 输出拼装（两路径共用） =====================

def _build_output(level, signals, lang, source):
    zh = str(lang).lower().startswith("zh")
    score = {"low": 0.15, "medium": 0.55, "high": 0.9}[level]

    if level == "high":
        banner = ("⚠️ [Safety Router] 风险较高 → 已暂停自动推送音乐，改为关怀与求助提示。"
                  if zh else
                  "⚠️ [Safety Router] elevated risk → auto music paused; showing care & help prompt.")
        care = _CARE_ZH if zh else _CARE_EN
    elif level == "medium":
        banner = ("🟡 [Safety Router] 检测到一些压力信号 → 正常推送，音乐偏向舒缓陪伴。"
                  if zh else
                  "🟡 [Safety Router] some distress signals → normal push, leaning soothing.")
        care = ""
    else:
        banner = ("🟢 [Safety Router] 风险较低 → 正常推送音乐。" if zh
                  else "🟢 [Safety Router] low risk → normal music push.")
        care = ""

    return {"risk_level": level, "score": score, "signals": signals,
            "banner": banner, "care_message": care,
            "pause_music": (level == "high"), "source": source}


# ============================= 对外入口 =============================

def screen(state, text=None, lang="en"):
    text = str(text or "")
    rule_level, rule_sig = _rule_risk(state, text)

    # 省钱短路：没文本且情绪风险低 → 直接 low，不调 LLM
    if not text.strip() and rule_level == "low":
        return _build_output("low", rule_sig, lang, "rule(skip-llm)")

    # 危机词硬兜底：规则判 high 就一定 high，不给 LLM 下调的机会
    if rule_level == "high":
        return _build_output("high", rule_sig, lang, "rule(crisis-override)")

    # 其余情况优先 LLM；失败 → 规则
    llm = _llm_risk(state, text, lang)
    if llm is not None:
        level, signals = llm
        return _build_output(level, signals, lang, "llm")
    return _build_output(rule_level, rule_sig, lang, "rule")


# ============================= 自测 =============================

if __name__ == "__main__":
    print(screen({"dominant_emotion": "neutral", "fatigue": "low", "confidence": 0.9}, "", "zh"))
    print(screen({"dominant_emotion": "sad", "fatigue": "high", "confidence": 0.8}, "我好累，压力好大", "zh"))
    print(screen({"dominant_emotion": "sad", "fatigue": "high", "confidence": 0.8}, "我不想活了", "en"))
