"""
EmotiCompanion —— 多模态情绪感知的实时音乐陪伴系统
====================================================
AIAA 3800 课程项目 · HKUST(GZ) · June 2026

模块 ⑧：Gradio UI + 集成（M4 主导）—— app.py

────────────────────────────────────────────────────────────────────────
这是整个系统的「胶水层」，本身不实现任何模型。它做三件事：
  1. 动态导入各成员各自文件里的真实实现；若该文件还没写好，自动回退到 mock
  2. 串联完整 pipeline：感知(①②③) → 融合(④) → LLM 推理(⑤) → 音乐生成(⑥)
     并把 GradCAM(⑦) 旁挂在人脸模块上
  3. 两种工作模式：
     · 自动模式（默认）：打开网页后摄像头/麦克风持续工作，每 60s 捕捉一次，
       音乐陪伴一直存在，直到关闭网页。情绪与压力都不变时不更换音乐。
     · 手动模式（视频版）：点击「● 录制」开始录一段视频，点「■ 停止」结束录制；
       结束后自动从视频里随机抽一帧当人脸图像、分离音频当语音，直接跑单次 pipeline。

常量与 schema 见 config.py；mock 实现见 mocks.py。
各成员只需在自己的文件里实现约定的函数签名，app.py 会自动接上。
────────────────────────────────────────────────────────────────────────

INTERFACE CONTRACT（接口契约）
  ① 人脸情绪 (M1)  face_emotion.py : predict(image) -> {"emotion", "confidence"}
  ⑦ GradCAM  (M1)  face_emotion.py : gradcam(image) -> 热力图 image
  ② 疲劳检测 (M4)  fatigue.py      : predict(image) -> {"fatigue_level", "confidence"}
  ③ 语音情绪 (M2)  speech_emotion.py: predict(audio_path) -> {"emotion","confidence","reasoning"}
  ④ 融合 (M2/M4)   fusion.py       : fuse(face, speech, fatigue) -> 统一状态 JSON
  ⑤ LLM 推理 (M3)  llm_reason.py   : infer(state) -> {"need","reasoning","music_spec"}
  ⑥ 音乐生成 (M3)  music_gen.py    : generate(music_spec) -> 音频(路径 str 或 (sr, ndarray))

运行：
    pip install -r requirements.txt      # 注意 gradio 需 >= 4.39
    python app.py
    # 打开浏览器里的本地地址，默认进入自动模式
"""

import json
import os
import tempfile
import time
import wave
import concurrent.futures

from dotenv import load_dotenv

load_dotenv(override=True)

# 让本地回环地址绕过系统/环境代理。否则开了科学上网（系统代理指向
# 127.0.0.1:xxxx）时，gradio 启动后自检 http://127.0.0.1 的请求会被代理
# 拦截返回 502，导致 app.launch() 直接抛异常退出。在这里把 localhost 加进
# no_proxy，对所有组员（无论是否开代理）都安全生效。
for _v in ("no_proxy", "NO_PROXY"):
    _cur = os.environ.get(_v, "")
    _need = "127.0.0.1,localhost,::1"
    os.environ[_v] = f"{_cur},{_need}" if _cur else _need

# 绕过 Anaconda 常见的 OpenMP 运行时冲突（OMP: Error #15: libiomp5md.dll
# already initialized）。多个库（numpy/mkl、torch、mediapipe 等）各自静态链了一份
# OpenMP，加载时会撞车导致进程直接崩退。这里在任何重依赖 import 之前把它设成 TRUE，
# 允许重复加载（官方标注为 unsafe 但对本项目实测稳定，不设则常常起不来）。
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

# 语音后端默认用 emotion2vec+ —— 跨所有测试数据集表现最好（见 docs/experiment_plan.md §4）。
# 用 setdefault，想临时切回 API/其它后端时 `SPEECH_BACKEND=api python app.py` 仍可覆盖。
# （需已装 funasr/modelscope/torchaudio；缺依赖时 speech_emotion 会回退，见 requirements.txt）
os.environ.setdefault("SPEECH_BACKEND", "emotion2vec")

import numpy as np
import gradio as gr

from config import AUTO_INTERVAL_SEC as CONFIG_AUTO_INTERVAL_SEC
import mocks
from audio_extract import extract_audio


def _ensure_ffmpeg_on_path():
    """确保命令名 `ffmpeg` 能在 PATH 上被找到。

    背景：Gradio 的 gr.Video 预处理（webcam 录制时默认做一次 hflip / 格式转换）内部
    直接以命令名 `ffmpeg` 调用；本项目的 extract_audio 也优先用系统 ffmpeg。很多机器
    只装了 pip 的 imageio-ffmpeg —— 它的二进制名带版本号（如 ffmpeg-win-x86_64-v7.1.exe），
    PATH 上按 `ffmpeg` 认不出，于是 gr.Video 会抛 FFExecutableNotFoundError。
    这里把 imageio-ffmpeg 的二进制复制成标准名 `ffmpeg(.exe)` 放到固定临时目录并加进 PATH，
    让所有按命令名调用 ffmpeg 的地方（Gradio 内部 + extract_audio）都能正常工作。失败静默。
    """
    import shutil

    if shutil.which("ffmpeg"):
        return
    try:
        import imageio_ffmpeg
        src = imageio_ffmpeg.get_ffmpeg_exe()
    except Exception as e:
        print(f"[ffmpeg] imageio-ffmpeg 不可用，gr.Video/抽音频可能失败：{e}")
        return
    exe_name = "ffmpeg.exe" if os.name == "nt" else "ffmpeg"
    shim_dir = os.path.join(tempfile.gettempdir(), "emoti_ffmpeg_bin")
    dst = os.path.join(shim_dir, exe_name)
    try:
        os.makedirs(shim_dir, exist_ok=True)
        if not os.path.exists(dst) or os.path.getsize(dst) != os.path.getsize(src):
            shutil.copy2(src, dst)
        os.environ["PATH"] = shim_dir + os.pathsep + os.environ.get("PATH", "")
        print(f"[ffmpeg] 已用 imageio-ffmpeg 提供 `ffmpeg`：{dst}")
    except Exception as e:
        print(f"[ffmpeg] 建立 ffmpeg 别名失败：{e}")


_ensure_ffmpeg_on_path()

# Auto mode timing:
# AUTO_INTERVAL_SEC: perception/fusion refresh interval
# MUSIC_REFRESH_SEC: when state is unchanged, generate a fresh music variation after this many seconds
AUTO_INTERVAL_SEC = int(os.getenv("AUTO_INTERVAL_SEC", str(CONFIG_AUTO_INTERVAL_SEC)))
MUSIC_REFRESH_SEC = int(os.getenv("MUSIC_REFRESH_SEC", "90"))

# ⑥ 音乐生成时长（秒）：UI 上一个 0~MAX 可拖动的滑条来选择，默认 DEFAULT。
MUSIC_MAX_DURATION_SEC = int(os.getenv("MUSICGEN_MAX_DURATION_SEC", "120"))
MUSIC_DEFAULT_DURATION_SEC = max(
    0, min(MUSIC_MAX_DURATION_SEC, int(os.getenv("MUSICGEN_DURATION_SEC", "15"))))


# =============================================================================
# 动态加载各成员模块 —— 有真实文件就用真实的，没有就用 mock
# =============================================================================

def _load(module_name, mock, label):
    """尝试 import 成员的模块文件；失败则回退 mock 并打印提示。"""
    try:
        mod = __import__(module_name)
        print(f"[OK ] {label}: loaded real module {module_name}.py")
        return mod
    except Exception as e:  # 文件不存在 / 依赖未装 / 导入报错，都回退 mock
        print(f"[mock] {label}: using mock ({module_name}.py not ready: {e})")
        return mock


FACE = _load("face_emotion", mocks.MockFace, "① face + ⑦ GradCAM (M1)")
FATIGUE = _load("fatigue", mocks.MockFatigue, "② fatigue (M4)")
SPEECH = _load("speech_emotion", mocks.MockSpeech, "③ speech (M2)")
FUSION = _load("fusion", mocks.MockFusion, "④ fusion (M2/M4)")
LLM = _load("llm_reason", mocks.MockLLM, "⑤ LLM reasoning (M3)")
MUSIC = _load("music_gen", mocks.MockMusic, "⑥ music gen (M3)")
SAFETY = _load("safety", mocks.MockSafety, "⑨ safety router (M?)")


# =============================================================================
# Pipeline 辅助函数
# =============================================================================

def _state_key(state):
    """从融合状态里抽出「情绪 + 压力」二元组，作为是否更换音乐的判断依据。"""
    return (state.get("dominant_emotion"), state.get("fatigue"))


# ---- 语音转写（供 ⑨ 风险筛查）的开销控制 ----
# 只有音频时长 ≥ 此值才转写，跳过静音/极短片段，省一次 Omni 调用。
MIN_TRANSCRIBE_SEC = float(os.getenv("SAFETY_MIN_TRANSCRIBE_SEC", "1.5"))
# 自动模式是否转写：默认开（仍受时长门控）；设 0 则自动模式完全不转写，只用情绪+疲劳筛查。
AUTO_TRANSCRIBE = os.getenv("SAFETY_AUTO_TRANSCRIBE", "1").strip().lower() not in {"0", "false", "no", ""}
# 复用的线程池：让「语音转写(网络)」与本地感知①②③⑦并发，而不是串行等待。
_PERCEPTION_POOL = concurrent.futures.ThreadPoolExecutor(max_workers=2)


def _should_transcribe(audio):
    """音频足够长（可能含人声）才值得转写；读不出时长时保守放行。"""
    if not audio:
        return False
    try:
        with wave.open(str(audio), "rb") as wf:
            dur = wf.getnframes() / float(wf.getframerate() or 1)
        return dur >= MIN_TRANSCRIBE_SEC
    except Exception:
        return True


def _perceive_and_fuse(image, audio, speech_mode=None, mode=None):
    """跑 ①②③ 感知 + ⑦ 热力图 + ④ 融合，返回 (perception, heatmap, state)。

    mode：'auto' / 'manual' / None，仅用于决定是否做语音转写（供 ⑨ 风险筛查）。
    语音转写走 Omni API（网络），这里与本地感知并发执行以省串行等待。
    """
    backend = _set_speech_backend_from_ui(speech_mode)

    # 先并发启动「语音转写」（网络耗时），与下面的本地感知①②③⑦重叠。
    # 仅在音频够长、且未被开关关闭时才转写（自动模式可用 SAFETY_AUTO_TRANSCRIBE=0 关掉）。
    want_transcribe = (
        hasattr(SPEECH, "transcribe")
        and _should_transcribe(audio)
        and not (mode == "auto" and not AUTO_TRANSCRIBE)
    )
    transcribe_future = _PERCEPTION_POOL.submit(SPEECH.transcribe, audio) if want_transcribe else None

    face = FACE.predict(image)          # ① 人脸情绪 (M1)
    heatmap = FACE.gradcam(image)       # ⑦ GradCAM (M1)
    fatigue = FATIGUE.predict(image)    # ② 疲劳/压力 (M4)
    speech = SPEECH.predict(audio)      # ③ 语音情绪 (M2)

    # 在 UI 输出里显式记录本次使用的 speech backend，方便 user study / debug 对齐。
    if isinstance(speech, dict):
        speech = dict(speech)
        speech.setdefault("backend", backend)

    state = FUSION.fuse(face, speech, fatigue)  # ④ 融合 (M2/M4)

    # 取回并发的转写结果（失败/超时 → 空串，风险筛查退化为只看情绪+疲劳）。
    transcript = ""
    if transcribe_future is not None:
        try:
            transcript = transcribe_future.result(timeout=90) or ""
        except Exception as e:
            print(f"[safety] transcribe failed: {e}")

    perception = {"face": face, "speech": speech, "fatigue": fatigue, "transcript": transcript}
    return perception, heatmap, state


# 会话日志：每次生成音乐时把「系统输出」追加进一个 JSONL，供 user study 事后对齐。
# 记录完整的 ①②③ 感知 / ④ 融合状态 / ⑤ 推理(need+reasoning+music_spec)，全是标签/
# 置信度/文本，**不含**人脸图像或录音。为兼容 docs/user_study.md 的分析口径，保留旧的
# sys_emotion/sys_fatigue/sys_need/sys_music_spec 概要字段，另加 perception/state/reasoning 全量。
# 路径默认 results/session_log.jsonl（results/ 已 gitignore）；可用 STUDY_LOG 覆盖。
# 用「会话标签」区分数据：实验时填参与者编号(P01…)，平时测试保持 test/自己的名字。
SESSION_LOG_PATH = os.getenv("STUDY_LOG", os.path.join("results", "session_log.jsonl"))


def _log_session(state, need, session_label, perception=None, mode=None):
    """把一次生成的系统输出（完整感知/融合/推理）追加进会话日志（失败静默，绝不影响主流程）。"""
    try:
        os.makedirs(os.path.dirname(SESSION_LOG_PATH) or ".", exist_ok=True)
        rec = {
            "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
            "label": (str(session_label).strip() or "test") if session_label else "test",
            "mode": mode,                                 # 推理模式：tom_cot / standard
            # ---- 概要字段（保留旧口径，docs/user_study.md 直接用）----
            "sys_emotion": state.get("dominant_emotion"),
            "sys_fatigue": state.get("fatigue"),
            "sys_need": need.get("need"),
            "sys_music_spec": need.get("music_spec"),
            # ---- 全量字段（UI 上显示的内容都落盘）----
            "perception": perception,                     # ①②③ face / speech / fatigue 全部
            "state": state,                               # ④ 多模态融合统一状态 JSON
            "reasoning": need.get("reasoning"),           # ⑤ LLM 推理链（ToM + CoT）
            "latency": need.get("_latency"),              # LLM / MusicGen 延迟
        }
        with open(SESSION_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    except Exception as e:  # 日志不是核心功能，出错只提示不中断
        print(f"[study-log] skip ({e})")


# 可视化留存：user study 时保存 GradCAM 热力图(⑦) + 疲劳关键点可视化(②) 到 <media>/<label>/。
# ⚠️ 这两张都是**人脸衍生图**（含被试面容），与「不保存人脸」相冲突，故**默认关闭**；
#    只有显式 STUDY_SAVE_MEDIA=1 且**已取得被试知情同意**时才开启（见 docs/user_study.md 附录 A）。
STUDY_SAVE_MEDIA = os.getenv("STUDY_SAVE_MEDIA", "0").strip().lower() in {"1", "true", "yes", "on"}
SESSION_MEDIA_DIR = os.getenv("STUDY_MEDIA_DIR", os.path.join("results", "session_media"))


def _save_media(image, heatmap, session_label):
    """保存本次的 GradCAM 热力图 + 疲劳可视化（默认关闭；失败静默）。"""
    if not STUDY_SAVE_MEDIA or image is None:
        return
    try:
        import cv2
        label = (str(session_label).strip() or "test") if session_label else "test"
        d = os.path.join(SESSION_MEDIA_DIR, label)
        os.makedirs(d, exist_ok=True)
        ts = time.strftime("%H%M%S")
        if heatmap is not None:                      # ⑦ GradCAM（RGB → BGR 存盘）
            cv2.imwrite(os.path.join(d, f"gradcam_{ts}.png"),
                        cv2.cvtColor(np.asarray(heatmap), cv2.COLOR_RGB2BGR))
        fviz = FATIGUE.visualize(image) if hasattr(FATIGUE, "visualize") else None
        if fviz is not None:                         # ② 疲劳关键点可视化
            cv2.imwrite(os.path.join(d, f"fatigue_{ts}.png"),
                        cv2.cvtColor(np.asarray(fviz), cv2.COLOR_RGB2BGR))
    except Exception as e:
        print(f"[study-media] skip ({e})")


def _compose_one(state, mode, session_label=None, duration_sec=None, perception=None):
    """跑单个推理模式的 ⑤ LLM + ⑥ 音乐生成，返回 (need_dict, music)。

    mode：'tom_cot' 或 'standard'（已归一化）。perception 仅用于会话日志留存。
    """
    t0 = time.perf_counter()

    if hasattr(LLM, "infer_with_mode"):
        need = LLM.infer_with_mode(state, mode)   # ⑤ (M3) 支持模式切换
    else:
        need = LLM.infer(state)                   # mock 或旧接口：回退默认

    t1 = time.perf_counter()

    music_spec = str(need.get("music_spec", "") or "")
    print("=" * 60)
    print(f"[M3] mode={mode}")
    print(f"[M3] UI duration_sec={duration_sec}")
    print(f"[M3] music_spec={music_spec[:500]}")
    print("=" * 60)

    # ⑥ MusicGen (M3)，按 UI 滑条时长生成；兼容尚未支持 duration 参数的旧签名。
    try:
        music = MUSIC.generate(music_spec, duration_sec)
    except TypeError:
        music = MUSIC.generate(music_spec)

    t2 = time.perf_counter()

    need["_latency"] = {
        "llm_sec": round(t1 - t0, 2),
        "music_sec": round(t2 - t1, 2),
        "llm_music_total_sec": round(t2 - t0, 2),
    }

    # 记录系统输出（完整感知/融合/推理，user study 用）
    _log_session(state, need, session_label, perception=perception, mode=mode)
    return need, music


def _one_mode_text(need, cot=True):
    tag = "Reasoning (CoT)" if cot else "Reasoning"
    lat = need.get("_latency") or {}
    latency_text = ""
    if lat:
        latency_text = (
            "\n\n[Latency]\n"
            f"LLM reasoning: {lat.get('llm_sec')}s\n"
            f"Music generation: {lat.get('music_sec')}s\n"
            f"LLM + Music total: {lat.get('llm_music_total_sec')}s"
        )
    return (
        f"[Need] {need['need']}\n\n"
        f"[{tag}]\n{need['reasoning']}\n\n"
        f"[Music spec]\n{need['music_spec']}"
        f"{latency_text}"
    )


# ---- 面向用户的友好摘要（根据 ④ 状态 + ⑤ 推荐音乐，生成一段大白话；支持中/英切换）----
_EMOTION_ZH = {
    "neutral": "平静", "happy": "开心", "sad": "低落", "angry": "烦躁",
    "fear": "紧张不安", "surprise": "惊讶", "disgust": "厌烦",
}
_FATIGUE_ZH = {"low": "较低", "medium": "中等", "high": "较高"}
_EMOTION_EN = {
    "neutral": "calm", "happy": "happy", "sad": "down", "angry": "irritable",
    "fear": "anxious", "surprise": "surprised", "disgust": "put off",
}
_FATIGUE_EN = {"low": "low", "medium": "moderate", "high": "high"}


def _norm_lang(lang):
    """把 UI 传入的语言选项归一化为 'zh' / 'en'（默认英文）。"""
    t = str(lang or "").strip().lower()
    return "zh" if ("中" in str(lang or "")) or ("zh" in t) or ("chinese" in t) else "en"


def _music_style(music_spec, lang="en"):
    """把 ⑤ 给出的英文 music_spec 粗分成一句风格描述（按语言）。"""
    t = str(music_spec or "").lower()
    if any(k in t for k in ("slow", "calm", "ambient", "soft", "warm", "soothing", "gentle", "low energy")):
        return "舒缓、放松的轻音乐" if lang == "zh" else "soothing, relaxing music"
    if any(k in t for k in ("upbeat", "bright", "energetic", "positive", "uplifting", "cheerful", "playful")):
        return "明快、提振心情的音乐" if lang == "zh" else "bright, mood-lifting music"
    if any(k in t for k in ("focus", "neutral", "balanced", "steady", "minimal", "unobtrusive")):
        return "平和、适合专注的背景音乐" if lang == "zh" else "calm background music for focus"
    return "温暖、平衡的轻音乐" if lang == "zh" else "warm, balanced music"


def _summary_head(state, lang="en"):
    """情绪 + 疲劳的说明；置信度低时用更谨慎的措辞。"""
    emo = state.get("dominant_emotion", "neutral")
    fat = state.get("fatigue", "medium")
    conf = state.get("confidence", None)
    low = isinstance(conf, (int, float)) and conf < 0.45
    if lang == "zh":
        emo_zh = _EMOTION_ZH.get(emo, emo)
        fat_zh = _FATIGUE_ZH.get(fat, fat)
        if low:
            return f"🌿 此刻的你，或许有一点「{emo_zh}」，疲劳程度大概是{fat_zh}。"
        return f"🌿 此刻的你看起来有些「{emo_zh}」，疲劳程度{fat_zh}。"
    emo_en = _EMOTION_EN.get(emo, emo)
    fat_en = _FATIGUE_EN.get(fat, fat)
    if low:
        return f"🌿 Right now, you might be feeling a little {emo_en}, with {fat_en} energy."
    return f"🌿 Right now you seem {emo_en}, with {fat_en} fatigue."


def _build_summary(ctx, lang="en"):
    """根据上一次 pipeline 存下的 ctx + 语言，(重新)生成面向用户的说明。

    ctx 结构：
      单模式  {"both": False, "state": <④状态>, "spec": <⑤ music_spec>}
      both    {"both": True,  "state": <④状态>, "spec_tom": ..., "spec_std": ...}
    ctx 为空（还没跑过 pipeline）→ 返回 gr.skip()，即不改动当前显示。
    这样「切语言即时重刷」和「跑 pipeline 生成」共用同一套文案，绝不跑偏。
    """
    if not ctx:
        return gr.skip()
    state = ctx.get("state", {})
    if ctx.get("both"):
        style_a = _music_style(ctx.get("spec_tom", ""), lang)
        style_b = _music_style(ctx.get("spec_std", ""), lang)
        if lang == "zh":
            return (_summary_head(state, "zh") + "\n"
                    "🎵 这次为你准备了两段音乐，可以静静对比一下：\n"
                    f"　· A（ToM+CoT）：{style_a}\n"
                    f"　· B（Standard）：{style_b}")
        return (_summary_head(state, "en") + "\n"
                "🎵 Two pieces for you this time — take a moment to compare:\n"
                f"　· A (ToM+CoT): {style_a}\n"
                f"　· B (Standard): {style_b}")
    style = _music_style(ctx.get("spec", ""), lang)
    if lang == "zh":
        return _summary_head(state, "zh") + "\n" + f"🎵 为你挑了一段{style}，愿它能陪伴此刻的你。"
    return _summary_head(state, "en") + "\n" + f"🎵 Here's some {style}, chosen to keep you company right now."


def _summary_text(ctx, lang="en"):
    """给用户的话：优先大模型生成（单模式 & Both 都走 DeepSeek），失败/未配置 → 模板兜底。

    ctx 为空（还没跑过 pipeline）→ gr.skip()，不改动当前显示。
    """
    if not ctx:
        return gr.skip()
    # 高风险：语言切换时也保持关怀语（按新语言重取），不落回普通推荐语。
    triage = ctx.get("triage")
    if triage and triage.get("risk_level") == "high":
        return SAFETY.screen(ctx.get("state"), ctx.get("user_text", ""), lang).get(
            "care_message") or triage.get("care_message")
    try:
        if ctx.get("both"):
            if hasattr(LLM, "summarize_both_for_user"):
                msg = LLM.summarize_both_for_user(
                    ctx.get("state"), ctx.get("need_tom"), ctx.get("need_std"), lang)
                if msg and str(msg).strip():
                    return str(msg).strip()
        elif hasattr(LLM, "summarize_for_user"):
            msg = LLM.summarize_for_user(ctx.get("state"), ctx.get("need"), lang)
            if msg and str(msg).strip():
                return str(msg).strip()
    except Exception as e:  # 任何问题都不影响主流程，回退模板
        print(f"[summary] LLM summarize failed ({e}), using template.")
    return _build_summary(ctx, lang)


def _reason_and_compose(state, session_label=None, reasoning_mode="tom_cot", duration_sec=None, perception=None, lang="en"):
    """跑 ⑤ LLM 推理 + ⑥ 音乐生成。

    返回 (summary_text, summary_ctx, reasoning_text, music1_update, music2_update)：
      · summary_text  面向用户的友好摘要（按 lang 生成，'en'/'zh'）
      · summary_ctx   重建摘要所需的最小上下文（存进 gr.State，供切语言时即时重刷）

    reasoning_mode：'tom_cot'（默认）/ 'standard' / 'both'（两种都跑，生成两段音乐）。
    duration_sec：UI 滑条选择的音乐生成时长（秒，0~120）；None 时用 music_gen 的默认值。
    perception：①②③ 感知结果，用于会话日志留存；其中 transcript（语音转写）供 ⑨ 风险筛查。
    lang：摘要语言，'en'（默认）或 'zh'。

    music1_update / music2_update 都是 gr.update()：动态设置每个播放器的音频与标注，
    并在非 both 模式下隐藏第二个播放器，保证「标清楚 + 不误导」。
    """
    # ⑨ Safety Router：先做心理风险分流（结合 采集语音的转写 + ④情绪/疲劳/置信度）。
    user_text = str((perception or {}).get("transcript") or "")
    triage = SAFETY.screen(state, user_text, lang)

    if triage.get("risk_level") == "high" and triage.get("pause_music"):
        # 风险较高：显示关怀与求助，暂停自动推送音乐（跳过 ⑤ LLM + ⑥ 音乐）。
        reasoning_text = (
            f"{triage['banner']}\n\n"
            f"[Safety Router] risk=high  score={triage['score']:.2f}  source={triage.get('source')}\n"
            "signals:\n  - " + "\n  - ".join(triage.get("signals") or ["(none)"])
        )
        ctx = {"both": False, "state": state, "spec": "", "need": None,
               "triage": triage, "user_text": user_text}
        # 显式停掉两个播放器（value=None）。播放器现在 loop=True 会无限循环，
        # 若仍用 gr.skip()「保留当前」，高风险时旧音乐会一直响，违背「暂停推送」的本意。
        return (triage["care_message"], ctx, reasoning_text,
                gr.update(value=None), gr.update(value=None))

    text = str(reasoning_mode).lower()

    if "both" in text:
        # both：ToM+CoT 放上面播放器，Standard 放下面播放器，两段都清楚标注。
        need_tom, music_tom = _compose_one(state, "tom_cot", session_label, duration_sec, perception)
        need_std, music_std = _compose_one(state, "standard", session_label, duration_sec, perception)
        reasoning_text = (
            "===== BOTH modes · 两种推理对比（各生成一段音乐）=====\n\n"
            "########## 上方播放器：ToM+CoT (default) ##########\n"
            f"{_one_mode_text(need_tom, cot=True)}\n\n\n"
            "########## 下方播放器：Standard (baseline) ##########\n"
            f"{_one_mode_text(need_std, cot=False)}"
        )
        reasoning_text = triage["banner"] + "\n\n" + reasoning_text
        ctx = {"both": True, "state": state,
               "spec_tom": need_tom.get("music_spec", ""),
               "spec_std": need_std.get("music_spec", ""),
               "need_tom": need_tom, "need_std": need_std,
               "triage": triage, "user_text": user_text}
        m1 = gr.update(value=music_tom, label="⑥-A Music", visible=True)
        m2 = gr.update(value=music_std, label="⑥-B Music", visible=True)
        return _summary_text(ctx, lang), ctx, reasoning_text, m1, m2

    # 单模式
    mode = "standard" if text.startswith(("standard", "base")) else "tom_cot"
    need, music = _compose_one(state, mode, session_label, duration_sec, perception)
    if mode == "standard":
        reasoning_text = _one_mode_text(need, cot=False)
        label = "⑥ Music · Standard (baseline)"
    else:
        reasoning_text = _one_mode_text(need, cot=True)
        label = "⑥ Music · ToM+CoT (default)"
    reasoning_text = triage["banner"] + "\n\n" + reasoning_text
    ctx = {"both": False, "state": state, "spec": need.get("music_spec", ""), "need": need,
           "triage": triage, "user_text": user_text}
    m1 = gr.update(value=music, label=label, visible=True)
    # 非 both：清空并隐藏第二个播放器，避免看到上一次的残留。
    m2 = gr.update(value=None, label="⑥-B Music · (Both 模式第二段)", visible=False)
    return _summary_text(ctx, lang), ctx, reasoning_text, m1, m2


def _dump(obj):
    return json.dumps(obj, ensure_ascii=False, indent=2)



# =============================================================================
# Speech backend UI helper
# =============================================================================

def _speech_mode_value_from_env():
    """Map SPEECH_BACKEND env value to the radio's stable value ('emotion2vec'/'api')."""
    backend = os.getenv("SPEECH_BACKEND", "emotion2vec").strip().lower()
    return "api" if backend in {"api", "qwen", "qwen-omni", "qwen_omni"} else "emotion2vec"


def _set_speech_backend_from_ui(speech_mode):
    """Set SPEECH_BACKEND before calling speech_emotion.predict()."""
    text = str(speech_mode or "").strip().lower()
    if "api" in text or "qwen" in text:
        backend = "api"
    else:
        backend = "emotion2vec"

    os.environ["SPEECH_BACKEND"] = backend
    return backend


# =============================================================================
# 界面文案（中/英）—— 全 UI 语言切换的唯一真源。构建时用 en，切换时整体替换。
#   · 单选框用 (显示文本, 稳定值) 元组：显示随语言变，值不变，所有解析逻辑照常工作。
# =============================================================================

def _texts():
    """返回 {'en': {...}, 'zh': {...}}，含每个可切换组件的 label/info/choices/markdown。"""
    a, mx = AUTO_INTERVAL_SEC, MUSIC_MAX_DURATION_SEC
    en = {
        "app_title": (
            "# 🎵 EmotiCompanion\n"
            "### Your real-time music companion — AIAA 3800 · HKUST(GZ)\n"
            "We quietly read your face, voice, and energy level, then compose a piece of music "
            "just for this moment. No input needed — just be yourself."
        ),
        "mode_label": "How would you like to use it?",
        "mode_info": (f"Continuous: camera & mic stay on, music refreshes every {a}s as your mood shifts. "
                      "One-shot: record a short clip and we'll read it once."),
        "mode_choices": [("🔄  Continuous (always on)", "auto"), ("🎬  One-shot (record a clip)", "manual")],
        "reasoning_label": "Reasoning mode",
        "reasoning_info": ("ToM+CoT reads between the lines of your emotional state (recommended). "
                           "Standard maps emotion directly to style. Both runs both and lets you compare."),
        "reasoning_choices": [("✨  ToM+CoT — reads between the lines (recommended)", "tom_cot"),
                              ("⚡  Standard — direct mapping (baseline)", "standard"),
                              ("🔬  Both — compare the two", "both")],
        "duration_label": "Music length",
        "duration_info": (f"How long should the piece be? 0–{mx}s. Longer takes more time to generate. "
                          "Takes effect on the next run."),
        "lang_label": "🌐 Language",
        "lang_info": "Switch the whole interface and your personal note between English and Chinese.",
        "lang_choices": [("English", "en"), ("中文", "zh")],
        "auto_cam_label": f"📷 Camera — captured every {a}s",
        "auto_mic_label": "🎙️ Microphone — listening",
        "auto_media_acc_label": "📹 Live view & attention map (click to hide)",
        "auto_note": (f"> Continuous mode is **on**. EmotiCompanion checks in every {a}s. "
                      "The music **loops seamlessly** — it keeps playing while the next piece "
                      "is being composed, and only switches when your mood shifts."),
        "manual_note": ("> **How to use:** tap **● Record** on the video box below, "
                        "say a few words or just sit naturally for a moment, then tap **■ Stop**. "
                        "We'll pick a frame as your face snapshot and separate the audio — "
                        "then the full analysis runs automatically."),
        "man_video_label": "🎥 Record a short clip",
        "man_frame_label": "🖼️ Face snapshot (auto-sampled)",
        "manual_media_acc_label": "📹 Recording & attention map (click to hide)",
        "run_btn_label": "↺  Re-analyse the same clip",
        "summary_label": "✨ A note just for you",
        "summary_placeholder": ("Once the analysis runs, you'll see a warm, plain-language note here — "
                                "what we sensed about your mood and energy, and what music we chose for you."),
        "heatmap_label": "Attention map (GradCAM)",
        "accordion_label": "🔬 Under the hood — perception · fusion · reasoning",
        "perception_label": "Raw perception  ①②③",
        "fusion_label": "Fused state  ④",
        "reasoning_out_label": "LLM reasoning chain  ⑤",
        "music_label": "🎵 Your music companion",
        "music2_label": "🎵 Companion B  (Both mode)",
    }
    zh = {
        "app_title": (
            "# 🎵 EmotiCompanion\n"
            "### 你的实时情绪音乐陪伴 — AIAA 3800 · 香港科技大学（广州）\n"
            "我们轻轻感知你的表情、声音和疲劳状态，为此刻的你即兴创作一段音乐。"
            "不需要任何操作，做自己就好。"
        ),
        "mode_label": "你想怎么使用？",
        "mode_info": (f"持续模式：摄像头和麦克风保持开启，每 {a} 秒感知一次，情绪变化时音乐随之更新。"
                      "单次模式：录一段短视频，我们读取一次。"),
        "mode_choices": [("🔄  持续陪伴（一直开着）", "auto"), ("🎬  单次体验（录一段）", "manual")],
        "reasoning_label": "推理方式",
        "reasoning_info": ("ToM+CoT 会读懂情绪背后的深层需求（推荐）。"
                           "Standard 直接把情绪映射到曲风。Both 两种都跑，可以对比。"),
        "reasoning_choices": [("✨  ToM+CoT — 读懂你的深层需求（推荐）", "tom_cot"),
                              ("⚡  Standard — 直接映射（基线）", "standard"),
                              ("🔬  Both — 两种都生成，对比看看", "both")],
        "duration_label": "音乐时长",
        "duration_info": (f"这段音乐生成多长？0~{mx}s。越长生成越慢，下一次运行生效。"),
        "lang_label": "🌐 语言",
        "lang_info": "在中/英之间切换整个界面和给你的便签，立即生效。",
        "lang_choices": [("English", "en"), ("中文", "zh")],
        "auto_cam_label": f"📷 摄像头 — 每 {a} 秒捕捉一次",
        "auto_mic_label": "🎙️ 麦克风 — 持续聆听",
        "auto_media_acc_label": "📹 实时画面与热力图（点击可收起）",
        "auto_note": (f"> 持续模式**已开启**。EmotiCompanion 每 {a} 秒悄悄感知一次。"
                      "音乐**循环不间断**——下一段还在创作时当前音乐持续播放，"
                      "只有情绪或疲劳变化时才悄悄换成新的。"),
        "manual_note": ("> **使用方法：** 点击下方视频框的 **● 录制**，"
                        "说几句话或自然地坐一会儿，再点 **■ 停止**。"
                        "我们会自动抽取一帧作为人脸快照、分离音频，然后完整分析自动开始。"),
        "man_video_label": "🎥 录一段短视频",
        "man_frame_label": "🖼️ 人脸快照（自动抽取）",
        "manual_media_acc_label": "📹 录制画面与热力图（点击可收起）",
        "run_btn_label": "↺  重新分析这段视频",
        "summary_label": "✨ 写给你的便签",
        "summary_placeholder": ("分析完成后，这里会出现一段温柔的话——"
                                "告诉你我们感知到了什么情绪和疲劳状态，以及为你选了什么样的音乐陪伴。"),
        "heatmap_label": "注意力热力图（GradCAM）",
        "accordion_label": "🔬 技术细节 — 感知 · 融合 · 推理",
        "perception_label": "原始感知输出  ①②③",
        "fusion_label": "多模态融合状态  ④",
        "reasoning_out_label": "LLM 推理链  ⑤",
        "music_label": "🎵 此刻的旋律",
        "music2_label": "🎵 旋律 B（Both 模式）",
    }
    return {"en": en, "zh": zh}

# =============================================================================
# 手动模式 —— 单次运行完整 pipeline（总是重新生成音乐）
# =============================================================================

def run_manual(image, audio, session_label=None, reasoning_mode="tom_cot", speech_mode=None, duration_sec=None, summary_lang="en"):
    perception, heatmap, state = _perceive_and_fuse(image, audio, speech_mode, mode="manual")
    summary_text, summary_ctx, reasoning_text, music, music2 = _reason_and_compose(
        state, session_label, reasoning_mode, duration_sec, perception, _norm_lang(summary_lang))
    _save_media(image, heatmap, session_label)   # user study 时留存 ⑦/② 可视化（默认关闭）
    # 同时把本次 state_key 写回 last_key，避免切到自动模式时重复生成
    return (summary_text, summary_ctx, heatmap, _dump(perception), _dump(state),
            reasoning_text, music, music2, _state_key(state), time.time())


# =============================================================================
# 手动模式（视频版）—— 录制一段视频，随机抽帧当图像 + 分离音频当语音，跑单次 pipeline
# =============================================================================

def _extract_random_frame(video_path):
    """从录制视频里随机抽一帧作为人脸图像输入，返回 RGB numpy（失败返回 None）。

    优先按帧号 seek 直接取；有些 webm/mp4 容器 seek 不准，回退到「顺序读全部帧再随机挑」。
    """
    import random
    import cv2

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None
    try:
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame = None
        if total > 0:
            idx = random.randint(0, total - 1)
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, frame = cap.read()
            if not ok:                       # seek 不准 → 回退顺序读
                frame = None
        if frame is None:                    # 帧数未知或 seek 失败：读全部帧再随机挑
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            frames = []
            while True:
                ok, f = cap.read()
                if not ok:
                    break
                frames.append(f)
            if not frames:
                return None
            frame = random.choice(frames)
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    finally:
        cap.release()


def run_manual_video(video_path, session_label=None, reasoning_mode="tom_cot", speech_mode=None, duration_sec=None, summary_lang="en"):
    """手动模式（视频版）单次运行：结束录制后自动触发。

    ① 从视频随机抽一帧当人脸图像输入（①②⑦ 用）
    ② 用 ffmpeg 从视频分离音频当语音输入（③ 用）
    然后跑完整 pipeline：感知融合 → LLM 推理 → 音乐生成（按 UI 滑条时长）。

    输出顺序对齐 UI（比 run_manual 末尾多一个「随机抽取的输入帧」预览）：
      summary, summary_ctx, heatmap, perception, state, reasoning, music, music2, last_key, last_music_ts, sampled_frame
    """
    lang = _norm_lang(summary_lang)
    if not video_path:
        note = ("还没有视频：请先点击视频框的「● 录制」，录制一段后点「■ 停止」结束录制。"
                if lang == "zh" else
                "No video yet: click ● Record on the video box, record a clip, then click ■ Stop.")
        return (note, gr.skip(), gr.skip(), gr.skip(), gr.skip(), gr.skip(), gr.skip(),
                gr.skip(), gr.skip(), gr.skip(), gr.skip())

    # ① 随机抽帧当图像输入
    image = _extract_random_frame(video_path)

    # ② 从视频分离音频当语音输入（ffmpeg → 16k 单声道 wav）
    try:
        audio = extract_audio(video_path)
    except Exception as e:  # 视频没声音 / ffmpeg 缺失时不阻塞，语音模块会拿到 None
        print(f"[manual-video] audio extract failed: {e}")
        audio = None

    perception, heatmap, state = _perceive_and_fuse(image, audio, speech_mode, mode="manual")
    summary_text, summary_ctx, reasoning_text, music, music2 = _reason_and_compose(
        state, session_label, reasoning_mode, duration_sec, perception, lang)
    _save_media(image, heatmap, session_label)   # user study 时留存 ⑦/② 可视化（默认关闭）
    return (summary_text, summary_ctx, heatmap, _dump(perception), _dump(state),
            reasoning_text, music, music2, _state_key(state), time.time(), image)


# =============================================================================
# 自动模式 —— 摄像头流式输入驱动（每帧直接送进来，不经 State 中转，避免读到 None）
#   情绪与压力都不变 → 跳过 ⑤⑥，保持当前音乐继续播放
#   不论是否变化，都刷新一个时间戳「心跳」，让你一眼看出 pipeline 在跑
# =============================================================================

def auto_step(image, audio_buffer, last_key, last_music_ts, session_label=None, reasoning_mode="tom_cot", speech_mode=None, duration_sec=None, summary_lang="en"):
    """摄像头流式回调：拿当前帧 + 最近一段录音，感知融合，按需更换音乐。

    输出顺序对齐 UI：
      summary, summary_ctx, heatmap, perception, state, reasoning, music, music2, last_key, last_music_ts, audio_buffer(清空)
    """
    lang = _norm_lang(summary_lang)
    ts = time.strftime("%H:%M:%S")
    if image is None:
        # 摄像头还没就绪：刷新心跳提示，其余不动
        note = (f"[{ts}] 正在等待摄像头画面…" if lang == "zh"
                else f"[{ts}] Waiting for camera frame…")
        return (note, gr.skip(), gr.skip(), gr.skip(), gr.skip(), gr.skip(), gr.skip(), gr.skip(), last_key, last_music_ts, [])

    audio = buffer_to_wav(audio_buffer)
    perception, heatmap, state = _perceive_and_fuse(image, audio, speech_mode, mode="auto")
    key = _state_key(state)
    transcript = str(perception.get("transcript") or "")

    # ⑨ 先做一次风险筛查（_reason_and_compose 内部会命中同参数缓存，不重复请求）。
    triage = SAFETY.screen(state, transcript, lang)
    high_risk = (triage.get("risk_level") == "high" and triage.get("pause_music"))

    now = time.time()
    unchanged = (key == last_key)
    within_window = (now - float(last_music_ts or 0.0)) < MUSIC_REFRESH_SEC

    # 非高风险 + 状态没变 + 距上次生成不到 MUSIC_REFRESH_SEC → 跳过最贵的 ⑤LLM+⑥音乐，
    # 保持当前音乐继续播（省时的关键）。摘要/播放器都用 gr.skip() 维持不变。
    if not high_risk and unchanged and within_window:
        left = int(MUSIC_REFRESH_SEC - (now - float(last_music_ts or 0.0)))
        reasoning_text = (
            f"[{ts}] state unchanged (emotion={key[0]}, fatigue={key[1]}); "
            f"within {int(MUSIC_REFRESH_SEC)}s refresh window (~{left}s left) → "
            f"kept current music, skipped ⑤ LLM + ⑥ music.\n\n{triage.get('banner', '')}"
        )
        return (gr.skip(), gr.skip(), heatmap, _dump(perception), _dump(state),
                reasoning_text, gr.skip(), gr.skip(), key, last_music_ts, [])

    # 否则（状态变了 / 刷新窗口已到 / 高风险）→ 走完整流程。
    # 高风险时 _reason_and_compose 会给关怀语并暂停音乐；否则重生成 ⑤⑥。
    summary_text, summary_ctx, reasoning_text, music, music2 = _reason_and_compose(
        state, session_label, reasoning_mode, duration_sec, perception, lang)
    _save_media(image, heatmap, session_label)   # user study 时留存 ⑦/② 可视化（默认关闭）

    if high_risk:
        # 暂停推送：把计时归零，强制下一 tick 重新完整评估（风险解除后立刻恢复音乐、不残留旧说明）。
        new_ts = 0.0
    else:
        new_ts = now                      # 生成了新音乐 → 重置刷新窗口
        if unchanged:
            reasoning_text = (
                f"[{ts}] state unchanged but {int(MUSIC_REFRESH_SEC)}s refresh window elapsed "
                f"→ regenerated a same-state music variation.\n\n" + reasoning_text
            )
        else:
            reasoning_text = f"[{ts}] state CHANGED → regenerated music.\n\n" + reasoning_text

    return (summary_text, summary_ctx, heatmap, _dump(perception), _dump(state),
            reasoning_text, music, music2, key, new_ts, [])


def accumulate_audio(new_chunk, buffer):
    """麦克风流式输入累积到 buffer。

    流式音频用 type="numpy"，每个 chunk 是 (sample_rate, np.ndarray)。
    这里把所有 chunk 收进 buffer（近似「最近一段连续有效声音」），
    定时器触发时再合并写成临时 wav 交给语音模块。
    """
    buffer = buffer or []
    if new_chunk is not None:
        buffer = buffer + [new_chunk]
    return buffer


def buffer_to_wav(buffer):
    """把累积的 numpy 音频 chunk 合并写成临时 wav 文件，返回路径（无则 None）。"""
    chunks = [c for c in (buffer or []) if c is not None]
    if not chunks:
        return None
    sr = chunks[0][0]
    data = np.concatenate([np.asarray(c[1]).flatten() for c in chunks])
    if data.size == 0:
        return None
    # 归一化到 int16 PCM
    if data.dtype != np.int16:
        peak = np.max(np.abs(data)) or 1.0
        data = (data / peak * 32767).astype(np.int16)
    path = os.path.join(tempfile.gettempdir(), "emoti_audio.wav")
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(int(sr))
        wf.writeframes(data.tobytes())
    return path


# =============================================================================
# Gradio UI
# =============================================================================

# ---- 视觉包装（主题 + 自定义 CSS）：只影响外观，不触碰任何 pipeline / 回调逻辑 ----
# 设计语言：柔和极光渐变背景 + 玻璃拟态卡片 + 流动渐变主视觉，
# 呼应「情绪陪伴音乐」的治愈系气质（紫 → 粉 → 蓝）；自动适配明/暗色模式。
EMOTI_THEME = gr.themes.Soft(
    primary_hue=gr.themes.colors.violet,
    secondary_hue=gr.themes.colors.pink,
    neutral_hue=gr.themes.colors.slate,
    radius_size=gr.themes.sizes.radius_lg,
    font=[gr.themes.GoogleFont("Outfit"), gr.themes.GoogleFont("Noto Sans SC"),
          "system-ui", "sans-serif"],
)

EMOTI_CSS = """
/* ================= EmotiCompanion · Aurora Glass ================= */
* { -webkit-font-smoothing: antialiased; }
footer { display: none !important; }        /* 隐藏 gradio 默认页脚 */

/* ---------- 页面底色 ---------- */
body, .gradio-container {
    background: linear-gradient(180deg, #f7f5ff 0%, #fdf4fa 55%, #f2f8ff 100%) !important;
}
.dark body, .dark .gradio-container {
    background: linear-gradient(180deg, #131020 0%, #1a1226 55%, #10131f 100%) !important;
}
.gradio-container { max-width: 1140px !important; margin: 0 auto !important; position: relative; }
/* 让内容层始终盖在装饰光斑之上 */
.gradio-container > * { position: relative; z-index: 1; }

/* ---------- 两团缓慢漂移的极光光斑（固定，不随滚动） ---------- */
.gradio-container::before, .gradio-container::after {
    content: "";
    position: fixed;
    width: 60vmax; height: 60vmax;
    border-radius: 50%;
    filter: blur(90px);
    opacity: .5;
    z-index: 0;
    pointer-events: none;
}
.gradio-container::before {
    background: radial-gradient(circle at 30% 30%, rgba(139,120,255,.38), rgba(255,126,179,.20) 55%, transparent 72%);
    top: -22vmax; left: -18vmax;
    animation: emoti-drift-a 26s ease-in-out infinite alternate;
}
.gradio-container::after {
    background: radial-gradient(circle at 70% 70%, rgba(96,165,250,.32), rgba(192,93,240,.18) 55%, transparent 72%);
    bottom: -24vmax; right: -20vmax;
    animation: emoti-drift-b 32s ease-in-out infinite alternate;
}
.dark .gradio-container::before, .dark .gradio-container::after { opacity: .38; }
@keyframes emoti-drift-a { from { transform: translate(0,0) scale(1); }    to { transform: translate(9vmax,6vmax) scale(1.15); } }
@keyframes emoti-drift-b { from { transform: translate(0,0) scale(1.1); }  to { transform: translate(-8vmax,-7vmax) scale(.95); } }

/* ---------- 细腻胶片噪点，压住渐变的塑料感 ---------- */
body::after {
    content: ""; position: fixed; inset: 0; z-index: 2; pointer-events: none;
    opacity: .035; mix-blend-mode: overlay;
    background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='160' height='160'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.8' numOctaves='2'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23n)'/%3E%3C/svg%3E");
}

/* ---------- 主视觉：流动渐变横幅 + 漂浮音符 ---------- */
#emoti-hero {
    position: relative;
    overflow: hidden;
    background: linear-gradient(120deg, #6d5dfc, #a45de2, #ff7eb3, #5da9fc, #6d5dfc);
    background-size: 340% 340%;
    animation: emoti-flow 16s ease infinite;
    border-radius: 28px;
    padding: 46px 40px 38px;
    text-align: center;
    box-shadow: 0 20px 50px rgba(109,93,252,.30), inset 0 1px 0 rgba(255,255,255,.35);
    margin-bottom: 4px;
}
@keyframes emoti-flow { 0% { background-position: 0% 50%; } 50% { background-position: 100% 50%; } 100% { background-position: 0% 50%; } }
#emoti-hero::before, #emoti-hero::after {   /* 漂浮的音符装饰 */
    position: absolute; color: rgba(255,255,255,.30); pointer-events: none; z-index: 0;
}
#emoti-hero::before { content: "♪"; font-size: 3.4rem; left: 7%;  top: 16%;
                      animation: emoti-float 7s ease-in-out infinite; }
#emoti-hero::after  { content: "♫"; font-size: 2.6rem; right: 8%; bottom: 14%;
                      animation: emoti-float 9s ease-in-out 1.2s infinite; }
@keyframes emoti-float { 0%,100% { transform: translateY(0) rotate(-6deg); }
                         50%     { transform: translateY(-14px) rotate(8deg); } }
#emoti-hero h1 { color: #fff !important; font-size: 2.6rem; font-weight: 700;
                 letter-spacing: .04em; margin: 0 0 .2em;
                 text-shadow: 0 2px 16px rgba(0,0,0,.20); }
#emoti-hero h3 { color: rgba(255,255,255,.95) !important; font-weight: 500;
                 font-size: 1.25rem; margin: 0 0 .6em; letter-spacing: .02em; }
#emoti-hero p, #emoti-hero li { color: rgba(255,255,255,.90) !important;
                 max-width: 640px; margin: .35em auto 0; font-size: 1.04rem; line-height: 1.8; }
#emoti-hero em { color: rgba(255,255,255,.75) !important; font-size: .9rem; }
#emoti-hero strong { color: #fff !important; }

/* ---------- 玻璃拟态卡片 ---------- */
.emoti-card {
    background: rgba(255, 255, 255, .58) !important;
    backdrop-filter: blur(18px) saturate(1.3);
    -webkit-backdrop-filter: blur(18px) saturate(1.3);
    border: 1px solid rgba(255, 255, 255, .80) !important;
    border-radius: 24px !important;
    box-shadow: 0 12px 32px rgba(92, 80, 180, .10) !important;
    padding: 12px !important;
}
.dark .emoti-card {
    background: rgba(255, 255, 255, .05) !important;
    border-color: rgba(255, 255, 255, .09) !important;
    box-shadow: 0 12px 32px rgba(0, 0, 0, .32) !important;
}
.emoti-card .block, .emoti-card .form {
    background: transparent !important; border: none !important; box-shadow: none !important;
}
.emoti-card img, .emoti-card video { border-radius: 16px !important; }
.emoti-controls { gap: 14px !important; }

/* ---------- 「给你的便签」：流光渐变描边 + 信纸质感（视觉主角一号） ---------- */
#emoti-summary {
    border: 2px solid transparent !important;
    border-radius: 24px !important;
    background:
        linear-gradient(rgba(255,255,255,.94), rgba(255,255,255,.94)) padding-box,
        linear-gradient(120deg, #8b78ff, #ff7eb3, #60a5fa, #8b78ff) border-box !important;
    background-size: 100% 100%, 300% 300% !important;
    animation: emoti-border 10s linear infinite;
    box-shadow: 0 16px 38px rgba(124, 107, 255, .16) !important;
    padding: 6px 10px !important;
}
.dark #emoti-summary {
    background:
        linear-gradient(rgba(26,20,38,.94), rgba(26,20,38,.94)) padding-box,
        linear-gradient(120deg, #8b78ff, #ff7eb3, #60a5fa, #8b78ff) border-box !important;
    background-size: 100% 100%, 300% 300% !important;
}
@keyframes emoti-border { 0% { background-position: 0 0, 0% 50%; } 100% { background-position: 0 0, 300% 50%; } }
#emoti-summary textarea {
    background: transparent !important; border: none !important;
    font-size: 1.14rem !important; line-height: 2.0 !important;
}
#emoti-summary label span { font-size: 1.05rem !important; font-weight: 700 !important;
                            letter-spacing: .02em; }

/* ---------- 「此刻的旋律」：呼吸光晕（视觉主角二号） ---------- */
#emoti-music {
    border-radius: 24px !important;
    animation: emoti-breathe 5.5s ease-in-out infinite;
}
@keyframes emoti-breathe {
    0%, 100% { box-shadow: 0 12px 32px rgba(109, 93, 252, .16); }
    50%      { box-shadow: 0 12px 48px rgba(255, 126, 179, .32); }
}
#emoti-music label span { font-size: 1.02rem !important; font-weight: 700 !important; }

/* ---------- 主按钮：渐变 + 悬浮流光扫过 ---------- */
#emoti-run-btn {
    position: relative; overflow: hidden;
    background: linear-gradient(135deg, #7c6bff 0%, #c05df0 55%, #ff7eb3 100%) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 16px !important;
    font-weight: 600; font-size: 1.02rem; letter-spacing: .04em;
    padding: 12px 18px !important;
    box-shadow: 0 10px 26px rgba(150, 90, 240, .34);
    transition: transform .16s ease, box-shadow .16s ease;
}
#emoti-run-btn::after {                     /* 流光扫过 */
    content: ""; position: absolute; top: 0; left: -80%;
    width: 50%; height: 100%;
    background: linear-gradient(105deg, transparent, rgba(255,255,255,.45), transparent);
    transform: skewX(-20deg);
    transition: left .5s ease;
}
#emoti-run-btn:hover { transform: translateY(-2px);
                       box-shadow: 0 16px 34px rgba(150, 90, 240, .46); }
#emoti-run-btn:hover::after { left: 130%; }
#emoti-run-btn:active { transform: translateY(0); }

/* ---------- 底部研究/调试折叠区：刻意弱化，让给主角 ---------- */
#emoti-details {
    border-radius: 18px !important;
    border: 1px dashed rgba(124, 107, 255, .28) !important;
    background: rgba(255, 255, 255, .30) !important;
    opacity: .82;
    transition: opacity .2s ease;
    margin-top: 6px;
}
#emoti-details:hover { opacity: 1; }
.dark #emoti-details { background: rgba(255, 255, 255, .03) !important; }
#emoti-details .label-wrap span { font-size: .92rem !important; opacity: .85; }

/* ---------- 提示文字与滚动条细节 ---------- */
#emoti-note-auto p, #emoti-note-manual p { opacity: .82; font-size: .96rem; line-height: 1.8; }
::-webkit-scrollbar { width: 9px; height: 9px; }
::-webkit-scrollbar-thumb { background: rgba(124,107,255,.30); border-radius: 8px; }
::-webkit-scrollbar-thumb:hover { background: rgba(124,107,255,.50); }
"""


def build_ui():
    with gr.Blocks(title="EmotiCompanion") as demo:
        T = _texts()
        t0 = T["en"]   # 初始英文；切换语言时由 set_language 整体替换

        title_md = gr.Markdown(t0["app_title"], elem_id="emoti-hero")

        # 跨回调共享的会话状态
        last_key = gr.State(None)    # 上一次的 (emotion, fatigue)，用于判断是否换音乐
        last_music_ts = gr.State(0.0) # 上一次生成音乐的时间戳，用于状态不变时定期刷新 variation
        audio_buf = gr.State([])     # 麦克风流式音频缓冲（numpy chunk 列表）
        summary_ctx = gr.State(None) # 上一次摘要的最小上下文，切语言时即时重刷用

        # ---------------- 控制区（卡片包装）----------------
        # 只放用户真正会关心的三项：使用方式 / 音乐时长 / 语言。
        # 「推理模式」偏研究对比，移到底部技术折叠区，避免干扰普通用户。
        with gr.Group(elem_classes=["emoti-card"]):
            with gr.Row(elem_classes=["emoti-controls"]):
                # 单选框统一用 (显示文本, 稳定值)：显示随语言变，value 恒定，所有解析逻辑不受影响。
                mode = gr.Radio(t0["mode_choices"], value="manual",
                                label=t0["mode_label"], info=t0["mode_info"])

                # 全局语言开关：默认英文，选中文后整个界面 + 给用户的话 + 标签都变中文（立即生效）。
                summary_lang = gr.Radio(t0["lang_choices"], value="en",
                                        label=t0["lang_label"], info=t0["lang_info"])

            music_duration = gr.Slider(
                minimum=0, maximum=MUSIC_MAX_DURATION_SEC, step=1,
                value=MUSIC_DEFAULT_DURATION_SEC,
                label=t0["duration_label"], info=t0["duration_info"])

        # 会话标签与语音后端不再作为可见控件：固定为 test / 环境默认，用隐藏 State 承载，
        # 保持各回调 inputs 的位置不变（值照常流入函数）。
        session_label = gr.State("test")
        speech_mode = gr.State(_speech_mode_value_from_env())

        # ---------------- 自动模式区 ----------------
        with gr.Group(visible=False, elem_classes=["emoti-card"]) as auto_group:
            # 摄像头 · 麦克风 · ⑦ GradCAM 热力图 并排（统一缩小）。
            # 包在默认展开的 Accordion 里：不想看到画面时可点标题收起（采集照常进行）。
            with gr.Accordion(t0["auto_media_acc_label"], open=True) as auto_media_acc:
                with gr.Row(equal_height=True):
                    auto_cam = gr.Image(label=t0["auto_cam_label"],
                                        type="numpy", sources=["webcam"], streaming=True, height=240)
                    auto_mic = gr.Audio(label=t0["auto_mic_label"],
                                        type="numpy", sources=["microphone"], streaming=True)
                    heatmap_auto = gr.Image(label=t0["heatmap_label"], height=240)
            auto_note_md = gr.Markdown(t0["auto_note"], elem_id="emoti-note-auto")

        # ---------------- 手动模式区（视频版：录制 → 抽帧+分离音频 → 单次 pipeline）----------------
        with gr.Group(visible=True, elem_classes=["emoti-card"]) as manual_group:
            manual_note_md = gr.Markdown(t0["manual_note"], elem_id="emoti-note-manual")
            # 录制视频 · 随机抽取的输入帧 · ⑦ GradCAM 热力图 并排（统一缩小，避免过于突兀）。
            # 同样包在默认展开的 Accordion 里，可随时收起。
            with gr.Accordion(t0["manual_media_acc_label"], open=True) as manual_media_acc:
                with gr.Row(equal_height=True):
                    man_video = gr.Video(label=t0["man_video_label"],
                                         sources=["webcam"], include_audio=True, height=240,
                                         webcam_options=gr.WebcamOptions(mirror=False))
                    man_frame = gr.Image(label=t0["man_frame_label"], type="numpy", height=240)
                    heatmap_manual = gr.Image(label=t0["heatmap_label"], height=240)
            run_btn = gr.Button(t0["run_btn_label"], variant="primary", elem_id="emoti-run-btn")

        # ---------------- 输出区（两模式共用）----------------
        # 面向用户的友好摘要：由大模型生成（未配置 LLM 后端时回退模板），显眼常显。
        summary_out = gr.Textbox(label=t0["summary_label"], lines=3, interactive=False,
                                 placeholder=t0["summary_placeholder"],
                                 elem_id="emoti-summary")

        # 中间产物（①②③ 感知 / ④ 融合 / ⑤ 推理）默认收起：一般用户不需要看，需要时展开。
        # 「推理模式」也一并放进来——它偏研究对比，普通用户用默认即可。
        with gr.Accordion(t0["accordion_label"], open=False,
                          elem_id="emoti-details") as details_accordion:
            reasoning_mode = gr.Radio(t0["reasoning_choices"], value="tom_cot",
                                      label=t0["reasoning_label"], info=t0["reasoning_info"])
            perception_out = gr.Code(label=t0["perception_label"], language="json")
            fusion_out = gr.Code(label=t0["fusion_label"], language="json")
            reasoning_out = gr.Textbox(label=t0["reasoning_out_label"], lines=8)

        with gr.Group(elem_classes=["emoti-card"], elem_id="emoti-music"):
            # loop=True：一段播完自动从头重播 → 自动模式下「音乐不断」。
            # 下一段还在生成时旧的一直循环；新音乐推过来时播放器自动换成新的。
            # 状态不变被跳过（gr.skip()）时播放器不被触碰，循环也不中断。
            music_out = gr.Audio(label=t0["music_label"], autoplay=True, loop=True)
            # 第二个播放器：只在 Both 模式出现，默认隐藏；不 autoplay，避免与上面重叠出声。
            music_out_2 = gr.Audio(label=t0["music2_label"], autoplay=False, visible=False)

        # 输出列表按模式区分 heatmap 目标：auto_step 的热力图刷到自动组里的 heatmap_auto，
        # run_manual_video 的刷到手动组里的 heatmap_manual（其余位置完全一致）。
        outputs_auto = [summary_out, summary_ctx, heatmap_auto, perception_out, fusion_out, reasoning_out, music_out, music_out_2]
        outputs_manual = [summary_out, summary_ctx, heatmap_manual, perception_out, fusion_out, reasoning_out, music_out, music_out_2]

        # ---------------- 自动模式的数据流 ----------------
        # 关键（本项目踩过的坑，见 README.md §6.0，勿改回）：
        #   摄像头 stream 必须「直接驱动」pipeline，绝不能用 gr.Timer 去读中转 State。
        #   早期用 Timer 读 last_frame(State) 时，Timer 在独立事件里读回的是 None，
        #   auto_step 一直走「image is None → 全 skip」分支，输出区永远不刷新
        #   （症状：相机/麦克风都正常，但没有运行与输出）。
        #   正确做法：auto_cam.stream(fn=auto_step, ..., stream_every=AUTO_INTERVAL_SEC)，
        #   每 AUTO_INTERVAL_SEC 秒把「最新一帧」作为参数直接送进来，不会是 None。

        # 麦克风：numpy chunk 持续累积进 audio_buf（触发时由 auto_step 合并成 wav）
        auto_mic.stream(
            fn=accumulate_audio,
            inputs=[auto_mic, audio_buf],
            outputs=[audio_buf],
            stream_every=1,
            show_progress="hidden",
        )

        # 摄像头：每 AUTO_INTERVAL_SEC 秒把最新帧直接送进 auto_step 跑完整 pipeline，
        # 并带上累积的 audio_buf 与上一轮 last_key；auto_step 跑完会清空 audio_buf。
        # trigger_mode="always_last"：pipeline（LLM+音乐生成）一次要几十秒，远超推流
        # 间隔，事件会在队列里堆积、越跑越滞后；always_last 让堆积时只保留最新一次
        # 触发，跑完当前立刻处理「最新帧」，不追旧账。
        auto_cam.stream(
            fn=auto_step,
            inputs=[auto_cam, audio_buf, last_key, last_music_ts, session_label, reasoning_mode, speech_mode, music_duration, summary_lang],
            outputs=outputs_auto + [last_key, last_music_ts, audio_buf],
            stream_every=AUTO_INTERVAL_SEC,
            trigger_mode="always_last",
            show_progress="hidden",
        )

        # 手动模式（视频版）：
        #   结束录制(stop_recording) → 自动跑一次 pipeline；按钮 → 用当前视频重跑（可选）。
        # 输出比自动模式多一个「随机抽取的输入帧」预览(man_frame)。
        video_outputs = outputs_manual + [last_key, last_music_ts, man_frame]
        man_video.stop_recording(
            fn=run_manual_video,
            inputs=[man_video, session_label, reasoning_mode, speech_mode, music_duration, summary_lang],
            outputs=video_outputs,
        )
        run_btn.click(
            fn=run_manual_video,
            inputs=[man_video, session_label, reasoning_mode, speech_mode, music_duration, summary_lang],
            outputs=video_outputs,
        )

        # ---------------- 模式切换 ----------------
        # 只切换两个区域的可见性即可：切到手动时自动区（含流式摄像头）隐藏，
        # 摄像头停止推流，auto_step 自然不再触发；切回自动时恢复。
        def switch_mode(m):
            is_auto = (m == "auto")
            return (gr.update(visible=is_auto),       # auto_group
                    gr.update(visible=not is_auto))   # manual_group

        mode.change(fn=switch_mode, inputs=[mode],
                    outputs=[auto_group, manual_group])

        # ---------------- 全 UI 语言切换 ----------------
        # 切语言：整体替换所有文字组件（label/info/choices/markdown/按钮/占位），
        # 并用上一次的 ctx 即时重写「给你的说明」（不重跑 pipeline）。
        # 单选框只换 choices 的显示文本、value 稳定，故下游解析逻辑完全不受影响。
        lang_targets = [
            title_md, mode, reasoning_mode, music_duration,
            summary_lang, auto_cam, auto_mic, heatmap_auto, auto_note_md, manual_note_md, man_video,
            man_frame, heatmap_manual, run_btn, summary_out, details_accordion,
            perception_out, fusion_out, reasoning_out, music_out, music_out_2,
            auto_media_acc, manual_media_acc,
        ]

        def set_language(lang_val, ctx):
            lang = _norm_lang(lang_val)
            t = T[lang]
            if ctx:   # 已跑过 → 顺带用新语言重写说明（可能走大模型）
                summary_update = gr.update(label=t["summary_label"],
                                           placeholder=t["summary_placeholder"],
                                           value=_summary_text(ctx, lang))
            else:     # 还没跑过 → 只换 label/placeholder，不动内容
                summary_update = gr.update(label=t["summary_label"],
                                           placeholder=t["summary_placeholder"])
            return [
                gr.update(value=t["app_title"]),                                             # title_md
                gr.update(choices=t["mode_choices"], label=t["mode_label"], info=t["mode_info"]),
                gr.update(choices=t["reasoning_choices"], label=t["reasoning_label"], info=t["reasoning_info"]),
                gr.update(label=t["duration_label"], info=t["duration_info"]),
                gr.update(label=t["lang_label"], info=t["lang_info"]),                        # summary_lang
                gr.update(label=t["auto_cam_label"]),
                gr.update(label=t["auto_mic_label"]),
                gr.update(label=t["heatmap_label"]),                                          # heatmap_auto
                gr.update(value=t["auto_note"]),
                gr.update(value=t["manual_note"]),
                gr.update(label=t["man_video_label"]),
                gr.update(label=t["man_frame_label"]),
                gr.update(label=t["heatmap_label"]),                                          # heatmap_manual
                gr.update(value=t["run_btn_label"]),
                summary_update,                                                              # summary_out
                gr.update(label=t["accordion_label"]),
                gr.update(label=t["perception_label"]),
                gr.update(label=t["fusion_label"]),
                gr.update(label=t["reasoning_out_label"]),
                gr.update(label=t["music_label"]),
                gr.update(label=t["music2_label"]),
                gr.update(label=t["auto_media_acc_label"]),                                   # auto_media_acc
                gr.update(label=t["manual_media_acc_label"]),                                 # manual_media_acc
            ]

        summary_lang.change(fn=set_language, inputs=[summary_lang, summary_ctx], outputs=lang_targets)

    return demo


if __name__ == "__main__":
    print("=" * 60)
    print("EmotiCompanion module loading status:")
    print("=" * 60)
    app = build_ui()
    # Gradio 6.x：theme / css 从 Blocks 构造器移到了 launch()（只影响外观包装）。
    app.launch(theme=EMOTI_THEME, css=EMOTI_CSS)
