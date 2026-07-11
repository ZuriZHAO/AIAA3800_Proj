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

from dotenv import load_dotenv

load_dotenv()

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


# =============================================================================
# Pipeline 辅助函数
# =============================================================================

def _state_key(state):
    """从融合状态里抽出「情绪 + 压力」二元组，作为是否更换音乐的判断依据。"""
    return (state.get("dominant_emotion"), state.get("fatigue"))


def _perceive_and_fuse(image, audio, speech_mode=None):
    """跑 ①②③ 感知 + ⑦ 热力图 + ④ 融合，返回 (perception, heatmap, state)。"""
    backend = _set_speech_backend_from_ui(speech_mode)
    face = FACE.predict(image)          # ① 人脸情绪 (M1)
    heatmap = FACE.gradcam(image)       # ⑦ GradCAM (M1)
    fatigue = FATIGUE.predict(image)    # ② 疲劳/压力 (M4)
    speech = SPEECH.predict(audio)      # ③ 语音情绪 (M2)

    # 在 UI 输出里显式记录本次使用的 speech backend，方便 user study / debug 对齐。
    if isinstance(speech, dict):
        speech = dict(speech)
        speech.setdefault("backend", backend)

    state = FUSION.fuse(face, speech, fatigue)  # ④ 融合 (M2/M4)
    perception = {"face": face, "speech": speech, "fatigue": fatigue}
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
    if hasattr(LLM, "infer_with_mode"):
        need = LLM.infer_with_mode(state, mode)   # ⑤ (M3) 支持模式切换
    else:
        need = LLM.infer(state)                   # mock 或旧接口：回退默认
    # ⑥ MusicGen (M3)，按 UI 滑条时长生成；兼容尚未支持 duration 参数的旧签名。
    try:
        music = MUSIC.generate(need["music_spec"], duration_sec)
    except TypeError:
        music = MUSIC.generate(need["music_spec"])
    # 记录系统输出（完整感知/融合/推理，user study 用）
    _log_session(state, need, session_label, perception=perception, mode=mode)
    return need, music


def _one_mode_text(need, cot=True):
    tag = "Reasoning (CoT)" if cot else "Reasoning"
    return (
        f"[Need] {need['need']}\n\n"
        f"[{tag}]\n{need['reasoning']}\n\n"
        f"[Music spec]\n{need['music_spec']}"
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
        hedge = "可能" if low else ""
        return f"🙂 现在{hedge}检测到你的情绪偏「{emo_zh}」，疲劳程度{fat_zh}。"
    emo_en = _EMOTION_EN.get(emo, emo)
    fat_en = _FATIGUE_EN.get(fat, fat)
    verb = "may be feeling" if low else "seem"
    return f"🙂 Right now you {verb} {emo_en}, with {fat_en} fatigue."


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
                    "🎵 这次生成了两段可对比的音乐：\n"
                    f"　· A（ToM+CoT）：{style_a}\n"
                    f"　· B（Standard）：{style_b}")
        return (_summary_head(state, "en") + "\n"
                "🎵 Two versions to compare:\n"
                f"　· A (ToM+CoT): {style_a}\n"
                f"　· B (Standard): {style_b}")
    style = _music_style(ctx.get("spec", ""), lang)
    if lang == "zh":
        return _summary_head(state, "zh") + "\n" + f"🎵 为你推荐了一段{style}，希望能陪伴此刻的你。"
    return _summary_head(state, "en") + "\n" + f"🎵 We've picked some {style} to keep you company right now."


def _summary_text(ctx, lang="en"):
    """给用户的话：优先大模型生成（单模式 & Both 都走 DeepSeek），失败/未配置 → 模板兜底。

    ctx 为空（还没跑过 pipeline）→ gr.skip()，不改动当前显示。
    """
    if not ctx:
        return gr.skip()
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
    perception：①②③ 感知结果，仅用于会话日志留存。
    lang：摘要语言，'en'（默认）或 'zh'。

    music1_update / music2_update 都是 gr.update()：动态设置每个播放器的音频与标注，
    并在非 both 模式下隐藏第二个播放器，保证「标清楚 + 不误导」。
    """
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
        ctx = {"both": True, "state": state,
               "spec_tom": need_tom.get("music_spec", ""),
               "spec_std": need_std.get("music_spec", ""),
               "need_tom": need_tom, "need_std": need_std}
        m1 = gr.update(value=music_tom, label="⑥-A Music · ToM+CoT (default)", visible=True)
        m2 = gr.update(value=music_std, label="⑥-B Music · Standard (baseline)", visible=True)
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
    ctx = {"both": False, "state": state, "spec": need.get("music_spec", ""), "need": need}
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
            "### Real-time Music Companion with Multimodal Emotion Perception "
            "— AIAA 3800 · HKUST(GZ)\n"
            "We **perceive** your state (face + speech + fatigue), **reason** about "
            "what you need, and **generate** music for this very moment."
        ),
        "mode_label": "Mode",
        "mode_info": (f"Auto: camera & mic keep running; capture every {a}s, music stays on "
                      "until you close the page. Manual: record a video, then process once."),
        "mode_choices": [("Auto (continuous)", "auto"), ("Manual (one-shot)", "manual")],
        "session_label_label": "🧪 Session label",
        "session_label_info": ("Enter a participant ID (e.g. P03) during the study; keep 'test' "
                               "otherwise. Used to separate rows in the session log."),
        "reasoning_label": "⑤ Reasoning mode",
        "reasoning_info": ("ToM+CoT: two-stage reasoning (default). Standard: direct emotion→style "
                           "baseline. Both: generate both, with a second player below."),
        "reasoning_choices": [("ToM+CoT (default)", "tom_cot"),
                              ("Standard (baseline)", "standard"),
                              ("Both (both modes)", "both")],
        "speech_label": "③ Speech backend",
        "speech_info": "Default emotion2vec; switch to API/Qwen-Omni to compare. Applies on the next run.",
        "speech_choices": [("emotion2vec (local, recommended)", "emotion2vec"),
                           ("API / Qwen-Omni", "api")],
        "duration_label": "⑥ Music length (seconds)",
        "duration_info": (f"Drag to choose the music length, 0–{mx}s; 0 = no music (silent). "
                          "Longer = slower. Applies on the next run."),
        "lang_label": "🌐 Language",
        "lang_info": ("Switch the whole interface and the message for you between English and "
                      "Chinese. Applies instantly."),
        "lang_choices": [("English", "en"), ("中文", "zh")],
        "auto_cam_label": f"📷 Live camera (auto-captured every {a}s)",
        "auto_mic_label": "🎙️ Live microphone (recording continuously)",
        "auto_note": (f"> Auto mode is **on**. The pipeline runs every {a}s. "
                      "Music only changes when emotion or fatigue changes."),
        "manual_note": ("> Manual mode (video): click **● Record** on the video box, record a clip, "
                        "then click **■ Stop**. On stop we automatically sample a frame as the face "
                        "image and split the audio as speech, then run the full pipeline."),
        "man_video_label": "🎥 Record video (with audio)",
        "man_frame_label": "🖼️ Sampled frame",
        "run_btn_label": "▶ Reprocess current video (optional)",
        "summary_label": "💬 For you",
        "summary_placeholder": ("After a run, this tells you in plain words how you seem and "
                                "what music we picked."),
        "heatmap_label": "⑦ GradCAM facial heatmap",
        "accordion_label": "🔎 Details (①②③ perception / ④ fusion / ⑤ reasoning)",
        "perception_label": "①②③ Perception output",
        "fusion_label": "④ Multimodal fusion · unified state JSON",
        "reasoning_out_label": "⑤ LLM need inference (ToM + CoT) · live status",
        "music_label": "⑥ Music companion (auto-plays)",
        "music2_label": "⑥-B Music · (second clip, Both mode)",
        "footer": (
            "---\n"
            "> Modules live in each member's own file: `face_emotion.py`(M1) · "
            "`fatigue.py`(M4) · `speech_emotion.py`(M2) · `fusion.py`(M2/M4) · "
            "`llm_reason.py`(M3) · `music_gen.py`(M3). Not-ready modules fall back to "
            "mock automatically; load status is printed to the console on startup. "
            "Shared constants are in `config.py`."
        ),
    }
    zh = {
        "app_title": (
            "# 🎵 EmotiCompanion\n"
            "### 多模态情绪感知实时音乐陪伴 — AIAA 3800 · 香港科技大学（广州）\n"
            "我们**感知**你的状态（人脸 + 语音 + 疲劳），**推理**你此刻的需要，"
            "并为这一刻**生成**音乐。"
        ),
        "mode_label": "模式",
        "mode_info": (f"自动：摄像头/麦克风持续工作，每 {a} 秒捕捉一次，音乐一直陪伴直到关闭网页。"
                      "手动：录一段视频再处理一次。"),
        "mode_choices": [("自动（持续）", "auto"), ("手动（单次）", "manual")],
        "session_label_label": "🧪 会话标签",
        "session_label_info": "实验时填参与者编号（如 P03）；平时保持 test。用于区分会话日志里的数据。",
        "reasoning_label": "⑤ 推理模式",
        "reasoning_info": ("ToM+CoT：两段式推理（默认）。Standard：情绪→曲风直接查表基线。"
                           "Both：两种都生成，下方出现第二个播放器。"),
        "reasoning_choices": [("ToM+CoT（默认）", "tom_cot"),
                              ("Standard（基线）", "standard"),
                              ("Both（两种都生成）", "both")],
        "speech_label": "③ 语音后端",
        "speech_info": "默认 emotion2vec；可切 API/Qwen-Omni 做对比。下一次运行生效。",
        "speech_choices": [("emotion2vec（本地，推荐）", "emotion2vec"),
                           ("API / Qwen-Omni", "api")],
        "duration_label": "⑥ 音乐长度（秒）",
        "duration_info": (f"拖动选择生成音乐的时长，0~{mx}s；0=不生成（静音）。越长越慢，下一次运行生效。"),
        "lang_label": "🌐 语言",
        "lang_info": "在中/英之间切换整个界面和「给你的说明」；立即生效。",
        "lang_choices": [("English", "en"), ("中文", "zh")],
        "auto_cam_label": f"📷 实时摄像头（每 {a} 秒自动捕捉）",
        "auto_mic_label": "🎙️ 实时麦克风（持续录音）",
        "auto_note": (f"> 自动模式**已开启**。每 {a} 秒跑一次 pipeline。"
                      "只有情绪或疲劳变化时才更换音乐。"),
        "manual_note": ("> 手动模式（视频）：点击视频框的 **● 录制** 开始，录一段后点 **■ 停止**。"
                        "停止后会**自动**从视频随机抽一帧当人脸图像、分离音频当语音，直接跑完整 pipeline。"),
        "man_video_label": "🎥 录制视频（含音频）",
        "man_frame_label": "🖼️ 随机抽取的输入帧",
        "run_btn_label": "▶ 重新处理当前视频（可选）",
        "summary_label": "💬 给你的说明",
        "summary_placeholder": "运行后这里会用大白话告诉你现在的情绪/疲劳，以及为你推荐了什么音乐。",
        "heatmap_label": "⑦ GradCAM 人脸热力图",
        "accordion_label": "🔎 详细中间输出（①②③ 感知 / ④ 融合 / ⑤ 推理）",
        "perception_label": "①②③ 感知输出",
        "fusion_label": "④ 多模态融合 · 统一状态 JSON",
        "reasoning_out_label": "⑤ LLM 需求推理（ToM + CoT）· 实时状态",
        "music_label": "⑥ 音乐陪伴（自动播放）",
        "music2_label": "⑥-B 音乐 · (Both 模式第二段)",
        "footer": (
            "---\n"
            "> 各模块在各自成员的文件里：`face_emotion.py`(M1) · `fatigue.py`(M4) · "
            "`speech_emotion.py`(M2) · `fusion.py`(M2/M4) · `llm_reason.py`(M3) · "
            "`music_gen.py`(M3)。未就绪的模块会自动回退到 mock；启动时控制台会打印加载状态。"
            "共享常量在 `config.py`。"
        ),
    }
    return {"en": en, "zh": zh}

# =============================================================================
# 手动模式 —— 单次运行完整 pipeline（总是重新生成音乐）
# =============================================================================

def run_manual(image, audio, session_label=None, reasoning_mode="tom_cot", speech_mode=None, duration_sec=None, summary_lang="en"):
    perception, heatmap, state = _perceive_and_fuse(image, audio, speech_mode)
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

    perception, heatmap, state = _perceive_and_fuse(image, audio, speech_mode)
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
    perception, heatmap, state = _perceive_and_fuse(image, audio, speech_mode)
    key = _state_key(state)

    # Auto mode: every valid detection generates a fresh music clip.
    # Even if emotion/fatigue are unchanged, we regenerate a same-state variation
    # so the companion does not become silent or feel static after the previous clip ends.
    summary_text, summary_ctx, reasoning_text, music, music2 = _reason_and_compose(
        state, session_label, reasoning_mode, duration_sec, perception, lang)
    _save_media(image, heatmap, session_label)   # user study 时留存 ⑦/② 可视化（默认关闭）

    if key == last_key:
        reasoning_text = (
            f"[{ts}] state unchanged "
            f"(emotion={key[0]}, fatigue={key[1]}) → regenerated a new same-state music variation.\n\n"
            + reasoning_text
        )
    else:
        reasoning_text = f"[{ts}] state CHANGED → regenerated music.\n\n" + reasoning_text

    return (summary_text, summary_ctx, heatmap, _dump(perception), _dump(state),
            reasoning_text, music, music2, key, time.time(), [])


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

def build_ui():
    with gr.Blocks(title="EmotiCompanion") as demo:
        T = _texts()
        t0 = T["en"]   # 初始英文；切换语言时由 set_language 整体替换

        title_md = gr.Markdown(t0["app_title"])

        # 跨回调共享的会话状态
        last_key = gr.State(None)    # 上一次的 (emotion, fatigue)，用于判断是否换音乐
        last_music_ts = gr.State(0.0) # 上一次生成音乐的时间戳，用于状态不变时定期刷新 variation
        audio_buf = gr.State([])     # 麦克风流式音频缓冲（numpy chunk 列表）
        summary_ctx = gr.State(None) # 上一次摘要的最小上下文，切语言时即时重刷用

        # 单选框统一用 (显示文本, 稳定值)：显示随语言变，value 恒定，所有解析逻辑不受影响。
        mode = gr.Radio(t0["mode_choices"], value="auto",
                        label=t0["mode_label"], info=t0["mode_info"])

        session_label = gr.Textbox(label=t0["session_label_label"], value="test",
                                   info=t0["session_label_info"])

        reasoning_mode = gr.Radio(t0["reasoning_choices"], value="tom_cot",
                                  label=t0["reasoning_label"], info=t0["reasoning_info"])

        speech_mode = gr.Radio(t0["speech_choices"], value=_speech_mode_value_from_env(),
                               label=t0["speech_label"], info=t0["speech_info"])

        music_duration = gr.Slider(
            minimum=0, maximum=MUSIC_MAX_DURATION_SEC, step=1,
            value=MUSIC_DEFAULT_DURATION_SEC,
            label=t0["duration_label"], info=t0["duration_info"])

        # 全局语言开关：默认英文，选中文后整个界面 + 给用户的话 + 标签都变中文（立即生效）。
        summary_lang = gr.Radio(t0["lang_choices"], value="en",
                                label=t0["lang_label"], info=t0["lang_info"])

        # ---------------- 自动模式区 ----------------
        with gr.Group(visible=True) as auto_group:
            with gr.Row():
                auto_cam = gr.Image(label=t0["auto_cam_label"],
                                    type="numpy", sources=["webcam"], streaming=True)
                auto_mic = gr.Audio(label=t0["auto_mic_label"],
                                    type="numpy", sources=["microphone"], streaming=True)
            auto_note_md = gr.Markdown(t0["auto_note"])

        # ---------------- 手动模式区（视频版：录制 → 抽帧+分离音频 → 单次 pipeline）----------------
        with gr.Group(visible=False) as manual_group:
            manual_note_md = gr.Markdown(t0["manual_note"])
            with gr.Row():
                man_video = gr.Video(label=t0["man_video_label"],
                                     sources=["webcam"], include_audio=True,
                                     webcam_options=gr.WebcamOptions(mirror=False))
                man_frame = gr.Image(label=t0["man_frame_label"], type="numpy")
            run_btn = gr.Button(t0["run_btn_label"], variant="primary")

        # ---------------- 输出区（两模式共用）----------------
        # 面向用户的友好摘要：由大模型生成（未配置 LLM 后端时回退模板），显眼常显。
        summary_out = gr.Textbox(label=t0["summary_label"], lines=3, interactive=False,
                                 placeholder=t0["summary_placeholder"])
        heatmap_out = gr.Image(label=t0["heatmap_label"])

        # 中间产物（①②③ 感知 / ④ 融合 / ⑤ 推理）默认收起：一般用户不需要看，需要时展开。
        with gr.Accordion(t0["accordion_label"], open=False) as details_accordion:
            perception_out = gr.Code(label=t0["perception_label"], language="json")
            fusion_out = gr.Code(label=t0["fusion_label"], language="json")
            reasoning_out = gr.Textbox(label=t0["reasoning_out_label"], lines=8)

        music_out = gr.Audio(label=t0["music_label"], autoplay=True)
        # 第二个播放器：只在 Both 模式出现，默认隐藏；不 autoplay，避免与上面重叠出声。
        music_out_2 = gr.Audio(label=t0["music2_label"], autoplay=False, visible=False)

        outputs = [summary_out, summary_ctx, heatmap_out, perception_out, fusion_out, reasoning_out, music_out, music_out_2]

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
        auto_cam.stream(
            fn=auto_step,
            inputs=[auto_cam, audio_buf, last_key, last_music_ts, session_label, reasoning_mode, speech_mode, music_duration, summary_lang],
            outputs=outputs + [last_key, last_music_ts, audio_buf],
            stream_every=AUTO_INTERVAL_SEC,
            show_progress="hidden",
        )

        # 手动模式（视频版）：
        #   结束录制(stop_recording) → 自动跑一次 pipeline；按钮 → 用当前视频重跑（可选）。
        # 输出比自动模式多一个「随机抽取的输入帧」预览(man_frame)。
        video_outputs = outputs + [last_key, last_music_ts, man_frame]
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

        footer_md = gr.Markdown(t0["footer"])

        # ---------------- 全 UI 语言切换 ----------------
        # 切语言：整体替换所有文字组件（label/info/choices/markdown/按钮/占位），
        # 并用上一次的 ctx 即时重写「给你的说明」（不重跑 pipeline）。
        # 单选框只换 choices 的显示文本、value 稳定，故下游解析逻辑完全不受影响。
        lang_targets = [
            title_md, mode, session_label, reasoning_mode, speech_mode, music_duration,
            summary_lang, auto_cam, auto_mic, auto_note_md, manual_note_md, man_video,
            man_frame, run_btn, summary_out, heatmap_out, details_accordion,
            perception_out, fusion_out, reasoning_out, music_out, music_out_2, footer_md,
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
                gr.update(label=t["session_label_label"], info=t["session_label_info"]),
                gr.update(choices=t["reasoning_choices"], label=t["reasoning_label"], info=t["reasoning_info"]),
                gr.update(choices=t["speech_choices"], label=t["speech_label"], info=t["speech_info"]),
                gr.update(label=t["duration_label"], info=t["duration_info"]),
                gr.update(label=t["lang_label"], info=t["lang_info"]),                        # summary_lang
                gr.update(label=t["auto_cam_label"]),
                gr.update(label=t["auto_mic_label"]),
                gr.update(value=t["auto_note"]),
                gr.update(value=t["manual_note"]),
                gr.update(label=t["man_video_label"]),
                gr.update(label=t["man_frame_label"]),
                gr.update(value=t["run_btn_label"]),
                summary_update,                                                              # summary_out
                gr.update(label=t["heatmap_label"]),
                gr.update(label=t["accordion_label"]),
                gr.update(label=t["perception_label"]),
                gr.update(label=t["fusion_label"]),
                gr.update(label=t["reasoning_out_label"]),
                gr.update(label=t["music_label"]),
                gr.update(label=t["music2_label"]),
                gr.update(value=t["footer"]),
            ]

        summary_lang.change(fn=set_language, inputs=[summary_lang, summary_ctx], outputs=lang_targets)

    return demo


if __name__ == "__main__":
    print("=" * 60)
    print("EmotiCompanion module loading status:")
    print("=" * 60)
    app = build_ui()
    app.launch()
