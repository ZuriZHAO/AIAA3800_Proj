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
     · 手动模式：保留原来的拍照/录音窗口，手动点击运行单次 pipeline。

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

# Auto mode timing:
# AUTO_INTERVAL_SEC: perception/fusion refresh interval
# MUSIC_REFRESH_SEC: when state is unchanged, generate a fresh music variation after this many seconds
AUTO_INTERVAL_SEC = int(os.getenv("AUTO_INTERVAL_SEC", str(CONFIG_AUTO_INTERVAL_SEC)))
MUSIC_REFRESH_SEC = int(os.getenv("MUSIC_REFRESH_SEC", "90"))


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
# 只存标签/文本（情绪/疲劳/需求/music_spec）+ 会话标签，**不含**人脸图像或录音。
# 路径默认 results/session_log.jsonl（results/ 已 gitignore）；可用 STUDY_LOG 覆盖。
# 用「会话标签」区分数据：实验时填参与者编号(P01…)，平时测试保持 test/自己的名字。
SESSION_LOG_PATH = os.getenv("STUDY_LOG", os.path.join("results", "session_log.jsonl"))


def _log_session(state, need, session_label):
    """把一次生成的系统输出追加进会话日志（失败静默，绝不影响主流程）。"""
    try:
        os.makedirs(os.path.dirname(SESSION_LOG_PATH) or ".", exist_ok=True)
        rec = {
            "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
            "label": (str(session_label).strip() or "test") if session_label else "test",
            "sys_emotion": state.get("dominant_emotion"),
            "sys_fatigue": state.get("fatigue"),
            "sys_need": need.get("need"),
            "sys_music_spec": need.get("music_spec"),
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


def _reason_and_compose(state, session_label=None, reasoning_mode="tom_cot"):
    """跑 ⑤ LLM 推理 + ⑥ 音乐生成，返回 (reasoning_text, music)。同时写会话日志。

    reasoning_mode：'tom_cot'（默认，ToM+CoT 两段式）或 'standard'（情绪→曲风直接查表基线）。
    供 user study 的 Q3 对比用——主试切到 standard 再跑一次即可，无需旁边写代码。
    """
    mode = "standard" if str(reasoning_mode).lower().startswith(("standard", "base")) else "tom_cot"
    if hasattr(LLM, "infer_with_mode"):
        need = LLM.infer_with_mode(state, mode)   # ⑤ (M3) 支持模式切换
    else:
        need = LLM.infer(state)                   # mock 或旧接口：回退默认
    music = MUSIC.generate(need["music_spec"])  # ⑥ MusicGen (M3)
    _log_session(state, need, session_label)    # 记录系统输出（user study 用）
    reasoning_text = (
        f"[Need] {need['need']}\n\n"
        f"[Reasoning (CoT)]\n{need['reasoning']}\n\n"
        f"[Music spec]\n{need['music_spec']}"
    )
    return reasoning_text, music


def _dump(obj):
    return json.dumps(obj, ensure_ascii=False, indent=2)



# =============================================================================
# Speech backend UI helper
# =============================================================================

SPEECH_MODE_CHOICES = [
    "emotion2vec (local, recommended)",
    "API / Qwen-Omni",
]


def _speech_mode_label_from_env():
    """Map SPEECH_BACKEND env value to the Gradio radio label."""
    backend = os.getenv("SPEECH_BACKEND", "emotion2vec").strip().lower()
    if backend in {"api", "qwen", "qwen-omni", "qwen_omni"}:
        return "API / Qwen-Omni"
    return "emotion2vec (local, recommended)"


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
# 手动模式 —— 单次运行完整 pipeline（总是重新生成音乐）
# =============================================================================

def run_manual(image, audio, session_label=None, reasoning_mode="tom_cot", speech_mode=None):
    perception, heatmap, state = _perceive_and_fuse(image, audio, speech_mode)
    reasoning_text, music = _reason_and_compose(state, session_label, reasoning_mode)
    _save_media(image, heatmap, session_label)   # user study 时留存 ⑦/② 可视化（默认关闭）
    # 同时把本次 state_key 写回 last_key，避免切到自动模式时重复生成
    return (heatmap, _dump(perception), _dump(state),
            reasoning_text, music, _state_key(state), time.time())


# =============================================================================
# 自动模式 —— 摄像头流式输入驱动（每帧直接送进来，不经 State 中转，避免读到 None）
#   情绪与压力都不变 → 跳过 ⑤⑥，保持当前音乐继续播放
#   不论是否变化，都刷新一个时间戳「心跳」，让你一眼看出 pipeline 在跑
# =============================================================================

def auto_step(image, audio_buffer, last_key, last_music_ts, session_label=None, reasoning_mode="tom_cot", speech_mode=None):
    """摄像头流式回调：拿当前帧 + 最近一段录音，感知融合，按需更换音乐。

    输出顺序对齐 UI：
      heatmap, perception, state, reasoning, music, last_key(state), audio_buffer(清空)
    """
    ts = time.strftime("%H:%M:%S")
    if image is None:
        # 摄像头还没就绪：刷新心跳提示，其余不动
        note = f"[{ts}] waiting for camera frame..."
        return (gr.skip(), gr.skip(), gr.skip(), note, gr.skip(), last_key, last_music_ts, [])

    audio = buffer_to_wav(audio_buffer)
    perception, heatmap, state = _perceive_and_fuse(image, audio, speech_mode)
    key = _state_key(state)

    # Auto mode: every valid detection generates a fresh music clip.
    # Even if emotion/fatigue are unchanged, we regenerate a same-state variation
    # so the companion does not become silent or feel static after the previous clip ends.
    reasoning_text, music = _reason_and_compose(state, session_label, reasoning_mode)
    _save_media(image, heatmap, session_label)   # user study 时留存 ⑦/② 可视化（默认关闭）

    if key == last_key:
        reasoning_text = (
            f"[{ts}] state unchanged "
            f"(emotion={key[0]}, fatigue={key[1]}) → regenerated a new same-state music variation.\n\n"
            + reasoning_text
        )
    else:
        reasoning_text = f"[{ts}] state CHANGED → regenerated music.\n\n" + reasoning_text

    return (heatmap, _dump(perception), _dump(state),
            reasoning_text, music, key, time.time(), [])


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
        gr.Markdown(
            "# 🎵 EmotiCompanion\n"
            "### Real-time Music Companion with Multimodal Emotion Perception "
            "— AIAA 3800 · HKUST(GZ)\n"
            "We **perceive** your state (face + speech + fatigue), **reason** about "
            "what you need, and **generate** music for this very moment."
        )

        # 跨回调共享的会话状态
        last_key = gr.State(None)    # 上一次的 (emotion, fatigue)，用于判断是否换音乐
        last_music_ts = gr.State(0.0) # 上一次生成音乐的时间戳，用于状态不变时定期刷新 variation
        audio_buf = gr.State([])     # 麦克风流式音频缓冲（numpy chunk 列表）

        mode = gr.Radio(
            ["Auto (continuous)", "Manual (one-shot)"],
            value="Auto (continuous)",
            label="Mode",
            info=f"Auto: camera & mic keep running; capture every {AUTO_INTERVAL_SEC}s, "
                 f"music stays on until you close the page. Manual: snapshot/upload then run once.",
        )

        # 会话标签：区分「实验数据」与「平时测试数据」。实验时填参与者编号(如 P03)，
        # 平时保持 test。写进 results/session_log.jsonl 的 label 列，事后按此筛选。
        session_label = gr.Textbox(
            label="🧪 Session label / 会话标签",
            value="test",
            info="实验时填参与者编号（如 P03）；平时测试保持 test。用于区分会话日志里的数据。",
        )

        # 推理模式：默认 ToM+CoT（部署走这个）；切 Standard 得到「情绪→曲风」直接查表基线。
        # user study Q3 对比用：主试切到 Standard、用同一输入再点一次即可，无需旁边写代码。
        reasoning_mode = gr.Radio(
            ["ToM+CoT (default)", "Standard (baseline)"],
            value="ToM+CoT (default)",
            label="⑤ Reasoning mode",
            info="对比用：默认 ToM+CoT 两段式推理；切 Standard 看直接查表基线（user study Q3）。",
        )

        # Speech backend：用于 demo / 消融 / debug 切换三种语音情绪识别模式。
        speech_mode = gr.Radio(
            SPEECH_MODE_CHOICES,
            value=_speech_mode_label_from_env(),
            label="③ Speech backend",
            info="默认 emotion2vec；可切 API/Qwen-Omni 做对比。切换后下一次运行生效。",
        )

        # ---------------- 自动模式区 ----------------
        with gr.Group(visible=True) as auto_group:
            with gr.Row():
                auto_cam = gr.Image(
                    label=f"📷 Live camera (auto-captured every {AUTO_INTERVAL_SEC}s)",
                    type="numpy", sources=["webcam"], streaming=True)
                auto_mic = gr.Audio(
                    label="🎙️ Live microphone (recording continuously)",
                    type="numpy", sources=["microphone"], streaming=True)
            gr.Markdown(
                f"> Auto mode is **on**. The pipeline runs every {AUTO_INTERVAL_SEC}s. "
                "Music only changes when emotion or fatigue changes."
            )

        # ---------------- 手动模式区（保留原拍照/录音窗口）----------------
        with gr.Group(visible=False) as manual_group:
            with gr.Row():
                man_img = gr.Image(label="Face image (snapshot / upload)", type="numpy",
                                   sources=["webcam", "upload"])
                man_audio = gr.Audio(label="Voice (record / upload)", type="filepath",
                                     sources=["microphone", "upload"])
            run_btn = gr.Button("▶ Run Pipeline once", variant="primary")

        # ---------------- 输出区（两模式共用）----------------
        with gr.Row():
            heatmap_out = gr.Image(label="⑦ GradCAM facial heatmap")
            perception_out = gr.Code(label="①②③ Perception output", language="json")
        fusion_out = gr.Code(label="④ Multimodal fusion · unified state JSON", language="json")
        reasoning_out = gr.Textbox(label="⑤ LLM need inference (ToM + CoT) · live status", lines=8)
        music_out = gr.Audio(label="⑥ Music companion (auto-plays)", autoplay=True)

        outputs = [heatmap_out, perception_out, fusion_out, reasoning_out, music_out]

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
            inputs=[auto_cam, audio_buf, last_key, last_music_ts, session_label, reasoning_mode, speech_mode],
            outputs=outputs + [last_key, last_music_ts, audio_buf],
            stream_every=AUTO_INTERVAL_SEC,
            show_progress="hidden",
        )

        # 手动模式按钮
        run_btn.click(
            fn=run_manual,
            inputs=[man_img, man_audio, session_label, reasoning_mode, speech_mode],
            outputs=outputs + [last_key, last_music_ts],
        )

        # ---------------- 模式切换 ----------------
        # 只切换两个区域的可见性即可：切到手动时自动区（含流式摄像头）隐藏，
        # 摄像头停止推流，auto_step 自然不再触发；切回自动时恢复。
        def switch_mode(m):
            is_auto = m.startswith("Auto")
            return (gr.update(visible=is_auto),       # auto_group
                    gr.update(visible=not is_auto))   # manual_group

        mode.change(fn=switch_mode, inputs=[mode],
                    outputs=[auto_group, manual_group])

        gr.Markdown(
            "---\n"
            "> Modules live in each member's own file: `face_emotion.py`(M1) · "
            "`fatigue.py`(M4) · `speech_emotion.py`(M2) · `fusion.py`(M2/M4) · "
            "`llm_reason.py`(M3) · `music_gen.py`(M3). Not-ready modules fall back to "
            "mock automatically; load status is printed to the console on startup. "
            "Shared constants are in `config.py`."
        )

    return demo


if __name__ == "__main__":
    print("=" * 60)
    print("EmotiCompanion module loading status:")
    print("=" * 60)
    app = build_ui()
    app.launch()
