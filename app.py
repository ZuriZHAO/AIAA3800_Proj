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

# 让本地回环地址绕过系统/环境代理。否则开了科学上网（系统代理指向
# 127.0.0.1:xxxx）时，gradio 启动后自检 http://127.0.0.1 的请求会被代理
# 拦截返回 502，导致 app.launch() 直接抛异常退出。在这里把 localhost 加进
# no_proxy，对所有组员（无论是否开代理）都安全生效。
for _v in ("no_proxy", "NO_PROXY"):
    _cur = os.environ.get(_v, "")
    _need = "127.0.0.1,localhost,::1"
    os.environ[_v] = f"{_cur},{_need}" if _cur else _need

import numpy as np
import gradio as gr

from config import AUTO_INTERVAL_SEC
import mocks


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


def _perceive_and_fuse(image, audio):
    """跑 ①②③ 感知 + ⑦ 热力图 + ④ 融合，返回 (perception, heatmap, state)。"""
    face = FACE.predict(image)          # ① 人脸情绪 (M1)
    heatmap = FACE.gradcam(image)       # ⑦ GradCAM (M1)
    fatigue = FATIGUE.predict(image)    # ② 疲劳/压力 (M4)
    speech = SPEECH.predict(audio)      # ③ 语音情绪 (M2)
    state = FUSION.fuse(face, speech, fatigue)  # ④ 融合 (M2/M4)
    perception = {"face": face, "speech": speech, "fatigue": fatigue}
    return perception, heatmap, state


def _reason_and_compose(state):
    """跑 ⑤ LLM 推理 + ⑥ 音乐生成，返回 (reasoning_text, music)。"""
    need = LLM.infer(state)             # ⑤ ToM + CoT 推理 (M3)
    music = MUSIC.generate(need["music_spec"])  # ⑥ MusicGen (M3)
    reasoning_text = (
        f"[Need] {need['need']}\n\n"
        f"[Reasoning (CoT)]\n{need['reasoning']}\n\n"
        f"[Music spec]\n{need['music_spec']}"
    )
    return reasoning_text, music


def _dump(obj):
    return json.dumps(obj, ensure_ascii=False, indent=2)


# =============================================================================
# 手动模式 —— 单次运行完整 pipeline（总是重新生成音乐）
# =============================================================================

def run_manual(image, audio):
    perception, heatmap, state = _perceive_and_fuse(image, audio)
    reasoning_text, music = _reason_and_compose(state)
    # 同时把本次 state_key 写回 last_key，避免切到自动模式时重复生成
    return (heatmap, _dump(perception), _dump(state),
            reasoning_text, music, _state_key(state))


# =============================================================================
# 自动模式 —— 摄像头流式输入驱动（每帧直接送进来，不经 State 中转，避免读到 None）
#   情绪与压力都不变 → 跳过 ⑤⑥，保持当前音乐继续播放
#   不论是否变化，都刷新一个时间戳「心跳」，让你一眼看出 pipeline 在跑
# =============================================================================

def auto_step(image, audio_buffer, last_key):
    """摄像头流式回调：拿当前帧 + 最近一段录音，感知融合，按需更换音乐。

    输出顺序对齐 UI：
      heatmap, perception, state, reasoning, music, last_key(state), audio_buffer(清空)
    """
    ts = time.strftime("%H:%M:%S")
    if image is None:
        # 摄像头还没就绪：刷新心跳提示，其余不动
        note = f"[{ts}] waiting for camera frame..."
        return (gr.skip(), gr.skip(), gr.skip(), note, gr.skip(), last_key, [])

    audio = buffer_to_wav(audio_buffer)
    perception, heatmap, state = _perceive_and_fuse(image, audio)
    key = _state_key(state)

    # 情绪与压力都没变 → 不重新推理/生成，音乐继续播；仍刷新感知/状态/心跳
    if key == last_key:
        note = (f"[{ts}] running · state unchanged "
                f"(emotion={key[0]}, fatigue={key[1]}) · keeping current music.")
        return (heatmap, _dump(perception), _dump(state),
                note, gr.skip(), last_key, [])

    # 状态变化 → 重新推理并生成新音乐
    reasoning_text, music = _reason_and_compose(state)
    reasoning_text = f"[{ts}] state CHANGED → regenerated music.\n\n" + reasoning_text
    return (heatmap, _dump(perception), _dump(state),
            reasoning_text, music, key, [])


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
        audio_buf = gr.State([])     # 麦克风流式音频缓冲（numpy chunk 列表）

        mode = gr.Radio(
            ["Auto (continuous)", "Manual (one-shot)"],
            value="Auto (continuous)",
            label="Mode",
            info=f"Auto: camera & mic keep running; capture every {AUTO_INTERVAL_SEC}s, "
                 f"music stays on until you close the page. Manual: snapshot/upload then run once.",
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
        # 麦克风流式输入：numpy chunk 持续累积进缓冲
        auto_mic.stream(
            fn=accumulate_audio,
            inputs=[auto_mic, audio_buf],
            outputs=[audio_buf],
            stream_every=1,
        )

        # 摄像头流式输入直接驱动 pipeline：每帧直接送进 auto_step（不经 State 中转，
        # 避免定时器读到 None 导致输出一直不刷新）。stream_every 控制处理频率。
        cam_stream = auto_cam.stream(
            fn=auto_step,
            inputs=[auto_cam, audio_buf, last_key],
            outputs=outputs + [last_key, audio_buf],
            stream_every=AUTO_INTERVAL_SEC,
            show_progress="hidden",
        )

        # 手动模式按钮
        run_btn.click(
            fn=run_manual,
            inputs=[man_img, man_audio],
            outputs=outputs + [last_key],
        )

        # ---------------- 模式切换 ----------------
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
