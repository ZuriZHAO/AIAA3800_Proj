# 🎵 EmotiCompanion

**多模态情绪感知的实时音乐陪伴系统**
AIAA 3800 课程项目 · HKUST(GZ) · June 2026

> 我们让系统先**感知**你的状态（人脸 + 语音 + 疲劳），再**推理**你需要什么，最后**生成**一段专属于这一刻的音乐。

---

## 1. 项目简介

人和音乐的交互还停留在静态模式——你选一次曲风，它就固定了，系统不知道你现在是什么状态。EmotiCompanion 做三件事：

1. **感知（Perception）**：多模态自动感知用户情绪，无需手动告诉系统你的心情
   - 人脸表情情绪（DeepFace）
   - 语音情绪（EmotionThinker / Qwen2.5-Omni）
   - 疲劳状态（L9P1 疲劳检测模型）
2. **推理（Reasoning）**：知道情绪还不够，要推断需求。用 LLM + Theory of Mind 提示 + CoT 推理，从「情绪标签」推断「心理需求」（例如：考试前焦虑 → 需要舒缓降低唤醒；演讲前焦虑 → 需要振奋提升状态）
3. **生成（Generation）**：根据推理出的音乐描述词，用 MusicGen 生成专属音频片段

---

## 2. 系统架构

```
  人脸图像 ──┬──► ① 人脸情绪 (DeepFace)      ──┐
            │                                  │
            └──► ② 疲劳检测 (L9P1)           ──┤
                                               ├──► ④ 多模态融合 ──► 统一状态 JSON
  语音音频 ────► ③ 语音情绪 (EmotionThinker) ──┘            │
                                                            ▼
                                              ⑤ LLM 需求推理 (ToM + CoT)
                                                            │
                                                            ▼
                                              ⑥ 音乐生成 (MusicGen) ──► 🎵
  ⑦ GradCAM 热力图旁挂在①上，⑧ Gradio app.py 串联全部
```

### 模块一览

| # | 模块 | 类型 | 技术/工具 | 对应 Practice | 负责人 |
|---|------|------|-----------|---------------|--------|
| ① | 人脸情绪感知 | 基于 Lab | DeepFace → `{emotion, confidence}` | L4P1 | M1 |
| ② | 疲劳状态检测 | 基于 Lab | L9P1 疲劳检测 → `{fatigue_level}` | L9P1 | M4 |
| ③ | 语音情绪感知 | 基于 Lab | EmotionThinker (Qwen2.5-Omni) → `{emotion, confidence}` | L4P3 | M2 |
| ④ | 多模态融合 | 自研 | 置信度加权融合三路输入 → 统一状态 JSON | — | M2 主导 / M4 协助 |
| ⑤ | LLM 需求推理 | 基于 Lab | ToM 增强提示 + CoT 推理 → 心理需求 + 音乐描述词 | L10P1 + L10P2 | M3 |
| ⑥ | 音乐生成 | 基于 Lab | MusicGen (text-to-music) → 音频片段 | L5P3 | M3 |
| ⑦ | GradCAM 可解释 | 基于 Lab | 对 DeepFace 卷积层做 Grad-CAM → 面部热力图 | L12P1 | M1 |
| ⑧ | Gradio UI + 集成 | 自研 | `app.py` 定义接口，串联所有模块 | — | M4 主导 |

---

## 3. 快速开始

```bash
# 安装框架依赖（注意 gradio 需 >= 4.39，自动模式用到 gr.skip / 流式输入 stream_every）
pip install -r requirements.txt

# 启动（即使各模块尚未实现，也能用 mock 数据端到端跑通）
python app.py
```

启动后控制台会打印每个模块的加载状态，浏览器自动打开 Gradio 页面（界面文字为英文），**默认进入自动模式**。

```
============================================================
EmotiCompanion module loading status:
============================================================
[OK ] ① face + ⑦ GradCAM (M1): loaded real module face_emotion.py
[mock] ③ speech (M2): using mock (speech_emotion.py not ready: No module named 'speech_emotion')
...
```

- `[OK ]` = 找到该成员的真实文件，用真实实现
- `[mock]` = 没找到该文件，用假数据顶上（不影响整体运行）

### 两种工作模式

页面顶部 **Mode** 单选框切换：

- **自动模式 Auto（默认）**：打开网页后摄像头/麦克风持续运行，每 `AUTO_INTERVAL_SEC` 秒（当前 5，见 `config.py`）自动捕捉一帧画面 + 一段录音跑一次 pipeline，生成的音乐自动播放，**陪伴一直存在直到关闭网页**。**情绪与压力都不变时不更换音乐**（当前曲子继续播不打断）。模块未实现时走 mock：感知固定返回舒缓/中性状态（不随机），音乐栏留空白（不乱生成占位音符）。
- **手动模式 Manual**：保留传统的拍照/上传图片 + 录音/上传音频窗口，点击 **▶ Run Pipeline once** 运行单次 pipeline。

> 详细的运行、内部原理与逐文件讲解见 [代码说明.md](代码说明.md)。

---

## 4. 协作方式（重要）

**`app.py` 是「胶水层」，本身不实现任何模型。** 它只动态导入各成员的文件、串联 pipeline、提供双模式 UI。常量与 schema 统一放在 `config.py`，mock 实现统一放在 `mocks.py`。

每位成员**在自己的文件里**实现约定好的函数签名即可，`app.py` 会自动接上——**未就绪的模块自动回退到 mock，无需改动 `app.py`，互不冲突。**

### 接口契约（请严格遵守文件名与函数签名）

| 文件名 | 负责人 | 需实现的函数 | 返回 |
|--------|--------|--------------|------|
| `face_emotion.py` | M1 | `predict(image)` | `{"emotion": str, "confidence": float}` |
| `face_emotion.py` | M1 | `gradcam(image)` | 热力图 image (np.ndarray / PIL.Image) |
| `fatigue.py` | M4 | `predict(image)` | `{"fatigue_level": str, "confidence": float}` |
| `speech_emotion.py` | M2 | `predict(audio_path)` | `{"emotion": str, "confidence": float, "reasoning": str}` |
| `fusion.py` | M2 主导 / M4 协助 | `fuse(face, speech, fatigue)` | 统一状态 JSON（见下） |
| `llm_reason.py` | M3 | `infer(state)` | `{"need": str, "reasoning": str, "music_spec": str}` |
| `music_gen.py` | M3 | `generate(music_spec)` | 音频（文件路径 str 或 `(sr, np.ndarray)`） |

### 统一状态 JSON（融合层 ④ 输出格式）

```json
{
  "dominant_emotion": "fear",
  "confidence": 0.74,
  "fatigue": "high",
  "face_conf": 0.78,
  "speech_conf": 0.61,
  "fatigue_conf": 0.85
}
```

三路输入（人脸情绪、语音情绪、疲劳状态）各自带 confidence score，动态加权后输出统一状态 JSON，交给 ⑤ 的 LLM 推理层。

> **情绪词表（全队契约）**：`dominant_emotion` 取值统一为 Ekman 标准 7 类 —— `neutral / happy / sad / angry / fear / surprise / disgust`（见 `config.EMOTION_LABELS`）。各感知模块把自己模型的原生标签**映射进这套词表**：人脸 ①（AffectNet-8）的映射见 `face_emotion.py`，语音 ③（EmotionThinker）的映射由 M2 在 `speech_emotion.py` 实现。困倦不属于情绪轴，单独走 `fatigue` 字段。

---

## 5. 目录结构

```
AIAA3800_Proj/
├── app.py              # ⑧ Gradio UI + 集成（M4）—— 加载模块 + 串 pipeline + 双模式 UI
├── config.py           # 全局常量与 schema（情绪标签、压力等级、捕捉间隔 60s 等）
├── mocks.py            # 6 个模块的 mock 占位实现（真实文件未就绪时自动启用）
├── requirements.txt    # 框架运行最小依赖（gradio>=4.39, numpy）
├── face_emotion.py     # ① 人脸情绪 + ⑦ GradCAM（M1）—— 待 M1 创建
├── fatigue.py          # ② 疲劳检测（M4）—— 待 M4 创建
├── speech_emotion.py   # ③ 语音情绪（M2）—— 待 M2 创建
├── fusion.py           # ④ 多模态融合（M2 主导 / M4 协助）—— 待创建
├── llm_reason.py       # ⑤ LLM 需求推理（M3）—— 待 M3 创建
├── music_gen.py        # ⑥ 音乐生成（M3）—— 待 M3 创建
├── 代码说明.md         # 集成层全部 .py 代码的开发者说明
└── README.md
```

> 目前仓库里已有 `app.py` / `config.py` / `mocks.py` / `requirements.txt`；6 个模块文件由各成员陆续创建，缺失时自动走 mock。

---

## 6. 时间线

| 阶段 | 时间 | 任务 |
|------|------|------|
| 一 · 框架搭建 | 6/22–6/28 | M4 搭 `app.py` 框架定义接口（mock 可用）；M1/M2/M3 各自跑通 lab 代码 |
| 二 · 模块实现 | 6/29–7/10 | 各成员完成各自 pipeline，融合模块接通 |
| 三 · 系统集成 | 7/11–7/17 | 各模块接入 `app.py`，端到端 demo 可运行 |
| 四 · 实验 + User Study | 7/18–7/23 | 消融实验（~50 段视频）+ user study（共 12 人） |
| 五 · Presentation | 7/24–7/27 | slides、排练、现场实时 demo |
| 六 · Report | 7/28–8/2 | 各自写 section → 7/31 互审 → 8/2 定稿（截止 8/3） |

> **总体原则**：系统集成在 7/17 前完成，留足一周做实验和 user study，7/24 前所有数据收集完毕，不把风险留到最后一周。

---

## 7. 团队分工

| 成员 | 核心模块 | Report 分工 |
|------|----------|-------------|
| M1 | 人脸情绪 + GradCAM | Introduction、Related Work |
| M2 | 语音情绪 + 融合模块 | Methodology §感知 + 融合 |
| M3 | LLM 推理 + 音乐生成 | Methodology §LLM + 生成 |
| M4 | 疲劳检测 + UI 集成 | Experiments、User Study、Conclusion（统稿） |

---

*AIAA 3800 小组项目 — HKUST(GZ) — June 2026*
