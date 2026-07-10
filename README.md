# 🎵 EmotiCompanion

**多模态情绪感知的实时音乐陪伴系统**
AIAA 3800 课程项目 · HKUST(GZ) · 2026

> 我们让系统先**感知**你的状态（人脸 + 语音 + 疲劳），再**推理**你需要什么，最后**生成**一段专属于这一刻的音乐。

---

## 1. 项目简介

人和音乐的交互还停留在静态模式——你选一次曲风，它就固定了，系统不知道你现在是什么状态。EmotiCompanion 做三件事：

1. **感知 Perception**：多模态自动感知用户情绪，无需手动告诉系统你的心情
   - 人脸表情情绪（HSEmotion / AffectNet）
   - 语音情绪（Qwen-Omni，经 API）
   - 疲劳状态（面部关键点 EAR/MAR/PERCLOS）
2. **推理 Reasoning**：知道情绪还不够，要推断需求。用 LLM + Theory-of-Mind 提示 + CoT 推理，从「情绪+疲劳」推断「心理需求」（例如：焦虑且疲惫 → 低唤醒舒缓；焦虑但清醒 → 稳定接地）
3. **生成 Generation**：根据推理出的音乐描述词，用 MusicGen 生成专属音频片段

---

## 2. 系统架构

```
  人脸图像 ──┬──► ① 人脸情绪 (HSEmotion)      ──┐
            │                                  │
            └──► ② 疲劳检测 (EAR/MAR/PERCLOS) ──┤
                                               ├──► ④ 多模态融合 ──► 统一状态 JSON
  语音音频 ────► ③ 语音情绪 (Qwen-Omni)      ──┘            │
                                                            ▼
                                              ⑤ LLM 需求推理 (ToM + CoT)
                                                            │
                                                            ▼
                                              ⑥ 音乐生成 (MusicGen) ──► 🎵
  ⑦ GradCAM 热力图旁挂在①上，⑧ Gradio app.py 串联全部
```

| # | 模块 | 文件 | 技术 | 负责人 |
|---|------|------|------|--------|
| ① | 人脸情绪 | `face_emotion.py` | HSEmotion (EfficientNet-B0, AffectNet-8) | M1 |
| ⑦ | GradCAM 可解释 | `face_emotion.py` | 对①同一模型做 Grad-CAM 热力图 | M1 |
| ② | 疲劳检测 | `fatigue.py` | MediaPipe FaceLandmarker → EAR/MAR/PERCLOS | M4 |
| ③ | 语音情绪 | `speech_emotion.py` | Qwen-Omni（智增增网关，OpenAI 兼容 API） | M2 |
| — | 音频抽取 | `audio_extract.py` | ffmpeg 从视频抽 wav（实验用） | M2 |
| ④ | 多模态融合 | `fusion.py` | naive / weighted / weighted_cam / bayes 四模式 | M2/M4 |
| ⑤ | LLM 需求推理 | `llm_reason.py` | ToM + 两段式 CoT，规则版 / DeepSeek-OpenAI 后端 | M3 |
| ⑥ | 音乐生成 | `music_gen.py` | MusicGen (Hugging Face Transformers) | M3 |
| ⑧ | UI + 集成 | `app.py` | Gradio 双模式，动态加载各模块 | M4 |

---

## 3. 快速开始

**环境**：conda 环境 `3800`（**Python 3.10**）。torch 按本机 CUDA 从 <https://pytorch.org> 装（本机有 GPU）。

```bash
# 完整系统（四人真实模块）
pip install -r requirements.txt

# 只想看 UI 空壳（mock 数据端到端跑通）：只需框架最小依赖
#   pip install gradio numpy      # 缺依赖的模块 app.py 会自动回退 mock

# 启动
python app.py
```

**两个额外前置**：

- **ffmpeg**（系统级程序，非 pip 包）：语音③抽音频、消融实验读视频音轨要用。
  `conda install -n 3800 -c conda-forge ffmpeg`
- **API key**（语音③ Qwen + LLM⑤ DeepSeek）：`cp .env.example .env` 后填 key（`.env` 已 gitignore）。见 [§9 常见问题](#9-常见问题)。

启动后控制台打印每个模块的加载状态：

```
============================================================
EmotiCompanion module loading status:
============================================================
[OK ] ① face + ⑦ GradCAM (M1): loaded real module face_emotion.py
[mock] ③ speech (M2): using mock (speech_emotion.py not ready: ...)
...
```

- `[OK ]` = 找到真实文件且依赖齐全，用真实实现
- `[mock]` = 文件缺失或依赖没装，用假数据顶上（不影响启动）

### 两种工作模式

页面顶部 **Mode** 单选框切换：

- **自动模式 Auto（默认）**：摄像头/麦克风持续运行，每 `AUTO_INTERVAL_SEC` 秒（`config.py`，当前 5）自动捕捉一帧 + 一段录音跑一次 pipeline，音乐自动播放、陪伴一直存在直到关网页。**情绪与疲劳都不变时不更换音乐**（当前曲子继续播不打断）。
- **手动模式 Manual**：传统拍照/上传 + 录音/上传窗口，点 **▶ Run Pipeline once** 跑单次。

---

## 4. 目录结构

```
AIAA3800_Proj/
├── README.md              # 本文档（项目介绍 + 开发者指南 + FAQ）
├── app.py                 # ⑧ 集成层：动态加载模块 + 串 pipeline + 双模式 UI
├── config.py              # 全局常量与 schema（情绪词表、疲劳等级、捕捉间隔…）
├── mocks.py               # 6 模块的 mock 占位（真实文件未就绪时自动启用）
├── face_emotion.py        # ① 人脸情绪 + ⑦ GradCAM (M1)
├── fatigue.py             # ② 疲劳检测 (M4)
├── speech_emotion.py      # ③ 语音情绪 (M2)
├── audio_extract.py       # 视频→wav 音频抽取工具 (M2)
├── fusion.py              # ④ 多模态融合 naive/weighted/weighted_cam/bayes (M2/M4)
├── llm_reason.py          # ⑤ LLM 需求推理 (M3)
├── music_gen.py           # ⑥ 音乐生成 (M3)
├── requirements.txt       # 完整依赖清单（含最小 mock 说明）
├── .env.example           # 环境变量模板（复制成 .env 填 key）
├── requirements/          # 各成员依赖拆分 requirements_m1..m4.txt
├── scripts/
│   ├── prepare_ravdess.py  # RAVDESS 数据准备 → labels.csv（语音 in-domain，对照）
│   ├── prepare_cremad.py   # CREMA-D 数据准备 → labels.csv（公平跨域，主实验）
│   ├── prepare_enterface.py# eNTERFACE'05 数据准备 → labels.csv（正脸+语义）
│   ├── run_ablation.py     # 感知层五臂消融评测（A-E）
│   ├── fusion_bayes.py     # 臂 F 逐类可靠性贝叶斯融合（读缓存 LOOCV / 导出先验）
│   ├── gradcam_analysis.py # ⑦ GradCAM 可解释性实验 + 热力图（路线A）
│   └── viz_fatigue.py      # ② 疲劳关键点可视化（pre 素材）
├── docs/
│   ├── experiment_plan.md # 感知层消融实验记录（报告 Experiments 章草稿）
│   └── user_study.md      # 推理⑤/生成⑥/疲劳② 的 user study 方案（≤10 人）
└── models/
    ├── face_landmarker.task     # MediaPipe 疲劳检测模型（3.7MB）
    └── fusion_bayes_priors.json # 臂 F 逐类可靠性先验（离线学好、fusion.py bayes 模式加载，2.5KB）
```

---

## 5. 接口契约

**`app.py` 是「胶水层」，本身不实现任何模型。** 它动态导入各成员文件、串 pipeline、提供双模式 UI。每位成员**在自己的文件里**实现约定函数即可，`app.py` 自动接上——**未就绪的模块自动回退 mock，无需改 `app.py`。**

| 文件 | 负责人 | 需实现的函数 | 返回 |
|------|--------|--------------|------|
| `face_emotion.py` | M1 | `predict(image)` | `{"emotion": str, "confidence": float}` |
| `face_emotion.py` | M1 | `gradcam(image)` | 热力图 image (np.ndarray) |
| `fatigue.py` | M4 | `predict(image)` | `{"fatigue_level": str, "confidence": float}` |
| `speech_emotion.py` | M2 | `predict(audio_path)` | `{"emotion": str, "confidence": float, "reasoning": str}` |
| `fusion.py` | M2/M4 | `fuse(face, speech, fatigue)` | 统一状态 JSON（见下） |
| `llm_reason.py` | M3 | `infer(state)` | `{"need": str, "reasoning": str, "music_spec": str}` |
| `music_gen.py` | M3 | `generate(music_spec)` | 音频（路径 str 或 `(sr, np.ndarray)` 或 None） |

### 统一状态 JSON（融合层 ④ 输出）

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

### 情绪词表（全队契约）

`dominant_emotion` 与各模态的 `emotion` 统一取 **Ekman 标准 7 类**：

```
neutral / happy / sad / angry / fear / surprise / disgust
```

见 `config.EMOTION_LABELS`。**各感知模块把自己模型的原生标签映射进这套词表**，融合层 ④ 的按情绪投票才能跨模态互证：

- **① 人脸（AffectNet-8）** → `face_emotion.py` 的 `AFFECTNET_TO_CONFIG`：`Contempt→disgust`，`Fear→fear`，`Surprise→surprise`，其余 1:1。
- **③ 语音（Qwen-Omni）** → `speech_emotion.py` 的 `RAW_TO_CONFIG`：`joy→happy`，`anxious/nervous/worried→fear`，`contempt→disgust` 等近义合并。
- **困倦不在情绪轴**：由 ② 疲劳走独立的 `fatigue`（low/medium/high）字段，任何模块都不输出 `tired`。

---

## 6. 模块详解（开发者指南）

### 6.0 框架层 `app.py` / `config.py` / `mocks.py`

- **动态加载**：`_load()` 尝试 `import` 成员文件，成功用真实模块，失败（文件不存在 / 依赖没装 / 报错）回退 `mocks.py` 对应类。启动日志的 `[OK ]`/`[mock]` 就是这一步的结果。
- **常量集中在 `config.py`**：`EMOTION_LABELS`（7 类词表）、`FATIGUE_LEVELS`、`FUSION_SCHEMA_EXAMPLE`、`AUTO_INTERVAL_SEC`（=5）、`SOOTHE_EMOTIONS`、`MOCK_EMOTION/MOCK_FATIGUE`。改这些改一处，别在别的文件另写一份。
- **不换音乐的判断**：`_state_key(state)` 取 `(dominant_emotion, fatigue)`；`auto_step` 比较本轮与上轮，相同 → 音乐输出 `gr.skip()` 继续播，不同 → 重新推理+生成。
- **自动模式两个已修复的坑（勿改回）**：① 摄像头流式直接驱动 pipeline（`auto_cam.stream(fn=auto_step, stream_every=AUTO_INTERVAL_SEC)`），不要用 `gr.Timer`+State 中转（会读到 None 导致输出不刷新）；② 流式麦克风必须 `type="numpy"`（`filepath` 会每秒 422 报错）。

### 6.1 ① 人脸情绪 + ⑦ GradCAM `face_emotion.py`（M1）

- **模型**：HSEmotion `enet_b0_8_best_vgaf`（EfficientNet-B0，AffectNet-8）。相比 DeepFace(FER2013) 精度更高，且是 PyTorch，让 ⑦ Grad-CAM 能解释①用的**同一个模型**。
- **`predict`**：Haar 检测人脸 → 裁剪 → HSEmotion 分类 → 映射进 7 类词表；额外回传 `raw_emotion`/`face_detected` 便于调试。
- **`gradcam`**：对 `model.bn2`（最后一层卷积）挂 hook 做 Grad-CAM → 热力图叠加回人脸区域。
- **兼容坑（已固化，见 `requirements/requirements_m1.txt`）**：hsemotion 权重是 timm 0.9.x 的整模型 pickle → **timm 必须 0.9.16**；torch≥2.6 加载时需临时 `weights_only=False`（代码已处理）。

### 6.2 ② 疲劳检测 `fatigue.py`（M4）

- **原理（L9P1 + 眼动）**：**EAR**（眼纵横比，闭眼下降）、**MAR**（嘴纵横比，打哈欠升高）、**PERCLOS**（闭眼帧占比，业界金标准）。模块内滚动窗口 `deque(maxlen=12)` 聚合近 ~1 分钟，避免单帧误报；实验可传 `use_history=False` 只看单帧。
- **关键点后端**：首选 MediaPipe **FaceLandmarker（Tasks API）**，需模型 `models/face_landmarker.task`；未装 mediapipe 或模型缺失时自动退回 OpenCV Haar 睁/闭眼粗判。
- **定位（重要 · 报告口径）**：疲劳是与情绪**正交**的唤醒/精力维度，**不进情绪投票、不做定量消融**，只作为独立信号透传给 ⑤。采用预训练/启发式，不声称在疲劳检测本身有贡献；价值由 user study 定性评估。详见 [docs/experiment_plan.md](docs/experiment_plan.md) §5.1。

### 6.3 ③ 语音情绪 `speech_emotion.py` + 音频抽取 `audio_extract.py`（M2）

> **与原计划的差异（写报告注意）**：语音③ 由原计划的 **EmotionThinker**（本地跑的 Qwen2.5-Omni 情绪微调版）改为**通用 Qwen-Omni（`qwen3-omni-flash`，经 API）+ prompt 工程**，理由是笔记本友好、无需本地下大模型/占 GPU。Related Work 与 Methodology 按此实际实现描述，勿照抄旧计划的 EmotionThinker。

- **`predict(audio_path)`**：调智增增网关的 `qwen3-omni-flash`，prompt 要求**同时结合语音内容与声学线索**（pitch/energy/rhythm/pause/tone…），返回 `{emotion, confidence, reasoning}`，emotion 映射进 7 类词表。空音频/文件不存在/API 失败/解析失败 → 安全回退 `neutral, 0.0`。
- **置信度处理**：Omni 模型易对 `neutral` 过度自信，`_calibrate_confidence` 在 reasoning 出现模板化「平淡」描述时压低其置信度，避免融合阶段错误的高置信 neutral 压过其他模态。这是「模型自评置信度 + 规则校准」方案。
- **`audio_extract.extract_audio(video_path)`**：用 ffmpeg 抽单声道 16kHz wav，供 RAVDESS/MELD 这类视频数据接入语音模块（消融用）。

### 6.4 ④ 多模态融合 `fusion.py`（M2/M4）

- **`fuse(face, speech, fatigue, mode="weighted")`**，四模式对应消融四臂：

  | mode | 实验臂 | 说明 |
  |------|--------|------|
  | `naive` | C 朴素融合 | 各投一票，不用 confidence 定主导；冲突用固定规则打破平票 |
  | `weighted` | D 置信度加权 | `score = 模态权重 × 置信度`，默认 face 0.45 / speech 0.55 |
  | `weighted_cam` | E 加权+CAM门控 | 人脸权重 × GradCAM 可靠性（注意力跑到脸外→压低人脸） |
  | `bayes` | F 贝叶斯逐类 | 按**离线学到的逐类可靠性**做朴素贝叶斯融合（`models/fusion_bayes_priors.json`）；唯一能越过最强单模态的模式（§3.7） |

- **默认 `weighted`**（`app.py` 走这个）。`bayes` 是可选增强：需要 `models/fusion_bayes_priors.json`（用 `scripts/fusion_bayes.py --export` 从验证集离线生成），缺文件时自动回退 `weighted`、不阻塞；输出情绪类别受限于先验的训练类，收益小且依赖领域匹配，故不设默认。`fatigue` **不参与情绪投票**，只透传。额外回传 `fusion_mode`/`face_emotion`/`speech_emotion`/`modal_agreement` 便于实验记录。

### 6.5 ⑤ LLM 需求推理 `llm_reason.py`（M3）

- **`infer(state)`**：两段式——Stage 1 用 ToM 从「情绪+疲劳」推断可能的内在状态与需求；Stage 2 规划音乐策略并产出 MusicGen 描述词。返回 `{need, reasoning, music_spec}`。
- **双模式（消融/对比用）**：`infer_with_mode(state, mode="tom_cot"|"standard")`，`standard` 是「情绪→曲风」直接查表的基线。
- **后端**：默认 `rule`（本地规则版 ToM+CoT，无网络）；设 `EMOTI_LLM_BACKEND=deepseek`（或 openai）走真实 LLM，失败自动回退 rule。内置安全护栏，禁止临床诊断类措辞。

### 6.6 ⑥ 音乐生成 `music_gen.py`（M3）

- **`generate(music_spec)`**：优先用 Hugging Face Transformers 的 **MusicGen**（`facebook/musicgen-small`）从文本生成 wav（写到 `outputs/`）；不可用时回退到简单合成音（仅供 UI/流程测试）。首次运行会下载 musicgen-small 权重。

---

## 7. 消融实验

**只做「感知层」定量消融**，证明我们自己的贡献——**融合④**：把预训练的人脸与语音融合，是否优于单模态。推理⑤/音乐⑥ 与疲劳② 归 user study / 定性评估。

实验臂：A 仅人脸 · B 仅语音 · C 朴素融合 · **D 置信度加权融合** · E 加权+GradCAM门控 · **F 贝叶斯逐类可靠性融合**（新，LOOCV 非泄漏）；指标 accuracy / macro-F1 / 逐类混淆矩阵。三假设：H1 融合>单模态、H2 加权>朴素、H3 模态互补。

### 三条主要结论（7 轮实验，详见 [docs/experiment_plan.md](docs/experiment_plan.md)）

- **H2 稳固成立**：置信度加权 D 在各种模态失衡下都 > 朴素 C（对失衡**鲁棒**、优雅降级）——这是最扎实的结论。
- **H1 有条件成立**：置信度加权从未越过最强单模态；直到用 **F 贝叶斯逐类融合**（按各模态**逐类历史可靠性**做朴素贝叶斯晚融合），才在**两模态势均力敌且互补**的 CREMA-D+emotion2vec 上首次越过（0.683 > 人脸 0.667）。即 **H1 需要「均衡+互补」+「逐类可靠性感知」两个条件**，margin 单段、作趋势呈现。
- **语音后端选型**：**emotion2vec+ 跨数据集一致最佳**；Qwen-Omni API 读不好语调、wav2vec2(RAVDESS 微调)跨域即崩。

### 语音后端与数据集

- **语音后端**：`SPEECH_BACKEND=emotion2vec`（首选）/ `ser`(wav2vec2) / `api`(Qwen-Omni)。
- **四个数据集**覆盖不同失衡与场景：**RAVDESS**(对照，语音 in-domain 碾压)、**CREMA-D**(公平跨域，主实验，两模态都不在其上训练)、**MELD**(真实对话有语义、但多人/侧脸致人脸失效)、**eNTERFACE'05**(正面单人脸+情绪语义句，验证 MELD 人脸问题是几何性的)。

```bash
# 主实验 · CREMA-D（公平跨域）——先下载视频版数据
python scripts/prepare_cremad.py --src <CREMA-D视频目录> --n 60 --out data/cremad
SPEECH_BACKEND=emotion2vec python scripts/run_ablation.py --data data/cremad --out results/ablation_cremad_e2v --refresh
#   --limit 6 快速冒烟；--fusion-only 只用缓存重算融合臂

# 逐类可靠性贝叶斯融合（arm F，读缓存、不重跑感知）—— H1 在此首次成立
python scripts/fusion_bayes.py --data data/cremad --cache results/ablation_cremad_e2v

# 可解释性 + pre 可视化（⑦ 与 ② 的展示素材，同样支持 --data）
python scripts/gradcam_analysis.py --data data/cremad
python scripts/viz_fatigue.py --data data/cremad
```

其他数据集准备：`prepare_ravdess.py`（自动下载）、`prepare_enterface.py --src <eNTERFACE目录>`。完整方案与踩坑见 **[docs/experiment_plan.md](docs/experiment_plan.md)**。

---

## 8. 团队分工

| 成员 | 核心模块 | Report 分工 |
|------|----------|-------------|
| M1 | 人脸情绪 + GradCAM | Introduction、Related Work |
| M2 | 语音情绪 + 融合 + 音频抽取 | Methodology §感知 + 融合 |
| M3 | LLM 推理 + 音乐生成 | Methodology §LLM + 生成 |
| M4 | 疲劳检测 + UI 集成 + 消融评测 | Experiments、User Study、Conclusion（统稿） |

> **消融实验（寻找能证明 H1 的实验设置）是 M1/M2/M4 的联合探索**（核心模块分工不变）：
> - **M1**：规划并运行了整套感知层消融、找到 **CREMA-D 数据集**与 **SER 语音模型**、并在 M4 的基础上发现了**贝叶斯逐类融合（arm F）**让 H1 成立的可能性。
> - **M4**：找到 **emotion2vec+** 语音模型（跨数据集最佳后端）。
> - **M2**：找到 **MELD 数据集**（真实对话/语义场景，暴露人脸在非正脸下失效）。

---

## 9. 常见问题

**Q：启动报 `... startup-events failed (code 502) ... proxy settings ...`？**
开了科学上网/系统代理，导致 gradio 自检本地请求被代理拦截。`app.py` 已内置把 `127.0.0.1,localhost` 加进 `no_proxy` 的修复，拉最新代码直接跑即可。仍遇到可运行前 `export no_proxy=127.0.0.1,localhost` 或临时关系统代理。

**Q：两个 API（qwen / deepseek）怎么配？**
项目根目录 `cp .env.example .env`，填两个 key：
```env
# 语音③ · Qwen（智增增网关）
ZHIZENGZENG_API_KEY=your_qwen_key
ZHIZENGZENG_BASE_URL=https://api.zhizengzeng.com/v1
AUDIO_MODEL=qwen3-omni-flash
# LLM⑤ · DeepSeek
EMOTI_LLM_BACKEND=deepseek
DEEPSEEK_API_KEY=your_deepseek_key
```
两个模块启动都会 `load_dotenv()` 读它。不填 key：语音回退 neutral、LLM 回退本地规则版，仍能跑。

**Q：版本踩坑？**
`timm` 必须 `0.9.16`（hsemotion 权重兼容）；`transformers 5.x` 已验证 MusicGen 可用；新版 `mediapipe` 已移除旧 `mp.solutions`，`fatigue.py` 用 Tasks API + `models/face_landmarker.task`。都写进了 `requirements.txt` / `requirements/`。

**Q：`audio_extract` 报找不到 ffmpeg？**
ffmpeg 是系统程序不是 pip 包，`conda install -n 3800 -c conda-forge ffmpeg`。

**Q：音乐栏没声音 / 自动模式音乐不换？**
真实 `music_gen` 不可用时回退合成音（或 mock 返回 None 为空）；自动模式「情绪+疲劳」和上轮相同就不换曲（预期行为）。

**Q：怎么确认用上了我的真实实现？**
看启动控制台那行是 `[OK ]` 还是 `[mock]`。缺依赖会掉 mock。

---

## 10. 局限性（写进报告 Limitations）

- **表演数据**：RAVDESS 是专业演员的表演情绪，与真实自发情绪有分布差异。
- **样本量小**：每个数据集 50–60 段 / 6–7 类 ≈ 每类 ~10 段，统计力有限（H1 的 arm F 越过仅 +1 段），结论以趋势呈现。
- **预训练编码器**：单模态准确率取决于预训练模型本身，不代表我们的工作；贡献在**融合策略**与**系统集成**。
- **语音置信度**：`api` 后端为模型自评+规则校准（非严格概率）；`ser`/`emotion2vec` 为 softmax 概率。
- **bayes 融合泛化性**：臂 F 的逐类先验在 CREMA-D 上学，迁移到真实部署分布的收益未验证，故默认仍用 weighted。
- **语音在线 API**：速度/稳定性受网络影响，自动模式每轮同步调用会有延迟（见 experiment_plan 与后续优化）。
- **疲劳单帧**：用跨调用滚动窗口近似 PERCLOS，换用户应调 `fatigue.reset_history()`。

---

*AIAA 3800 小组项目 — EmotiCompanion — HKUST(GZ) — 2026*
