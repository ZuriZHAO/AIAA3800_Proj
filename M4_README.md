# M4 模块说明：Fatigue Detection + Integration + Ablation Evaluation

本文档说明 M4 本次提交的模块与脚本，面向小组协作与实验复现。M4 负责系统的**集成层**与**疲劳分支**，并主导**感知层消融实验**的评测。所有代码遵循 `app.py` 的接口契约：只要文件名与函数签名一致，`app.py` 会自动加载真实模块，缺失时回退到 mock（见 [代码说明.md](代码说明.md)）。

## 1. 本次 / 已提交文件

| 文件 | 作用 | 主要接口 | 现状 |
|---|---|---|---|
| `fatigue.py` | ② 疲劳/困倦状态检测，面部关键点 → EAR/MAR/PERCLOS → `low/medium/high` | `predict(image)` | ✅ 本次新增 |
| `scripts/run_ablation.py` | 感知层四臂消融评测（A/B/C/D），出 accuracy/macro-F1/混淆矩阵 | `python scripts/run_ablation.py` | ✅ 本次新增 |
| `requirements_m4.txt` | M4 模块额外依赖（opencv-python，可选 mediapipe） | — | ✅ 本次新增 |
| `app.py` | ⑧ 集成层：加载模块 + 串 pipeline + 双模式 Gradio UI | — | ✅ 早前提交 |
| `config.py` / `mocks.py` | 全局常量/schema + 6 模块 mock 占位 | — | ✅ 早前提交 |
| `scripts/prepare_ravdess.py` | 消融数据准备（下载 RAVDESS、去 calm、均衡挑 N 段 → `labels.csv`）| — | ✅ 早前提交（M1/M4）|

## 2. 环境配置

```bash
pip install -r requirements_m4.txt
```

- `opencv-python`：**必需**——疲劳检测的 Haar 兜底、消融脚本读视频帧都要用。
- `mediapipe`：**可选增强**——装上后疲劳检测走 468 点精确关键点（EAR/MAR）；不装则自动退化为 Haar 睁/闭眼粗判，功能不缺失、精度略降。
- 消融脚本还会调用 M1 的 `face_emotion.py`、M2 的 `speech_emotion.py` / `audio_extract.py`（需 `ffmpeg` 与 `.env` 里的 API key）与 `fusion.py`，跑实验前请确保这些真实模块可用。

## 3. 疲劳检测模块 `fatigue.py`

### 3.1 接口

```python
def predict(image, use_history=True):
    ...
```

输入是 RGB 图（`np.ndarray`，来自 `app.py` 的摄像头帧，与人脸模块共用同一帧），输出：

```json
{
  "fatigue_level": "medium",
  "confidence": 0.83,
  "ear": 0.21, "mar": 0.12, "perclos": 0.25,
  "yawn_frac": 0.0, "face_detected": true, "backend": "mediapipe"
}
```

`fatigue_level ∈ config.FATIGUE_LEVELS = ["low","medium","high"]`。下游融合层 ④ 只读 `fatigue_level` / `confidence`，其余字段用于调试与报告出图。图为 `None`、检测不到人脸或任何异常时，安全回退到 `{"fatigue_level": "low", "confidence": 0.0}`（与 mock 一致，绝不冲掉 `app.py` 的自动流）。

### 3.2 原理（对应课程 L9P1 + Lecture 2 眼动）

疲劳是与「情绪类别」正交的一维（valence–arousal 里的**唤醒/精力**维度）。方法走经典生理指标链路：

- **EAR（Eye Aspect Ratio，眼纵横比）**：眼睛闭合时明显下降，是困倦最强信号；
- **MAR（Mouth Aspect Ratio，嘴纵横比）**：打哈欠时升高；
- **PERCLOS（闭眼帧占比）**：疲劳检测业界金标准。`app.py` 每 `AUTO_INTERVAL_SEC` 秒调一次 `predict`，模块内用一个滚动窗口（`deque(maxlen=12)`，约覆盖近 1 分钟）聚合 PERCLOS 与哈欠频率来判级，避免"某帧恰好眨眼"误报 high。实验里可传 `use_history=False` 只看单帧。

判级规则（见 `_classify`）：`perclos≥0.5 或 yawn_frac≥0.3 → high`；`perclos≥0.2 或 yawn_frac≥0.1 或 avg_ear<0.22 → medium`；否则 `low`。

### 3.3 与融合/实验的关系（重要 · 报告口径）

按 [experiment_plan.md](experiment_plan.md) §5.1，疲劳分支**不进情绪投票、也不做定量消融**：`fusion.fuse(face, speech, fatigue)` 只把 `fatigue` 作为独立信号透传进统一状态 JSON，交给 ⑤ LLM 做需求推断（同样"中性"，深夜疲惫 → 低唤醒舒缓乐，清醒 → 维持专注乐）。我们**采用轻量启发式/预训练关键点，不声称在疲劳检测本身上有贡献**；其价值由 **user study 定性评估**。

### 3.4 单独测试

```bash
python fatigue.py
```

会打印：`predict(None)` 与噪声图的安全回退、以及 `_classify` 在全闭眼/全睁眼/全哈欠三种窗口下的判级（应分别为 high / low / high）。无摄像头、未装 mediapipe 也能跑（走 Haar / 纯逻辑冒烟）。

## 4. 消融评测脚本 `scripts/run_ablation.py`

对应 experiment_plan §2 的四个实验臂，一条命令跑完并出指标：

| 臂 | 配置 | 取自 |
|---|---|---|
| A | 仅人脸 ① | `face_emotion.predict`（抽帧聚合）|
| B | 仅语音 ③ | `speech_emotion.predict` |
| C | 人脸+语音 · 朴素融合 | `fusion.fuse(..., mode="naive")` |
| **D** | 人脸+语音 · 置信度加权融合 | `fusion.fuse(..., mode="weighted")` |

### 4.1 用法

```bash
# 先准备数据（生成 data/ravdess/labels.csv）
python scripts/prepare_ravdess.py --actors 01 02 05 --n 50 --out data/ravdess

# 跑四臂评测
python scripts/run_ablation.py --data data/ravdess --frames 8 --out results/ablation
python scripts/run_ablation.py --limit 6          # 只跑前 6 段，快速冒烟
python scripts/run_ablation.py --fusion-only      # 只用缓存重算 C/D（不重跑感知/不再调 API）
python scripts/run_ablation.py --refresh          # 忽略缓存强制重算
```

### 4.2 机制与产物

- **每段视频**：人脸用 `cv2` 均匀抽 K 帧 → 逐帧 `predict` → 置信度加权聚合成一个人脸结果；语音用 `audio_extract` 抽 wav → `speech_emotion.predict`。
- **缓存**：每段的人脸/语音原始预测边跑边存到 `<out>/cache/predictions.json`，重跑默认命中缓存（**语音 API 昂贵**，避免重复计费）；改融合策略时用 `--fusion-only` 秒出新 C/D。
- **指标**：`accuracy`、`macro-F1`（只计有真值的类）、逐类 P/R/F1、混淆矩阵，**手写实现不引入 sklearn**。
- **产物**：`<out>/metrics.json` + `<out>/confusion_[A-D].csv`，并在终端打印 A/B 两臂混淆矩阵作为 **H3 互补性**证据。
- **结论对照**：H1 看 D/C vs A/B，H2 看 **D vs C**，H3 看 A/B 的逐类混淆矩阵互补。

## 5. 与 `app.py` 的联调

M4 模块不需要改 `app.py`。把 `fatigue.py` 放在项目根目录后，启动应看到：

```text
[OK ] ② fatigue (M4): loaded real module fatigue.py
```

Manual 模式下运行一次，`Perception output` 的 `fatigue` 字段应带 `fatigue_level / confidence / ear / mar / perclos`；`Multimodal fusion` 的统一状态里 `fatigue` 字段随之更新。未装 mediapipe 时 `backend` 显示 `haar`，其余不受影响。

## 6. 当前限制（写进报告 Limitations）

- 单帧关键点无法直接给出真实 PERCLOS，本模块用**跨调用滚动窗口近似**；换用户/新会话时应调 `fatigue.reset_history()` 清空窗口。
- 未装 mediapipe 时退化为 Haar「数眼睛」粗判，只能区分睁/闭、无哈欠信号，精度有限。
- RAVDESS 是**表演**数据且无疲劳标签，故疲劳分支只做理论定位 + user study 定性，不进 §4 的定量消融。

---

*AIAA 3800 小组项目 — EmotiCompanion — HKUST(GZ) — 2026*
