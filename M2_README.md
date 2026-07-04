# M2 模块说明：Speech Emotion + Multimodal Fusion

本文档说明 M2 本次提交的模块文件，面向小组成员协作与后续实验复现，主要覆盖语音情绪识别、音频提取与多模态融合三部分。本模块遵循现有 `app.py` 的接口约定，不需要修改集成层代码；只要文件名与函数签名保持一致，`app.py` 会自动加载真实模块，失败时回退到 mock。

## 1. 本次提交文件

| 文件 | 作用 | 主要接口 |
|---|---|---|
| `speech_emotion.py` | M2 语音情绪识别模块，调用智增增平台的 `qwen3-omni-flash`，根据语音内容、语调和声学线索输出情绪、置信度与解释 | `predict(audio_path)` |
| `fusion.py` | M2/M4 融合模块，支持朴素融合与置信度加权融合两种模式，对应消融实验中的 C/D 两个实验臂 | `fuse(face, speech, fatigue, mode="weighted")` |
| `audio_extract.py` | 从视频文件中提取 wav 音频，方便后续 RAVDESS 这类音视频数据接入语音模块 | `extract_audio(video_path)` |
| `requirements_m2.txt` | M2 模块额外依赖 | — |
| `.env.example` | 环境变量模板，组员复制为 `.env` 后填写自己的 API key | — |

## 2. 环境配置

在项目根目录安装 M2 依赖：

```bash
pip install -r requirements_m2.txt
```

如果需要从视频中抽取音频，建议本地安装 `ffmpeg`，Windows/Conda 环境可以使用：

```bash
conda install ffmpeg
```

复制 `.env.example` 为 `.env`，并填写自己的 API key：

```env
ZHIZENGZENG_API_KEY=your_api_key_here
ZHIZENGZENG_BASE_URL=https://api.zhizengzeng.com/v1
AUDIO_MODEL=qwen3-omni-flash
SPEECH_MAX_AUDIO_BYTES=8388608
```

注意：`.env` 不应提交到 GitHub，仓库只保留 `.env.example`。

## 3. 语音情绪模块 `speech_emotion.py`

### 3.1 接口

```python
def predict(audio_path):
    ...
```

输入是音频文件路径，通常来自 Gradio 麦克风录音，或由 `audio_extract.py` 从视频中提取。输出格式如下：

```json
{
  "emotion": "happy",
  "confidence": 0.92,
  "reasoning": "Vocal cues and spoken content both indicate a positive emotional state."
}
```

`emotion` 必须属于全队统一情绪词表：

```text
neutral / happy / sad / angry / fear / surprise / disgust
```

模块内部会把模型可能输出的近义标签映射到统一词表，例如 `joy -> happy`，`anxious / nervous / worried -> fear`，`contempt -> disgust`。如果音频为空、文件不存在、API 失败或结果无法解析，模块会安全回退到：

```json
{
  "emotion": "neutral",
  "confidence": 0.0,
  "reasoning": "fallback: ..."
}
```

### 3.2 Prompt 设计

当前 prompt 不只依赖纯声学特征，也不只依赖文本语义，而是要求模型同时结合 spoken content 与 vocal cues，包括语音内容、语义意图、语调、pitch、energy、rhythm、pauses、speaking speed、voice stability 等线索。这样更符合 EmotiCompanion 的真实使用场景：用户的情绪既可能体现在“说了什么”，也可能体现在“怎么说”。

### 3.3 置信度处理

由于部分 Omni 模型容易对 `neutral` 给出过高置信度，`speech_emotion.py` 中加入了简单的 confidence calibration：当模型把普通、平淡或不明确的语音判为 `neutral` 且 reasoning 中出现模板化描述时，会限制其置信度，避免在融合阶段让错误的高置信度 neutral 压制其他模态。这个策略对应 experiment plan 中提到的语音置信度阻塞项，目前采用的是“模型自评置信度 + 规则校准”的方案。

### 3.4 单独测试

如果只测试 fallback：

```bash
python speech_emotion.py
```

如果测试真实音频：

```powershell
$env:TEST_AUDIO_PATH="test_happy.m4a"
python speech_emotion.py
```

正常情况下会输出一个包含 `emotion / confidence / reasoning` 的 JSON。

## 4. 音频提取模块 `audio_extract.py`

### 4.1 接口

```python
def extract_audio(video_path):
    ...
```

输入是视频路径，输出是临时 wav 文件路径，默认使用 `ffmpeg` 抽取音频、转成单声道、16kHz wav，适合后续传入 `speech_emotion.predict(audio_path)`。这个模块主要用于 RAVDESS 或 MELD 这类视频数据集，因为它们的语音通常嵌在 mp4 文件中。

示例：

```python
from audio_extract import extract_audio
from speech_emotion import predict

video_path = "path/to/video.mp4"
audio_path = extract_audio(video_path)
result = predict(audio_path)
print(result)
```

## 5. 融合模块 `fusion.py`

### 5.1 接口

```python
def fuse(face, speech, fatigue, mode="weighted"):
    ...
```

输入示例：

```python
face = {"emotion": "neutral", "confidence": 0.50}
speech = {"emotion": "happy", "confidence": 0.92, "reasoning": "..."}
fatigue = {"fatigue_level": "low", "confidence": 0.80}
```

输出示例：

```json
{
  "dominant_emotion": "happy",
  "confidence": 0.92,
  "fatigue": "low",
  "face_conf": 0.5,
  "speech_conf": 0.92,
  "fatigue_conf": 0.8,
  "fusion_mode": "weighted",
  "face_emotion": "neutral",
  "speech_emotion": "happy",
  "modal_agreement": false
}
```

其中 `dominant_emotion` 是融合后的主导情绪，`confidence` 是融合置信度，`fatigue` 不参与 emotion 投票，只作为独立状态透传给后续 LLM 需求推理。这一点与 experiment plan 一致：本次定量消融只评估感知层的人脸与语音融合，疲劳分支不参与情绪识别准确率对比。

### 5.2 两种融合模式

`fusion.py` 支持两种模式，对应 experiment plan 中的两个融合实验臂：

| mode | 对应实验臂 | 说明 |
|---|---|---|
| `naive` | C. 人脸+语音 · 朴素融合 | 人脸和语音各投一票，不使用 confidence 决定主导情绪；若两者冲突，使用固定规则打破平票 |
| `weighted` | D. 人脸+语音 · 置信度加权融合 | 使用 `score = modality_weight × modality_confidence` 进行加权投票，默认 face 权重 0.45，speech 权重 0.55 |

默认模式是 `weighted`，因此 `app.py` 正常调用 `FUSION.fuse(face, speech, fatigue)` 时会使用置信度加权融合。实验脚本可以显式指定模式：

```python
from fusion import fuse

state_c = fuse(face_result, speech_result, fatigue_result, mode="naive")
state_d = fuse(face_result, speech_result, fatigue_result, mode="weighted")
```

如果只需要单独测试融合模块：

```bash
python fusion.py
```

## 6. 与 `app.py` 的联调

本模块不需要修改 `app.py`。只要 `speech_emotion.py` 和 `fusion.py` 放在项目根目录，启动时应看到类似输出：

```text
[OK ] ③ speech (M2): loaded real module speech_emotion.py
[OK ] ④ fusion (M2/M4): loaded real module fusion.py
```

Manual 模式下上传图片和音频后，`Perception output` 中应包含 speech 的 `emotion / confidence / reasoning`，`Multimodal fusion` 中应显示融合后的统一状态 JSON。如果人脸模型权重下载失败或其他成员模块暂未可用，`app.py` 会自动回退到 mock，这不影响 M2 模块本身测试。

## 7. 与消融实验的关系

根据 experiment plan，感知层消融实验需要比较四个实验臂：A 仅人脸，B 仅语音，C 朴素融合，D 置信度加权融合。本次 M2 模块对应其中 B/C/D 三部分：`speech_emotion.py` 提供 B 的语音预测，`fusion.py` 的 `mode="naive"` 提供 C，`mode="weighted"` 提供 D。后续评测脚本可以对同一批 RAVDESS 视频先提取人脸结果与语音结果，再分别计算各实验臂的 accuracy、macro-F1 与混淆矩阵。

## 8. 当前限制

当前语音模块依赖在线 API，因此运行速度和稳定性受网络影响；如果 API 不可用，模块会 fallback 到 neutral。语音 confidence 目前来自模型自评并经过规则校准，仍然不等价于严格概率，后续实验中需要观察它与真实正确率是否匹配。融合权重当前采用可解释的静态设定 `face=0.45, speech=0.55`，后续可以在验证集上做轻量网格搜索，但正式报告中需要说明权重选择方式，避免在同一批样本上既调参又汇报最终结果。
