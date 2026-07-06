# EmotiCompanion 消融实验方案

**多模态情绪感知的实时音乐陪伴系统 · 感知层消融**
AIAA 3800 课程项目 · HKUST(GZ) · 2026

> 本文档是消融实验的详细设计，也是报告 Experiments 章的草稿。核心结论先说：
> **这次消融只做「感知层」，用来证明"把预训练的人脸情绪与语音情绪融合起来，比单独用任何一个都好"，即验证我们自己设计的融合模块 ④ 的价值。** 推理/生成层（⑤⑥）与疲劳分支（②）不在定量消融范围内，理由见 §5。

---

## 1. 我们到底要证明什么

系统里人脸 ①（HSEmotion）、语音 ③（Qwen-Omni，经 API）都是**现成的预训练/托管模型**，不是我们的贡献；**真正原创的是融合策略 ④**。因此消融的意义不是"我们的识别器多准"，而是验证融合这个设计选择。三个假设：

| 编号 | 假设 | 怎么证明 |
|------|------|----------|
| **H1** | 多模态融合（人脸+语音）的情绪识别准确率 **高于任一单模态** | 对比单模态臂 vs 融合臂的 accuracy / macro-F1 |
| **H2** | 我们的**置信度加权融合 ④** **优于朴素融合**（等权平均 / 投票） | 对比朴素融合臂 vs 加权融合臂 |
| **H3** | 人脸与语音**模态互补**：各自擅长不同情绪，融合吃到并集 | 对比两个单模态的**逐类混淆矩阵**，指出各自强项 |

---

## 2. 实验设计（实验臂）

同一批视频，跑以下 5 种配置，比较情绪识别表现：

| 实验臂 | 用到的模块 | 角色 | 对应假设 |
|--------|-----------|------|----------|
| A. 仅人脸 | ① | 单模态基线 | H1 / H3 |
| B. 仅语音 | ③ | 单模态基线 | H1 / H3 |
| C. 人脸+语音 · 朴素融合（等权平均或投票） | ①③ | 对照 | H1 |
| **D. 人脸+语音 · 置信度加权融合 ④** | ①③④ | **主角** | H2 |
| **E. 加权 + GradCAM 人脸可靠性门控** | ①③④⑦ | 路线B：⑦ 是否有定量贡献 | E vs D |
| *(可选) O. Oracle 上界*：每样本取"猜对的那个模态" | — | 参照理论上限，看融合还差多少 | — |

> 臂 E＝D＋CAM 门控：注意力跑到脸外→人脸可靠性低→压低人脸权重（见 §5.3 路线B 与 `fusion.py` 的 `weighted_cam` 模式）。看 **E vs D** 是否提升。
> Oracle 臂不是真实方法，只用来给出"完美融合"的天花板，衬托 C/D 距离上限有多近。可放进报告，也可省略。

---

## 3. 数据集

- **RAVDESS**（Ryerson Audio-Visual Database of Emotional Speech and Song）
  - 下载：<https://zenodo.org/records/1188976>（CC BY-NC-SA 4.0，非商业，课程可用，需引用 PLOS ONE 论文）
  - 每段是**音视频**，同时含人脸与人声 → 一段视频可同时喂 ① 和 ③
  - 自带情绪标签作为 **ground truth**
- **规模**：约 **50 段**（README 时间线），按情绪均衡采样。
- **标签对齐**：RAVDESS 8 类映射到全队 7 类契约词表（`config.EMOTION_LABELS`），并**丢弃 calm**（不在 7 类内、且 calm≈neutral 太模糊）：

  | RAVDESS | 01 neutral | 02 calm | 03 happy | 04 sad | 05 angry | 06 fearful | 07 disgust | 08 surprised |
  |---------|-----------|---------|----------|--------|----------|-----------|-----------|-------------|
  | 契约词表 | neutral | *(丢弃)* | happy | sad | angry | fear | disgust | surprise |

- **数据准备脚本**：[scripts/prepare_ravdess.py](scripts/prepare_ravdess.py) —— 只下指定演员、筛出有声 AV 片段、去 calm、均衡挑 N 段并生成 `labels.csv`。

  ```bash
  python scripts/prepare_ravdess.py --actors 01 02 05 --n 50 --out data/ravdess
  ```

  演员奇数=男、偶数=女，多选几个演员可提升性别与说话人多样性。

---

## 4. 评测指标与协议

- **主指标**：整体 **Accuracy** 与 **macro-F1**（7 类，缓解类别不均）。
- **互补性证据（H3 核心图）**：A、B 两臂各画一张**逐类混淆矩阵**，指出"人脸在 X/Y 类更准、语音在 Z 类更准"。
- **协议**：
  1. 对每段视频抽若干帧喂 ① 得到人脸情绪 + 置信度；整段音频喂 ③ 得到语音情绪 + 置信度。
  2. A/B 直接取单模态预测；C/D 按对应融合策略合并。
  3. 与 `labels.csv` 的 ground truth 比对，统计各臂指标。
- **可复现**：固定采样与随机种子；`prepare_ravdess.py` 的挑选按文件名排序，结果确定。

---

## 5. 明确**不做**定量消融的部分（及其理由）

这几块**不是不重要，而是不适合用这 50 段视频做定量消融**，写报告时要讲清楚定位：

### 5.1 疲劳分支 ②（保留，但只做理论定位 + user study 定性）
- **为什么不做定量消融**：
  1. 疲劳检测用的是**预训练模型**，检测准确率不是我们的贡献，测它没有意义；
  2. 情绪数据集（RAVDESS）没有疲劳标签，无法在同一批数据上评估；
  3. 即便用瞌睡数据集（如 UTA-RLDD）测出疲劳准确率，也**证明不了"加疲劳比不加更有效"**——那是需求推断/音乐适配层面的价值。
- **合理定位（报告可直接用的措辞）**：
  > 在情感计算中，情绪与疲劳是两个正交的维度（valence–arousal 情感环状模型）：人脸与语音提供**情绪类别（效价维度）**，疲劳提供**精力/唤醒维度**。同一情绪标签在不同疲劳水平下对应不同的陪伴需求——例如同样"中性"，深夜疲惫时需要低唤醒的舒缓音乐，清醒时则适合维持专注的音乐。因此疲劳分支的作用不是提升情绪识别准确率，而是作为正交信号去**调制 ⑤ 的需求推断**。我们采用预训练疲劳检测模型，**不声称在疲劳检测本身上有贡献**；其价值体现在下游需求推断与音乐适配的合理性上，由 **user study 定性评估**。
- **可选：若想拿到"加疲劳更好"的证据**，唯一正确的做法是在 **user study 里做 A/B**（疲劳感知的推荐 vs 忽略疲劳的推荐，比较用户偏好），而**非** UTA 上的检测准确率。是否加此 A/B 臂由做 user study 的 M4 决定。

### 5.2 推理与生成层 ⑤⑥（→ user study）
- 最终输出是**生成的音乐**，没有客观 ground truth，无法用 50 段视频定量打分。
- 「ToM+CoT 需求推断是否比"情绪→曲风"直接查表更好」这类问题归 **user study**（12 人）或 LLM-judge 代理评测。

### 5.3 GradCAM ⑦ —— 可解释性实验（M1，非准确率消融臂）
GradCAM 是「解释」不是「预测」，不改变准确率，故**不作为准确率消融臂**，而是给 ⑦ 一块独立的可解释性实验（也让 M1 的贡献不止"套一个预训练脸模型"）。

- **路线 A（现在做）· 脚本 [scripts/gradcam_analysis.py](../scripts/gradcam_analysis.py)**：
  - 对每段视频取中间帧跑人脸①，产出 [原图 | Grad-CAM] 对照图，并按**预测对 / 错**分文件夹（用 labels.csv 的 ground truth）；接消融的错误分析。
  - **聚焦度指标 focus**＝CAM 能量最高的前 20% 像素占总能量比例（∈[0,1]，越高越集中）。假设：**预测正确时注意力更集中于面部表情区**（focus_mean_correct > focus_mean_incorrect）。产物 `results/gradcam/gradcam_metrics.json`。
  - 关键修正：Grad-CAM 目标是**情绪类别分数**（`features @ 分类器W.T`），不是特征通道——`face_emotion.cam_map()` 已按此实现，与 `predict` 的预测严格一致。
- **路线 B（已实现）**：把 CAM 变成**人脸可靠性信号** `reliability∈[0,1]` 去调制融合——`face_emotion.cam_reliability()` 度量 CAM 注意力集中于人脸中心区的程度（泄漏到裁剪边缘/背景则低）；`fusion.py` 的 `weighted_cam` 模式（arm E）把它乘到人脸权重上（低可靠→压低人脸、抬高语音）；`run_ablation.py` 逐帧算可靠性、聚合进人脸结果，新增第五臂 E。看 **E vs D** 是否提升。
  - **诚实预期**：RAVDESS 是干净正面表演脸，实测 reliability≈0.82–0.86、方差小，故 E 相对 D 增益**大概率微弱甚至持平**；这本身是一个如实的发现——CAM 门控在干净数据上作用有限，预计在遮挡/侧脸/暗光等**人脸可靠性方差大**的场景更有价值（接 user study / 未来工作 / 可掺入困难样本验证）。
- **另配 pre 可视化**：[scripts/viz_fatigue.py](../scripts/viz_fatigue.py) 把疲劳②用到的眼/嘴关键点画在脸上 + 标注 EAR/MAR/等级，直观展示"疲劳对应人脸哪些点"。

---

## 6. 语音置信度方案（已解决 · M2）

置信度加权融合 ④ 依赖每个模态给出置信度。人脸 HSEmotion 有 softmax 概率没问题；语音这边——

> **语音模型选型差异（写报告务必按实际实现）**：语音③ 已从原计划的 **EmotionThinker**（本地跑的 Qwen2.5-Omni 情绪微调版）改为**通用 Qwen-Omni（`qwen3-omni-flash`，经智增增 OpenAI 兼容 API）+ prompt 工程**，理由是笔记本友好、无需本地下大模型/占 GPU。Related Work 与 Methodology 都要按此描述，别照抄旧计划里的 EmotionThinker。

- **方案**：让模型在返回的 JSON 里**自评置信度**，再由 `speech_emotion._calibrate_confidence` 做规则校准——当模型把平淡/不明确语音判为 neutral 且 reasoning 出现模板化描述时压低其置信度，避免融合阶段错误的高置信 neutral 压过其他模态。这对应「模型自评 + 规则校准」，**H2（加权 > 朴素）因此可测**。
- **注意（跑实验 / 写报告时）**：
  1. 自评置信度**不等价于严格概率**，需在实验中观察它与真实正确率是否匹配（见 §7）；
  2. API 失败 / 无 key 时语音回退 `neutral, conf=0`，而融合把 conf=0 当作「模态缺失」→「带语音」的臂会**悄悄退化成纯人脸**。跑消融时必须**统计 fallback 率**，否则 H1/H2 的结论会被误读。

---

## 7. 局限性（写进报告 Limitations）

- **表演数据**：RAVDESS 是专业演员的**表演**情绪，与真实自发情绪存在分布差异。
- **样本量小**：约 50 段 / 7 类 ≈ 每类 7 段，统计力有限，结论以趋势性呈现；如成本允许可适当增大 N。
- **预训练编码器**：单模态准确率取决于预训练模型本身，不代表我们的工作；我们的贡献在**融合策略**与**系统集成**。
- **语音走在线 API**：用通用 Qwen-Omni（非情绪专用微调），置信度为模型自评+规则校准（非严格概率）；速度/稳定性受网络影响，失败会回退 neutral（需统计 fallback 率）。

---

## 8. 首轮结果、发现与语音「语义 vs 语调」的方法论

### 8.1 首轮五臂结果（50 段, actors 01/02/05）

| 臂 | accuracy | macro-F1 |
|----|----------|----------|
| A. 仅人脸 | **0.60** | 0.56 |
| B. 仅语音 | 0.16 | 0.07 |
| C. 朴素融合 | 0.16 | 0.07 |
| D. 加权融合 | 0.36 | 0.35 |
| E. 加权+CAM门控 | 0.26 | 0.23 |

**反常现象**：① 人脸 >> 融合（违反 H1）；② C(朴素)≡B(语音)；③ E(CAM门控) < D。

### 8.2 根因：语音在 RAVDESS 上恒判 neutral

缓存显示语音对 **47/50 段预测 `neutral`**（置信度 0.6–0.95，**非** API 失败）。原因是 **RAVDESS 只念两句固定中性句**（"Kids are talking by the door" / "Dogs are sitting by the door"），情绪**只在语调/韵律**里；而 `speech_emotion.py` 的 prompt 要求「结合语义 + 声学、不要只靠 tone」——在 RAVDESS 上恰好把一切拉回 neutral。加之 `_calibrate_confidence` 把模板化 neutral 只压到 0.6，仍高到能主导融合。

### 8.3 逐臂机制（都能被「语音≈随机」解释）

- **A >> 融合**：H1 的前提是两模态各自都有用；语音≈随机时，融合把好模态和废模态平均，必然被拉低。
- **C ≡ B**：朴素融合冲突时按固定优先级 `speech>face` 打破平票；语音(neutral)几乎总与人脸不一致 → 每次判给语音 → C 退化成「永远取语音」。
- **D < A**：加权按置信度，但语音对错误 neutral 给了 0.69 高置信 → 把融合拽到人脸之下。
- **E < D**：人脸是这里唯一的好模态，CAM 门控（reliability 均值 0.78<1）**下调人脸权重**、把权重推向坏语音 → 更差。且 reliability 方差小（0.69–0.88），门控无区分度、只会一味削人脸（路线B 待改进：见 8.5）。

### 8.4 「语音要不要靠语义」——能用消融调参吗？

**结论：不能只在 RAVDESS 上调这个权重，否则是对数据集过拟合。**

- RAVDESS **结构上只有语调**（字面永远中性），在它上面调「语义 vs 语调」权重，最优解永远是「纯语调、丢掉语义」——这是数据集人为造成的，不可泛化。
- 但产品初心是**情绪音乐陪伴**：真实用户可能直说「我今天很难过」（语义有情绪），也可能只是闲聊让系统看状态（只有语调）。**两种情形都要服务**，所以语义的价值不该被一个单一数据集抹掉。
- **正确做法**：把「语义 vs 语调」当成一个**在两类数据集上各自评测的设计权衡**，而不是单点调优——
  - 语调能力 → 在 **RAVDESS**（固定脚本、只有语调）上测；
  - 语义能力 → 在**有情绪语义的数据**（如 **MELD** 对话文本，M2 已有 `predict_text`）上测；
  - 跑几个离散 prompt 变体（纯语调 / 语调为主 / 均衡 / 语义为主），报告**在两个数据集上相反的最优点**——这是诚实且更强的发现（权衡真实存在），最终权重由**部署场景 / user study** 决定，而非最大化某一个 benchmark。
- 注意：prompt 权重是给 LLM 的**自然语言指令、非数值旋钮**，且 LLM 对措辞敏感有噪声，故只能做**少量离散变体**的小消融，不是梯度式精调。
- **对当前融合消融（H1/H2）的务实处理**：RAVDESS 的 A/B/C/D/E 实验里，应给语音一个**语调为主**的 prompt（因为 RAVDESS 本就只有语调），这样语音模态不再恒 neutral、H1/H2 才测得动；同时在报告里说明「部署系统保留语义、以服务真实闲聊场景」。**两件事分开讲，别混为一谈。**

### 8.5 后续行动

1. **[关键·M2]** 给 RAVDESS 消融改一版**语调为主**的语音 prompt（明确固定中性脚本、忽略字面语义），并核实音频真被模型听到（拿明显 angry 片段单测）。
2. **[重跑]** 语音结果已缓存，重跑需 `--refresh`（或删 `results/ablation/cache/`）以重新调用语音 API。
3. **[路线B·M1]** 改 `cam_reliability` 的映射：现在均值 0.78 会恒定给可靠人脸打八折、无区分度；应改成**只惩罚异常低值**（按分布中心化，典型→1.0）。
4. **[语义评测]** 若要论证「保留语义有价值」，在 MELD/文本上跑 `predict_text` 的语义能力评测（与 RAVDESS 的语调评测互补）。
5. **[报告口径]** 这个「失败」是好素材：实证了**融合只在两模态都有用时才赢**、**过度自信的坏模态会拖垮加权融合**——反向论证了**置信度/可靠性感知融合（④⑦）的必要性**，也说明语音置信度校准强度不够。

### 8.6 第二次发现：通用 Qwen-Omni API 读不好情绪语调 → 换本地 SER

改了「语调为主 + 语义自适应」prompt 后，拿 Actor_01 各情绪单段实测（含 intensity=strong）：

| 片段 | 预测 | conf | 模型 reasoning |
|------|------|------|----------------|
| angry(强) | surprise | 0.85 | "sudden sharp pitch rise and exclamation" |
| happy(强) | neutral | 0.2 | "prosody flat" |
| sad(强) | neutral | 0.2 | "prosody flat" |
| 其余 7 类(normal) | 几乎全 neutral | 0.2 | "prosody flat / words neutral" |

**诊断**：① 音频确实到达（模型能正确转写句子、强 angry 捕捉到"sharp pitch"，故非纯 ASR 丢韵律）；② 但 `qwen3-omni-flash` **对情绪语调又钝又不准**——多数强情绪读成"flat"→neutral，偶尔捕捉到又误分类（强 angry→surprise）。这是**通用 omni-flash 的能力上限，prompt 改不动**。

**prompt 改动仍有价值**：置信度从「错误的 0.69」变成「诚实的 0.2」，低置信语音会被加权融合自动降权，不再像首轮那样把 D 拖到 A 之下。但语音本身仍不可用。

**决定**：RAVDESS 消融的语音后端从 API 换成**本地专用 SER 模型**（wav2vec2，直接读韵律、给 softmax 置信度）；部署仍可保留 API 服务语义闲聊（双后端，`SPEECH_BACKEND` 切换）。
- **为什么不用 EmotionThinker**：它是 7B 情绪微调 LLM，重、慢、autoregressive、无标定概率（又回到置信度难题）；SER 专用分类器更轻更快、且**给 softmax 概率**（正好喂 H2 的置信度加权），RAVDESS 上准确率也更高。
- **⚠️ 报告须写的方法论 caveat**：所选 SER 若在 RAVDESS 上微调过，则对 RAVDESS 是**in-domain（可能训练集泄漏）**，其准确率是乐观上界、与人脸①（AffectNet→RAVDESS 跨域）**不对称**。诚实做法：如实标注；如需公平的泛化对比，改用**跨库 SER**（如 IEMOCAP/MSP 训练）在 RAVDESS 上测（准确率更低但公平）。

### 8.7 SER 修好后的第二轮结果 + 人脸调查

**SER 后端重跑（50 段）**：

| 臂 | acc | macro-F1 |
|----|-----|----------|
| A 仅人脸 | 0.60 | 0.56 |
| B 仅语音(SER) | **0.88** | 0.87 |
| C 朴素 | 0.88 | 0.867 |
| D 加权 | 0.88 | 0.869 |
| E 加权+CAM | 0.88 | 0.869 |

**新现象（H1 仍不成立，但方向相反）**：语音 SER 修好后**太强**、人脸相对弱，融合只**追平语音、无增益**。混淆矩阵显示**语音在几乎每一类都 ≥ 人脸、无互补空间** → 融合必然≈语音。

**人脸调查（排除 bug）**：检测全部 8/8（非 Haar 问题）；把抽帧改到情绪峰值段 acc 仅 0.60→0.62；`fear` 仍 0/7。**结论：人脸 0.60 是 AffectNet→RAVDESS 的真实跨域水平，非流水线 bug**。而 SER 0.88 是 **in-domain 上界**——**不对称是真实的**（一个开卷、一个闭卷）。

**核心症结**：要让融合真正体现价值，需两模态**势均力敌且互补**；当前 speech(in-domain) 碾压 face(cross-domain)，融合无从发挥。两条出路（待定）：
- **A. 跨库 SER**：换非 RAVDESS 训练的 SER（如 `superb/wav2vec2-base-superb-er`，IEMOCAP），语音降到与人脸相当。代价：仅 4 类（neu/hap/ang/sad），词表窄、且是另一种"不公平"（缺类）。
- **B. 换素材到 CREMA-D**（推荐）：人脸(AffectNet) 与 语音(RAVDESS-SER) **都没在 CREMA-D 上训练** → 两者都跨域、公平可比，6 类（无 surprise）。代价：要写 CREMA-D 的数据准备脚本 + 下载。这是最干净的"公平对比、给融合机会"的设置。

### 8.8 CREMA-D 第三轮结果：H2 强证实，H1 仍难（附三轮综合结论）

**CREMA-D（60 段、语音用 SER 跨域）：**

| 臂 | acc | macro-F1 |
|----|-----|----------|
| A 仅人脸 | **0.667** | 0.678 |
| B 仅语音(SER 跨域) | 0.267 | 0.257 |
| C 朴素融合 | 0.267 | 0.257 |
| **D 加权融合** | **0.517** | 0.575 |
| E 加权+CAM | 0.517 | 0.575 |

**不对称又反转了**：CREMA-D 上人脸(0.667)强、语音 SER 跨域(RAVDESS→CREMA-D 迁移差，0.267≈近随机)弱。

- **✅ H2 强证实（真正的好结果）**：**加权融合 0.517 ≈ 翻倍于朴素融合 0.267**。朴素因固定优先级退化成「取弱语音」(C≡B)；加权用置信度**把不可靠的低置信语音降权、大幅拉回人脸的表现**。这直接证明了融合④「置信度加权」的价值——**对模态失衡鲁棒**。
- **❌ H1 仍不成立**：融合 0.517 < 最强单模态人脸 0.667。语音太弱，即便加权也只能减少损害、无法超过人脸。
- **E ≡ D**：CREMA-D 人脸也基本正面清晰，`cam_reliability`≈1，门控几乎不触发（预期）。

**三轮综合结论（写进报告，是诚实且有力的叙事）**：三次实验是三种「模态失衡」regime——
| 数据/语音 | 语音 acc | 现象 | 教训 |
|-----------|---------|------|------|
| RAVDESS + API | 0.16 | 语音废、朴素崩 | 融合需**置信度感知** |
| RAVDESS + SER(in-domain) | 0.88 | 语音碾压、融合冗余 | 需**公平可比**的评测 |
| CREMA-D + SER(跨域) | 0.27 | 语音弱、加权>>朴素 | 加权融合**对失衡鲁棒**（H2）|

**核心洞见**：**融合超过最强单模态（H1）需要两模态势均力敌；而我们的贡献④（置信度加权）真正的价值是「对失衡的鲁棒性」（H2）——在任一模态不可靠时优雅降级、不被拖垮。** 这比"融合总是赢"更真实、也更有说服力。

**若仍想让 H1 成立**：需要一个在评测集上准确率与人脸相当（~0.6）的语音模型——要么用迁移更好的跨库 SER，要么用 CREMA-D 上 held-out 的 in-domain SER。属于额外实验，可留作 future work 或按需再做。

---

## 9. 分工与产物

| 事项 | 负责 | 产物 |
|------|------|------|
| 数据准备 | M1/M4 | `scripts/prepare_ravdess.py` → `data/ravdess/labels.csv` |
| 语音语义/语调 prompt + 置信度 | **M2** | `speech_emotion.py`（RAVDESS 用语调为主；部署保留语义） |
| 融合策略实现 | M2/M4 | `fusion.py`（naive / weighted / weighted_cam 三种） |
| 评测脚本与指标 | M4 | `scripts/run_ablation.py` 跑 5 臂、出 accuracy/F1/混淆矩阵 |
| ⑦ 可解释性 + 可视化 | M1 | `scripts/gradcam_analysis.py` / `scripts/viz_fatigue.py` |
| 报告 Experiments 章 | M4（统稿） | 基于本文件扩写 |

---

## 10. 复现与运行指南（交接 M2 · 从零跑通消融）

> 所有命令在**激活的 conda 环境 `3800`（Python 3.10）**、**项目根目录**下运行。激活后
> ffmpeg 自动在 PATH 上、`SPEECH_BACKEND=ser python …` 直接可用。

### 10.0 环境准备（一次性）

```bash
conda activate 3800
pip install -r requirements.txt
conda install -n 3800 -c conda-forge ffmpeg   # audio_extract 抽音频必需
# .env 仅在用 API 后端时需要（填 ZHIZENGZENG/DEEPSEEK key）；用本地 SER 不需要 key
```

### 10.1 下载 SER 语音模型（本地后端）

模型：`ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition`（RAVDESS 8 类，~1.2GB）。
**首次 `SPEECH_BACKEND=ser` 运行时会自动下载**到 HuggingFace 缓存（`~/.cache/huggingface`），无需手动下载。
- 想先预下载：`python -c "from huggingface_hub import snapshot_download; snapshot_download('ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition')"`
- 下载慢/走代理时耐心等；下好后可加 `HF_HUB_OFFLINE=1` 让后续跑得更快（纯离线读缓存）。
- 注意：该模型用**自定义分类头**，`speech_emotion.py` 里已做复刻加载（stock 类会得到随机头）；Windows 上 symlink 警告无害。

### 10.2 下载数据集

**RAVDESS（语音 in-domain，作对照）**：`prepare_ravdess.py` 会自动从 Zenodo 下载指定演员的视频 zip，无需手动。

**CREMA-D（公平跨域，主实验）**：**必须手动下载「视频版」**，普通 Kaggle 镜像只有音频！
- 视频版（推荐）：Kaggle `stefanogiannini/crema-d-video`（含人脸 .flv）
- 或官方 GitHub `CheyneyComputerScience/CREMA-D` 的 `VideoFlash/`（git-lfs）
- ⚠️ **不要**下 `ejlok1/cremad`（只有 AudioWAV，没有人脸，做不了融合）
- 下好解压到某目录（如 `data/cremad/video/`），`.flv` 里自带音轨，一段视频喂两个模态。

### 10.3 生成标签（→ labels.csv）

```bash
# RAVDESS：下 3 个演员、去 calm、均衡挑 50 段
python scripts/prepare_ravdess.py --actors 01 02 05 --n 50 --out data/ravdess

# CREMA-D：从下载的视频目录均衡挑 60 段（6 类各 10）
python scripts/prepare_cremad.py --src data/cremad/video --n 60 --out data/cremad
```

### 10.4 跑五臂消融（A/B/C/D/E）

```bash
# 主实验 · CREMA-D（语音用本地 SER；两模态都跨域、公平）
SPEECH_BACKEND=ser python scripts/run_ablation.py --data data/cremad --out results/ablation_cremad --refresh
#   --limit 6      先跑 6 段冒烟
#   --fusion-only  只用缓存重算 C/D/E（不重跑感知，改融合参数时秒出）

# 对照 · RAVDESS（语音 in-domain，会看到语音碾压人脸）
SPEECH_BACKEND=ser python scripts/run_ablation.py --data data/ravdess --out results/ablation --refresh
```
产物在 `<out>/`：`metrics.json`（五臂 acc/F1/逐类）、`confusion_[A-E].csv`、`cache/predictions.json`。

### 10.5 可解释性 + pre 可视化（⑦ 与 ②）

```bash
python scripts/gradcam_analysis.py --data data/cremad     # 热力图(对/错分组)+focus 指标 → results/gradcam
python scripts/viz_fatigue.py     --data data/cremad     # 眼/嘴关键点标注 → results/viz_fatigue
# 两者也支持 --images path/to/face.jpg（不依赖数据集，任意人脸图快速出图）
```

### 10.6 常见坑（务必注意）

- **`SPEECH_BACKEND`**：默认 `api`（qwen-omni，读不好语调）。**跑消融一定要 `SPEECH_BACKEND=ser`**。
- **`--refresh`**：切换语音后端 / 改感知后，旧的 `cache/predictions.json` 不会自动失效，**必须 `--refresh`**（或删 `<out>/cache/`）重算，否则用的是旧结果。
- **`--out` 分开**：RAVDESS 与 CREMA-D 用不同 `--out`，别互相覆盖。
- **ffmpeg**：没装或不在 PATH → `audio_extract` 报错、语音全 fallback。激活 3800 后应自动在 PATH。
- **CREMA-D 必须是视频版**（见 10.2），音频版做不了人脸/融合。
- **词表**：CREMA-D 无 surprise（6 类），`config.EMOTION_LABELS` 仍 7 类，surprise 列 support=0，不影响指标、不用改代码。

---

*AIAA 3800 小组项目 — EmotiCompanion — HKUST(GZ) — 2026*
