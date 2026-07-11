# EmotiCompanion 感知层消融 · 实验记录

**多模态情绪感知的实时音乐陪伴系统 · AIAA 3800 · HKUST(GZ) · 2026**

> 记录我们在**感知层**做的消融：验证「把预训练的人脸情绪①与语音情绪③融合，是否优于任一单模态」，即我们自己设计的**融合模块 ④** 的价值。推理/生成层（⑤⑥）与疲劳②不做定量消融（理由见 §6）。
> **一句话结论**：① 置信度加权 ④ 真正的价值是**对模态失衡的鲁棒性**（H2，跨所有轮次成立）；② 融合**超过**最强单模态（H1）需两模态势均力敌且互补、且融合对**逐类可靠性**敏感——满足这两条时（CREMA-D+emotion2vec + 贝叶斯逐类融合 arm F）H1 首次成立（0.683 > 0.667，见 §3.7）。

---

## 1. 目标与假设

人脸①（HSEmotion/AffectNet）、语音③（多种后端）都是**现成预训练模型**，不是我们的贡献；**原创的是融合策略 ④**。故消融验证的是这个设计选择，而非"识别器多准"。

| 编号 | 假设 | 证明方式 |
|------|------|----------|
| **H1** | 多模态融合准确率 **高于任一单模态** | 单模态臂 vs 融合臂的 acc / macro-F1 |
| **H2** | **置信度加权融合 ④** **优于朴素融合**（等权/投票） | 朴素臂 vs 加权臂 |
| **H3** | 人脸与语音**互补**：各擅长不同情绪 | 两单模态的**逐类混淆矩阵** |

---

## 2. 实验设置

**实验臂**（A–E 由 `run_ablation.py` 同批跑；**F 为读缓存的后置分析**，见 §3.7）：

| 臂 | 模块 | 角色 | 假设 |
|----|------|------|------|
| A. 仅人脸 | ① | 单模态基线 | H1/H3 |
| B. 仅语音 | ③ | 单模态基线 | H1/H3 |
| C. 朴素融合（等权/投票） | ①③ | 对照 | H1 |
| **D. 置信度加权融合 ④** | ①③④ | **主角** | H2 |
| E. 加权 + GradCAM 可靠性门控 | ①③④⑦ | ⑦ 是否有定量贡献 | E vs D |
| **F. 贝叶斯逐类可靠性融合** | ①③ | 逐类可靠性 late-fusion（LOOCV） | H1 |

> 臂 E＝D＋CAM 门控：注意力跑到脸外→人脸可靠性低→压低人脸权重（`fusion.py` 的 `weighted_cam` 模式）。
> 臂 F＝按各模态**逐类历史可靠性**做朴素贝叶斯融合（`fusion.py` 的 `bayes` 模式 / `scripts/fusion_bayes.py`），是唯一越过最强单模态的臂——详见 §3.7。

**数据集**：

| 数据集 | 性质 | 类数 | 用途 |
|--------|------|------|------|
| RAVDESS | 表演 AV；固定中性脚本、情绪只在语调 | 7（去 calm） | 对照（语音 in-domain） |
| **CREMA-D** | 表演 AV；人脸与语音**都未在其上训练** | 6（无 surprise） | **主实验**（两模态都跨域、公平可比） |
| MELD | 真实剧集(Friends)对话 AV；**含语义**、多人/镜头切换 | 7 | 语义场景（比语音后端；人脸不稳） |
| eNTERFACE'05 | 表演 AV；**正面单人脸 + 情绪一致语义句** | 6（无 neutral） | 兼顾人脸清晰+语义（见 §4.1） |

**语音后端**（`SPEECH_BACKEND` 切换；部署默认 `api` 保留语义闲聊能力）：

| 后端 | 模型 | 性质 |
|------|------|------|
| `api` | Qwen-Omni `qwen3-omni-flash`（OpenAI 兼容 API） | 通用多模态，读不好情绪语调 |
| `ser` | wav2vec2 `ehcalabres`（RAVDESS 微调） | 对 RAVDESS **in-domain**（乐观上界） |
| **`emotion2vec`** | **emotion2vec+ `iic/emotion2vec_plus_base`（FunASR）** | **自监督预训练+微调，跨域** |

**指标与协议**：整体 **Accuracy** + **macro-F1**（7 类）；A/B 两臂各出**逐类混淆矩阵**（H3）；每段抽若干帧喂①、整段音频喂③，与 `labels.csv` 比对。采样固定、可复现。**须统计 fallback 率**（语音失败会回退 `neutral, conf=0`，带语音的臂会悄悄退化成纯人脸）。

---

## 3. 结果（按轮次）

六轮实验（§3.1–3.6，每轮换语音后端 / 数据集）对应六种「模态失衡」regime，观察融合在不同失衡下的行为；§3.7 是横跨各轮缓存的逐类可靠性融合分析。

### 3.1 RAVDESS + Qwen-Omni API（50 段）

| 臂 | acc | macro-F1 |
|----|-----|----------|
| A 仅人脸 | **0.60** | 0.56 |
| B 仅语音 | 0.16 | 0.07 |
| C 朴素 | 0.16 | 0.07 |
| D 加权 | 0.36 | 0.35 |
| E 加权+CAM | 0.26 | 0.23 |

**发现**：语音对 **47/50 段恒判 neutral**（非 API 失败）——RAVDESS 只念两句固定中性句，情绪只在语调里，而通用 omni-flash 读不好语调。语音≈随机 → 融合被废模态拖垮（H1 崩）；朴素按固定优先级 `speech>face` → C 退化成"永远取语音"(C≡B)；坏语音过度自信(conf 0.69)把加权 D 也拽到人脸之下。**反证了置信度感知融合的必要性。**

### 3.2 RAVDESS + SER（wav2vec2 in-domain，50 段）

| 臂 | acc | macro-F1 |
|----|-----|----------|
| A 仅人脸 | 0.60 | 0.56 |
| B 仅语音(SER) | **0.88** | 0.87 |
| C 朴素 | 0.88 | 0.867 |
| D 加权 | 0.88 | 0.869 |
| E 加权+CAM | 0.88 | 0.869 |

**发现**：换 in-domain SER 后语音**太强**、人脸相对弱，融合只**追平语音、无增益**。混淆矩阵显示语音几乎每类 ≥ 人脸、**无互补空间**。人脸调查确认 0.60 是 AffectNet→RAVDESS 的**真实跨域水平**（检测 8/8、抽峰值帧仅 0.60→0.62，非 bug）。**不对称是真实的**：一个开卷（SER in-domain）、一个闭卷（face 跨域）——这不是公平对比，促使我们换 CREMA-D。

### 3.3 CREMA-D + SER（wav2vec2 跨域，60 段）

| 臂 | acc | macro-F1 |
|----|-----|----------|
| A 仅人脸 | **0.667** | 0.678 |
| B 仅语音(SER 跨域) | 0.267 | 0.257 |
| C 朴素 | 0.267 | 0.257 |
| **D 加权** | **0.517** | 0.575 |
| E 加权+CAM | 0.517 | 0.575 |

**发现**：CREMA-D 上不对称反转——人脸强(0.667)、SER 跨域弱(0.267≈近随机，RAVDESS→CREMA-D 迁移差）。
- **✅ H2 强证实**：加权融合 0.517 ≈ **翻倍于朴素 0.267**。朴素因固定优先级退化成"取弱语音"(C≡B)；加权用置信度**把不可靠的低置信语音降权、拉回人脸**——直接证明 ④ 对模态失衡鲁棒。
- **❌ H1 仍不成立**：0.517 < 最强单模态人脸 0.667。语音太弱，加权只能减损、无法超越人脸。
- **E ≡ D**：CREMA-D 人脸正面清晰，`cam_reliability`≈1，门控几乎不触发（预期）。

### 3.4 CREMA-D + emotion2vec+（自监督跨域，60 段）

> M4 引入 emotion2vec+（`iic/emotion2vec_plus_base`）——自监督预训练+微调的专用 SER，对 CREMA-D 跨域、与人脸①的跨域性质更对称，期望语音比 3.3 的 wav2vec2 更强、给融合更公平的机会。

| 臂 | acc | macro-F1 |
|----|-----|----------|
| A 仅人脸 | **0.667** | 0.678 |
| B 仅语音(emotion2vec 跨域) | 0.533 | 0.520 |
| C 朴素 | 0.533 | 0.520 |
| **D 加权** | **0.600** | 0.607 |
| E 加权+CAM | 0.600 | 0.607 |

> fallback 率 **0/60**（语音全部真跑，非缺失）；语音 conf 均值 0.91（softmax）、人脸 conf 0.68、`reliability` 均值 0.99。

**发现**：这是**各轮里最均衡的 regime**——emotion2vec 把跨域语音从 wav2vec2 的 0.267 提到 **0.533**，终于与人脸(0.667)**量级相当**。
- **✅ H2 再次证实**：加权 **0.600 > 朴素 0.533**。朴素按固定优先级 `speech>face` 又退化成"取语音"(C≡B)；加权用置信度**部分找回人脸判对的样本**，抬到两单模态之间。
- **❌ H1 仍差一点（但最接近）**：0.600 < 最强单模态人脸 0.667。语音够强了，但人脸的强项太集中，融合稀释了它。
- **✅ H3 首次两侧互补**：语音在 **angry(5>3)、neutral(10>8)** 胜过人脸；人脸在 **fear(9>2)、disgust(9>5)** 大幅领先；happy 打平(7=7)。互补真实存在，但人脸的领先幅度(fear +7、disgust +4)大于语音(angry/neutral 各 +2)，逐样本融合没能完美按类路由 → 融合未超过人脸。
- **E ≡ D**：人脸清晰、`reliability`≈0.99，CAM 门控几乎不触发（预期）。

> **给 M2 的提示**：这轮已把两模态拉到接近势均力敌，是**离 H1 成立最近的一次**。若想让融合真正超过人脸，方向是（a）让加权对**逐类可靠性**更敏感（人脸的 fear/disgust 权重更高、语音的 angry/neutral 权重更高），或（b）继续提升语音（如 emotion2vec_plus_**large**）。属加分项，非必需——H2 的鲁棒性结论已经成立。

### 3.5 MELD（真实对话、含语义 · 三后端 × 四臂，M2 跑）

MELD 取自美剧 Friends 的多人对话，**首次带真实语义**（不再是固定中性脚本），更贴近部署时"用户真的说一句有情绪的话"的场景。同一批样本，三个语音后端各跑 A–D：

| 语音后端 | A 仅人脸 | B 仅语音 | C 朴素 | D 加权 | 最佳臂 |
|----------|---------|---------|--------|--------|--------|
| API / Qwen-Omni | 0.183 / 0.180 | 0.414 / 0.409 | 0.415 / 0.409 | 0.391 / 0.381 | C 朴素 0.415 |
| SER / wav2vec2 | 0.183 / 0.180 | 0.201 / 0.189 | 0.201 / 0.189 | 0.209 / 0.193 | D 加权 0.209 |
| **emotion2vec** | 0.183 / 0.180 | **0.557 / 0.541** | 0.557 / 0.541 | 0.552 / 0.532 | B 仅语音 0.557 |

**发现**：

- **语音后端排序在有语义的数据上给出新信息**：`emotion2vec (0.557) > API/Qwen-Omni (0.414) ≫ wav2vec2 (0.201)`。
  - **emotion2vec 再次最强** → 跨各数据集，它都是最佳语音后端，选型可定。
  - **API(0.414) ≫ wav2vec2(0.201)** 是关键反转：MELD **有语义**，读语义的 Qwen-Omni 大幅反超**纯语调**的 wav2vec2（RAVDESS 微调、跨域到自然对话即崩）。这**实证了 §4 的「语义 vs 语调」权衡**——固定脚本数据上纯语调后端占优，含语义的真实数据上读语义的后端反超。印证了部署保留 API 语义后端服务真实闲聊是对的。
- **人脸恒 0.183（极低，且与语音后端无关）**：MELD 是多人剧集、镜头频繁切换，说话者常侧脸/小脸/被遮挡/多人同框，人脸①拿不到稳定正脸（M2 观察）→ 人脸参考价值低。
- **融合≈语音（与 CREMA-D 对称的失衡）**：人脸失效时融合退化成语音；加权在 emotion2vec 上甚至略低于纯语音（0.552<0.557，被坏人脸拖了一点）。CREMA-D 是"语音弱、人脸强"，MELD 是"人脸弱、语音强"——**两种相反的失衡，都印证融合无法凭空造出缺失模态的信息**。
- **⚠️ 局限**：MELD 因人脸不可用，**无法在其上演示融合价值**；它的价值在于确认语音后端选型 + 验证语义后端的必要性。

### 3.6 eNTERFACE'05（正面单人脸 + 情绪语义句 · emotion2vec，60 段）

为兼顾"人脸清晰 + 语音有语义"（CMU-MOSEI 原计划，但官方 raw 视频服务器已 502、镜像全是特征，故改用 eNTERFACE'05，见 §4.1）。受试者对镜头正面作答、说情绪一致的语义句（6 情绪，**无 neutral**）。

| 臂 | acc | macro-F1 |
|----|-----|----------|
| A 仅人脸 | **0.483** | 0.462 |
| B 仅语音(emotion2vec) | 0.300 | 0.340 |
| C 朴素 | 0.300 | 0.340 |
| **D 加权** | **0.400** | 0.439 |
| E 加权+CAM | 0.400 | 0.439 |

> fallback 0/60；语音 conf 均值 0.80、人脸 conf 0.65、**`reliability` 均值 0.999**（min 0.98）。

**发现**：

- **✅ 验证了 MELD 人脸失效是几何性的（非正脸）**：eNTERFACE 正面单人脸上 `reliability`≈**0.999**（对比 MELD 的多人/侧脸），人脸①能正常识别真实表情（happy 9/10、surprise 8/10、disgust 7/10）。**同一个人脸模型，换到正面数据就恢复了** → 坐实"MELD 的 0.18 是画面几何问题，不是模型问题"。
- **⚠️ eNTERFACE 自带的 neutral-onset 假象拉低了两个模态**：eNTERFACE 每段**从中性表情起、到句末才到情绪峰值**，而均匀抽帧/整段音频会吃到大量中性片段 → **两模态都把不少样本判成 `neutral`（人脸 10/60、语音 18/60），但 eNTERFACE 根本没有 neutral 类** → 这些预测必错，人为压低 acc。这是数据集时序结构造成的，非模型缺陷（可用"抽情绪峰值段/句末帧"缓解，见下）。
- **✅ H2 再次成立**：加权 0.400 > 朴素 0.300。朴素又退化成"取语音"(C≡B)，加权用置信度拉回部分人脸判对的样本。
- **❌ H1 仍未越过**：0.400 < 人脸 0.483。语音偏弱（受试者是 14 国**非母语英语**、口音重，削弱 emotion2vec；叠加 neutral-onset），互补有限（除 fear 外人脸几乎全面领先）→ 融合超不过人脸。
- **E ≡ D**：人脸干净、`reliability`≈1，CAM 门控不触发（预期）。

> **可选改进**：run_ablation 抽帧改成**取片段后半/句末峰值帧**、并考虑把 neutral 预测按"无此类"折算，能缓解 neutral-onset、预计抬高 face 与 speech。但当前结论（正面人脸恢复 + H2 成立）已不依赖它。

### 3.7 逐类可靠性贝叶斯融合（arm F · LOOCV）——H1 首次成立

前六轮里加权融合 ④ 从未干净超过最强单模态（H1），因为它只看**单模态自评置信度**，
不知道**每个模态在每一类上多可靠**（人脸①擅长 fear/disgust、语音③(emotion2vec)擅长 angry/neutral）。
新增 **arm F**：把"逐类可靠性"显式建模成**混淆似然**、做朴素贝叶斯晚融合
（`P(true=c|face,speech) ∝ P(c)·P(face|c)·P(speech|c)`，脚本 [scripts/fusion_bayes.py](../scripts/fusion_bayes.py)）。

> **⚠️ 防泄漏**：逐类可靠性**不能**从被评测的同一批样本估计（否则过拟合测试集、H1 假性通过）。
> 用**留一交叉验证 LOOCV**：每个样本只用其余样本估计似然/先验。部署等价于"在验证集上离线学好
> 逐类可靠性、作为固定先验上线"——**有原则、可部署、非泄漏**。直接读各轮缓存、不重跑感知。

| 数据 / 后端 | A 人脸 | B 语音 | D 加权(旧) | **F 贝叶斯逐类** | F 越过最强单模态？ |
|-------------|-------|-------|-----------|-----------------|-------------------|
| **CREMA-D + emotion2vec** | 0.667 | 0.533 | 0.600 | **0.683** | ✅ **是**（41>40 段）|
| eNTERFACE + emotion2vec | 0.483 | 0.300 | 0.400 | 0.433 | ❌ 否 |
| CREMA-D + SER(wav2vec2) | 0.667 | 0.267 | 0.517 | 0.650 | ❌ 否 |
| RAVDESS + SER | 0.600 | 0.880 | 0.880 | 0.840 | ❌ 否 |

**发现**：

- **✅ H1 首次成立（唯一一次）**：在**最均衡且两侧互补**的 CREMA-D+emotion2vec 上，`F=0.683 > 最强单模态(人脸 0.667)`。这是全部实验里融合第一次真正超过任一单模态。margin 是 **+1 段（41 vs 40）**、较薄，但**真实且方法有原则、无泄漏**。
- **只在"均衡 + 互补"时越过**：其余三处 F 都 < 最强单模态——语音太弱(SER 0.267)、太强(RAVDESS 0.88)、或无两侧互补(eNTERFACE)时，逐类可靠性也造不出增益。**证实 H1 需要两模态势均力敌且互补**（§4 的核心洞见）。
- **是"逐类可靠性"而非"置信度"解锁了 H1**：`--use-conf`（似然按置信度加权）反而回落到 0.667 持平——加权 ④ 的置信度不够，**F 的增量全来自逐类混淆似然**。
- **诚实 caveat**：F 在 CREMA-D+e2v 上 acc 赢（0.683>0.667）但 macro-F1 略低（0.669<0.678）；即赢在多数类、并非全类占优。样本量小（60/6 类），margin 单段，宜以趋势 + 方法论价值呈现，而非强主张。

**意义**：把 §3.4 的"给 M2 的提示(a)"落地了——**加权对逐类可靠性敏感后，H1 在均衡 regime 上可成立**。这也让贡献从"H2 鲁棒性"扩展到"有原则的逐类可靠性融合"（一个可部署的 late-fusion 方法）。

**已落地为部署可选模式**：`fusion.py` 新增 `mode="bayes"`（arm F），用 `scripts/fusion_bayes.py --export` 从验证集离线学好的 `models/fusion_bayes_priors.json`（当前用 CREMA-D+emotion2vec 学，6 类）。**默认仍是 `weighted`**——bayes 收益薄（单段）、且依赖先验与部署分布匹配、输出类别受限于训练类；缺先验文件自动回退 weighted。定位为"验证数据匹配时可开启的增强"，不围绕它过度工程。

---

## 4. 结论

**六轮 = 六种模态失衡 regime**：

| 数据 / 语音后端 | 语音 acc | 现象 | 教训 |
|-----------------|---------|------|------|
| RAVDESS + API | 0.16 | 语音废、朴素崩 | 融合需**置信度感知** |
| RAVDESS + SER(in-domain) | 0.88 | 语音碾压、融合冗余 | 需**公平可比**评测 |
| CREMA-D + SER(跨域) | 0.27 | 语音弱、加权>>朴素 | 加权对失衡**鲁棒**（H2）|
| CREMA-D + emotion2vec(跨域) | 0.53 | 两模态量级相当、加权>朴素、首现两侧互补 | **arm F 在此越过 H1**；H2 稳固 |
| MELD + emotion2vec(含语义) | 0.56 | **人脸失效**、语义后端反超语调 | 融合造不出缺失模态；需人脸清晰+语义的数据 |
| eNTERFACE + emotion2vec(正脸+语义) | 0.30 | 人脸恢复(rel≈1)、H2 成立、neutral-onset 拉低两模态 | 正脸让人脸复活；数据集时序结构成新变量 |

- **H2 成立（最扎实的结论）**：置信度加权融合 ④ 在三种失衡下都优于朴素融合——CREMA-D+SER 0.517 vs 0.267（近翻倍）、CREMA-D+emotion2vec 0.600 vs 0.533、eNTERFACE 0.400 vs 0.300。朴素因固定优先级退化成"取语音"，加权用置信度优雅降级、不被弱模态拖垮。
- **H1 可成立，但有严格前提（§3.7）**：置信度加权 ④ 从未越过最强单模态（最接近 CREMA-D+emotion2vec 0.600 vs 0.667）；直到引入**逐类可靠性贝叶斯融合 arm F**，才在**最均衡且两侧互补**的 CREMA-D+emotion2vec 上首次越过（0.683 > 0.667，LOOCV 非泄漏）。结论：**H1 需要 (i) 两模态势均力敌且互补 + (ii) 融合对逐类可靠性敏感**；缺任一条件（语音过弱/过强、无互补、只用置信度）都不成立。margin 单段、宜作趋势呈现。
- **H3 互补性**：仅在均衡 regime（CREMA-D+emotion2vec）才显现两侧互补——语音强于 angry/neutral、人脸强于 fear/disgust；eNTERFACE 上人脸几乎全面领先（除 fear）、互补有限。说明"互补"是融合有增益的前提，只有两模态都够强才谈得上。
- **GradCAM 可靠性信号（E）经受住了正/反例检验**：干净正面人脸（CREMA-D、eNTERFACE）上 `reliability`≈0.99–1.0、门控不触发、E≡D；而 MELD 的多人/侧脸正是它该压低人脸权重的场景。**eNTERFACE 用同一人脸模型从 MELD 的 0.18 恢复到 0.48，反证了 reliability 信号对"正脸 vs 非正脸"的判别是对的**。其定量增益仍需人脸可靠性方差大的数据才显现（future work）。
- **设计权衡「语义 vs 语调」**：RAVDESS 字面永远中性、只有语调，在其上调"语义vs语调"权重会过拟合数据集。MELD（含语义）上 API 后端反超纯语调 SER，实证了这个权衡真实存在。正确定位是**两类数据分别评测**，部署保留 API 语义后端服务真实闲聊、消融用本地 SER 测语调——两件事分开讲。
- **语音后端选型已定**：跨 RAVDESS/CREMA-D/MELD/eNTERFACE，**emotion2vec+ 一致最佳**（自监督、跨域稳、给 softmax 置信度），定为**本地 SER 的默认选择**（部署仍可切 `api` 服务语义闲聊）。

### 4.1 数据集的两难：没有一个同时满足「人脸清晰 + 语音有语义」

多轮暴露出一个结构性矛盾——现有数据集**两个要求（人脸清晰 / 语音有语义）常只能满足其一**：

| 类型 | 代表 | 人脸 | 语音语义 | 问题 |
|------|------|------|----------|------|
| 表演·固定脚本 | RAVDESS / CREMA-D | ✅ 干净正面 | ❌ 念中性句 | 语音只有语调，无语义；不像真实使用 |
| 真实·剧集对话 | MELD | ❌ 多人/侧脸/切换 | ✅ 有语义 | 人脸拿不到稳定正脸，融合演示不了 |
| 表演·正面独白 | **eNTERFACE'05** | ✅ 正面单人脸 | ~ 情绪一致句（有语义、但仍为脚本） | 兼顾两者，但 neutral-onset + 非母语口音（§3.6） |

我们真实的部署场景是**单个用户对着摄像头、边说边被观察**——最理想是一个既有**清晰正面单人脸**、又有**自然带情绪语义语音**的 AV 数据集。候选与实际选择：

- **CMU-MOSEI（原首选，但拿不到）**：YouTube **独白/vlog**（说话人正对镜头），单人正脸 + 自然语义语音 + Ekman 6 类标签，与部署场景生态效度最高。**⚠️ 但官方 raw 视频服务器已 502、Kaggle/HF 镜像全是预抽特征非视频**，无法喂我们自己的人脸/语音模型 → 放弃。
- **eNTERFACE'05（实际采用，见 §3.6）**：正面独白 + 情绪一致语义句，是**唯一可下载、同时（近似）满足两条**的 AV 数据集。已证实正面人脸让人脸①从 MELD 的 0.18 恢复到 0.48；但其 neutral-onset 时序 + 非母语口音仍拉低了数值。
- **SEND（Stanford Emotional Narratives）**：对镜头讲情绪故事，正面单人脸 + 自然语义；但标签是**连续 valence**，需转换（备选）。
- **不推荐**：IEMOCAP/MSP-IMPROV（侧脸+MoCap 标记，人脸不干净）、MSP-Podcast（纯音频无脸）。

**小结**：真实部署（用户对 webcam 说话）最贴近 CMU-MOSEI 的独白形态，但它拿不到；eNTERFACE 是可行替代并已跑（§3.6）。若日后能取到 MOSEI 原始视频，是最值得补的一轮——属加分实验，H1/H2 的核心结论已不依赖它。

---

## 5. 局限性（写进报告 Limitations）

- **表演数据**：RAVDESS/CREMA-D 均为演员表演情绪，与真实自发情绪有分布差异。
- **样本量小**：50–60 段 / 6–7 类 ≈ 每类 ~10 段，结论以趋势呈现。
- **预训练编码器**：单模态准确率取决于预训练模型本身，非我们的工作；贡献在**融合策略**与**系统集成**。
- **语音置信度**：API 后端为模型自评+规则校准（非严格概率）；SER/emotion2vec 为 softmax 概率。失败回退 neutral，需统计 fallback 率。

---

## 6. 不做定量消融的部分（及定位）

这几块不适合用这批视频做定量消融，报告里如实定位：

- **疲劳②**：预训练模型（检测准确率非我们的贡献），且情绪数据集无疲劳标签。定位为与情绪**正交的唤醒维度**（valence–arousal），作用是**调制 ⑤ 的需求推断**（同一情绪在不同疲劳下需求不同），由 **user study 定性评估**。若要"加疲劳更好"的证据，唯一正确做法是 user study 里做 A/B，而非疲劳检测准确率。
- **推理⑤/生成⑥**：最终输出是生成的音乐，无客观 ground truth → 归 **user study（≤10 人，见 [docs/user_study.md](user_study.md)）** 或 LLM-judge。
- **GradCAM⑦**：是"解释"不是"预测"，不作准确率消融臂，而是独立的可解释性实验——[scripts/gradcam_analysis.py](../scripts/gradcam_analysis.py) 产出 [原图|CAM] 对照图（按预测对/错分组）+ focus 聚焦度指标（CAM 前 20% 能量占比），验证"预测正确时注意力更集中于面部表情区"。CAM 目标是**情绪类别分数**（`features @ 分类器W.T`），与 `predict` 严格一致。

---

## 7. 复现指南（交接 M2 · 从零跑通）

> 所有命令在**激活的 conda 环境 `3800`（Python 3.10）**、**项目根目录**下运行。

### 7.0 环境（一次性）

```bash
conda activate 3800
pip install -r requirements.txt              # 含 librosa/funasr/modelscope（本地 SER 用）
conda install -n 3800 -c conda-forge ffmpeg  # 抽音频必需
# .env 仅 api 后端需要（ZHIZENGZENG/DEEPSEEK key）；本地 SER 不需要 key
```

### 7.1 语音模型（本地后端，首次运行自动下载）

- **`ser`** — wav2vec2 `ehcalabres/...`（~1.2GB，RAVDESS 8 类，HuggingFace 缓存）。用**自定义分类头**，`speech_emotion.py` 已做复刻加载（stock 类会得随机头）。
- **`emotion2vec`** — emotion2vec+ `iic/emotion2vec_plus_base`（FunASR/ModelScope，自动下载到 modelscope 缓存）。想换更高精度：`EMO2VEC_MODEL=iic/emotion2vec_plus_large`（1.95G，低内存机会 OOM）。
- 走代理时耐心等；下好后可加 `HF_HUB_OFFLINE=1`（HuggingFace）离线读缓存。

### 7.2 数据集

- **RAVDESS**（对照）：`prepare_ravdess.py` 自动从 Zenodo 下指定演员，无需手动。
- **CREMA-D**（主实验，**必须视频版**）：Kaggle `stefanogiannini/crema-d-video`（含人脸 .flv）或官方 GitHub `CheyneyComputerScience/CREMA-D` 的 `VideoFlash/`。⚠️ **不要**下 `ejlok1/cremad`（只有 AudioWAV，做不了人脸/融合）。解压到 `data/cremad/video/`。
- **eNTERFACE'05**（人脸清晰+语义）：Kaggle `unidpro/video-emotion-recognition-dataset`（1166 段，即 eNTERFACE'05）或官方 <https://enterface.net/enterface05/>（证书较旧，浏览器下）。⚠️ CMU-MOSEI 原计划更贴近部署，但**官方 raw 视频服务器已 502、镜像全是特征非视频**，故改用 eNTERFACE。解压到 `data/enterface/raw/`。

### 7.3 生成标签

```bash
python scripts/prepare_ravdess.py   --actors 01 02 05 --n 50 --out data/ravdess
python scripts/prepare_cremad.py    --src data/cremad/video --n 60 --out data/cremad     # 6 类各 10
python scripts/prepare_enterface.py --src data/enterface/raw --n 60 --out data/enterface  # 6 类各 10
```

### 7.4 跑五臂消融（A/B/C/D/E）+ 逐类可靠性融合（F）

```bash
# 五臂 A–E · 主实验 · CREMA-D · emotion2vec 后端（两模态都跨域）
SPEECH_BACKEND=emotion2vec python scripts/run_ablation.py --data data/cremad --out results/ablation_cremad_e2v --refresh
# 对照后端：SPEECH_BACKEND=ser（wav2vec2） / api（Qwen-Omni）
# 对照数据：--data data/ravdess --out results/ablation
#   --limit 6      先跑 6 段冒烟
#   --fusion-only  只用缓存重算 C/D/E（改融合参数时秒出）

# 臂 F · 逐类可靠性贝叶斯融合（读上面的缓存、LOOCV、不重跑感知）——H1 在此首次成立
python scripts/fusion_bayes.py --data data/cremad --cache results/ablation_cremad_e2v
#   --export models/fusion_bayes_priors.json   额外导出全数据先验，供 fusion.py mode="bayes" 部署加载
```

产物在 `<out>/`：`metrics.json`（五臂 acc/F1/逐类）、`confusion_[A-E].csv`、`cache/predictions.json`；F 直接打印、可选导出 `models/fusion_bayes_priors.json`。

### 7.5 可解释性 + pre 可视化

```bash
python scripts/gradcam_analysis.py --data data/cremad   # 热力图(对/错分组)+focus → results/gradcam
python scripts/viz_fatigue.py      --data data/cremad   # 眼/嘴关键点+EAR/MAR → results/viz_fatigue
# 两者也支持 --images path/to/face.jpg（不依赖数据集）
```

### 7.6 常见坑

- **`SPEECH_BACKEND`**：默认 `api`（读不好语调）。**跑消融务必设 `ser` 或 `emotion2vec`**。
- **`--refresh`**：换后端/改感知后旧缓存不会自动失效，**必须 `--refresh`**（或删 `<out>/cache/`）。
- **`--out` 分开**：不同数据集/后端用不同 `--out`，别互相覆盖。
- **ffmpeg**：没装/不在 PATH → 语音全 fallback。激活 3800 后应自动在 PATH。
- **CREMA-D 必须视频版**；词表仍 7 类，CREMA-D 的 surprise 列 support=0（不影响指标）。
- **emotion2vec 依赖 funasr/modelscope**：已加入 requirements.txt；首次运行联网下载权重。

---

*AIAA 3800 小组项目 — EmotiCompanion — HKUST(GZ) — 2026*
