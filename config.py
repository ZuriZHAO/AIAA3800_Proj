"""
config.py —— EmotiCompanion 全局配置与常量
================================================
集中存放所有模块共享的常量与 schema，供 app.py / mocks.py / 各成员模块 import。
这样常量只在一处维护，不散落在 app.py 里。

用法：
    from config import EMOTION_LABELS, FATIGUE_LEVELS, FUSION_SCHEMA_EXAMPLE, AUTO_INTERVAL_SEC
"""

# 全队统一的情绪「契约词表」—— Ekman 标准 7 类基本情绪。
# 人脸(①HSEmotion/AffectNet-8) 与 语音(③Qwen-Omni) 的原生输出都映射进这一套，
# 这样融合层 ④ 的按情绪投票才能跨模态互相印证。各感知模块负责把自己的原生标签
# 映射到这里（人脸映射见 face_emotion.py，语音映射由 M2 在 speech_emotion.py 实现）。
# 注意：疲劳/困倦是独立的一维，见下面 FATIGUE_LEVELS，不要混进情绪词表。
EMOTION_LABELS = ["neutral", "happy", "sad", "angry", "fear", "surprise", "disgust"]

# 疲劳/压力等级
FATIGUE_LEVELS = ["low", "medium", "high"]

# 融合输出的标准格式示例（来自规划文档 §2.1），所有成员对齐此 schema
FUSION_SCHEMA_EXAMPLE = {
    "dominant_emotion": "fear",
    "confidence": 0.74,
    "fatigue": "high",
    "face_conf": 0.78,
    "speech_conf": 0.61,
    "fatigue_conf": 0.85,
}

# 自动模式：每隔多少秒捕捉一次照片 / 处理一段录音
AUTO_INTERVAL_SEC = 5

# 被「舒缓类」音乐对待的负面/低效价情绪（用于 mock LLM 决策，真实模型可忽略）。
# surprise 唤醒度可正可负，不一概归入舒缓；困倦由 fatigue=="high" 单独触发。
SOOTHE_EMOTIONS = ("sad", "angry", "fear", "disgust")

# ---- mock 占位行为（真实模块未就绪时的固定回退，不使用随机）----
# 真实模块没写好时，感知一律返回下面这个固定的「舒缓」状态，
# LLM 一律给舒缓音乐，音乐生成则直接留空白（不乱生成占位音符）。
MOCK_EMOTION = "neutral"   # mock 感知固定返回的情绪
MOCK_FATIGUE = "low"       # mock 感知固定返回的疲劳/压力等级
