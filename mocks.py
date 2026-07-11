"""
mocks.py —— 各模块的 mock（占位）实现
================================================
当某成员的真实模块文件尚未就绪时，app.py 自动回退到这里的 mock，
保证整个框架任何时候都能端到端跑通（阶段一「mock 数据可用」）。

设计原则（与真实模型行为解耦）：
  - 感知类 mock 不使用随机，一律返回固定的「舒缓」状态（见 config.MOCK_*），
    这样在真实模块没写好时，画面稳定、不会乱跳情绪。
  - 音乐生成 mock 不再合成占位音符：真实音乐模块没写好时直接返回空白（None），
    UI 上音乐栏为空，避免误导。

每个 mock 类的函数名/签名/返回格式都与真实模块的接口契约一致，
因此 app.py 调用时无需关心拿到的是真实模块还是 mock。
"""

from config import MOCK_EMOTION, MOCK_FATIGUE, SOOTHE_EMOTIONS


class MockFace:
    """① 人脸情绪 + ⑦ GradCAM 的 mock（对应 face_emotion.py）"""

    @staticmethod
    def predict(image):
        # 固定返回舒缓/中性情绪，不随机
        return {"emotion": MOCK_EMOTION, "confidence": 0.0}

    @staticmethod
    def gradcam(image):
        # mock：直接返回原图占位；真实实现返回叠加热力图后的图
        return image


class MockFatigue:
    """② 疲劳检测的 mock（对应 fatigue.py）"""

    @staticmethod
    def predict(image):
        # 固定返回低压力，不随机
        return {"fatigue_level": MOCK_FATIGUE, "confidence": 0.0}


class MockSpeech:
    """③ 语音情绪的 mock（对应 speech_emotion.py）"""

    @staticmethod
    def predict(audio_path):
        # 固定返回舒缓/中性情绪，不随机
        return {"emotion": MOCK_EMOTION, "confidence": 0.0,
                "reasoning": "(mock) speech module not ready, returning calm/neutral."}


class MockFusion:
    """④ 多模态融合的 mock（对应 fusion.py）

    置信度加权投票：人脸/语音按各自 confidence 给候选情绪累加权重，
    取权重最高者为 dominant_emotion；疲劳作为独立维度透传。
    （感知 mock 的 confidence 为 0 时，这里会回退到固定的舒缓状态。）
    """

    @staticmethod
    def fuse(face, speech, fatigue):
        fc, sc, gc = (float(face.get("confidence", 0.0)),
                      float(speech.get("confidence", 0.0)),
                      float(fatigue.get("confidence", 0.0)))
        votes = {}
        votes[face["emotion"]] = votes.get(face["emotion"], 0.0) + fc
        votes[speech["emotion"]] = votes.get(speech["emotion"], 0.0) + sc

        if not votes or max(votes.values()) == 0.0:
            # 没有有效置信度（mock 状态）→ 固定舒缓/中性
            dominant, dom_w = MOCK_EMOTION, 0.0
        else:
            dominant = max(votes, key=votes.get)
            dom_w = votes[dominant]

        total = fc + sc
        conf = round(dom_w / total, 2) if total > 0 else 0.0
        return {
            "dominant_emotion": dominant,
            "confidence": conf,
            "fatigue": fatigue.get("fatigue_level", MOCK_FATIGUE),
            "face_conf": round(fc, 2),
            "speech_conf": round(sc, 2),
            "fatigue_conf": round(gc, 2),
        }


class MockLLM:
    """⑤ LLM 需求推理的 mock（对应 llm_reason.py）"""

    @staticmethod
    def infer(state):
        emo = state.get("dominant_emotion", MOCK_EMOTION)
        fatigue = state.get("fatigue", MOCK_FATIGUE)
        soothe = emo in SOOTHE_EMOTIONS or fatigue == "high"
        reasoning = (
            f"(mock CoT) User's dominant emotion is {emo}, fatigue level {fatigue}. "
            f"With Theory-of-Mind reasoning: the user likely needs "
            f"{'calming, low-arousal' if soothe else 'uplifting / state-sustaining'} "
            f"music companionship."
        )
        music_spec = (
            f"a {'calm, soft, low-arousal' if soothe else 'warm, uplifting'} "
            f"instrumental piece, gentle tempo, suited for a {emo} mood"
        )
        return {"need": f"regulate {emo}", "reasoning": reasoning, "music_spec": music_spec}


class MockMusic:
    """⑥ 音乐生成的 mock（对应 music_gen.py）"""

    @staticmethod
    def generate(music_spec, duration_sec=None):
        # 真实音乐模块未就绪时不乱生成占位音符，直接返回空白（None）。
        # UI 的音频栏会保持为空，不会播放任何声音。
        # 保留 duration_sec 参数只为与真实 generate() 的签名对齐（mock 忽略它）。
        return None
