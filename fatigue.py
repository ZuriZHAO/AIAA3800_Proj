"""
fatigue.py —— ② 疲劳 / 困倦状态检测（M4）
============================================================
EmotiCompanion · AIAA 3800 · HKUST(GZ)

本文件实现 app.py 约定的模块级函数（接口契约见 README / config.py）：
    predict(image) -> {"fatigue_level": str, "confidence": float}   # ② 疲劳检测

其中 fatigue_level ∈ config.FATIGUE_LEVELS = ["low", "medium", "high"]。

────────────────────────────────────────────────────────────────────────
为什么这样选型（汇报 / 报告可直接引用）
  · 任务定位：疲劳/困倦是与「情绪类别」正交的一维（valence–arousal 里的
    唤醒/精力维度，见 experiment_plan §5.1）。人脸①、语音③ 给情绪类别，② 给
    精力状态，二者一起交给 ⑤ 做需求推断——同样"中性"，深夜疲惫时要低唤醒的
    舒缓乐，清醒时适合维持专注的乐。所以 ② 不进情绪投票，只透传 fatigue 字段。
  · 方法：对应课程 L9P1，用「眼动/面部关键点 → 生理疲劳指标」这条经典链路：
      - EAR（Eye Aspect Ratio，眼纵横比）：眼睛闭合时明显下降，是困倦最强信号；
      - MAR（Mouth Aspect Ratio，嘴纵横比）：打哈欠时升高；
      - PERCLOS（一段时间内闭眼帧占比）：疲劳检测的业界金标准，比单帧稳健。
    正好呼应 Lecture 2「human eye movement」——闭眼/眨眼/哈欠都是眼动线索。
  · 关键点检测：首选 MediaPipe FaceMesh（468 点、CPU 友好、laptop-friendly，
    与本项目"语音走 API、人脸走轻量骨干"的低门槛路线一致）。未装 MediaPipe 时
    退化为 OpenCV Haar 眼睛级联的「睁/闭眼」粗判，零额外依赖也能出结果。
  · 时序聚合：predict 在自动模式里每 AUTO_INTERVAL_SEC 秒被调用一次，我们在
    模块内维护一个滚动窗口（deque），用最近若干次的 PERCLOS / 哈欠频率判级，
    避免"某一帧恰好眨眼"就误报 high。experiment 里可传 use_history=False 关掉。

设计原则（与 face_emotion.py / 框架约定一致）
  · 模型只在首次调用时懒加载一次（auto 模式每帧都会调用，禁止反复初始化）。
  · predict 全程 try/except 兜底：app.py 外层不包异常，这里一旦抛错会冲掉整个
    自动模式的流，所以出错一律回退到安全的 low / confidence 0.0（同 mock）。
  · 检测不到人脸 → low / 0.0（无从判断，交给融合层当作"无疲劳信号"）。
────────────────────────────────────────────────────────────────────────
"""

from collections import deque

import numpy as np

from config import FATIGUE_LEVELS, MOCK_FATIGUE

# ---- MediaPipe FaceMesh 关键点索引（468 点网格上的 6 点 EAR + 4 点 MAR）----
# EAR 采用 Soukupová & Čech 的 6 点定义：EAR = (|p2-p6|+|p3-p5|) / (2|p1-p4|)
# 左眼(画面右)：p1..p6
LEFT_EYE = (33, 160, 158, 133, 153, 144)     # (左角, 上1, 上2, 右角, 下2, 下1)
RIGHT_EYE = (362, 385, 387, 263, 373, 380)   # 右眼(画面左)
# 内唇：上13 下14 左角78 右角308；MAR = |13-14| / |78-308|，打哈欠时增大
MOUTH = (13, 14, 78, 308)

# ---- 阈值（可在报告里说明来源；均为 EAR/MAR 常用经验值）----
EAR_CLOSED = 0.18      # 低于此判为「闭眼」
EAR_DROWSY = 0.22      # 低于此判为「眯眼/困倦」（介于睁闭之间）
MAR_YAWN = 0.55        # 高于此判为「打哈欠」

# ---- 时序滚动窗口（自动模式每 ~5s 一帧 → 12 帧约覆盖近 1 分钟）----
WINDOW = 12
_HISTORY = deque(maxlen=WINDOW)   # 每项: {"closed": bool, "yawn": bool, "ear": float}

# ---- 懒加载的全局单例 ----
_BACKEND = None      # "mediapipe" / "haar" / None
_FACE_MESH = None    # mediapipe FaceMesh
_FACE_CASCADE = None # OpenCV Haar 人脸
_EYE_CASCADE = None  # OpenCV Haar 眼睛


def _lazy_init():
    """首次调用时初始化关键点后端：优先 MediaPipe，失败退回 OpenCV Haar。"""
    global _BACKEND, _FACE_MESH, _FACE_CASCADE, _EYE_CASCADE
    if _BACKEND is not None:
        return

    # 首选 MediaPipe FaceMesh
    try:
        import mediapipe as mp
        _FACE_MESH = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=True,     # 逐帧独立处理（帧之间不做跟踪）
            max_num_faces=1,
            refine_landmarks=False,     # 不需要虹膜细化点，468 点足够算 EAR/MAR
            min_detection_confidence=0.5,
        )
        _BACKEND = "mediapipe"
        return
    except Exception:
        _FACE_MESH = None

    # 退化方案：OpenCV Haar 睁/闭眼粗判（无 EAR，只能数眼睛）
    try:
        import cv2
        _FACE_CASCADE = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        _EYE_CASCADE = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_eye.xml")
        _BACKEND = "haar"
    except Exception:
        _BACKEND = None


def _to_rgb_uint8(image):
    """把 gradio 传入的图统一成 HxWx3 的 RGB uint8（与 face_emotion 一致）。"""
    import cv2
    rgb = np.asarray(image)
    if rgb.ndim == 2:                                   # 灰度 → RGB
        rgb = cv2.cvtColor(rgb, cv2.COLOR_GRAY2RGB)
    rgb = rgb[..., :3]                                  # 丢掉可能的 alpha 通道
    if rgb.dtype != np.uint8:
        rgb = np.clip(rgb, 0, 255).astype(np.uint8)
    return rgb


def _dist(a, b):
    return float(np.hypot(a[0] - b[0], a[1] - b[1]))


def _ear(pts):
    """6 点 EAR：(|p2-p6| + |p3-p5|) / (2|p1-p4|)。分母为 0 时返回 0。"""
    p1, p2, p3, p4, p5, p6 = pts
    horiz = _dist(p1, p4)
    if horiz <= 1e-6:
        return 0.0
    return (_dist(p2, p6) + _dist(p3, p5)) / (2.0 * horiz)


def _mar(pts):
    """4 点 MAR：|top-bottom| / |left-right|。分母为 0 时返回 0。"""
    top, bottom, left, right = pts
    width = _dist(left, right)
    if width <= 1e-6:
        return 0.0
    return _dist(top, bottom) / width


# =============================================================================
# 单帧测量：返回 (detected, ear, mar)；detected=False 表示没检测到脸
# =============================================================================

def _measure_mediapipe(rgb):
    """用 FaceMesh 关键点算平均 EAR 与 MAR。"""
    h, w = rgb.shape[:2]
    res = _FACE_MESH.process(rgb)               # 输入需为 RGB uint8
    if not res.multi_face_landmarks:
        return False, 0.0, 0.0
    lm = res.multi_face_landmarks[0].landmark   # 归一化坐标，转像素后算距离

    def px(i):
        return (lm[i].x * w, lm[i].y * h)

    ear = 0.5 * (_ear([px(i) for i in LEFT_EYE]) + _ear([px(i) for i in RIGHT_EYE]))
    mar = _mar([px(i) for i in MOUTH])
    return True, ear, mar


def _measure_haar(rgb):
    """Haar 退化方案：在人脸框里数眼睛，估一个"伪 EAR"。

    检测到 >=1 只睁开的眼 → 记 EAR≈0.30（睁）；人脸在但一只都没检到 → EAR≈0.10
    （多半闭眼）。没有 MAR，哈欠一律记 0。精度有限，仅在没装 MediaPipe 时兜底。
    """
    import cv2
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    faces = _FACE_CASCADE.detectMultiScale(gray, 1.1, 5, minSize=(60, 60))
    if len(faces) == 0:
        return False, 0.0, 0.0
    x, y, fw, fh = max(faces, key=lambda b: int(b[2]) * int(b[3]))
    roi = gray[y:y + fh, x:x + fw]
    eyes = _EYE_CASCADE.detectMultiScale(roi, 1.1, 5, minSize=(20, 20))
    pseudo_ear = 0.30 if len(eyes) >= 1 else 0.10
    return True, pseudo_ear, 0.0


# =============================================================================
# 时序判级：把最近若干帧的 PERCLOS / 哈欠频率映射到 low / medium / high
# =============================================================================

def _classify(history):
    """从滚动窗口聚合出 (level, perclos, yawn_frac, avg_ear)。"""
    n = len(history)
    if n == 0:
        return MOCK_FATIGUE, 0.0, 0.0, 0.0
    perclos = sum(1 for s in history if s["closed"]) / n     # 闭眼帧占比（PERCLOS）
    yawn_frac = sum(1 for s in history if s["yawn"]) / n      # 哈欠帧占比
    avg_ear = sum(s["ear"] for s in history) / n

    if perclos >= 0.50 or yawn_frac >= 0.30:
        level = "high"
    elif perclos >= 0.20 or yawn_frac >= 0.10 or avg_ear < EAR_DROWSY:
        level = "medium"
    else:
        level = "low"
    return level, perclos, yawn_frac, avg_ear


def _confidence(detected, n_window, level):
    """启发式置信度：检测到脸 + 窗口越满越可信；medium 较模糊略降。

    fatigue_conf 主要用于状态展示与 user study（融合层不用它做情绪投票），
    因此这里给一个合理、可复现的数值即可，不追求严格概率。
    """
    if not detected:
        return 0.0
    window_factor = min(1.0, n_window / max(1, WINDOW // 2))  # 半窗填满即到 1.0
    conf = 0.5 + 0.45 * window_factor                         # 单帧≈0.58，满窗≈0.95
    if level == "medium":
        conf -= 0.12                                          # 边界态，稍降
    return round(min(0.95, max(0.30, conf)), 2)


def reset_history():
    """清空时序窗口（切换用户 / 重新开始一段会话 / 单元测试时调用）。"""
    _HISTORY.clear()


# =============================================================================
# ② 疲劳检测 —— app.py 调用的公开接口
# =============================================================================

def predict(image, use_history=True):
    """输入 RGB 图（np.ndarray，可能为 None），输出 {"fatigue_level","confidence"}。

    额外返回 ear / mar / perclos / backend / face_detected 便于调试与报告出图，
    下游（融合层④）只读 fatigue_level / confidence，多余字段不影响。

    参数：
        use_history: True（默认，自动模式用）→ 用最近 WINDOW 帧做 PERCLOS 判级，稳；
                     False（实验/单帧评测用）→ 只看当前这一帧，不写入/不读取窗口。
    """
    if image is None:
        return {"fatigue_level": MOCK_FATIGUE, "confidence": 0.0, "face_detected": False}
    try:
        _lazy_init()
        if _BACKEND is None:                       # 关键点后端都没装好
            return {"fatigue_level": MOCK_FATIGUE, "confidence": 0.0,
                    "face_detected": False, "backend": None}

        rgb = _to_rgb_uint8(image)
        if _BACKEND == "mediapipe":
            detected, ear, mar = _measure_mediapipe(rgb)
        else:
            detected, ear, mar = _measure_haar(rgb)

        if not detected:                           # 没检测到脸 → 无疲劳信号
            return {"fatigue_level": MOCK_FATIGUE, "confidence": 0.0,
                    "face_detected": False, "backend": _BACKEND}

        sample = {"closed": ear < EAR_CLOSED, "yawn": mar > MAR_YAWN, "ear": ear}

        if use_history:
            _HISTORY.append(sample)
            window = _HISTORY
        else:
            window = [sample]                      # 单帧模式：只看这一帧

        level, perclos, yawn_frac, avg_ear = _classify(window)
        conf = _confidence(detected, len(window), level)
        return {
            "fatigue_level": level,
            "confidence": conf,
            "ear": round(ear, 3),
            "mar": round(mar, 3),
            "perclos": round(perclos, 2),
            "yawn_frac": round(yawn_frac, 2),
            "face_detected": True,
            "backend": _BACKEND,
        }
    except Exception as e:                          # 任何异常都兜底为 low，别冲掉 app 的流
        return {"fatigue_level": MOCK_FATIGUE, "confidence": 0.0, "error": str(e)}


# =============================================================================
# 自测（不依赖 app / 其他成员）：python fatigue.py
# =============================================================================

if __name__ == "__main__":
    import json

    print("predict(None) ->", predict(None))

    # 随机噪声图：检测不到人脸，应回退 low / 0.0
    dummy = (np.random.rand(240, 240, 3) * 255).astype(np.uint8)
    print("predict(noise) ->", json.dumps(predict(dummy, use_history=False), ensure_ascii=False))

    # 冒烟测试 _classify 的判级逻辑（不依赖是否装了 mediapipe）
    reset_history()
    closed = [{"closed": True, "yawn": False, "ear": 0.10} for _ in range(WINDOW)]
    open_ = [{"closed": False, "yawn": False, "ear": 0.30} for _ in range(WINDOW)]
    yawns = [{"closed": False, "yawn": True, "ear": 0.28} for _ in range(WINDOW)]
    print("all-closed ->", _classify(closed)[0], "(expect high)")
    print("all-open   ->", _classify(open_)[0], "(expect low)")
    print("all-yawn   ->", _classify(yawns)[0], "(expect high)")
