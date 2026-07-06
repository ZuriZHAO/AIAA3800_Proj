"""
face_emotion.py —— ① 人脸情绪感知 + ⑦ GradCAM 可解释（M1）
============================================================
EmotiCompanion · AIAA 3800 · HKUST(GZ)

本文件实现 app.py 约定的两个模块级函数（接口契约见 README / config.py）：
    predict(image) -> {"emotion": str, "confidence": float}      # ① 人脸情绪
    gradcam(image) -> np.ndarray (RGB 热力图叠加图)               # ⑦ Grad-CAM

────────────────────────────────────────────────────────────────────────
为什么这样选型（汇报时可直接引用）
  · 模型：HSEmotion 的 enet_b0_8_best_vgaf —— EfficientNet-B0，在 AffectNet 上
    训练的 8 类表情模型。相比课堂参考的 DeepFace（FER2013 上的小 CNN，准确率
    约 60–67%），AffectNet 模型精度更高，且骨干是 PyTorch / timm，CPU/GPU 都快。
  · 可解释性：关键在于「⑦ 解释的必须是 ① 用的同一个模型」。HSEmotion 是 PyTorch
    模型，因此 Grad-CAM 直接用 forward/backward hook 挂在它最后一层卷积（bn2）上，
    predict 与 gradcam 共用同一权重——这正是 DeepFace(Keras) 难以做到的（要另接
    tf-keras-vis、且通常是另一套模型）。
  · 检测：用 OpenCV 自带的 Haar 正面人脸级联裁出人脸再送模型，零额外依赖；检测
    不到时退化为整帧输入，保证永不崩。

兼容性说明（teammate 复现要点，已在 requirements/requirements_m1.txt 钉死）
  · HSEmotion 权重是「整模型 pickle」，torch≥2.6 默认 weights_only=True 会拒绝加载，
    故在加载那一刻临时设 weights_only=False（权重来自 HSEmotion 官方仓库，可信）。
  · 该 pickle 用 timm 0.9.x 序列化：timm 太新(1.0+)会在 forward 报 conv_s2d 缺失，
    太旧(0.6)又缺 timm.layers 模块 → 必须 timm==0.9.16。

设计原则（与框架约定一致）
  · 模型只在首次调用时懒加载一次（auto 模式每帧都会调用，禁止反复 from_pretrained）。
  · predict / gradcam 全程 try/except 兜底：app.py 没有在外层包异常处理，这里一旦
    抛错会冲掉整个自动模式的流，所以出错一律回退到中性状态 / 原图。
────────────────────────────────────────────────────────────────────────
"""

import numpy as np

from config import MOCK_EMOTION

# HSEmotion(AffectNet-8) 原生标签 → config.EMOTION_LABELS（Ekman 标准 7 类）的映射。
# 词表是全队契约（neutral/happy/sad/angry/fear/surprise/disgust），两模态都映射进去，
# 融合层 ④ 才能跨模态投票互证。除 Contempt 并入 disgust 外，其余 1:1 对应（几乎无损）。
# 困倦不在情绪轴，由 ② 疲劳检测走 fatigue 字段，人脸不输出它。
AFFECTNET_TO_CONFIG = {
    "Anger":     "angry",
    "Contempt":  "disgust",   # 蔑视并入 disgust（标准 7 类无 contempt）
    "Disgust":   "disgust",
    "Fear":      "fear",
    "Happiness": "happy",
    "Neutral":   "neutral",
    "Sadness":   "sad",
    "Surprise":  "surprise",
}

# ---- 懒加载的全局单例（首次调用时填充）----
_DEVICE = None      # 'cuda' or 'cpu'
_FER = None         # HSEmotionRecognizer
_CASCADE = None     # OpenCV Haar 人脸检测器
_TF = None          # torchvision 预处理（gradcam 手动前向时用）


def _lazy_init():
    """首次调用时加载模型、检测器、预处理；已加载则直接返回。"""
    global _DEVICE, _FER, _CASCADE, _TF
    if _FER is not None:
        return

    import torch
    import cv2
    from torchvision import transforms
    from hsemotion.facial_emotions import HSEmotionRecognizer

    _DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # 临时放开 weights_only 以加载 HSEmotion 的整模型 pickle（可信来源）
    _orig_load = torch.load

    def _patched_load(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return _orig_load(*args, **kwargs)

    torch.load = _patched_load
    try:
        _FER = HSEmotionRecognizer(model_name="enet_b0_8_best_vgaf", device=_DEVICE)
    finally:
        torch.load = _orig_load

    _FER.model.eval()

    _CASCADE = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    # 与 HSEmotion 训练一致：Resize 224 + ImageNet 归一化
    _TF = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


def _to_rgb_uint8(image):
    """把 gradio 传入的图统一成 HxWx3 的 RGB uint8。"""
    import cv2
    rgb = np.asarray(image)
    if rgb.ndim == 2:                                   # 灰度 → RGB
        rgb = cv2.cvtColor(rgb, cv2.COLOR_GRAY2RGB)
    rgb = rgb[..., :3]                                  # 丢掉可能的 alpha 通道
    if rgb.dtype != np.uint8:
        rgb = np.clip(rgb, 0, 255).astype(np.uint8)
    return rgb


def _detect_face_box(rgb):
    """返回最大人脸框 (x, y, w, h)；检测不到返回 None。"""
    import cv2
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    faces = _CASCADE.detectMultiScale(gray, scaleFactor=1.1,
                                      minNeighbors=5, minSize=(60, 60))
    if len(faces) == 0:
        return None
    return max(faces, key=lambda b: int(b[2]) * int(b[3]))


# =============================================================================
# ① 人脸情绪
# =============================================================================

def predict(image):
    """输入 RGB 图（np.ndarray，可能为 None），输出 {"emotion","confidence"}。

    额外返回 raw_emotion / face_detected 便于调试与可解释性展示，下游只读
    emotion / confidence，多余字段不影响融合层。
    """
    if image is None:
        return {"emotion": MOCK_EMOTION, "confidence": 0.0}
    try:
        _lazy_init()
        rgb = _to_rgb_uint8(image)
        box = _detect_face_box(rgb)
        face = rgb[box[1]:box[1] + box[3], box[0]:box[0] + box[2]] if box is not None else rgb

        _label, scores = _FER.predict_emotions(face, logits=False)  # logits=False → softmax 概率
        scores = np.asarray(scores).ravel()
        idx = int(np.argmax(scores))
        raw = _FER.idx_to_class[idx]
        emotion = AFFECTNET_TO_CONFIG.get(raw, "neutral")
        return {
            "emotion": emotion,
            "confidence": round(float(scores[idx]), 4),
            "raw_emotion": raw,
            "face_detected": bool(box is not None),
        }
    except Exception as e:                                # 任何异常都兜底为中性，别冲掉 app 的流
        return {"emotion": MOCK_EMOTION, "confidence": 0.0, "error": str(e)}


# =============================================================================
# ⑦ GradCAM —— 对 ① 用的同一个模型做可解释
# =============================================================================

def gradcam(image):
    """对 predict 所用的同一模型做 Grad-CAM，返回热力图叠加后的 RGB 图。"""
    if image is None:
        return None
    try:
        _lazy_init()
        import torch
        import cv2

        rgb = _to_rgb_uint8(image)
        box = _detect_face_box(rgb)
        face = rgb[box[1]:box[1] + box[3], box[0]:box[0] + box[2]] if box is not None else rgb

        model = _FER.model
        tensor = _TF(face).unsqueeze(0).to(_DEVICE)
        tensor.requires_grad_(True)

        # 在最后一层卷积后的特征图(bn2)上挂前向/反向 hook
        store = {}
        target = model.bn2
        h1 = target.register_forward_hook(
            lambda m, i, o: store.__setitem__("act", o))
        h2 = target.register_full_backward_hook(
            lambda m, gi, go: store.__setitem__("grad", go[0]))
        try:
            logits = model(tensor)                       # (1, 8)
            cls = int(logits.argmax(dim=1).item())
            model.zero_grad(set_to_none=True)
            logits[0, cls].backward()
        finally:
            h1.remove()
            h2.remove()

        acts = store["act"][0]                            # (C, h, w)
        grads = store["grad"][0]                          # (C, h, w)
        weights = grads.mean(dim=(1, 2))                  # (C,) 全局平均池化梯度
        cam = torch.relu((weights[:, None, None] * acts).sum(dim=0))  # (h, w)
        cam = cam.detach().cpu().numpy()
        cam -= cam.min()
        cam /= (cam.max() + 1e-8)                          # 归一化到 0..1

        fh, fw = face.shape[:2]
        cam = cv2.resize(cam, (fw, fh))
        heat = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
        heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)      # JET 是 BGR，转回 RGB
        overlay_face = np.clip(0.5 * face + 0.5 * heat, 0, 255).astype(np.uint8)

        # 把热力图贴回原图的人脸位置，并画框；没检测到脸就整图叠加
        out = rgb.copy()
        if box is not None:
            x, y, w, h = box
            out[y:y + h, x:x + w] = overlay_face
            cv2.rectangle(out, (x, y), (x + w, y + h), (0, 255, 0), 2)
        else:
            out = overlay_face
        return out
    except Exception:
        return image                                      # 出错就返回原图，保证 UI 不崩


# =============================================================================
# 自测（不依赖 app / 其他成员）：python face_emotion.py
# =============================================================================

if __name__ == "__main__":
    print("predict(None) ->", predict(None))
    dummy = (np.random.rand(240, 240, 3) * 255).astype(np.uint8)
    print("predict(dummy) ->", predict(dummy))
    cam = gradcam(dummy)
    print("gradcam(dummy) -> type:", type(cam).__name__,
          "shape:", None if cam is None else np.asarray(cam).shape)
