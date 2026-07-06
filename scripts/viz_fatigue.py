"""
viz_fatigue.py —— ② 疲劳检测的关键点可视化（pre 素材）
============================================================
EmotiCompanion · AIAA 3800 · HKUST(GZ)

把疲劳检测②实际用到的**面部关键点画在脸上**，直观展示「疲劳识别对应人脸哪些点」：
    · 眼睛 6 点（算 EAR，眼纵横比 → 闭眼下降）—— 绿色
    · 嘴巴 4 点（算 MAR，嘴纵横比 → 打哈欠升高）—— 品红
并标注该帧的 EAR / MAR / 疲劳等级。关键点索引与阈值直接从 fatigue.py 取，保证与
真实检测一致。

产物（默认 --out results/viz_fatigue）：
    <name>.png   画好关键点 + 标注的对照图

用法（3800 环境、项目根目录）：
    # 任意人脸图（现在就能跑）
    python scripts/viz_fatigue.py --images path/to/face.jpg
    # 有 RAVDESS 后批量取中间帧
    python scripts/viz_fatigue.py --data data/ravdess --limit 12
"""

import argparse
import csv
import glob
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import numpy as np

import fatigue as fa
from fatigue import LEFT_EYE, RIGHT_EYE, MOUTH


# =============================================================================
# 关键点后端（自建 FaceLandmarker，只为拿到像素坐标画点）
# =============================================================================

_LANDMARKER = None


def _get_landmarker():
    global _LANDMARKER
    if _LANDMARKER is not None:
        return _LANDMARKER
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision
    model_path = os.path.join(_ROOT, "models", "face_landmarker.task")
    _LANDMARKER = vision.FaceLandmarker.create_from_options(
        vision.FaceLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=model_path),
            num_faces=1,
            running_mode=vision.RunningMode.IMAGE,
        ))
    return _LANDMARKER


def _landmark_px(rgb):
    """返回 {idx: (x_px, y_px)}（只含 EAR/MAR 用到的点）；无脸返回 None。"""
    import mediapipe as mp
    h, w = rgb.shape[:2]
    res = _get_landmarker().detect(
        mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb)))
    if not res.face_landmarks:
        return None
    lm = res.face_landmarks[0]
    idxs = set(LEFT_EYE) | set(RIGHT_EYE) | set(MOUTH)
    return {i: (int(lm[i].x * w), int(lm[i].y * h)) for i in idxs}


# =============================================================================
# 出图
# =============================================================================

def _draw_and_save(out_path, rgb, pts, ear, mar, level):
    import cv2
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    img = rgb.copy()
    # 按人脸大小缩放点径/线宽，保证高分辨率图上也看得清
    xs = [p[0] for p in pts.values()]
    ys = [p[1] for p in pts.values()]
    face_w = max(1, max(xs) - min(xs))
    r = max(2, int(face_w * 0.03))
    t = max(1, r // 2)

    # 眼睛点（绿）+ 连线成环
    for eye in (LEFT_EYE, RIGHT_EYE):
        poly = [pts[i] for i in eye]
        for p in poly:
            cv2.circle(img, p, r, (0, 255, 0), -1)
        cv2.polylines(img, [np.array(poly, np.int32)], True, (0, 200, 0), t)
    # 嘴巴点（品红）
    for i in MOUTH:
        cv2.circle(img, pts[i], r, (255, 0, 255), -1)
    cv2.line(img, pts[MOUTH[0]], pts[MOUTH[1]], (255, 0, 255), t)  # 上下唇（MAR 分子）
    cv2.line(img, pts[MOUTH[2]], pts[MOUTH[3]], (255, 0, 255), t)  # 左右嘴角（MAR 分母）

    # 裁剪到人脸区域（关键点外扩 45% 边距），让标注的脸填满画面
    h, w = img.shape[:2]
    mx = int(face_w * 0.45)
    my = int((max(ys) - min(ys)) * 0.45)
    x0, x1 = max(0, min(xs) - mx), min(w, max(xs) + mx)
    y0, y1 = max(0, min(ys) - my), min(h, max(ys) + my)
    crop = img[y0:y1, x0:x1]

    fig, ax = plt.subplots(figsize=(4.2, 4.6))
    ax.imshow(crop)
    ax.axis("off")
    ax.set_title(f"EAR={ear:.3f}  MAR={mar:.3f}  →  fatigue: {level}\n"
                 "green = eye pts (EAR) · magenta = mouth pts (MAR)", fontsize=9)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


# =============================================================================
# 数据源
# =============================================================================

def _iter_data(data_dir, limit):
    import cv2
    labels = os.path.join(data_dir, "labels.csv")
    if not os.path.exists(labels):
        print(f"[FATAL] 找不到 {labels}，请先运行 scripts/prepare_ravdess.py。", file=sys.stderr)
        sys.exit(1)
    with open(labels, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if limit > 0:
        rows = rows[:limit]
    for row in rows:
        path = os.path.join(data_dir, row["path"])
        cap = cv2.VideoCapture(path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, total // 2))
        ok, bgr = cap.read()
        cap.release()
        if ok and bgr is not None:
            name = os.path.splitext(os.path.basename(row["path"]))[0]
            yield name, cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def _iter_images(paths, limit):
    import cv2
    expanded = []
    for p in paths:
        expanded.extend(sorted(glob.glob(p)) if any(c in p for c in "*?[") else [p])
    if limit > 0:
        expanded = expanded[:limit]
    for p in expanded:
        bgr = cv2.imread(p)
        if bgr is None:
            print(f"[skip] 读不了图片: {p}")
            continue
        yield os.path.splitext(os.path.basename(p))[0], cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


# =============================================================================
# 主流程
# =============================================================================

def main():
    ap = argparse.ArgumentParser(description="② 疲劳检测关键点可视化")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--data", help="含 labels.csv 的 RAVDESS 目录")
    src.add_argument("--images", nargs="+", help="人脸图片路径/通配符")
    ap.add_argument("--out", default="results/viz_fatigue", help="输出目录")
    ap.add_argument("--limit", type=int, default=0, help="只处理前 N 个（0=全部）")
    args = ap.parse_args()

    source = _iter_data(args.data, args.limit) if args.data else _iter_images(args.images, args.limit)

    n = 0
    for name, rgb in source:
        pts = _landmark_px(rgb)
        info = fa.predict(rgb, use_history=False)          # 复用真实 EAR/MAR/等级
        if pts is None or info.get("backend") != "mediapipe" or "ear" not in info:
            print(f"[skip] {name}: 未检测到人脸 / mediapipe 不可用（backend={info.get('backend')}）")
            continue
        _draw_and_save(os.path.join(args.out, f"{name}.png"),
                       rgb, pts, info["ear"], info["mar"], info["fatigue_level"])
        n += 1
        print(f"[{n}] {name}: EAR={info['ear']:.3f} MAR={info['mar']:.3f} -> {info['fatigue_level']}")

    print(f"\n共出图 {n} 张 -> {args.out}/")
    if n == 0:
        print("提示：没成功出图。确认 mediapipe 已装、models/face_landmarker.task 存在、图里有正面人脸。")


if __name__ == "__main__":
    main()
