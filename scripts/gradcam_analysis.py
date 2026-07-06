"""
gradcam_analysis.py —— ⑦ GradCAM 可解释性实验 + 可视化（M1 · 路线 A）
============================================================
EmotiCompanion · AIAA 3800 · HKUST(GZ)

GradCAM 是「解释」而非「预测」，不改变准确率，故不作为准确率消融臂，而是独立的
**可解释性实验**：证明人脸情绪模型①在做判断时，注意力确实落在有表情意义的面部区域，
并对比「预测对 vs 预测错」时注意力的差异（接消融的错误分析），同时产出 pre 用的热力图。

产物（默认 --out results/gradcam）：
    correct/<name>.png     预测正确样本的 [原图 | Grad-CAM] 对照图
    incorrect/<name>.png   预测错误样本的对照图
    all/<name>.png         无 ground truth（--images 模式）时的对照图
    gradcam_metrics.json   聚焦度指标 + 对/错分组均值 + 逐样本明细

聚焦度指标 focus = CAM 能量最高的前 20% 像素占总能量的比例（∈[0,1]，越高越集中）。
假设：预测正确时注意力更集中于面部表情区（focus 更高）。

用法（在 3800 环境、项目根目录下）：
    # 用任意人脸图快速试（现在就能跑，不依赖 RAVDESS）
    python scripts/gradcam_analysis.py --images path/to/face1.jpg path/to/face2.png

    # 有了 RAVDESS 后按 labels.csv 批量（取每段视频中间帧）
    python scripts/gradcam_analysis.py --data data/ravdess --limit 20
"""

import argparse
import csv
import glob
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import numpy as np


# =============================================================================
# 数据源：labels.csv（RAVDESS）或直接给图片
# =============================================================================

def _iter_data(data_dir, limit):
    """从 labels.csv 逐条产出 (name, rgb, gt_emotion)，取每段视频中间帧。"""
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
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, total // 2))  # 中段情绪最明显
        ok, bgr = cap.read()
        cap.release()
        if not ok or bgr is None:
            continue
        name = os.path.splitext(os.path.basename(row["path"]))[0]
        yield name, cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), row["emotion"]


def _iter_images(paths, limit):
    """直接读图片文件，逐条产出 (name, rgb, None)。gt 未知。"""
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
        name = os.path.splitext(os.path.basename(p))[0]
        yield name, cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), None


# =============================================================================
# 指标与出图
# =============================================================================

def cam_focus(cam):
    """CAM 聚焦度：能量最高的前 20% 像素占总能量比例（越高越集中）。"""
    flat = np.sort(cam.ravel())[::-1]
    k = max(1, int(0.20 * flat.size))
    return float(flat[:k].sum() / (flat.sum() + 1e-8))


def _save_panel(out_path, face, cam, title):
    """保存 [原始人脸 | Grad-CAM 叠加] 对照图。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import cv2

    heat = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)[:, :, ::-1] / 255.0
    overlay = np.clip(0.5 * face / 255.0 + 0.5 * heat, 0, 1)

    fig, ax = plt.subplots(1, 2, figsize=(6, 3.2))
    ax[0].imshow(face); ax[0].set_title("input face", fontsize=9); ax[0].axis("off")
    ax[1].imshow(overlay); ax[1].set_title("Grad-CAM", fontsize=9); ax[1].axis("off")
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


# =============================================================================
# 主流程
# =============================================================================

def main():
    ap = argparse.ArgumentParser(description="⑦ GradCAM 可解释性实验 + 可视化")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--data", help="含 labels.csv 的 RAVDESS 目录（每段取中间帧）")
    src.add_argument("--images", nargs="+", help="直接给人脸图片路径/通配符（无 gt）")
    ap.add_argument("--out", default="results/gradcam", help="输出目录")
    ap.add_argument("--limit", type=int, default=0, help="只处理前 N 个（0=全部）")
    args = ap.parse_args()

    import face_emotion as fe

    source = _iter_data(args.data, args.limit) if args.data else _iter_images(args.images, args.limit)

    records = []
    n_correct = n_wrong = n_noface = 0
    for name, rgb, gt in source:
        info = fe.cam_map(rgb)
        if info is None or info["box"] is None:
            n_noface += 1
            print(f"[skip] {name}: 未检测到人脸")
            continue

        pred = info["emotion"]
        focus = cam_focus(info["cam"])
        rec = {"name": name, "pred": pred, "gt": gt,
               "confidence": info["confidence"], "focus": round(focus, 4)}

        if gt is None:
            sub, tag = "all", f"pred={pred} conf={info['confidence']:.2f} focus={focus:.2f}"
        else:
            correct = (pred == gt)
            rec["correct"] = correct
            n_correct += int(correct); n_wrong += int(not correct)
            sub = "correct" if correct else "incorrect"
            mark = "OK" if correct else "X"
            tag = f"[{mark}] pred={pred} gt={gt} conf={info['confidence']:.2f} focus={focus:.2f}"

        _save_panel(os.path.join(args.out, sub, f"{name}.png"), info["face"], info["cam"], tag)
        records.append(rec)
        print(f"[{len(records)}] {name}: {tag}")

    # 聚合
    def _mean(xs):
        return round(sum(xs) / len(xs), 4) if xs else None

    summary = {
        "n_samples": len(records),
        "n_no_face": n_noface,
        "focus_mean_all": _mean([r["focus"] for r in records]),
    }
    graded = [r for r in records if "correct" in r]
    if graded:
        summary["n_correct"] = n_correct
        summary["n_incorrect"] = n_wrong
        summary["accuracy"] = round(n_correct / len(graded), 4)
        summary["focus_mean_correct"] = _mean([r["focus"] for r in graded if r["correct"]])
        summary["focus_mean_incorrect"] = _mean([r["focus"] for r in graded if not r["correct"]])

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "gradcam_metrics.json"), "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "records": records}, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 56)
    print("GradCAM 可解释性小结")
    print("=" * 56)
    for k, v in summary.items():
        print(f"  {k:22} {v}")
    print(f"\n对照图 -> {args.out}/(correct|incorrect|all)/")
    print(f"指标   -> {args.out}/gradcam_metrics.json")
    if graded and summary.get("focus_mean_correct") and summary.get("focus_mean_incorrect"):
        print("\n若 focus_mean_correct > focus_mean_incorrect，说明预测正确时注意力更集中于面部表情区。")


if __name__ == "__main__":
    main()
