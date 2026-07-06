"""
run_ablation.py —— 感知层消融实验评测脚本（M4）
============================================================
EmotiCompanion · AIAA 3800 · HKUST(GZ)

按 experiment_plan.md 跑「感知层」四个实验臂，比较情绪识别表现：
    A. 仅人脸 ①
    B. 仅语音 ③
    C. 人脸+语音 · 朴素融合 ④(mode="naive")
    D. 人脸+语音 · 置信度加权融合 ④(mode="weighted")   ← 主角
并对每个臂输出 Accuracy / macro-F1 / 逐类混淆矩阵（对应 H1/H2/H3）。
疲劳分支 ② 不在定量消融范围内（见 experiment_plan §5.1），本脚本不评它。

数据来源：scripts/prepare_ravdess.py 生成的 <data>/labels.csv
    列： path, emotion, actor, intensity     （path 相对 <data> 目录，emotion 为 7 类契约词表）

流程（每段视频一次）：
    人脸①：cv2 均匀抽 K 帧 → face_emotion.predict 逐帧 → 置信度加权聚合成一个人脸结果
    语音③：audio_extract.extract_audio 抽 wav → speech_emotion.predict
    A/B 直接取单模态；C/D 交给 fusion.fuse 的两种模式。
每段的人脸/语音原始预测会缓存到 <out>/cache/predictions.json，
重跑时默认命中缓存（语音 API 昂贵、人脸抽帧慢），改 --refresh 可强制重算。

用法（在 3800 环境、项目根目录下）：
    python scripts/run_ablation.py                          # 默认 data/ravdess，抽 8 帧
    python scripts/run_ablation.py --data data/ravdess --frames 8 --out results/ablation
    python scripts/run_ablation.py --limit 6                # 只跑前 6 段，快速冒烟
    python scripts/run_ablation.py --fusion-only            # 只用缓存重算 C/D（不重跑感知）

产物：
    <out>/cache/predictions.json     每段视频的人脸/语音原始预测（缓存）
    <out>/metrics.json               四臂的 accuracy / macro-F1 / 逐类 P,R,F1
    <out>/confusion_<arm>.csv        四臂各一张混淆矩阵
"""

import argparse
import csv
import json
import os
import sys

# 让脚本无论从项目根还是 scripts/ 下运行，都能 import 到根目录的成员模块
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from config import EMOTION_LABELS


# =============================================================================
# 真实模块加载（实验脚本必须用真实实现，不能拿 mock 出实验结论）
# =============================================================================

def _load_real_modules():
    """import 四个真实模块；缺哪个就报错说明，避免用 mock 跑出无意义的数字。"""
    missing = []
    mods = {}
    for name in ("face_emotion", "speech_emotion", "audio_extract", "fusion"):
        try:
            mods[name] = __import__(name)
        except Exception as e:
            missing.append(f"  - {name}.py: {e}")
    if missing:
        print("[FATAL] 消融实验需要真实模块，但以下模块无法导入：", file=sys.stderr)
        print("\n".join(missing), file=sys.stderr)
        print("请先装好各自依赖（requirements_m1/m2/m3/m4.txt）并配置 .env 后再跑。",
              file=sys.stderr)
        sys.exit(1)
    return mods


# =============================================================================
# 人脸：均匀抽帧 → 逐帧 predict → 置信度加权聚合成单个结果
# =============================================================================

def _sample_frame_indices(total, k):
    """在 [0, total) 里均匀取 k 个下标（total 不足则取全部）。"""
    if total <= 0:
        return []
    if total <= k:
        return list(range(total))
    step = total / float(k)
    return [min(total - 1, int(i * step + step / 2)) for i in range(k)]


def _face_predict_video(face_mod, video_path, k):
    """抽 k 帧跑人脸情绪，按置信度加权投票聚合成一个 {emotion, confidence, reliability}。

    reliability = 各帧 GradCAM 人脸可靠性的均值（路线B · arm E 用），无 cam_reliability
    实现时退回 1.0（=不门控，等价于 arm D）。
    """
    import cv2

    has_rel = hasattr(face_mod, "cam_reliability")
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    idxs = _sample_frame_indices(total, k)

    votes, conf_sum, n_used, rels = {}, 0.0, 0, []
    for idx in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame_bgr = cap.read()
        if not ok or frame_bgr is None:
            continue
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        r = face_mod.predict(rgb)
        emo = r.get("emotion", "neutral")
        c = float(r.get("confidence", 0.0) or 0.0)
        votes[emo] = votes.get(emo, 0.0) + c
        conf_sum += c
        n_used += 1
        if has_rel:
            rels.append(float(face_mod.cam_reliability(rgb)))   # 含无脸帧的 0，会拉低均值
    cap.release()

    reliability = round(sum(rels) / len(rels), 4) if rels else 1.0
    if not votes or conf_sum <= 0.0:
        return {"emotion": "neutral", "confidence": 0.0,
                "reliability": reliability, "frames_used": n_used}
    emotion = max(votes, key=votes.get)
    confidence = round(votes[emotion] / conf_sum, 4)   # 主导情绪占总置信度的比例
    return {"emotion": emotion, "confidence": confidence,
            "reliability": reliability, "frames_used": n_used}


# =============================================================================
# 每段视频的原始感知（带缓存）
# =============================================================================

def _perceive_video(mods, data_root, rel_path, k):
    """返回 {"face": {...}, "speech": {...}}（人脸抽帧聚合 + 语音一次）。"""
    video_path = os.path.join(data_root, rel_path)
    face = _face_predict_video(mods["face_emotion"], video_path, k)
    try:
        audio_path = mods["audio_extract"].extract_audio(video_path)
        speech = mods["speech_emotion"].predict(audio_path)
    except Exception as e:
        speech = {"emotion": "neutral", "confidence": 0.0, "reasoning": f"speech failed: {e}"}
    return {"face": face, "speech": speech}


def _load_cache(cache_path):
    if os.path.exists(cache_path):
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def _save_cache(cache_path, cache):
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)


# =============================================================================
# 四个实验臂：从一段视频的 (face, speech) 得到各臂的预测标签
# =============================================================================

def _arm_predictions(fusion_mod, face, speech):
    """返回 {"A"..."E": emo}。融合层不用疲劳，传 None。
    E = 置信度加权 + GradCAM 人脸可靠性门控（路线B），读 face["reliability"]。
    """
    c = fusion_mod.fuse(face, speech, None, mode="naive")
    d = fusion_mod.fuse(face, speech, None, mode="weighted")
    e = fusion_mod.fuse(face, speech, None, mode="weighted_cam")
    return {
        "A": face.get("emotion", "neutral"),
        "B": speech.get("emotion", "neutral"),
        "C": c.get("dominant_emotion", "neutral"),
        "D": d.get("dominant_emotion", "neutral"),
        "E": e.get("dominant_emotion", "neutral"),
    }


# =============================================================================
# 指标：accuracy / macro-F1 / 逐类 P,R,F1 / 混淆矩阵（手写，不引入 sklearn）
# =============================================================================

def _confusion(y_true, y_pred, labels):
    """labels×labels 混淆矩阵，cm[t][p] = 真值 t 被预测成 p 的次数。"""
    index = {lab: i for i, lab in enumerate(labels)}
    cm = [[0] * len(labels) for _ in labels]
    for t, p in zip(y_true, y_pred):
        if t in index and p in index:
            cm[index[t]][index[p]] += 1
    return cm


def _metrics(y_true, y_pred, labels):
    """整体 accuracy、逐类 precision/recall/f1、macro-F1。"""
    n = len(y_true)
    correct = sum(1 for t, p in zip(y_true, y_pred) if t == p)
    accuracy = correct / n if n else 0.0

    cm = _confusion(y_true, y_pred, labels)
    per_class, f1s = {}, []
    for i, lab in enumerate(labels):
        tp = cm[i][i]
        fp = sum(cm[r][i] for r in range(len(labels))) - tp
        fn = sum(cm[i]) - tp
        support = sum(cm[i])
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        # 只把「有真值样本的类」计入 macro-F1，避免空类把均值拉低
        if support > 0:
            f1s.append(f1)
        per_class[lab] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "support": support,
        }
    macro_f1 = sum(f1s) / len(f1s) if f1s else 0.0
    return {
        "accuracy": round(accuracy, 4),
        "macro_f1": round(macro_f1, 4),
        "n": n,
        "per_class": per_class,
        "confusion": cm,
    }


def _save_confusion_csv(path, cm, labels):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["true\\pred"] + labels)
        for i, lab in enumerate(labels):
            w.writerow([lab] + cm[i])


def _print_confusion(cm, labels):
    head = "true\\pred".ljust(9) + "".join(f"{l[:4]:>6}" for l in labels)
    print(head)
    for i, lab in enumerate(labels):
        print(lab[:9].ljust(9) + "".join(f"{cm[i][j]:>6}" for j in range(len(labels))))


# =============================================================================
# 主流程
# =============================================================================

def main():
    ap = argparse.ArgumentParser(description="EmotiCompanion 感知层四臂消融评测")
    ap.add_argument("--data", default="data/ravdess", help="含 labels.csv 的数据目录")
    ap.add_argument("--out", default="results/ablation", help="结果输出目录")
    ap.add_argument("--frames", type=int, default=8, help="每段视频抽多少帧喂人脸模型")
    ap.add_argument("--limit", type=int, default=0, help="只跑前 N 段（0=全部），用于冒烟测试")
    ap.add_argument("--refresh", action="store_true", help="忽略缓存，强制重跑感知")
    ap.add_argument("--fusion-only", action="store_true",
                    help="只用缓存里的人脸/语音结果重算 C/D，不重跑感知（无缓存则报错）")
    args = ap.parse_args()

    labels_csv = os.path.join(args.data, "labels.csv")
    if not os.path.exists(labels_csv):
        print(f"[FATAL] 找不到 {labels_csv}，请先运行 scripts/prepare_ravdess.py 生成。",
              file=sys.stderr)
        sys.exit(1)

    with open(labels_csv, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if args.limit > 0:
        rows = rows[:args.limit]
    print(f"共 {len(rows)} 段样本，标签词表 = {EMOTION_LABELS}")

    os.makedirs(args.out, exist_ok=True)
    cache_path = os.path.join(args.out, "cache", "predictions.json")
    cache = _load_cache(cache_path)

    # fusion-only 模式不需要人脸/语音/音频模块，只需 fusion
    if args.fusion_only:
        import fusion as fusion_mod
        mods = {"fusion": fusion_mod}
    else:
        mods = _load_real_modules()

    y_true = []
    preds = {"A": [], "B": [], "C": [], "D": [], "E": []}

    for i, row in enumerate(rows, 1):
        rel_path, gt = row["path"], row["emotion"]
        if gt not in EMOTION_LABELS:
            print(f"[skip] {rel_path}: 标签 {gt} 不在契约词表内")
            continue

        # 取感知结果：缓存命中直接用；否则跑真实模块（fusion-only 下必须有缓存）
        if not args.refresh and rel_path in cache:
            raw = cache[rel_path]
        elif args.fusion_only:
            print(f"[skip] {rel_path}: --fusion-only 但缓存缺失，跳过")
            continue
        else:
            print(f"[{i}/{len(rows)}] perceiving {rel_path} ...")
            raw = _perceive_video(mods, args.data, rel_path, args.frames)
            cache[rel_path] = raw
            _save_cache(cache_path, cache)   # 边跑边存，中断也不丢已算的（省 API 费）

        arm_pred = _arm_predictions(mods["fusion"], raw["face"], raw["speech"])
        y_true.append(gt)
        for arm in preds:
            preds[arm].append(arm_pred[arm])

    if not y_true:
        print("[FATAL] 没有可评测的样本（缓存为空且未跑感知？）", file=sys.stderr)
        sys.exit(1)

    # 算四臂指标并输出
    arm_names = {
        "A": "A. face only",
        "B": "B. speech only",
        "C": "C. naive fusion",
        "D": "D. weighted fusion",
        "E": "E. weighted + CAM-gate",
    }
    all_metrics = {}
    print("\n" + "=" * 60)
    print(f"Ablation results on {len(y_true)} samples")
    print("=" * 60)
    print(f"{'arm':<24}{'accuracy':>10}{'macro-F1':>10}")
    for arm in ("A", "B", "C", "D", "E"):
        m = _metrics(y_true, preds[arm], EMOTION_LABELS)
        all_metrics[arm] = {"name": arm_names[arm], **m}
        print(f"{arm_names[arm]:<24}{m['accuracy']:>10.4f}{m['macro_f1']:>10.4f}")
        _save_confusion_csv(
            os.path.join(args.out, f"confusion_{arm}.csv"), m["confusion"], EMOTION_LABELS)

    # H3 互补性证据：打印 A、B 两个单模态臂的混淆矩阵
    for arm in ("A", "B"):
        print(f"\n[{arm_names[arm]}] confusion matrix (H3 complementarity):")
        _print_confusion(all_metrics[arm]["confusion"], EMOTION_LABELS)

    metrics_path = os.path.join(args.out, "metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(all_metrics, f, ensure_ascii=False, indent=2)

    print("\n已保存：")
    print(f"  指标      -> {metrics_path}")
    print(f"  混淆矩阵  -> {os.path.join(args.out, 'confusion_[A-E].csv')}")
    print(f"  感知缓存  -> {cache_path}")
    print("\n结论对照 experiment_plan：H1 看 D/C vs A/B，H2 看 D vs C，"
          "H3 看 A/B 混淆矩阵互补，路线B 看 E vs D（CAM 门控是否有用）。")


if __name__ == "__main__":
    main()
