"""
eval_speech.py —— 只评「语音单模态」的情绪识别(轻量、低内存)
============================================================
EmotiCompanion · AIAA 3800 · HKUST(GZ)

背景：run_ablation.py 每段都要抽 8 帧跑人脸 + Grad-CAM，人脸(hsemotion)叠加
语音后端(尤其 ser=wav2vec2 会拉起 TensorFlow)在低内存机器上会 OOM / 段错误。
但要对比「emotion2vec vs wav2vec2」其实只需要 run_ablation 的 B 臂(speech only)。
本脚本因此**只跑语音**：不加载人脸、不做 Grad-CAM、内存占用小，专门给语音后端打分。

评的就是 speech_emotion.predict()，用哪个后端由环境变量 SPEECH_BACKEND 决定：
    SPEECH_BACKEND=emotion2vec  → emotion2vec+
    SPEECH_BACKEND=ser          → wav2vec2 (ehcalabres)
    SPEECH_BACKEND=api          → Qwen-Omni 云 API

⚠️ Windows 上必须在**启动 python 之前**在 shell 里设好防 OpenMP 段错误的环境变量
（os.environ 在脚本里设太晚，OpenMP DLL 已加载）。用 run_emo2vec.ps1 会自动设好；
手动跑则先执行：
    $env:KMP_DUPLICATE_LIB_OK='TRUE'; $env:OMP_NUM_THREADS='1'; $env:MKL_NUM_THREADS='1'

用法(在项目根目录，用装了依赖的解释器)：
    $env:SPEECH_BACKEND='emotion2vec'
    D:\\python3.12\\python.exe scripts\\eval_speech.py --data data\\ravdess --limit 6
    D:\\python3.12\\python.exe scripts\\eval_speech.py --data data\\ravdess          # 全量

产物：
    <out>/speech_<backend>.json   accuracy / macro-F1 / 逐类 P,R,F1 / 混淆矩阵 / 逐条预测
"""

import argparse
import csv
import json
import os
import sys

# Windows 上多份 OpenMP 运行时冲突会段错误(见 run_ablation.py 顶部详解)。
# KMP_DUPLICATE_LIB_OK 必须在 OpenMP DLL 加载前就位，脚本里设太晚 → 用「设好环境后
# 自我重启一次」的守卫，让用户直接 `python scripts/eval_speech.py ...` 也不崩、无需先设变量。
if os.environ.get("_EVAL_SPEECH_REEXEC") != "1":
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["USE_TF"] = "0"                      # 禁 transformers 的 TF 后端(ser 用)
    os.environ["USE_TORCH"] = "1"
    os.environ.setdefault("TQDM_DISABLE", "1")
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    os.environ["_EVAL_SPEECH_REEXEC"] = "1"
    import subprocess
    sys.exit(subprocess.run([sys.executable] + sys.argv, env=os.environ).returncode)

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from config import EMOTION_LABELS


def _confusion(y_true, y_pred, labels):
    """labels×labels 混淆矩阵，cm[t][p] = 真值 t 被预测成 p 的次数。"""
    index = {lab: i for i, lab in enumerate(labels)}
    cm = [[0] * len(labels) for _ in labels]
    for t, p in zip(y_true, y_pred):
        if t in index and p in index:
            cm[index[t]][index[p]] += 1
    return cm


def _metrics(y_true, y_pred, labels):
    """整体 accuracy、逐类 precision/recall/f1、macro-F1(只算有真值样本的类)。"""
    n = len(y_true)
    accuracy = (sum(1 for t, p in zip(y_true, y_pred) if t == p) / n) if n else 0.0
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
        if support > 0:
            f1s.append(f1)
        per_class[lab] = {"precision": round(precision, 4), "recall": round(recall, 4),
                          "f1": round(f1, 4), "support": support}
    macro_f1 = sum(f1s) / len(f1s) if f1s else 0.0
    return {"accuracy": round(accuracy, 4), "macro_f1": round(macro_f1, 4),
            "n": n, "per_class": per_class, "confusion": cm}


def _print_confusion(cm, labels):
    print("true\\pred".ljust(9) + "".join(f"{l[:4]:>6}" for l in labels))
    for i, lab in enumerate(labels):
        print(lab[:9].ljust(9) + "".join(f"{cm[i][j]:>6}" for j in range(len(labels))))


def main():
    ap = argparse.ArgumentParser(description="EmotiCompanion 语音单模态情绪评测(轻量)")
    ap.add_argument("--data", default="data/ravdess", help="含 labels.csv 的数据目录")
    ap.add_argument("--out", default="results/speech", help="结果输出目录")
    ap.add_argument("--limit", type=int, default=0, help="只跑前 N 段(0=全部)")
    args = ap.parse_args()

    backend = os.getenv("SPEECH_BACKEND", "api").strip().lower()

    labels_csv = os.path.join(args.data, "labels.csv")
    if not os.path.exists(labels_csv):
        print(f"[FATAL] 找不到 {labels_csv}，请先运行 scripts/prepare_ravdess.py 生成。",
              file=sys.stderr)
        sys.exit(1)

    # 只 import 语音相关模块，不碰人脸(省内存、避开 TF+hsemotion 一起 OOM)
    import audio_extract
    import speech_emotion

    with open(labels_csv, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if args.limit > 0:
        rows = rows[:args.limit]
    print(f"[backend={backend}] 共 {len(rows)} 段样本，词表 = {EMOTION_LABELS}")

    y_true, y_pred, records = [], [], []
    for i, row in enumerate(rows, 1):
        rel_path, gt = row["path"], row["emotion"]
        if gt not in EMOTION_LABELS:
            print(f"[skip] {rel_path}: 标签 {gt} 不在词表内")
            continue
        video_path = os.path.join(args.data, rel_path)
        try:
            wav = audio_extract.extract_audio(video_path)
            res = speech_emotion.predict(wav)
            pred = res.get("emotion", "neutral")
            conf = res.get("confidence", 0.0)
        except Exception as e:
            pred, conf = "neutral", 0.0
            res = {"error": str(e)}
        ok = "✓" if pred == gt else "✗"
        print(f"[{i}/{len(rows)}] {ok} gt={gt:<9} pred={pred:<9} conf={conf}  {rel_path}")
        y_true.append(gt)
        y_pred.append(pred)
        records.append({"path": rel_path, "gt": gt, "pred": pred, "confidence": conf})

    if not y_true:
        print("[FATAL] 没有可评测的样本。", file=sys.stderr)
        sys.exit(1)

    m = _metrics(y_true, y_pred, EMOTION_LABELS)
    print("\n" + "=" * 52)
    print(f"Speech-only [{backend}] on {m['n']} samples")
    print("=" * 52)
    print(f"  accuracy = {m['accuracy']:.4f}    macro-F1 = {m['macro_f1']:.4f}")
    print("\nconfusion matrix:")
    _print_confusion(m["confusion"], EMOTION_LABELS)

    os.makedirs(args.out, exist_ok=True)
    out_path = os.path.join(args.out, f"speech_{backend}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"backend": backend, **m, "records": records}, f,
                  ensure_ascii=False, indent=2)
    print(f"\n已保存 -> {out_path}")


if __name__ == "__main__":
    main()
