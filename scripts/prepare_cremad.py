"""
prepare_cremad.py —— 为「公平对比」消融准备 CREMA-D 视频子集
============================================================
EmotiCompanion · AIAA 3800 · HKUST(GZ)

为什么用 CREMA-D：人脸①(AffectNet) 与 语音③(RAVDESS-SER) **都没在 CREMA-D 上训练** →
两者都是跨域、公平可比，融合才有机会真正体现价值（见 docs/experiment_plan.md §8.7）。

CREMA-D 是本地已下载的数据（不自动下载）。本脚本从你下载的视频目录里，按文件名解析情绪、
均衡挑 N 段、复制到 <out>/clips/ 并生成 labels.csv（与 prepare_ravdess 一致，run_ablation 直接用）。

下载 CREMA-D（任选其一，约 1–2GB）：
  · GitHub: https://github.com/CheyneyComputerScience/CREMA-D  → VideoFlash/(.flv) 或 VideoMP4/
  · 下好后把视频所在目录传给 --src。

CREMA-D 文件名：`1001_DFA_ANG_XX.flv` = 演员ID_句子_情绪_强度
  情绪码：ANG/DIS/FEA/HAP/NEU/SAD（6 类，无 surprise）

用法（3800 环境、项目根目录）：
    python scripts/prepare_cremad.py --src path/to/CREMA-D/VideoFlash --n 60 --out data/cremad

产物：
    <out>/clips/<原文件名>        复制来的视频
    <out>/labels.csv             path, emotion, actor, level（emotion 为 7 类契约词表中的 6 类）
"""

import argparse
import collections
import csv
import glob
import os
import shutil

# CREMA-D 情绪码 → 全队 7 类契约词表（config.EMOTION_LABELS）。CREMA-D 无 surprise。
CREMAD_EMOTION = {
    "ANG": "angry", "DIS": "disgust", "FEA": "fear",
    "HAP": "happy", "NEU": "neutral", "SAD": "sad",
}

VIDEO_EXTS = (".flv", ".mp4", ".avi", ".mov", ".mkv")


def _scan(src):
    """递归找 CREMA-D 视频，解析文件名 → [(path, emotion, actor, level)]。"""
    items = []
    for ext in VIDEO_EXTS:
        for p in glob.glob(os.path.join(src, "**", f"*{ext}"), recursive=True):
            parts = os.path.splitext(os.path.basename(p))[0].split("_")
            if len(parts) < 3:
                continue
            actor, _sentence, emo_code = parts[0], parts[1], parts[2]
            level = parts[3] if len(parts) > 3 else "XX"
            emotion = CREMAD_EMOTION.get(emo_code.upper())
            if emotion is None:                    # 非 CREMA-D 情绪码 → 跳过
                continue
            items.append((p, emotion, actor, level))
    return items


def _balanced_pick(items, n):
    """按情绪轮询均衡挑 n 段（可复现：桶内按路径排序）。"""
    buckets = collections.defaultdict(list)
    for it in items:
        buckets[it[1]].append(it)
    for e in buckets:
        buckets[e].sort(key=lambda x: x[0])
    picked = []
    while len(picked) < n and any(buckets.values()):
        for e in sorted(buckets):
            if buckets[e]:
                picked.append(buckets[e].pop(0))
                if len(picked) >= n:
                    break
    return picked


def main():
    ap = argparse.ArgumentParser(description="准备 CREMA-D 消融子集（公平跨域对比）")
    ap.add_argument("--src", required=True, help="已下载的 CREMA-D 视频目录（递归搜索）")
    ap.add_argument("--n", type=int, default=60, help="挑选片段数，默认 60")
    ap.add_argument("--out", default="data/cremad", help="输出目录，默认 data/cremad")
    args = ap.parse_args()

    if not os.path.isdir(args.src):
        raise SystemExit(f"[FATAL] --src 目录不存在: {args.src}")

    items = _scan(args.src)
    if not items:
        raise SystemExit(
            f"[FATAL] 在 {args.src} 下没找到 CREMA-D 视频（*.flv/*.mp4…）。"
            "确认已下载 VideoFlash/ 或 VideoMP4/。")
    print(f"扫描到 {len(items)} 段，情绪分布：",
          dict(collections.Counter(e for _p, e, _a, _l in items)))

    picked = _balanced_pick(items, args.n)
    clips_dir = os.path.join(args.out, "clips")
    os.makedirs(clips_dir, exist_ok=True)

    labels_path = os.path.join(args.out, "labels.csv")
    with open(labels_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["path", "emotion", "actor", "level"])
        for src_path, emotion, actor, level in picked:
            fn = os.path.basename(src_path)
            dst = os.path.join(clips_dir, fn)
            if not os.path.exists(dst):
                shutil.copy2(src_path, dst)
            w.writerow([f"clips/{fn}", emotion, actor, level])

    dist = collections.Counter(e for _p, e, _a, _l in picked)
    print(f"已挑选 {len(picked)} 段 -> {labels_path}")
    print("情绪分布：", dict(sorted(dist.items())))
    if len(picked) < args.n:
        print(f"⚠️ 只凑到 {len(picked)}/{args.n} 段。")
    print("\n下一步（语音用本地 SER，与 RAVDESS 一致；两模态在 CREMA-D 上都跨域）：")
    print("  SPEECH_BACKEND=ser python scripts/run_ablation.py --data", args.out, "--refresh --limit 6")


if __name__ == "__main__":
    main()
