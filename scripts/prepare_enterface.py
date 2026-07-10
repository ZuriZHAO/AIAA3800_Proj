"""
prepare_enterface.py —— 为「人脸清晰 + 语音有语义」消融准备 eNTERFACE'05 子集
============================================================
EmotiCompanion · AIAA 3800 · HKUST(GZ)

为什么用 eNTERFACE'05：现有数据集「人脸清晰」与「语音有语义」只能满足其一——
  · RAVDESS/CREMA-D：人脸干净，但念固定中性句（无语义）
  · MELD：有语义，但多人/侧脸/切换，人脸拿不到稳定正脸
eNTERFACE'05 是**唯一可下载、同时满足两者**的 AV 数据集：受试者对镜头正面作答，
且说的是**与情绪一致的语义句**（如 anger→"What?! No, that's impossible!"），
比 RAVDESS 的中性句多了语义信号。见 docs/experiment_plan.md §4.1。

eNTERFACE'05 是本地已下载的数据（不自动下载）。本脚本递归扫描你下载的目录，
从**文件夹名或文件名**解析情绪、均衡挑 N 段、复制到 <out>/clips/ 并生成 labels.csv
（与 prepare_cremad/prepare_ravdess 一致，run_ablation 直接用）。

下载 eNTERFACE'05（约 1–2GB，任选其一）：
  · Kaggle: `unidpro/video-emotion-recognition-dataset`（1166 段视频，即 eNTERFACE'05）
  · 官方: https://enterface.net/enterface05/  （FTP/HTTP，证书较旧，浏览器下载）
  下好解压到某目录（如 data/enterface/raw/），把该目录传给 --src。

eNTERFACE'05：42 名受试者 × 6 情绪 × 5 句 ≈ 1166 段 .avi（DivX 编码、含音轨）。
  6 情绪（**无 neutral**）：anger / disgust / fear / happiness / sadness / surprise
  常见目录/命名两种，本脚本都支持：
    (a) 嵌套目录：subject 1/anger/sentence 1/s1_an_1.avi   → 用文件夹名判情绪
    (b) 扁平编码：s1_an_1.avi / s12_su_3.avi               → 用文件名 2 字母码判情绪

用法（3800 环境、项目根目录）：
    python scripts/prepare_enterface.py --src data/enterface/raw --n 60 --out data/enterface

产物：
    <out>/clips/<原文件名>        复制来的视频
    <out>/labels.csv             path, emotion, actor, level（emotion 为 7 类契约词表中的 6 类）
"""

import argparse
import collections
import csv
import glob
import os
import re
import shutil

# eNTERFACE'05 情绪 → 全队 7 类契约词表（config.EMOTION_LABELS）。eNTERFACE 无 neutral。
# 同时覆盖「完整词（文件夹名）」与「2 字母码（文件名）」两种写法。
ENTERFACE_EMOTION = {
    # anger
    "anger": "angry", "angry": "angry", "an": "angry",
    # disgust
    "disgust": "disgust", "disgusted": "disgust", "di": "disgust",
    # fear
    "fear": "fear", "fearful": "fear", "afraid": "fear", "fe": "fear",
    # happiness
    "happiness": "happy", "happy": "happy", "happie": "happy", "joy": "happy", "ha": "happy",
    # sadness
    "sadness": "sad", "sad": "sad", "sa": "sad",
    # surprise
    "surprise": "surprise", "surprised": "surprise", "su": "surprise",
}

VIDEO_EXTS = (".avi", ".mp4", ".mov", ".mkv", ".flv")

# 文件名里的 2 字母情绪码，形如 s1_an_1 / s12-su-3 / subj3_ha_5
_CODE_RE = re.compile(r"(?:^|[_\-\s])(an|di|fe|ha|sa|su)(?:[_\-\s]|\d|$)", re.IGNORECASE)
# 受试者编号：s1 / s12 / subject 3 / subj_7
_SUBJ_RE = re.compile(r"(?:subject|subj|s)[ _\-]?(\d{1,2})", re.IGNORECASE)
# 句子编号：sentence 1 / _1 结尾
_SENT_RE = re.compile(r"(?:sentence[ _\-]?(\d)|[_\-](\d)(?:\.[a-z0-9]+)?$)", re.IGNORECASE)


def _emotion_from_path(path, src):
    """先看路径里的文件夹名有没有完整情绪词，再退回文件名的 2 字母码。"""
    rel = os.path.relpath(path, src)
    parts = re.split(r"[\\/]", rel)
    # (a) 文件夹名 = 完整情绪词（嵌套结构）
    for comp in parts[:-1]:
        key = comp.strip().lower()
        if key in ENTERFACE_EMOTION and len(key) > 2:   # 只认完整词，避免误匹配
            return ENTERFACE_EMOTION[key]
    # (b) 文件名里的 2 字母码（扁平结构）
    stem = os.path.splitext(parts[-1])[0]
    m = _CODE_RE.search(stem)
    if m:
        return ENTERFACE_EMOTION.get(m.group(1).lower())
    # (c) 兜底：文件名里出现完整情绪词
    low = stem.lower()
    for key, val in ENTERFACE_EMOTION.items():
        if len(key) > 2 and key in low:
            return val
    return None


def _meta_from_path(path, src):
    """解析 (subject, sentence)，取不到给占位符。"""
    rel = os.path.relpath(path, src)
    ms = _SUBJ_RE.search(rel)
    subject = ms.group(1) if ms else "XX"
    stem = os.path.splitext(os.path.basename(path))[0]
    mt = _SENT_RE.search(stem)
    sentence = (mt.group(1) or mt.group(2)) if mt else "XX"
    return subject, sentence


def _scan(src):
    """递归找视频，解析情绪 → [(path, emotion, actor, level)]。"""
    items = []
    for ext in VIDEO_EXTS:
        for p in glob.glob(os.path.join(src, "**", f"*{ext}"), recursive=True):
            emotion = _emotion_from_path(p, src)
            if emotion is None:                    # 认不出情绪 → 跳过
                continue
            actor, level = _meta_from_path(p, src)
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
    ap = argparse.ArgumentParser(description="准备 eNTERFACE'05 消融子集（人脸清晰 + 语义语音）")
    ap.add_argument("--src", required=True, help="已下载的 eNTERFACE'05 目录（递归搜索）")
    ap.add_argument("--n", type=int, default=60, help="挑选片段数，默认 60")
    ap.add_argument("--out", default="data/enterface", help="输出目录，默认 data/enterface")
    args = ap.parse_args()

    if not os.path.isdir(args.src):
        raise SystemExit(f"[FATAL] --src 目录不存在: {args.src}")

    items = _scan(args.src)
    if not items:
        raise SystemExit(
            f"[FATAL] 在 {args.src} 下没识别出 eNTERFACE'05 视频。"
            "确认已解压（含 *.avi），且目录名或文件名带情绪信息（anger.../ 或 s1_an_1）。")
    print(f"扫描到 {len(items)} 段，情绪分布：",
          dict(sorted(collections.Counter(e for _p, e, _a, _l in items).items())))

    picked = _balanced_pick(items, args.n)
    clips_dir = os.path.join(args.out, "clips")
    os.makedirs(clips_dir, exist_ok=True)

    labels_path = os.path.join(args.out, "labels.csv")
    with open(labels_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["path", "emotion", "actor", "level"])
        for src_path, emotion, actor, level in picked:
            fn = os.path.basename(src_path)
            # 扁平命名可能重名（不同 subject 同 sentence），加 subject 前缀防覆盖
            safe = fn if fn.lower().startswith(("s", "subj")) else f"s{actor}_{fn}"
            dst = os.path.join(clips_dir, safe)
            if not os.path.exists(dst):
                shutil.copy2(src_path, dst)
            w.writerow([f"clips/{safe}", emotion, actor, level])

    dist = collections.Counter(e for _p, e, _a, _l in picked)
    print(f"已挑选 {len(picked)} 段 -> {labels_path}")
    print("情绪分布：", dict(sorted(dist.items())))
    if len(picked) < args.n:
        print(f"⚠️ 只凑到 {len(picked)}/{args.n} 段。")
    print("\n下一步（语音用 emotion2vec，两模态都跨域、且人脸正面 + 语音有语义）：")
    print("  SPEECH_BACKEND=emotion2vec python scripts/run_ablation.py --data",
          args.out, "--out results/ablation_enterface --refresh")


if __name__ == "__main__":
    main()
