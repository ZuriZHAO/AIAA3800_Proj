"""
prepare_ravdess.py —— 为感知层消融实验准备 RAVDESS 视频子集
============================================================
EmotiCompanion · AIAA 3800 · HKUST(GZ)

RAVDESS 的视频按演员分成 24 个独立 zip（Video_Speech_Actor_01..24.zip，各 ~550MB），
本脚本只下你指定的几个演员，解压后筛出「有声的音视频片段」，去掉 calm 类，
并按情绪均衡地挑出 N 段，生成 labels.csv（路径 → 全队 7 类情绪词表）。

用法（在 3800 环境里）：
    python scripts/prepare_ravdess.py                       # 默认：演员 01,02,05，挑 50 段
    python scripts/prepare_ravdess.py --actors 01 02 05 08  # 指定演员
    python scripts/prepare_ravdess.py --n 60 --out data/ravdess

产物：
    <out>/zips/Video_Speech_Actor_XX.zip   下载的原始 zip
    <out>/extracted/Actor_XX/*.mp4         解压后的视频
    <out>/labels.csv                       挑选出的片段清单 (path, emotion, actor, intensity)

RAVDESS 文件名 7 段编码，例：01-01-06-01-02-01-12.mp4
    1 模态:  01=音视频(有声)  02=纯视频(无声)  03=纯音频   → 只留 01
    2 声道:  01=speech        02=song                      → 只留 01
    3 情绪:  01中性 02平静 03开心 04悲伤 05愤怒 06恐惧 07厌恶 08惊讶
    4 强度:  01=normal 02=strong
    5 语句 / 6 重复 / 7 演员(奇男偶女)
"""

import argparse
import collections
import csv
import os
import urllib.request
import zipfile

# RAVDESS 情绪码 → 全队契约词表（config.EMOTION_LABELS，Ekman 标准 7 类）。
# calm(02) 不在 7 类里、且 calm≈neutral 太模糊，直接丢弃（值设为 None）。
RAVDESS_EMOTION = {
    "01": "neutral",
    "02": None,        # calm —— 丢弃
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fear",
    "07": "disgust",
    "08": "surprise",
}

ZENODO_BASE = "https://zenodo.org/records/1188976/files"


def download_actor(actor, zips_dir):
    """下载单个演员的 Video_Speech zip（已存在则跳过），返回本地路径。"""
    fname = f"Video_Speech_Actor_{actor}.zip"
    dst = os.path.join(zips_dir, fname)
    if os.path.exists(dst) and os.path.getsize(dst) > 0:
        print(f"[skip] {fname} 已存在")
        return dst
    url = f"{ZENODO_BASE}/{fname}?download=1"
    print(f"[down] {fname} <- {url}")

    def _progress(block, block_size, total):
        if total > 0:
            pct = min(100, block * block_size * 100 // total)
            print(f"\r       {pct:3d}%", end="", flush=True)

    urllib.request.urlretrieve(url, dst, _progress)
    print()  # 换行
    return dst


def extract_actor(zip_path, extract_dir):
    """解压演员 zip（对应文件夹已存在则跳过）。"""
    with zipfile.ZipFile(zip_path) as zf:
        names = zf.namelist()
        top = names[0].split("/")[0] if names else None
        if top and os.path.isdir(os.path.join(extract_dir, top)):
            print(f"[skip] {os.path.basename(zip_path)} 已解压")
            return
        print(f"[unzip] {os.path.basename(zip_path)}")
        zf.extractall(extract_dir)


def collect_av_clips(extract_dir):
    """遍历解压目录，筛出「有声音视频 + speech + 非 calm」的片段，返回 [(path, meta)]。"""
    clips = []
    for root, _dirs, files in os.walk(extract_dir):
        for fn in files:
            if not fn.lower().endswith(".mp4"):
                continue
            parts = os.path.splitext(fn)[0].split("-")
            if len(parts) != 7:
                continue
            modality, vocal, emo_code, intensity, _stmt, _rep, actor = parts
            if modality != "01" or vocal != "01":       # 只要有声的语音 AV
                continue
            emotion = RAVDESS_EMOTION.get(emo_code)
            if emotion is None:                          # 丢弃 calm / 未知
                continue
            clips.append((os.path.join(root, fn),
                          {"emotion": emotion, "actor": actor, "intensity": intensity}))
    return clips


def balanced_pick(clips, n):
    """按情绪轮询均衡地挑 n 段（各类尽量均匀）。"""
    buckets = collections.defaultdict(list)
    for path, meta in clips:
        buckets[meta["emotion"]].append((path, meta))
    # 每桶内按文件名排序，保证可复现
    for emo in buckets:
        buckets[emo].sort(key=lambda x: x[0])

    picked = []
    while len(picked) < n and any(buckets.values()):
        for emo in sorted(buckets):
            if buckets[emo]:
                picked.append(buckets[emo].pop(0))
                if len(picked) >= n:
                    break
    return picked


def main():
    ap = argparse.ArgumentParser(description="下载并准备 RAVDESS 感知层消融子集")
    ap.add_argument("--actors", nargs="+", default=["01", "02", "05"],
                    help="演员编号 01-24（奇男偶女），默认 01 02 05")
    ap.add_argument("--n", type=int, default=50, help="最终挑选的片段数，默认 50")
    ap.add_argument("--out", default="data/ravdess", help="输出目录，默认 data/ravdess")
    args = ap.parse_args()

    zips_dir = os.path.join(args.out, "zips")
    extract_dir = os.path.join(args.out, "extracted")
    os.makedirs(zips_dir, exist_ok=True)
    os.makedirs(extract_dir, exist_ok=True)

    for actor in args.actors:
        zip_path = download_actor(actor, zips_dir)
        extract_actor(zip_path, extract_dir)

    clips = collect_av_clips(extract_dir)
    print(f"\n共找到 {len(clips)} 段有效 AV 语音片段（已排除 calm / 无声）")

    picked = balanced_pick(clips, args.n)
    dist = collections.Counter(m["emotion"] for _p, m in picked)

    labels_path = os.path.join(args.out, "labels.csv")
    with open(labels_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["path", "emotion", "actor", "intensity"])
        for path, meta in picked:
            w.writerow([os.path.relpath(path, args.out).replace("\\", "/"),
                        meta["emotion"], meta["actor"], meta["intensity"]])

    print(f"已挑选 {len(picked)} 段 -> {labels_path}")
    print("情绪分布：", dict(sorted(dist.items())))
    if len(picked) < args.n:
        print(f"⚠️ 只凑到 {len(picked)}/{args.n} 段，加更多 --actors 可提升数量与多样性。")


if __name__ == "__main__":
    main()
