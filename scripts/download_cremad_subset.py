"""
download_cremad_subset.py —— 从官方 GitHub 只下载 CREMA-D 的一小撮视频子集
============================================================
EmotiCompanion · AIAA 3800 · HKUST(GZ)

CREMA-D 官方仓库 (CheyneyComputerScience/CREMA-D) 用 git-lfs 托管，整库 git clone 会拉下
上万个 .flv（好几个 G）。但消融只需要每种情绪凑够十几段。本脚本因此：
  1. 用 GitHub git-trees 递归接口「一次」拿到 VideoFlash/ 下全部 .flv 清单（省限流）；
  2. 按 6 种情绪均衡、且尽量跨不同演员挑 K 段/类；
  3. 用 git-lfs 的 media 直链逐个下载到 <out>（默认 data/cremad/video）。
已存在的文件跳过（可断点续传）。下完即可跑：
    python scripts/prepare_cremad.py --src data/cremad/video --n 60 --out data/cremad

用法（项目根目录）：
    python scripts/download_cremad_subset.py                 # 默认每类 12 段 → data/cremad/video
    python scripts/download_cremad_subset.py --per-emotion 15 --out data/cremad/video

代理：脚本会自动读取 HTTP(S)_PROXY 环境变量；没有则读 Windows 系统代理（注册表
Internet Settings 的 ProxyServer）。也可用 --proxy http://127.0.0.1:7890 手动指定，
或 --proxy "" 强制不走代理。
"""

import argparse
import collections
import json
import os
import sys
import time
import urllib.request

REPO = "CheyneyComputerScience/CREMA-D"
BRANCH = "master"
TREE_API = f"https://api.github.com/repos/{REPO}/git/trees/{BRANCH}?recursive=1"
MEDIA_BASE = f"https://media.githubusercontent.com/media/{REPO}/{BRANCH}/"

# CREMA-D 情绪码（6 类，无 surprise），与 prepare_cremad.py 一致
CREMAD_EMOTIONS = ("ANG", "DIS", "FEA", "HAP", "NEU", "SAD")


def detect_proxy(cli_proxy):
    """确定代理：命令行 > 环境变量 > Windows 系统代理（注册表）。返回 url 或 None。"""
    if cli_proxy is not None:                      # 显式传了（含空串 = 强制不走代理）
        return cli_proxy or None
    for k in ("HTTPS_PROXY", "HTTP_PROXY", "ALL_PROXY"):
        v = os.environ.get(k) or os.environ.get(k.lower())
        if v:
            return v
    if sys.platform == "win32":
        try:
            import winreg
            key = winreg.OpenKey(
                winreg.HKEY_CURRENT_USER,
                r"Software\Microsoft\Windows\CurrentVersion\Internet Settings")
            enable, _ = winreg.QueryValueEx(key, "ProxyEnable")
            if enable:
                server, _ = winreg.QueryValueEx(key, "ProxyServer")
                if server:
                    return server if "://" in server else "http://" + server
        except Exception:
            pass
    return None


def make_opener(proxy):
    handlers = []
    if proxy:
        handlers.append(urllib.request.ProxyHandler({"http": proxy, "https": proxy}))
    op = urllib.request.build_opener(*handlers)
    op.addheaders = [("User-Agent", "crema-d-subset-downloader")]
    return op


def fetch_flv_list(opener):
    """一次 git-trees 调用，返回 [(path, actor, emo_code)]（仅 VideoFlash/*.flv）。"""
    with opener.open(TREE_API, timeout=60) as r:
        data = json.load(r)
    if data.get("truncated"):
        print("[warn] git-trees 返回被截断，清单可能不全（一般不会发生）。", file=sys.stderr)
    items = []
    for node in data.get("tree", []):
        path = node.get("path", "")
        if node.get("type") != "blob":
            continue
        if not (path.startswith("VideoFlash/") and path.endswith(".flv")):
            continue
        parts = os.path.splitext(os.path.basename(path))[0].split("_")
        if len(parts) < 3:
            continue
        actor, emo = parts[0], parts[2].upper()
        if emo not in CREMAD_EMOTIONS:
            continue
        items.append((path, actor, emo))
    return items


def balanced_pick(items, per_emotion):
    """每种情绪挑 per_emotion 段，尽量来自不同演员（按演员轮询），可复现（排序后取）。"""
    by_emo = collections.defaultdict(lambda: collections.defaultdict(list))
    for path, actor, emo in items:
        by_emo[emo][actor].append(path)

    picked = []
    for emo in CREMAD_EMOTIONS:
        actors = sorted(by_emo[emo].keys())
        for a in actors:
            by_emo[emo][a].sort()
        chosen, ai = [], 0
        # 按演员轮询：每轮每个演员取一段，直到凑够 per_emotion 或取尽
        while len(chosen) < per_emotion and any(by_emo[emo][a] for a in actors):
            a = actors[ai % len(actors)]
            if by_emo[emo][a]:
                chosen.append(by_emo[emo][a].pop(0))
            ai += 1
        if len(chosen) < per_emotion:
            print(f"[warn] 情绪 {emo} 只凑到 {len(chosen)}/{per_emotion} 段", file=sys.stderr)
        picked.extend(chosen)
    return picked


def download(opener, path, out_dir, retries=3):
    """下载单个 media 直链到 out_dir/<basename>，已存在则跳过。返回 'skip'/'ok'/'fail'。"""
    fn = os.path.basename(path)
    dst = os.path.join(out_dir, fn)
    if os.path.exists(dst) and os.path.getsize(dst) > 1000:   # >1KB 视为已下好（避开 LFS 指针残留）
        return "skip"
    url = MEDIA_BASE + path
    tmp = dst + ".part"
    for attempt in range(1, retries + 1):
        try:
            with opener.open(url, timeout=120) as r, open(tmp, "wb") as f:
                f.write(r.read())
            if os.path.getsize(tmp) <= 1000:
                raise IOError("文件过小，疑似 LFS 指针而非视频")
            os.replace(tmp, dst)
            return "ok"
        except Exception as e:
            if attempt == retries:
                if os.path.exists(tmp):
                    os.remove(tmp)
                print(f"[fail] {fn}: {e}", file=sys.stderr)
                return "fail"
            time.sleep(2 * attempt)
    return "fail"


def main():
    ap = argparse.ArgumentParser(description="从 GitHub 下载 CREMA-D 视频子集（VideoFlash/.flv）")
    ap.add_argument("--out", default="data/cremad/video", help="下载目录，默认 data/cremad/video")
    ap.add_argument("--per-emotion", type=int, default=12,
                    help="每种情绪下多少段（默认 12，给 prepare_cremad 的 60 段留余量）")
    ap.add_argument("--proxy", default=None,
                    help='代理，如 http://127.0.0.1:7890；传 "" 强制不走代理；不传则自动探测')
    args = ap.parse_args()

    proxy = detect_proxy(args.proxy)
    print(f"[proxy] {proxy or '（不走代理，直连）'}")
    opener = make_opener(proxy)

    print("[1/3] 拉取 VideoFlash 文件清单（git-trees，一次调用）…")
    items = fetch_flv_list(opener)
    print(f"      共 {len(items)} 段 .flv，情绪分布：",
          dict(collections.Counter(e for _p, _a, e in items)))

    picked = balanced_pick(items, args.per_emotion)
    print(f"[2/3] 已挑选 {len(picked)} 段，情绪分布：",
          dict(collections.Counter(os.path.basename(p).split('_')[2] for p in picked)))

    os.makedirs(args.out, exist_ok=True)
    print(f"[3/3] 下载到 {args.out} …")
    stats = collections.Counter()
    for i, path in enumerate(picked, 1):
        res = download(opener, path, args.out)
        stats[res] += 1
        tag = {"ok": "✓", "skip": "·", "fail": "✗"}[res]
        print(f"  [{i:>3}/{len(picked)}] {tag} {os.path.basename(path)}")

    print(f"\n完成：新下载 {stats['ok']}，跳过 {stats['skip']}，失败 {stats['fail']}。")
    have = len([f for f in os.listdir(args.out) if f.lower().endswith('.flv')])
    print(f"目录 {args.out} 现有 {have} 段 .flv。")
    if stats["fail"]:
        print("有失败项，重跑本脚本会自动只补下失败/缺失的。")
    print("\n下一步：")
    print(f"  python scripts/prepare_cremad.py --src {args.out} --n 60 --out data/cremad")


if __name__ == "__main__":
    main()
