"""
fusion_bayes.py —— 逐类可靠性感知融合（朴素贝叶斯晚融合 · 留一交叉验证）
============================================================
EmotiCompanion · AIAA 3800 · HKUST(GZ)

动机（docs/experiment_plan.md §3.4/§4）：置信度加权融合 ④ 对模态失衡鲁棒（H2 成立），
但从未干净超过最强单模态（H1）。原因是加权只看**单模态自评置信度**，不知道**每个模态
在每一类上到底多可靠**——例如人脸①擅长 fear/disgust、语音③(emotion2vec)擅长 angry/neutral。
本脚本把"逐类可靠性"显式建模：用**混淆似然**做朴素贝叶斯融合。

  P(true=c | face=f, speech=s) ∝ P(c) · P(face=f | true=c) · P(speech=s | true=c)

其中 P(face=f|true=c)、P(speech=s|true=c) 就是**各模态的逐类混淆似然**（谁在哪类上可靠，
似然就集中）。这正是"让融合对逐类可靠性敏感"的**有原则、可部署**版本。

⚠️ 防数据泄漏：逐类可靠性**不能**从被评测的同一批样本估计（否则是对测试集过拟合、H1 会
假性通过）。这里用**留一交叉验证（LOOCV）**：对每个样本，仅用**其余样本**估计似然/先验，
再预测该样本。部署时等价于"在验证集上离线学好逐类可靠性、作为固定先验上线"。

用法（3800 环境、项目根目录，读已有缓存、不重跑感知）：
    python scripts/fusion_bayes.py --data data/cremad     --cache results/ablation_cremad_e2v
    python scripts/fusion_bayes.py --data data/enterface  --cache results/ablation_enterface
    # --alpha 0.5   拉普拉斯平滑强度（默认 0.5）
    # --use-conf    似然按模态置信度做几何加权（默认关，先看纯逐类可靠性）

输出：A 仅人脸 / B 仅语音 / D 加权(读 metrics.json) / F 贝叶斯逐类融合 的 acc + macro-F1，
以及 F 是否**首次越过最强单模态**（H1）。
"""

import argparse
import collections
import csv
import json
import math
import os
import sys

try:                                    # 控制台可能是 GBK，强制 utf-8 免得 emoji/中文炸
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass


def _load_labels(data_dir):
    path = os.path.join(data_dir, "labels.csv")
    labels = {}
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            labels[row["path"]] = row["emotion"]
    return labels


def _load_cache(cache_dir):
    path = os.path.join(cache_dir, "cache", "predictions.json")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _join(labels, cache):
    """→ [(true, face_pred, face_conf, speech_pred, speech_conf)]，按 path 对齐。"""
    rows = []
    for p, true in labels.items():
        rec = cache.get(p)
        if not rec:
            continue
        fa, sp = rec.get("face", {}), rec.get("speech", {})
        rows.append((
            true,
            fa.get("emotion", "neutral"), float(fa.get("confidence", 0.0) or 0.0),
            sp.get("emotion", "neutral"), float(sp.get("confidence", 0.0) or 0.0),
        ))
    return rows


def _macro_f1(true, pred, classes):
    f1s = []
    for c in classes:
        tp = sum(1 for t, p in zip(true, pred) if t == c and p == c)
        fp = sum(1 for t, p in zip(true, pred) if t != c and p == c)
        fn = sum(1 for t, p in zip(true, pred) if t == c and p != c)
        if tp == 0 and (fp == 0 or fn == 0) and sum(1 for t in true if t == c) == 0:
            continue  # 该类在 GT 中不存在 → 不计入 macro
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1s.append(2 * prec * rec / (prec + rec) if (prec + rec) else 0.0)
    return sum(f1s) / len(f1s) if f1s else 0.0


def _acc(true, pred):
    return sum(1 for t, p in zip(true, pred) if t == p) / len(true)


def _bayes_loocv(rows, alpha, use_conf):
    """留一交叉验证的朴素贝叶斯逐类融合。返回每个样本的预测（与 rows 同序）。"""
    # 训练类别集合 = GT 里出现过的类（避免对 support=0 的类瞎猜）
    classes = sorted({r[0] for r in rows})
    K = len(classes)
    N = len(rows)
    preds = []
    for i in range(N):
        # ---- 用「除 i 外」的样本估计先验与逐类似然（防泄漏）----
        prior = collections.Counter()
        lf = collections.defaultdict(collections.Counter)  # lf[c][face_pred]
        ls = collections.defaultdict(collections.Counter)  # ls[c][speech_pred]
        for j in range(N):
            if j == i:
                continue
            t, fp, fc, sp, sc = rows[j]
            prior[t] += 1
            # 置信度可选做软计数（几何加权的近似）：默认权重 1
            wf = fc if use_conf else 1.0
            ws = sc if use_conf else 1.0
            lf[t][fp] += wf
            ls[t][sp] += ws
        t_i, fp_i, fc_i, sp_i, sc_i = rows[i]
        best_c, best_score = None, -math.inf
        for c in classes:
            nc = prior[c]
            # 拉普拉斯平滑的 log 概率
            log_prior = math.log((nc + alpha) / (N - 1 + alpha * K))
            denom_f = sum(lf[c].values())
            denom_s = sum(ls[c].values())
            log_lf = math.log((lf[c][fp_i] + alpha) / (denom_f + alpha * K))
            log_ls = math.log((ls[c][sp_i] + alpha) / (denom_s + alpha * K))
            score = log_prior + log_lf + log_ls
            if score > best_score:
                best_score, best_c = score, c
        preds.append(best_c)
    return preds, classes


def fit_full(rows, alpha):
    """用**全部**样本估计先验与逐类似然（部署用：离线学好、作为固定先验上线）。

    返回可直接 json.dump 的字典：{classes, prior, face_lik, speech_lik, alpha, ...}。
    与 LOOCV 不同——这里不留一，因为部署时"训练集"就是我们所有带标签的验证数据。
    """
    classes = sorted({r[0] for r in rows})
    prior = collections.Counter()
    lf = collections.defaultdict(collections.Counter)
    ls = collections.defaultdict(collections.Counter)
    for t, fp, _fc, sp, _sc in rows:
        prior[t] += 1
        lf[t][fp] += 1
        ls[t][sp] += 1
    N, K = len(rows), len(classes)
    face_lik, speech_lik = {}, {}
    for c in classes:
        df, ds = sum(lf[c].values()), sum(ls[c].values())
        # 存平滑后的条件概率 P(pred|true=c)；查不到的 pred 在使用端用 alpha/(denom+alpha*K) 兜底
        face_lik[c] = {f: (lf[c][f] + alpha) / (df + alpha * K) for f in lf[c]}
        speech_lik[c] = {s: (ls[c][s] + alpha) / (ds + alpha * K) for s in ls[c]}
        face_lik[c]["__floor__"] = alpha / (df + alpha * K)
        speech_lik[c]["__floor__"] = alpha / (ds + alpha * K)
    return {
        "classes": classes,
        "prior": {c: (prior[c] + alpha) / (N + alpha * K) for c in classes},
        "face_lik": face_lik,
        "speech_lik": speech_lik,
        "alpha": alpha,
        "n_samples": N,
    }


def main():
    ap = argparse.ArgumentParser(description="逐类可靠性贝叶斯融合（LOOCV，读缓存）")
    ap.add_argument("--data", required=True, help="含 labels.csv 的数据目录")
    ap.add_argument("--cache", required=True, help="含 cache/predictions.json 的结果目录")
    ap.add_argument("--alpha", type=float, default=0.5, help="拉普拉斯平滑（默认 0.5）")
    ap.add_argument("--use-conf", action="store_true", help="似然按置信度软加权")
    ap.add_argument("--export", metavar="PATH", default=None,
                    help="把全数据学到的先验/似然导出为 JSON（供 fusion.py 的 bayes 模式加载）")
    args = ap.parse_args()

    labels = _load_labels(args.data)
    cache = _load_cache(args.cache)
    rows = _join(labels, cache)
    if not rows:
        raise SystemExit("[FATAL] labels 与 cache 对不上（path 不匹配？）")

    true = [r[0] for r in rows]
    face = [r[1] for r in rows]
    speech = [r[3] for r in rows]
    classes = sorted(set(true))

    a_acc, a_f1 = _acc(true, face), _macro_f1(true, face, classes)
    b_acc, b_f1 = _acc(true, speech), _macro_f1(true, speech, classes)
    preds_f, _ = _bayes_loocv(rows, args.alpha, args.use_conf)
    f_acc, f_f1 = _acc(true, preds_f), _macro_f1(true, preds_f, classes)

    # 读 run_ablation 记录的 D（加权融合）作对照
    d_acc = d_f1 = None
    mp = os.path.join(args.cache, "metrics.json")
    if os.path.exists(mp):
        try:
            m = json.load(open(mp, encoding="utf-8"))
            for k, v in (m.items() if isinstance(m, dict) else []):
                kl = str(k).lower()
                if "weighted" in kl and "cam" not in kl and "gate" not in kl:
                    d_acc = v.get("accuracy") if isinstance(v, dict) else None
                    d_f1 = v.get("macro_f1") or v.get("macro-F1") if isinstance(v, dict) else None
        except Exception:
            pass

    n = len(rows)
    print("=" * 60)
    print(f"逐类可靠性贝叶斯融合（LOOCV, N={n}, alpha={args.alpha}, use_conf={args.use_conf})")
    print(f"数据={args.data}  缓存={args.cache}")
    print("=" * 60)
    print(f"{'臂':<28}{'accuracy':>10}{'macro-F1':>10}")
    print(f"{'A. 仅人脸':<24}{a_acc:>12.4f}{a_f1:>10.4f}")
    print(f"{'B. 仅语音':<24}{b_acc:>12.4f}{b_f1:>10.4f}")
    if d_acc is not None:
        print(f"{'D. 加权融合(旧)':<22}{d_acc:>12.4f}{(d_f1 or 0):>10.4f}")
    print(f"{'F. 贝叶斯逐类融合':<20}{f_acc:>12.4f}{f_f1:>10.4f}")
    print("-" * 60)
    best_single = max(a_acc, b_acc)
    if f_acc > best_single + 1e-9:
        print(f"✅ H1 越过！F={f_acc:.4f} > 最强单模态={best_single:.4f}")
    else:
        print(f"❌ H1 未越过：F={f_acc:.4f} ≤ 最强单模态={best_single:.4f}")

    if args.export:
        priors = fit_full(rows, args.alpha)
        priors["source"] = f"{args.data} | {args.cache}"
        with open(args.export, "w", encoding="utf-8") as f:
            json.dump(priors, f, ensure_ascii=False, indent=2)
        print(f"\n已导出全数据先验/似然 -> {args.export}（供 fusion.py mode='bayes' 加载）")


if __name__ == "__main__":
    main()
