"""Step 2 of the baseline: a linear head on the frozen DINOv3 features (CPU is enough; ~1 min).

Usage: python baseline/train_linear.py [--epochs 100] [--lr 1e-3] [--wd 1e-2] [--seed 0]
Each image is one sample; target y = log(1 + discharge_lps). Features are standardised with the training mean / std.
Model: one linear layer, MSE loss, AdamW, full-batch-free minibatches of 256. Train / val / test losses are recorded
every epoch; the epoch with the lowest val loss is kept (test is never used for any choice).
A day's prediction = mean of its frames. "Accuracy" = share of days predicted within x2 of the measured discharge,
i.e. |log(1 + pred) - log(1 + measured)| <= log 2.
Writes baseline/results/: loss_curves.png, test_accuracy_by_month.png, metrics.json, predictions_<split>.csv
"""
import argparse
import json
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

ROOT = pathlib.Path(__file__).resolve().parent.parent
SPLITS = ["train", "val", "test", "test_oksr"]
COLOURS = {"train": "#2a78d6", "val": "#eb6834", "test": "#0b0b0b", "test_oksr": "#1baf7a"}


def load(split):
    rows = pd.read_csv(ROOT / "data" / "discharge" / f"{split}.csv")
    f = np.load(ROOT / "baseline" / "features" / f"{split}.npz", allow_pickle=True)
    assert (f["image_id"] == rows.image_id.to_numpy()).all()
    return rows, f["feats"].astype(np.float32), np.log1p(rows.discharge_lps.to_numpy()).astype(np.float32)


def day_scores(rows, pred):
    d = rows.assign(p=pred, y=np.log1p(rows.discharge_lps)).groupby("date").agg(y=("y", "mean"), p=("p", "mean"))
    d.index = pd.to_datetime(d.index)
    d["hit"] = (d.p - d.y).abs() <= np.log(2)
    r2 = 1 - ((d.y - d.p) ** 2).sum() / ((d.y - d.y.mean()) ** 2).sum()
    return d, {"days": int(len(d)), "within_x2": float(d.hit.mean()), "r2_log": float(r2),
               "rmse_log": float(np.sqrt(((d.y - d.p) ** 2).mean())),
               "spearman": float(d.y.rank().corr(d.p.rank()))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1e-2)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    out = ROOT / "baseline" / "results"
    out.mkdir(parents=True, exist_ok=True)

    data = {s: load(s) for s in SPLITS}
    mu, sd = data["train"][1].mean(0), data["train"][1].std(0) + 1e-6
    X = {s: torch.from_numpy((data[s][1] - mu) / sd) for s in SPLITS}
    Y = {s: torch.from_numpy(data[s][2]) for s in SPLITS}

    model = torch.nn.Linear(X["train"].shape[1], 1)
    torch.nn.init.zeros_(model.weight)
    model.bias.data.fill_(float(Y["train"].mean()))
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    hist = {s: [] for s in SPLITS}
    best = (np.inf, None, -1)
    for ep in range(args.epochs):
        model.train()
        for b in torch.randperm(len(X["train"])).split(args.batch):
            loss = ((model(X["train"][b]).squeeze(1) - Y["train"][b]) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            for s in SPLITS:
                hist[s].append(float(((model(X[s]).squeeze(1) - Y[s]) ** 2).mean()))
        if hist["val"][-1] < best[0]:
            best = (hist["val"][-1], {k: v.clone() for k, v in model.state_dict().items()}, ep)
    model.load_state_dict(best[1])
    print(f"best epoch {best[2] + 1} (val loss {best[0]:.3f})")

    # loss curves
    fig, ax = plt.subplots(figsize=(9, 4.6))
    ep = np.arange(1, args.epochs + 1)
    for s, name in [("train", "train (HOPB 2019–23)"), ("val", "val (HOPB 2024)"), ("test", "test (HOPB 2025)")]:
        ax.plot(ep, hist[s], color=COLOURS[s], lw=2, label=name)
    ax.axvline(best[2] + 1, color="#8a8984", ls="--", lw=1)
    ax.text(best[2] + 1.5, ax.get_ylim()[0] * 1.08, "kept epoch (lowest val loss)", fontsize=9, color="#52514e")
    ax.set_yscale("log")
    ax.set_xlabel("epoch")
    ax.set_ylabel("MSE loss on log(1 + discharge), per image")
    ax.set_title("Frozen DINOv3 ViT-B/16 + linear head: loss per epoch", loc="left")
    ax.grid(alpha=0.3)
    ax.legend(frameon=False)
    fig.savefig(out / "loss_curves.png", dpi=130, bbox_inches="tight")
    plt.close(fig)

    # per-split metrics and predictions with the kept model
    metrics = {"best_epoch": best[2] + 1, "args": vars(args), "loss_history": hist}
    days = {}
    with torch.no_grad():
        for s in SPLITS:
            pred = model(X[s]).squeeze(1).numpy()
            rows = data[s][0]
            rows.assign(pred_discharge_lps=np.expm1(pred)).to_csv(out / f"predictions_{s}.csv", index=False)
            days[s], metrics[s] = day_scores(rows, pred)
            print(f"{s:10s} days {metrics[s]['days']:4d}  within x2 {metrics[s]['within_x2']:.0%}  "
                  f"R2 {metrics[s]['r2_log']:.2f}")
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")

    # test accuracy by month
    t = days["test"]
    m = t.groupby(t.index.month).agg(acc=("hit", "mean"), n=("hit", "size"), q=("y", "median"))
    fig, ax = plt.subplots(figsize=(10, 4.6))
    bars = ax.bar(m.index, 100 * m.acc, color=COLOURS["test"], width=0.7)
    for x, a, n in zip(m.index, m.acc, m.n):
        ax.text(x, 100 * a + 1.5, f"{a:.0%}", ha="center", fontsize=9)
        ax.text(x, 2, f"n={n}", ha="center", fontsize=8, color="white")
    ax.axhline(100 * metrics["test"]["within_x2"], color=COLOURS["val"], ls="--", lw=1.5,
               label=f"whole year: {metrics['test']['within_x2']:.0%}")
    ax2 = ax.twinx()
    ax2.plot(m.index, np.expm1(m.q), color="#1baf7a", marker="o", lw=1.5, label="median measured discharge")
    ax2.set_yscale("log")
    ax2.set_ylabel("median measured discharge (L/s, log)", color="#1baf7a")
    ax.set_xticks(range(1, 13), ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"])
    ax.set_ylim(0, 105)
    ax.set_ylabel("% of days within ×2 of measured")
    ax.set_title("Test accuracy by month (HOPB 2025): share of days predicted within ×2", loc="left")
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=2, fontsize=9)
    fig.savefig(out / "test_accuracy_by_month.png", dpi=130, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
