"""
train and evaluate a cnn or lstm classifier that predicts which llm family wrote a response.

rq1: --input response
rq2: --input prompt | response | both

usage:
    python train.py --model cnn --input response
    python train.py --model lstm --input both --seeds 1 2 3
    python train.py --model cnn --input response --nomask      # keep self-identification (ablation)

outputs go to results/<model>-<input>[-nomask]/ and one summary row is appended to results/summary.csv
"""
import argparse
import copy
import json
import os
import random
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from torch.utils.data import DataLoader

from src.data import TextDataset, buildVocab, collate, encode, maskSelfId, toTokens
from src.models import buildModel


# helper functions

def setSeed(seed):
    # make runs reproducible
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def makeSeqs(df, args):
    # turn a dataframe split into token lists for the chosen input mode
    prompts = df["prompt"].tolist()
    resps = df["response"].tolist()
    if not args.noMask:
        prompts = [maskSelfId(p) for p in prompts]
        resps = [maskSelfId(r) for r in resps]
    return [toTokens(p, r, args.input, args.maxLen, args.promptLen) for p, r in zip(prompts, resps)]


def runEpoch(mod, load, crit, opt, dev):
    # one pass of training; returns mean loss
    mod.train()
    total, count = 0.0, 0
    for x, lens, y in load:
        x, y = x.to(dev), y.to(dev)
        opt.zero_grad()
        loss = crit(mod(x, lens), y)
        loss.backward()
        nn.utils.clip_grad_norm_(mod.parameters(), 1.0)
        opt.step()
        total += loss.item() * len(y)
        count += len(y)
    return total / count


@torch.no_grad()
def predict(mod, load, crit, dev):
    # returns (mean loss, true labels, predicted labels)
    mod.eval()
    total, trues, preds = 0.0, [], []
    for x, lens, y in load:
        x, y = x.to(dev), y.to(dev)
        out = mod(x, lens)
        total += crit(out, y).item() * len(y)
        trues += y.tolist()
        preds += out.argmax(1).tolist()
    return total / len(trues), trues, preds


def plotConfusion(trues, preds, names, path, title):
    # row-normalized confusion matrix (each row sums to 1)
    cm = confusion_matrix(trues, preds, labels=range(len(names)), normalize="true")
    fig, ax = plt.subplots(figsize=(5, 4.5))
    ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(names)), names, rotation=45, ha="right")
    ax.set_yticks(range(len(names)), names)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(title)
    for i in range(len(names)):
        for j in range(len(names)):
            ax.text(j, i, f"{cm[i, j]:.2f}", ha="center", va="center", color="white" if cm[i, j] > 0.5 else "black", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plotCurves(hist, path, title):
    # train/val loss and val macro-f1 per epoch
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9, 3.5))
    eps = range(1, len(hist["trainLoss"]) + 1)
    a1.plot(eps, hist["trainLoss"], label="train")
    a1.plot(eps, hist["valLoss"], label="val")
    a1.set_xlabel("epoch"); a1.set_ylabel("loss"); a1.legend()
    a2.plot(eps, hist["valF1"], color="tab:green")
    a2.set_xlabel("epoch"); a2.set_ylabel("val macro-f1")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def trainOnce(seed, data, names, args, dev, outDir):
    # train one model with one seed, early-stop on val macro-f1, evaluate on test
    setSeed(seed)
    voc = buildVocab(data["train"]["seqs"], args.vocab)
    sets = {k: TextDataset([encode(s, voc) for s in v["seqs"]], v["labs"]) for k, v in data.items()}
    trainLoad = DataLoader(sets["train"], batch_size=args.bs, shuffle=True, collate_fn=collate)
    valLoad = DataLoader(sets["val"], batch_size=args.bs, collate_fn=collate)
    testLoad = DataLoader(sets["test"], batch_size=args.bs, collate_fn=collate)

    mod = buildModel(args.model, len(voc), len(names)).to(dev)
    opt = torch.optim.Adam(mod.parameters(), lr=args.lr)
    crit = nn.CrossEntropyLoss()

    hist = {"trainLoss": [], "valLoss": [], "valF1": []}
    bestF1, bestState, bestEp, wait = -1.0, None, 0, 0
    for ep in range(1, args.epochs + 1):
        start = time.time()
        trainLoss = runEpoch(mod, trainLoad, crit, opt, dev)
        valLoss, vt, vp = predict(mod, valLoad, crit, dev)
        valF1 = f1_score(vt, vp, average="macro")
        hist["trainLoss"].append(trainLoss); hist["valLoss"].append(valLoss); hist["valF1"].append(valF1)
        print(f"  seed {seed} ep {ep:2d} | train loss {trainLoss:.3f} | val loss {valLoss:.3f} | val f1 {valF1:.3f} | {time.time() - start:.0f}s")
        if valF1 > bestF1:
            bestF1, bestEp, wait = valF1, ep, 0
            bestState = copy.deepcopy(mod.state_dict())
        else:
            wait += 1
            if wait >= args.patience: break

    mod.load_state_dict(bestState)
    _, tt, tp = predict(mod, testLoad, crit, dev)
    res = {
        "seed": seed,
        "bestEpoch": bestEp,
        "vocabSize": len(voc),
        "params": sum(p.numel() for p in mod.parameters()),
        "testAcc": accuracy_score(tt, tp),
        "testF1": f1_score(tt, tp, average="macro"),
        "report": classification_report(tt, tp, labels=range(len(names)), target_names=names, output_dict=True, zero_division=0),
        "history": hist,
    }
    tag = os.path.basename(outDir)
    plotConfusion(tt, tp, names, os.path.join(outDir, f"confusion-seed{seed}.png"), f"{tag} (seed {seed})")
    plotCurves(hist, os.path.join(outDir, f"curves-seed{seed}.png"), f"{tag} (seed {seed})")
    with open(os.path.join(outDir, f"seed{seed}.json"), "w") as f: json.dump(res, f, indent=2)
    print(f"  seed {seed} test acc {res['testAcc']:.3f} | test macro-f1 {res['testF1']:.3f} (best epoch {bestEp})")
    return res


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/arena.csv")
    parser.add_argument("--model", choices=["cnn", "lstm"], required=True)
    parser.add_argument("--input", choices=["prompt", "response", "both"], default="response")
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--patience", type=int, default=3, help="stop after this many epochs without val f1 improving")
    parser.add_argument("--bs", type=int, default=64, help="batch size")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--maxlen", dest="maxLen", type=int, default=256, help="max tokens per example")
    parser.add_argument("--promptlen", dest="promptLen", type=int, default=64, help="prompt tokens kept in --input both")
    parser.add_argument("--vocab", type=int, default=30000)
    parser.add_argument("--nomask", dest="noMask", action="store_true", help="keep model/company names in the text")
    parser.add_argument("--out", default="results")
    args = parser.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    df = pd.read_csv(args.data, keep_default_na=False)
    names = sorted(df["family"].unique())
    labOf = {n: i for i, n in enumerate(names)}

    data = {}
    for split in ("train", "val", "test"):
        part = df[df["split"] == split]
        data[split] = {"seqs": makeSeqs(part, args), "labs": part["family"].map(labOf).tolist()}

    tag = f"{args.model}-{args.input}" + ("-nomask" if args.noMask else "")
    outDir = os.path.join(args.out, tag)
    os.makedirs(outDir, exist_ok=True)
    print(f"{tag} on {dev} | classes {names} | train {len(data['train']['labs'])} val {len(data['val']['labs'])} test {len(data['test']['labs'])}")

    allRes = [trainOnce(s, data, names, args, dev, outDir) for s in args.seeds]

    accs = np.array([r["testAcc"] for r in allRes])
    f1s = np.array([r["testF1"] for r in allRes])
    row = {
        "model": args.model,
        "input": args.input,
        "mask": not args.noMask,
        "seeds": len(args.seeds),
        "accMean": round(accs.mean(), 4), "accStd": round(accs.std(), 4),
        "f1Mean": round(f1s.mean(), 4), "f1Std": round(f1s.std(), 4),
    }
    sumPath = os.path.join(args.out, "summary.csv")
    pd.DataFrame([row]).to_csv(sumPath, mode="a", header=not os.path.exists(sumPath), index=False)
    print(f"\n{tag}: acc {accs.mean():.3f} ± {accs.std():.3f} | macro-f1 {f1s.mean():.3f} ± {f1s.std():.3f}  (chance = {1 / len(names):.3f})")


if __name__ == "__main__":
    main()
