"""
rq3: does an llm-family classifier generalize to a task it never saw in training?

for each held-out task (coding, math, writing):
  - seen:   a model trained on ALL tasks, scored on the test rows of that task
  - unseen: a model trained on every task EXCEPT that one, scored on the same test rows
the drop from seen to unseen is how much the classifier depends on having seen that task.
the "seen" model is trained once per seed and reused for every task.

usage:
    python rq3.py                          # cnn, response input, 3 seeds
    python rq3.py --model lstm --seeds 1

outputs:
    results/rq3/rq3-<model>.csv    one row per (task, seed)
    results/rq3/summary-<model>.csv mean +/- std per task
"""
import argparse
import copy
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader

from src.data import TextDataset, buildVocab, collate, encode
from src.models import buildModel
from train import makeSeqs, predict, runEpoch, setSeed

heldTasks = ["coding", "math", "writing"]


# helper functions

def fitModel(train, val, names, args, dev, seed):
    # train one model on the given rows (early stopping on val macro-f1); returns (model, vocab)
    setSeed(seed)
    labOf = {n: i for i, n in enumerate(names)}
    trainSeqs, valSeqs = makeSeqs(train, args), makeSeqs(val, args)
    voc = buildVocab(trainSeqs, args.vocab)
    trainSet = TextDataset([encode(s, voc) for s in trainSeqs], train["family"].map(labOf).tolist())
    valSet = TextDataset([encode(s, voc) for s in valSeqs], val["family"].map(labOf).tolist())
    trainLoad = DataLoader(trainSet, batch_size=args.bs, shuffle=True, collate_fn=collate)
    valLoad = DataLoader(valSet, batch_size=args.bs, collate_fn=collate)

    mod = buildModel(args.model, len(voc), len(names)).to(dev)
    opt = torch.optim.Adam(mod.parameters(), lr=args.lr)
    crit = nn.CrossEntropyLoss()
    bestF1, bestState, wait = -1.0, None, 0
    for ep in range(1, args.epochs + 1):
        runEpoch(mod, trainLoad, crit, opt, dev)
        _, vt, vp = predict(mod, valLoad, crit, dev)
        valF1 = f1_score(vt, vp, average="macro")
        if valF1 > bestF1:
            bestF1, wait = valF1, 0
            bestState = copy.deepcopy(mod.state_dict())
        else:
            wait += 1
            if wait >= args.patience: break
    mod.load_state_dict(bestState)
    return mod, voc


def scoreOn(mod, voc, rows, names, args, dev):
    # accuracy and macro-f1 of a trained model on some test rows
    labOf = {n: i for i, n in enumerate(names)}
    testSet = TextDataset([encode(s, voc) for s in makeSeqs(rows, args)], rows["family"].map(labOf).tolist())
    load = DataLoader(testSet, batch_size=args.bs, collate_fn=collate)
    _, tt, tp = predict(mod, load, nn.CrossEntropyLoss(), dev)
    return accuracy_score(tt, tp), f1_score(tt, tp, average="macro", labels=range(len(names)), zero_division=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/arena.csv")
    parser.add_argument("--model", choices=["cnn", "lstm"], default="cnn")
    parser.add_argument("--input", choices=["prompt", "response", "both"], default="response")
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--bs", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--maxlen", dest="maxLen", type=int, default=256)
    parser.add_argument("--promptlen", dest="promptLen", type=int, default=64)
    parser.add_argument("--vocab", type=int, default=30000)
    parser.add_argument("--nomask", dest="noMask", action="store_true")
    parser.add_argument("--out", default="results/rq3")
    args = parser.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    df = pd.read_csv(args.data, keep_default_na=False)
    names = sorted(df["family"].unique())
    train, val, test = (df[df["split"] == s] for s in ("train", "val", "test"))
    os.makedirs(args.out, exist_ok=True)

    print(f"rq3 with {args.model} on {dev} | test rows per task: " + ", ".join(f"{t} {int((test['task'] == t).sum())}" for t in heldTasks))
    rows = []
    for seed in args.seeds:
        seenMod, seenVoc = fitModel(train, val, names, args, dev, seed)
        for task in heldTasks:
            testTask = test[test["task"] == task]
            seenAcc, seenF1 = scoreOn(seenMod, seenVoc, testTask, names, args, dev)
            heldMod, heldVoc = fitModel(train[train["task"] != task], val[val["task"] != task], names, args, dev, seed)
            heldAcc, heldF1 = scoreOn(heldMod, heldVoc, testTask, names, args, dev)
            rows.append({"task": task, "seed": seed, "nTest": len(testTask), "nTrainHeld": int((train["task"] != task).sum()),
                         "seenAcc": seenAcc, "seenF1": seenF1, "unseenAcc": heldAcc, "unseenF1": heldF1, "dropF1": seenF1 - heldF1})
            print(f"  seed {seed} | {task:8s} | seen f1 {seenF1:.3f} | unseen f1 {heldF1:.3f} | drop {seenF1 - heldF1:+.3f}")

    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(args.out, f"rq3-{args.model}.csv"), index=False)
    agg = res.groupby("task").agg(nTest=("nTest", "first"), seenF1=("seenF1", "mean"), seenStd=("seenF1", "std"),
                                  unseenF1=("unseenF1", "mean"), unseenStd=("unseenF1", "std"), dropF1=("dropF1", "mean")).reindex(heldTasks)
    agg = agg.fillna(0).round(3)
    agg.to_csv(os.path.join(args.out, f"summary-{args.model}.csv"))
    print("\nmacro-f1 on each task's test rows (mean over seeds):")
    print(agg.to_string())


if __name__ == "__main__":
    main()
