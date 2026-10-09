"""
first training script: textcnn on the response text, one seed, fixed number of epochs.

usage:
    python train.py
    python train.py --epochs 5
"""
import argparse
import random
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader

from src.data import TextDataset, buildVocab, collate, encode, tokenize
from src.models import TextCNN


# helper functions

def setSeed(seed):
    # make runs reproducible
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def runEpoch(mod, load, crit, opt, dev):
    # one pass of training; returns mean loss
    mod.train()
    total, count = 0.0, 0
    for x, lens, y in load:
        x, y = x.to(dev), y.to(dev)
        opt.zero_grad()
        loss = crit(mod(x, lens), y)
        loss.backward()
        opt.step()
        total += loss.item() * len(y)
        count += len(y)
    return total / count


@torch.no_grad()
def predict(mod, load, dev):
    # returns (true labels, predicted labels)
    mod.eval()
    trues, preds = [], []
    for x, lens, y in load:
        out = mod(x.to(dev), lens)
        trues += y.tolist()
        preds += out.argmax(1).tolist()
    return trues, preds


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/arena.csv")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--bs", type=int, default=64, help="batch size")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--maxlen", dest="maxLen", type=int, default=256, help="max tokens per example")
    parser.add_argument("--vocab", type=int, default=30000)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()

    setSeed(args.seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    df = pd.read_csv(args.data, keep_default_na=False)
    names = sorted(df["family"].unique())
    labOf = {n: i for i, n in enumerate(names)}

    seqs, labs = {}, {}
    for split in ("train", "val", "test"):
        part = df[df["split"] == split]
        seqs[split] = [tokenize(r)[: args.maxLen] for r in part["response"]]
        labs[split] = part["family"].map(labOf).tolist()

    voc = buildVocab(seqs["train"], args.vocab)
    sets = {k: TextDataset([encode(s, voc) for s in seqs[k]], labs[k]) for k in seqs}
    trainLoad = DataLoader(sets["train"], batch_size=args.bs, shuffle=True, collate_fn=collate)
    valLoad = DataLoader(sets["val"], batch_size=args.bs, collate_fn=collate)
    testLoad = DataLoader(sets["test"], batch_size=args.bs, collate_fn=collate)

    mod = TextCNN(len(voc), len(names)).to(dev)
    opt = torch.optim.Adam(mod.parameters(), lr=args.lr)
    crit = nn.CrossEntropyLoss()
    print(f"textcnn on {dev} | classes {names} | vocab {len(voc)} | train {len(labs['train'])}")

    for ep in range(1, args.epochs + 1):
        start = time.time()
        loss = runEpoch(mod, trainLoad, crit, opt, dev)
        vt, vp = predict(mod, valLoad, dev)
        print(f"ep {ep:2d} | train loss {loss:.3f} | val acc {accuracy_score(vt, vp):.3f} | {time.time() - start:.0f}s")

    tt, tp = predict(mod, testLoad, dev)
    print(f"\ntest acc {accuracy_score(tt, tp):.3f} | test macro-f1 {f1_score(tt, tp, average='macro'):.3f} (chance = {1 / len(names):.3f})")


if __name__ == "__main__":
    main()
