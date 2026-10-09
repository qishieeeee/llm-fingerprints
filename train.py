"""
train and evaluate a cnn or lstm classifier that predicts which llm family wrote a response.

rq1: --input response
rq2: --input prompt | response | both

usage:
    python train.py --model cnn --input response
    python train.py --model lstm --input both
    python train.py --model cnn --input response --nomask      # keep self-identification (ablation)
"""
import argparse
import copy
import random
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, classification_report, f1_score
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/arena.csv")
    parser.add_argument("--model", choices=["cnn", "lstm"], required=True)
    parser.add_argument("--input", choices=["prompt", "response", "both"], default="response")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--patience", type=int, default=3, help="stop after this many epochs without val f1 improving")
    parser.add_argument("--bs", type=int, default=64, help="batch size")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--maxlen", dest="maxLen", type=int, default=256, help="max tokens per example")
    parser.add_argument("--promptlen", dest="promptLen", type=int, default=64, help="prompt tokens kept in --input both")
    parser.add_argument("--vocab", type=int, default=30000)
    parser.add_argument("--nomask", dest="noMask", action="store_true", help="keep model/company names in the text")
    args = parser.parse_args()

    setSeed(args.seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    df = pd.read_csv(args.data, keep_default_na=False)
    names = sorted(df["family"].unique())
    labOf = {n: i for i, n in enumerate(names)}

    seqs, labs = {}, {}
    for split in ("train", "val", "test"):
        part = df[df["split"] == split]
        seqs[split] = makeSeqs(part, args)
        labs[split] = part["family"].map(labOf).tolist()

    voc = buildVocab(seqs["train"], args.vocab)
    sets = {k: TextDataset([encode(s, voc) for s in seqs[k]], labs[k]) for k in seqs}
    trainLoad = DataLoader(sets["train"], batch_size=args.bs, shuffle=True, collate_fn=collate)
    valLoad = DataLoader(sets["val"], batch_size=args.bs, collate_fn=collate)
    testLoad = DataLoader(sets["test"], batch_size=args.bs, collate_fn=collate)

    mod = buildModel(args.model, len(voc), len(names)).to(dev)
    opt = torch.optim.Adam(mod.parameters(), lr=args.lr)
    crit = nn.CrossEntropyLoss()
    tag = f"{args.model}-{args.input}" + ("-nomask" if args.noMask else "")
    print(f"{tag} on {dev} | classes {names} | vocab {len(voc)} | train {len(labs['train'])} val {len(labs['val'])} test {len(labs['test'])}")

    bestF1, bestState, bestEp, wait = -1.0, None, 0, 0
    for ep in range(1, args.epochs + 1):
        start = time.time()
        trainLoss = runEpoch(mod, trainLoad, crit, opt, dev)
        valLoss, vt, vp = predict(mod, valLoad, crit, dev)
        valF1 = f1_score(vt, vp, average="macro")
        print(f"ep {ep:2d} | train loss {trainLoss:.3f} | val loss {valLoss:.3f} | val f1 {valF1:.3f} | {time.time() - start:.0f}s")
        if valF1 > bestF1:
            bestF1, bestEp, wait = valF1, ep, 0
            bestState = copy.deepcopy(mod.state_dict())
        else:
            wait += 1
            if wait >= args.patience: break

    mod.load_state_dict(bestState)
    _, tt, tp = predict(mod, testLoad, crit, dev)
    print(f"\n{tag}: test acc {accuracy_score(tt, tp):.3f} | test macro-f1 {f1_score(tt, tp, average='macro'):.3f} (best epoch {bestEp}, chance = {1 / len(names):.3f})")
    print(classification_report(tt, tp, labels=range(len(names)), target_names=names, zero_division=0))


if __name__ == "__main__":
    main()
