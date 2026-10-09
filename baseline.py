"""
non-neural reference points: majority class and tf-idf + logistic regression,
for each input mode. also writes the top n-grams per family (useful for rq4).

usage:
    python baseline.py
"""
import argparse
import os

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

from src.data import SEP, maskSelfId, tokenize


# helper functions

def makeText(df, mode, noMask):
    # build the raw text for each row given the input mode
    prompts, resps = df["prompt"], df["response"]
    if not noMask:
        prompts = prompts.map(maskSelfId)
        resps = resps.map(maskSelfId)
    if mode == "prompt": return prompts.tolist()
    if mode == "response": return resps.tolist()
    return (prompts + f" {SEP} " + resps).tolist()


def topFeatures(vec, clf, names, k=25):
    # highest-weight n-grams for each class
    vocab = np.array(vec.get_feature_names_out())
    lines = []
    for i, name in enumerate(names):
        top = vocab[np.argsort(clf.coef_[i])[::-1][:k]]
        lines.append(f"{name}: " + " | ".join(top))
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/arena.csv")
    parser.add_argument("--nomask", dest="noMask", action="store_true")
    parser.add_argument("--out", default="results")
    args = parser.parse_args()

    df = pd.read_csv(args.data, keep_default_na=False)
    train, test = df[df["split"] == "train"], df[df["split"] == "test"]
    names = sorted(df["family"].unique())
    os.makedirs(args.out, exist_ok=True)
    rows = []

    dummy = DummyClassifier(strategy="most_frequent").fit(np.zeros(len(train)), train["family"])
    pred = dummy.predict(np.zeros(len(test)))
    rows.append({"model": "majority", "input": "-", "mask": not args.noMask, "seeds": 1,
                 "accMean": round(accuracy_score(test["family"], pred), 4), "accStd": 0,
                 "f1Mean": round(f1_score(test["family"], pred, average="macro"), 4), "f1Std": 0})

    for mode in ("prompt", "response", "both"):
        vec = TfidfVectorizer(tokenizer=tokenize, lowercase=False, token_pattern=None,
                              ngram_range=(1, 2), min_df=2, max_features=100000, sublinear_tf=True)
        xTrain = vec.fit_transform(makeText(train, mode, args.noMask))
        xTest = vec.transform(makeText(test, mode, args.noMask))
        clf = LogisticRegression(max_iter=2000, C=4.0).fit(xTrain, train["family"])
        pred = clf.predict(xTest)
        acc, f1 = accuracy_score(test["family"], pred), f1_score(test["family"], pred, average="macro")
        print(f"tfidf-lr | {mode:8s} | acc {acc:.3f} | macro-f1 {f1:.3f}")
        rows.append({"model": "tfidf-lr", "input": mode, "mask": not args.noMask, "seeds": 1,
                     "accMean": round(acc, 4), "accStd": 0, "f1Mean": round(f1, 4), "f1Std": 0})
        if mode == "response":
            with open(os.path.join(args.out, "tfidf-top-features.txt"), "w") as f: f.write(topFeatures(vec, clf, list(clf.classes_)))

    sumPath = os.path.join(args.out, "summary.csv")
    pd.DataFrame(rows).to_csv(sumPath, mode="a", header=not os.path.exists(sumPath), index=False)
    print(f"\nmajority baseline acc {rows[0]['accMean']:.3f} | chance {1 / len(names):.3f}")
    print(f"top features per family saved to {args.out}/tfidf-top-features.txt")


if __name__ == "__main__":
    main()
