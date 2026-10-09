"""
non-neural reference points: majority class and tf-idf + logistic regression
on the response text.

usage:
    python baseline.py
"""
import argparse

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

from src.data import tokenize


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/arena.csv")
    args = parser.parse_args()

    df = pd.read_csv(args.data, keep_default_na=False)
    train, test = df[df["split"] == "train"], df[df["split"] == "test"]
    names = sorted(df["family"].unique())

    dummy = DummyClassifier(strategy="most_frequent").fit(np.zeros(len(train)), train["family"])
    pred = dummy.predict(np.zeros(len(test)))
    print(f"majority  | acc {accuracy_score(test['family'], pred):.3f} | macro-f1 {f1_score(test['family'], pred, average='macro'):.3f}")

    vec = TfidfVectorizer(tokenizer=tokenize, lowercase=False, token_pattern=None,
                          ngram_range=(1, 2), min_df=2, max_features=100000, sublinear_tf=True)
    xTrain = vec.fit_transform(train["response"])
    xTest = vec.transform(test["response"])
    clf = LogisticRegression(max_iter=2000, C=4.0).fit(xTrain, train["family"])
    pred = clf.predict(xTest)
    print(f"tfidf-lr  | acc {accuracy_score(test['family'], pred):.3f} | macro-f1 {f1_score(test['family'], pred, average='macro'):.3f}")
    print(f"\nchance = {1 / len(names):.3f}")


if __name__ == "__main__":
    main()
