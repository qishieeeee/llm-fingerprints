"""
build the dataset for the project from the chatbot arena human preference data.

source: lmarena-ai/arena-human-preference-55k (hugging face, apache-2.0)
each arena battle gives two (llm name, prompt, response) examples, one per side.
we keep only the first turn, map model names to families, balance the classes,
tag each prompt with a rough task type, and split train/val/test by prompt.

usage:
    python prepare.py                         # download from hugging face
    python prepare.py --csv train.csv         # use a local copy (e.g. the kaggle file)
    python prepare.py --families gpt claude llama mistral --perclass 3000
"""
import argparse
import json
import os
import re

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

# model name prefixes for each family (anything not matched is dropped,
# e.g. vicuna / koala / wizardlm, which are fine-tunes on other models' outputs)
famRules = {
    "gpt": r"^gpt-(3\.5|4)",
    "claude": r"^claude",
    "llama": r"^llama",
    "mistral": r"^(mistral|mixtral)",
    "gemini": r"^gemini",
    "qwen": r"^qwen",
    "deepseek": r"^deepseek",
}

codeRe = re.compile(r"```|\b(python|java|javascript|typescript|c\+\+|c#|sql|html|css|regex|bash|code|function|script|program|compile|bug|debug|api|class|def|import|json|react|docker)\b", re.I)
mathRe = re.compile(r"\b(solve|equation|calculate|compute|integral|derivative|probability|algebra|math|sum of|how many|prove|theorem)\b|\d+\s*[-+*/^x]\s*\d+", re.I)
writeRe = re.compile(r"\b(write|poem|story|essay|letter|email|haiku|song|lyrics|rewrite|paraphrase|summari[sz]e|tweet|joke|limerick|blog|caption|slogan)\b", re.I)


# helper functions

def familyOf(name):
    # map a raw arena model name (e.g. "claude-2.1") to its family, or none
    for fam, pat in famRules.items():
        if re.match(pat, name): return fam
    return None


def firstTurn(raw):
    # arena fields are json lists of turns; return the first turn as a string, or none
    try:
        turns = json.loads(raw)
    except (TypeError, json.JSONDecodeError):
        return None
    if not turns or not isinstance(turns[0], str): return None
    return turns[0].strip() or None


def tagTask(prompt):
    # rough task label from the prompt text (used later for rq3)
    if codeRe.search(prompt): return "coding"
    if mathRe.search(prompt): return "math"
    if writeRe.search(prompt): return "writing"
    return "other"


def loadArena(csvPath):
    # load the raw arena battles from a local csv or from hugging face
    if csvPath: return pd.read_csv(csvPath)
    from datasets import load_dataset
    return load_dataset("lmarena-ai/arena-human-preference-55k", split="train").to_pandas()


def explode(raw):
    # turn each battle into two rows: (model, prompt, response)
    rows = []
    for _, r in raw.iterrows():
        prompt = firstTurn(r["prompt"])
        if prompt is None: continue
        for side in ("a", "b"):
            resp = firstTurn(r[f"response_{side}"])
            if resp is None: continue
            rows.append({"battle": r["id"], "model": r[f"model_{side}"], "prompt": prompt, "response": resp})
    return pd.DataFrame(rows)


def groupSplit(df, seed):
    # 70/15/15 split where all rows sharing a prompt land in the same split
    groups = df["prompt"].str.lower().str.strip()
    outer = GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=seed)
    trainIdx, restIdx = next(outer.split(df, groups=groups))
    rest = df.iloc[restIdx]
    inner = GroupShuffleSplit(n_splits=1, test_size=0.50, random_state=seed)
    valIdx, testIdx = next(inner.split(rest, groups=groups.iloc[restIdx]))
    split = pd.Series("train", index=df.index)
    split.iloc[restIdx[valIdx]] = "val"
    split.iloc[restIdx[testIdx]] = "test"
    return split


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default=None, help="local arena csv (skip the download)")
    parser.add_argument("--families", nargs="+", default=["gpt", "claude", "llama", "mistral", "gemini"])
    parser.add_argument("--perclass", type=int, default=3000, help="max examples per family")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default="data/arena.csv")
    args = parser.parse_args()

    raw = loadArena(args.csv)
    print(f"battles loaded: {len(raw)}")

    df = explode(raw)
    df["family"] = df["model"].map(familyOf)
    print("\nexamples per family (before filtering):")
    print(df["family"].value_counts(dropna=False).to_string())

    df = df[df["family"].isin(args.families)]
    df = df[df["response"].str.split().str.len() >= 1]
    df = df.drop_duplicates(subset=["family", "prompt", "response"])

    # balance: every family gets the same number of examples
    counts = df["family"].value_counts()
    n = min(args.perclass, counts.min())
    print(f"\nbalancing to {n} examples per family (smallest class: {counts.idxmin()} with {counts.min()})")
    df = df.groupby("family", group_keys=False).sample(n=n, random_state=args.seed).reset_index(drop=True)

    df["task"] = df["prompt"].map(tagTask)
    df["split"] = groupSplit(df, args.seed)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    df[["family", "model", "task", "split", "prompt", "response"]].to_csv(args.out, index=False)

    print(f"\nsaved {len(df)} rows to {args.out}")
    print("\nrows per split:")
    print(df["split"].value_counts().to_string())
    print("\nfamily x split:")
    print(pd.crosstab(df["family"], df["split"]).to_string())
    print("\ntask x family:")
    print(pd.crosstab(df["task"], df["family"]).to_string())
    print("\nmodels included:")
    print(df.groupby("family")["model"].unique().to_string())


if __name__ == "__main__":
    main()
