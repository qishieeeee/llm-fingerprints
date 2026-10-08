"""
tokenization, vocabulary, and pytorch dataset utilities.

we deliberately keep case, punctuation, and newlines: formatting (markdown
headers, bullets, bold, line breaks) is a big part of an llm's fingerprint.
"""
import re
from collections import Counter

import torch
from torch.utils.data import Dataset

PAD, UNK, NL = "<pad>", "<unk>", "<nl>"

tokRe = re.compile(r"<\w+>|\w+|[^\w\s]")


# helper functions

def tokenize(text):
    # split text into words and punctuation, turning line breaks into a <nl> token
    return tokRe.findall(text.replace("\n", f" {NL} "))


def buildVocab(seqs, maxSize, minFreq=2):
    # map the most frequent training tokens to ids (0 = pad, 1 = unk)
    counts = Counter(tok for seq in seqs for tok in seq)
    words = [w for w, c in counts.most_common(maxSize) if c >= minFreq and w not in (PAD, UNK)]
    itos = [PAD, UNK] + words
    return {w: i for i, w in enumerate(itos)}


def encode(seq, voc):
    # token list -> list of ids (never empty)
    ids = [voc.get(tok, 1) for tok in seq]
    return ids or [1]


class TextDataset(Dataset):
    # holds encoded sequences and integer labels

    def __init__(self, seqs, labs):
        self.seqs = [torch.tensor(s, dtype=torch.long) for s in seqs]
        self.labs = torch.tensor(labs, dtype=torch.long)

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, i):
        return self.seqs[i], self.labs[i]


def collate(batch, minLen=5):
    # pad a batch to the longest sequence (at least minLen so the widest conv filter fits)
    seqs, labs = zip(*batch)
    lens = torch.tensor([len(s) for s in seqs])
    width = max(int(lens.max()), minLen)
    x = torch.zeros(len(seqs), width, dtype=torch.long)
    for i, s in enumerate(seqs): x[i, : len(s)] = s
    return x, lens, torch.stack(labs)
