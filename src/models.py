"""
the two required models: a textcnn (kim, 2014) and a bidirectional lstm.
both take (token ids, lengths) and return class logits.
"""
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class TextCNN(nn.Module):
    # embedding -> parallel 1d convs (several widths) -> masked max-pool -> dropout -> linear

    def __init__(self, vocSize, numClass, embedDim=200, kernelSizes=(3, 4, 5), numFilt=128, drop=0.5, padIdx=0):
        super().__init__()
        self.emb = nn.Embedding(vocSize, embedDim, padding_idx=padIdx)
        self.convs = nn.ModuleList([nn.Conv1d(embedDim, numFilt, k) for k in kernelSizes])
        self.kernelSizes = kernelSizes
        self.drop = nn.Dropout(drop)
        self.fc = nn.Linear(numFilt * len(kernelSizes), numClass)

    def forward(self, x, lens):
        emb = self.emb(x).transpose(1, 2)  # (batch, embed, seq)
        pooled = []
        for conv, k in zip(self.convs, self.kernelSizes):
            out = torch.relu(conv(emb))  # (batch, filters, seq - k + 1)
            # ignore positions whose window runs into padding
            pos = torch.arange(out.size(2), device=x.device)
            valid = pos[None, :] <= (lens.to(x.device)[:, None] - k).clamp(min=0)
            out = out.masked_fill(~valid[:, None, :], float("-inf"))
            pooled.append(out.max(dim=2).values)
        feat = torch.cat(pooled, dim=1)
        return self.fc(self.drop(feat))


class BiLSTM(nn.Module):
    # embedding -> stacked bidirectional lstm (packed) -> masked max-pool over time -> dropout -> linear

    def __init__(self, vocSize, numClass, embedDim=200, hiddenDim=128, numLayer=2, drop=0.5, padIdx=0):
        super().__init__()
        self.emb = nn.Embedding(vocSize, embedDim, padding_idx=padIdx)
        self.lstm = nn.LSTM(embedDim, hiddenDim, num_layers=numLayer, batch_first=True,
                            bidirectional=True, dropout=drop if numLayer > 1 else 0.0)
        self.drop = nn.Dropout(drop)
        self.fc = nn.Linear(2 * hiddenDim, numClass)

    def forward(self, x, lens):
        emb = self.drop(self.emb(x))
        packed = pack_padded_sequence(emb, lens.cpu(), batch_first=True, enforce_sorted=False)
        out, _ = self.lstm(packed)
        out, _ = pad_packed_sequence(out, batch_first=True, total_length=x.size(1))
        mask = torch.arange(x.size(1), device=x.device)[None, :] < lens.to(x.device)[:, None]
        out = out.masked_fill(~mask[:, :, None], float("-inf"))
        feat = out.max(dim=1).values
        return self.fc(self.drop(feat))


def buildModel(name, vocSize, numClass):
    # factory used by train.py
    if name == "cnn": return TextCNN(vocSize, numClass)
    if name == "lstm": return BiLSTM(vocSize, numClass)
    raise ValueError(f"unknown model: {name}")
