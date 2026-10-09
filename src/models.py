"""
textcnn (kim, 2014) for llm family classification.
takes (token ids, lengths) and returns class logits.
"""
import torch
import torch.nn as nn


class TextCNN(nn.Module):
    # embedding -> parallel 1d convs (several widths) -> max-pool -> dropout -> linear

    def __init__(self, vocSize, numClass, embedDim=200, kernelSizes=(3, 4, 5), numFilt=128, drop=0.5, padIdx=0):
        super().__init__()
        self.emb = nn.Embedding(vocSize, embedDim, padding_idx=padIdx)
        self.convs = nn.ModuleList([nn.Conv1d(embedDim, numFilt, k) for k in kernelSizes])
        self.drop = nn.Dropout(drop)
        self.fc = nn.Linear(numFilt * len(kernelSizes), numClass)

    def forward(self, x, lens):
        emb = self.emb(x).transpose(1, 2)  # (batch, embed, seq)
        pooled = [torch.relu(conv(emb)).max(dim=2).values for conv in self.convs]
        feat = torch.cat(pooled, dim=1)
        return self.fc(self.drop(feat))
