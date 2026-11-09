import torch
import torch.nn as nn
import torch.nn.functional as F


class CPM(nn.Module):

    def __init__(self, c1, c2=None):
        super().__init__()
        self.amp = nn.AdaptiveMaxPool2d(1)
        self.aap = nn.AdaptiveAvgPool2d(1)
        self.wm = nn.Parameter(torch.tensor(0.5))
        self.wa = nn.Parameter(torch.tensor(0.5))

    def forward(self, x):
        fm = self.amp(x)
        fm_o = x + x * F.softmax(fm, dim=1)
        fa = self.aap(x)
        fa_o = x - x * F.softmax(fa, dim=1)
        return self.wm * fm_o + self.wa * fa_o


class CPMSigmoid(nn.Module):

    def __init__(self, c1, c2=None):
        super().__init__()
        self.amp = nn.AdaptiveMaxPool2d(1)
        self.aap = nn.AdaptiveAvgPool2d(1)
        self.wm = nn.Parameter(torch.tensor(0.5))
        self.wa = nn.Parameter(torch.tensor(0.5))

    def forward(self, x):
        fm_o = x + x * torch.sigmoid(self.amp(x))
        fa_o = x - x * torch.sigmoid(self.aap(x))
        return self.wm * fm_o + self.wa * fa_o
