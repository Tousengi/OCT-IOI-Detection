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

    def forward(self.x):
        fm = self.amp(x)
        fm_w = x * F.softmax(fm, dim=1)
        fm_o = x + fm_w

        fa = self.aap(x)
        fa_w = x * F.softmax(fa, dim=1)
        fa_o = x - fa_w

        return self.wm * fm_o + self.wa * fa_o

class PPM(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()

        c_low, c_ref = in_channels

        self.conv_low = nn.Conv2d(c_low, 1, kernel_size=1, bias=False)
        self.conv_ref = nn.Conv2d(c_ref, 1, kernel_size=1, bias=False)

        self.conv_out_low = nn.Conv2d(c_low, out_channels, kernel_size=1, bias=False)
        self.conv_channels = out_channels

    def forward(self, x):
        p_low, p_ref = x
        p_low_ds = F.interpolate(p_low, size=p_ref.shape[2:], mode='bilinear', align_corners=False)

        pc_low = self.conv_low(p_low_ds)
        pc_ref = self.conv_ref(p_ref)

        pw = torch.cat([pc_low, pc_ref], dim=1)
        pw = F.softmax(pw, dim=1)

        pw_low, pw_ref = pw.split(1, dim=1)
        po_low = pw_low * self.conv_out_low(p_low_ds)
        po_ref = pw_ref * po_ref

        return po_low + po_ref
