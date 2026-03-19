import torch
import torch.nn as nn
import torch.nn.functional as F


class SmallBasicBlock(nn.Module):
    """Paper Table 2: Lightweight building block inspired by SqueezeNet Fire
    and Inception blocks. BN + ReLU applied after each convolution (Sec. 3.1).

    Flow: 1x1 -> 3x1 -> 1x3 -> 1x1, all with BN + ReLU.
    Preserves spatial dimensions via padding on the 3x1 and 1x3 convolutions.
    """

    def __init__(self, ch_in, ch_out):
        super().__init__()
        mid = ch_out // 4
        self.block = nn.Sequential(
            nn.Conv2d(ch_in, mid, kernel_size=1),
            nn.BatchNorm2d(mid),
            nn.ReLU(),
            nn.Conv2d(mid, mid, kernel_size=(3, 1), padding=(1, 0)),
            nn.BatchNorm2d(mid),
            nn.ReLU(),
            nn.Conv2d(mid, mid, kernel_size=(1, 3), padding=(0, 1)),
            nn.BatchNorm2d(mid),
            nn.ReLU(),
            nn.Conv2d(mid, ch_out, kernel_size=1),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.block(x)


class LocNet(nn.Module):
    """Paper Table 1: Localization network for Spatial Transformer.

    Two parallel conv branches with different strides applied on the
    average-pooled input, concatenated and mapped to 6 affine parameters.
    """

    def __init__(self):
        super().__init__()
        self.pool = nn.AvgPool2d(kernel_size=3, stride=2)
        self.branch1 = nn.Conv2d(3, 32, kernel_size=5, stride=3)
        self.branch2 = nn.Conv2d(3, 32, kernel_size=5, stride=5)
        self.dropout = nn.Dropout(0.5)
        # For 94x24 input: pool→3x11x46, branch1→32x3x14=1344, branch2→32x2x9=576
        self.fc1 = nn.Linear(1920, 32)
        self.fc2 = nn.Linear(32, 6)
        self.fc2.weight.data.zero_()
        self.fc2.bias.data.zero_()

    def forward(self, x):
        pooled = self.pool(x)
        b1 = self.branch1(pooled).flatten(1)
        b2 = self.branch2(pooled).flatten(1)
        features = torch.cat([b1, b2], dim=1)
        features = self.dropout(features)
        features = torch.tanh(self.fc1(features))
        theta = torch.tanh(self.fc2(features)) * 0.1  # scaled TanH (Table 1)
        identity = torch.tensor(
            [[1, 0, 0], [0, 1, 0]], dtype=x.dtype, device=x.device
        ).unsqueeze(0)
        theta = theta.view(-1, 2, 3) + identity
        grid = F.affine_grid(theta, x.size(), align_corners=True)
        return F.grid_sample(x, grid, align_corners=True)


class LPRNet(nn.Module):
    """LPRNet: License Plate Recognition via Deep Neural Networks.

    Architecture from arXiv:1806.10447, Section 3.1:
      - Optional STN alignment via LocNet (Table 1)
      - Lightweight CNN backbone (Table 3) with SmallBasicBlocks (Table 2)
      - Global context embedding (ParseNet-style, ref [12])
      - Per-position classification head for CTC decoding

    MaxPool3d is used to simultaneously pool across the channel dimension
    and spatial dimensions, matching the paper's Table 3 channel counts.

    Input:  (N, 3, 24, 94) — 94x24 RGB license plate image
    Output: (N, class_num, T) — per-position class logits (T ≈ 74)
    """

    def __init__(self, class_num, dropout_rate=0.5, use_stn=False):
        super().__init__()
        self.class_num = class_num
        self.use_stn = use_stn
        self.stn_enabled = use_stn

        if use_stn:
            self.stn = LocNet()

        # Paper Table 3 — Backbone Network
        # MaxPool3d on 4D Conv2d output: (N, C, H, W) is treated as
        # unbatched (C_pool=N, D=C_conv, H, W); stride order is (D, H, W).
        # Paper stride (2,1) means H_stride=2, W_stride=1; D_stride is set
        # to produce 64 output channels from the preceding layer's channel count.
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1),                       # Conv #64
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 1, 1)),           # MaxPool #64 s1
            SmallBasicBlock(64, 128),                                         # SBB #128
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(2, 2, 1)),           # MaxPool #64 s(2,1)
            SmallBasicBlock(64, 256),                                         # SBB #256
            SmallBasicBlock(256, 256),                                        # SBB #256
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(4, 2, 1)),           # MaxPool #64 s(2,1)
            nn.Dropout(dropout_rate),
            nn.Conv2d(64, 256, kernel_size=(4, 1), stride=1),                # Conv #256 4x1
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Conv2d(256, class_num, kernel_size=(1, 13), stride=1),        # Conv #class 1x13
        )

        # Global context embedding (Sec. 3.1, ref [12] ParseNet):
        # FC over backbone output → tile → concat → 1x1 conv
        self.global_fc = nn.Linear(class_num, 128)
        self.output_conv = nn.Conv2d(class_num + 128, class_num, kernel_size=1)

    def forward(self, x):
        if self.use_stn and self.stn_enabled:
            x = self.stn(x)

        x = self.backbone(x)
        # x shape: (N, class_num, 1, W)

        # Global context: avg-pool → FC → tile → concat → 1x1 conv
        ctx = x.mean(dim=[2, 3])                                              # (N, class_num)
        ctx = F.relu(self.global_fc(ctx))                                     # (N, 128)
        ctx = ctx.unsqueeze(2).unsqueeze(3).expand(-1, -1, x.size(2), x.size(3))
        x = torch.cat([x, ctx], dim=1)                                       # (N, class_num+128, 1, W)
        x = self.output_conv(x)                                               # (N, class_num, 1, W)

        logits = x.squeeze(2)                                                 # (N, class_num, W)
        return logits


def build_lprnet(class_num=66, dropout_rate=0.5, use_stn=False, training=True):
    """Factory helper matching the paper's default configuration."""
    net = LPRNet(class_num, dropout_rate, use_stn)
    return net.train() if training else net.eval()
