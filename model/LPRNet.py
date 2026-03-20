import torch
import torch.nn as nn
import torch.nn.functional as F


class LCNetBlock(nn.Module):
    """Depthwise-separable block used for lightweight local feature extraction."""

    def __init__(self, ch_in, ch_out, stride=(1, 1)):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(
                ch_in, ch_in, kernel_size=3, stride=stride, padding=1,
                groups=ch_in, bias=False
            ),
            nn.BatchNorm2d(ch_in),
            nn.Hardswish(),
            nn.Conv2d(ch_in, ch_out, kernel_size=1, bias=False),
            nn.BatchNorm2d(ch_out),
            nn.Hardswish(),
        )

    def forward(self, x):
        return self.block(x)


class SVTRMixBlock(nn.Module):
    """Minimal global-mixing block over flattened spatial tokens."""

    def __init__(self, dim, num_heads=4, mlp_ratio=2.0, dropout=0.1):
        super().__init__()
        hidden_dim = int(dim * mlp_ratio)
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        # x: (N, C, H, W) -> tokens: (N, H*W, C)
        n, c, h, w = x.shape
        tokens = x.flatten(2).transpose(1, 2)
        attn_out, _ = self.attn(self.norm1(tokens), self.norm1(tokens), self.norm1(tokens))
        tokens = tokens + attn_out
        tokens = tokens + self.mlp(self.norm2(tokens))
        return tokens.transpose(1, 2).reshape(n, c, h, w)


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


class LPRNetBackbone(nn.Module):
    """Original LPRNet paper-faithful backbone."""

    def __init__(self, class_num, dropout_rate):
        super().__init__()
        self.net = nn.Sequential(
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

    def forward(self, x):
        return self.net(x)


class HybridSVTRLCNetBackbone(nn.Module):
    """Incremental hybrid backbone: LCNet stem + lightweight SVTR mixing."""

    def __init__(self, class_num, dropout_rate, mix_dim=192, mix_blocks=2, mix_heads=4):
        super().__init__()
        if mix_dim % mix_heads != 0:
            raise ValueError("mix_dim must be divisible by mix_heads")
        if mix_blocks < 1:
            raise ValueError("mix_blocks must be >= 1")

        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.Hardswish(),
            LCNetBlock(32, 64, stride=(2, 1)),   # 24x94 -> 12x94
            LCNetBlock(64, 128, stride=(2, 1)),  # 12x94 -> 6x94
            LCNetBlock(128, 192, stride=(2, 2)), # 6x94 -> 3x47
            nn.Conv2d(192, mix_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(mix_dim),
            nn.Hardswish(),
        )
        self.mix_blocks = nn.Sequential(
            *[SVTRMixBlock(mix_dim, num_heads=mix_heads, dropout=dropout_rate * 0.2)
              for _ in range(mix_blocks)]
        )
        self.head = nn.Sequential(
            nn.Conv2d(mix_dim, 256, kernel_size=(3, 1), stride=1),  # 3x47 -> 1x47
            nn.BatchNorm2d(256),
            nn.Hardswish(),
            nn.Dropout(dropout_rate),
            nn.Conv2d(256, class_num, kernel_size=(1, 5), stride=1, padding=(0, 2)),
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.mix_blocks(x)
        x = self.head(x)
        return x


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

    def __init__(
        self,
        class_num,
        dropout_rate=0.5,
        use_stn=False,
        backbone_type="lprnet",
        mix_dim=192,
        mix_blocks=2,
        mix_heads=4,
    ):
        super().__init__()
        self.class_num = class_num
        self.use_stn = use_stn
        self.stn_enabled = use_stn
        self.backbone_type = backbone_type

        if use_stn:
            self.stn = LocNet()

        if backbone_type == "lprnet":
            self.backbone = LPRNetBackbone(class_num, dropout_rate)
        elif backbone_type == "svtr_lcnet":
            self.backbone = HybridSVTRLCNetBackbone(
                class_num=class_num,
                dropout_rate=dropout_rate,
                mix_dim=mix_dim,
                mix_blocks=mix_blocks,
                mix_heads=mix_heads,
            )
        else:
            raise ValueError(f"Unsupported backbone_type: {backbone_type}")

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


def build_lprnet(
    class_num=66,
    dropout_rate=0.5,
    use_stn=False,
    training=True,
    backbone_type="lprnet",
    mix_dim=192,
    mix_blocks=2,
    mix_heads=4,
):
    """Factory helper matching the paper's default configuration."""
    net = LPRNet(
        class_num=class_num,
        dropout_rate=dropout_rate,
        use_stn=use_stn,
        backbone_type=backbone_type,
        mix_dim=mix_dim,
        mix_blocks=mix_blocks,
        mix_heads=mix_heads,
    )
    return net.train() if training else net.eval()
