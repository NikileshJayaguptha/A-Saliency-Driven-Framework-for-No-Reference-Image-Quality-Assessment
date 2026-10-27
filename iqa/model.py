"""
Saliency-guided IQA network (Swin-B global branch + saliency-modulated VGG16
local branch + channel self-attention + patch-wise regressor).

The architecture is a 1:1 port of the notebook "saliency unet try-Kadid.ipynb",
so a checkpoint saved by the notebook (bestnatmodel.pth with keys
'salmodel', 'vgg', 'swin', 'attn', 'regressor') loads without changes.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import timm


def _vgg16_features(pretrained: bool):
    weights = models.VGG16_Weights.IMAGENET1K_V1 if pretrained else None
    return models.vgg16(weights=weights).features


# --------------------------------------------------------------------------
# Saliency U-Net (VGG16 encoder)
# --------------------------------------------------------------------------
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class UpBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = DoubleConv(out_channels + skip_channels, out_channels)

    def forward(self, x, skip):
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, out_classes=1, pretrained=True):
        super().__init__()
        vgg = _vgg16_features(pretrained)
        self.out_classes = out_classes
        self.enc1 = vgg[:5]
        self.enc2 = vgg[5:10]
        self.enc3 = vgg[10:17]
        self.enc4 = vgg[17:24]
        self.enc5 = vgg[24:31]
        self.up4 = UpBlock(512, 512, 256)
        self.up3 = UpBlock(256, 256, 128)
        self.up2 = UpBlock(128, 128, 64)
        self.up1 = UpBlock(64, 64, 32)
        self.final = nn.Conv2d(32, out_classes, kernel_size=1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        e5 = self.enc5(e4)
        d4 = self.up4(e5, e4)
        d3 = self.up3(d4, e3)
        d2 = self.up2(d3, e2)
        d1 = self.up1(d2, e1)
        return self.final(d1)


# --------------------------------------------------------------------------
# Saliency-modulated VGG16 local branch
# --------------------------------------------------------------------------
class VGG16_Saliency_Features(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        vgg = _vgg16_features(pretrained)
        self.block1_3 = vgg[:17]
        self.block4_5 = vgg[17:]
        self.alpha = nn.Parameter(torch.tensor(1.0))
        self.beta = nn.Parameter(torch.tensor(1.0))

    def forward(self, image, saliency):
        x = self.block1_3(image)
        saliency = F.interpolate(saliency, size=x.shape[2:], mode="bilinear", align_corners=False)
        x = self.alpha * x + self.beta * saliency
        return self.block4_5(x)


# --------------------------------------------------------------------------
# Channel self-attention + patch-wise regressor
# --------------------------------------------------------------------------
class ChannelSelfAttention(nn.Module):
    def __init__(self, C, H, W, d=128, drop=0.4):
        super().__init__()
        self.HW = H * W
        self.d = d
        self.q = nn.Linear(self.HW, d)
        self.k = nn.Linear(self.HW, d)
        self.v = nn.Linear(self.HW, d)
        self.proj = nn.Linear(d, self.HW)
        self.attn_drop = nn.Dropout(drop)
        self.proj_drop = nn.Dropout(drop)
        self.scale = d ** -0.5

    def forward(self, x):
        B, C, H, W = x.shape
        x_flat = x.reshape(B, C, -1)
        Q, K, V = self.q(x_flat), self.k(x_flat), self.v(x_flat)
        attn = torch.softmax(Q @ K.transpose(-2, -1) * self.scale, dim=-1)
        attn = self.attn_drop(attn)
        out = self.proj(attn @ V)
        out = self.proj_drop(out)
        return x + out.view(B, C, H, W)


class PatchWiseRegressor(nn.Module):
    def __init__(self, in_channels=512, hidden_dim=256, use_softmax=False, dropout_prob=0.4):
        super().__init__()
        self.use_softmax = use_softmax

        def head(sigmoid):
            layers = [
                nn.Linear(in_channels, hidden_dim), nn.ReLU(inplace=True), nn.Dropout(dropout_prob),
                nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU(inplace=True), nn.Dropout(dropout_prob),
                nn.Linear(hidden_dim // 2, 1),
            ]
            if sigmoid:
                layers.append(nn.Sigmoid())
            return nn.Sequential(*layers)

        self.score_head = head(sigmoid=False)
        self.attn_head = head(sigmoid=True)

    def forward(self, x, return_maps=False):
        B, C, H, W = x.shape
        x = x.reshape(B, C, -1).permute(0, 2, 1)  # (B, HW, C)
        s = self.score_head(x)
        w = self.attn_head(x)
        if self.use_softmax:
            w = F.softmax(w, dim=1)
            score = (s * w).sum(dim=1)
        else:
            score = (s * w).sum(dim=1) / (w.sum(dim=1) + 1e-6)
        score = score.squeeze(-1)
        if return_maps:
            return score, s.view(B, H, W), w.view(B, H, W)
        return score


# --------------------------------------------------------------------------
# Full model
# --------------------------------------------------------------------------
SWIN_NAME = "swin_base_patch4_window7_224"
CKPT_KEYS = {"salmodel": "salmodel", "vgg": "vgg", "swin": "swin", "attn": "attn", "regressor": "regressor"}


class SaliencyIQA(nn.Module):
    """
    forward(image, salimg=None) -> normalized quality score (B,)

    image  : (B,3,224,224) ImageNet-normalized
    salimg : (B,3,256,256) un-normalized; only used when sal_input == "salimg"
    """

    def __init__(self, pretrained=True, sal_input="image"):
        super().__init__()
        assert sal_input in ("image", "salimg")
        self.sal_input = sal_input
        self.salmodel = UNet(out_classes=1, pretrained=pretrained)
        self.vgg = VGG16_Saliency_Features(pretrained=pretrained)
        self.swin = timm.create_model(SWIN_NAME, pretrained=pretrained)
        self.swin.norm = nn.Identity()
        self.swin.head = nn.Identity()
        self.attn = ChannelSelfAttention(C=1536, H=7, W=7, d=49)
        self.regressor = PatchWiseRegressor(in_channels=1536)

    def backbones(self):
        return [self.salmodel, self.vgg, self.swin]

    def heads(self):
        return [self.attn, self.regressor]

    def forward(self, image, salimg=None, return_maps=False):
        sal_in = salimg if (self.sal_input == "salimg" and salimg is not None) else image
        saliency = self.salmodel(sal_in)
        local_f = self.vgg(image, saliency)                     # (B,512,7,7)
        global_f = self.swin.forward_features(image)            # (B,7,7,1024) NHWC
        if global_f.dim() == 4 and global_f.shape[-1] == 1024:
            global_f = global_f.permute(0, 3, 1, 2)
        f = torch.cat([global_f, local_f], dim=1)               # (B,1536,7,7)
        f = self.attn(f)
        if return_maps:
            score, patch_scores, patch_weights = self.regressor(f, return_maps=True)
            return score, saliency, patch_scores, patch_weights
        return self.regressor(f)

    # ---- checkpoint helpers (same layout as the notebook) ----
    def checkpoint_dict(self, meta=None):
        ckpt = {k: getattr(self, attr).state_dict() for k, attr in CKPT_KEYS.items()}
        if meta:
            ckpt["meta"] = meta
        return ckpt

    def load_checkpoint_dict(self, ckpt, strict=True):
        for k, attr in CKPT_KEYS.items():
            if k not in ckpt:
                raise KeyError(f"Checkpoint is missing '{k}' (has: {list(ckpt.keys())})")
            try:
                getattr(self, attr).load_state_dict(ckpt[k], strict=strict)
            except RuntimeError as e:
                raise RuntimeError(
                    f"Could not load '{k}' weights. If this is the Swin part, the installed "
                    f"timm version probably names layers differently from the one used for "
                    f"training; install the same timm version you trained with.\n{e}"
                ) from e
        return ckpt.get("meta", {})
