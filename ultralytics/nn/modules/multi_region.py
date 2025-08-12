import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiRegionExpand(nn.Module):
    def __init__(self, resize_mode='bilinear'):
        super().__init__()
        self.mode = resize_mode

    def forward(self, x):
        # x: (B,3,H,W)
        b,c,h,w = x.shape
        h5 = h // 5
        top    = x[:, :, :h5, :]
        bottom = x[:, :, -h5:, :]
        w5 = w // 5
        left   = x[:, :, :, :w5]
        right  = x[:, :, :, -w5:]
        # 统一尺寸
        def up(t): 
            return F.interpolate(t, size=(h,w), mode=self.mode, align_corners=False if self.mode=='bilinear' else None)
        parts = [x, up(top), up(bottom), up(left), up(right)]
        out = torch.cat(parts, dim=1)  # (B,15,H,W)
        return out