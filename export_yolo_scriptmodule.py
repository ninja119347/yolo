#!/usr/bin/env python3
import torch
import torch.nn as nn
from ultralytics import YOLO
import os

# ================== 配置 ====================
QAT_BEST = "/home/visuals/Code/yolo/runs/classify/train_qat_head_U/weights/best.pt"
ARCH_YAML = "/home/visuals/Code/yolo/ultralytics/cfg/models/11/yolo11l-model.yaml"
FP32_CLEAN = "/home/visuals/Code/yolo/yolo11l_clean_fp32.pt"
OUTPUT_SCRIPTMODULE = "/home/visuals/Code/yolo/yolo11l_cls_scriptmodule_single.pt"
INPUT_SIZE = 416
# ===========================================


# ---------- ① 替换 HardSigmoid ----------
class HardSigmoidReplacement(nn.Module):
    def __init__(self):
        super().__init__()
        self.alpha = 1.0/6.0
        self.beta = 0.5
    def forward(self, x):
        return torch.clamp(x * self.alpha + self.beta, 0.0, 1.0)

def replace_hs(m):
    for name, module in m.named_modules():
        if hasattr(module,'act') and isinstance(module.act, nn.Hardsigmoid):
            module.act = HardSigmoidReplacement()
        if isinstance(module, nn.Hardsigmoid):
            pass
    return m


# ---------- ② SingleOutputWrapper ----------
class SingleOutputWrapper(nn.Module):
    def __init__(self, m): super().__init__(); self.m=m
    def forward(self,x):
        out = self.m(x)
        if isinstance(out,(tuple,list)): return out[0]
        return out


# ---------- ③ 脱壳 FP32 ----------
def strip_qat_to_fp32():
    m_qat = YOLO(QAT_BEST)
    m_clean = YOLO(ARCH_YAML, task="classify")

    sd_qat = m_qat.model.state_dict()
    sd_clean = m_clean.model.state_dict()
    filtered = {k:v for k,v in sd_qat.items() if k in sd_clean and sd_clean[k].shape==v.shape}
    sd_clean.update(filtered)
    m_clean.model.load_state_dict(sd_clean, strict=False)
    torch.save(m_clean.model.state_dict(), FP32_CLEAN)
    return FP32_CLEAN


def main():

    if not os.path.exists(FP32_CLEAN):
        print(">> FP32 clean 不存在，先脱壳…")
        strip_qat_to_fp32()
        print("✅ 脱壳完成:", FP32_CLEAN)

    print(">> 加载 FP32 clean …")
    y = YOLO(ARCH_YAML, task="classify")
    y.model.load_state_dict(torch.load(FP32_CLEAN,map_location='cpu'), strict=False)
    model = y.model.eval()

    print(">> 替换 HardSigmoid …")
    model = replace_hs(model)

    print(">> 包装单输出 …")
    model = SingleOutputWrapper(model)

    print(">> Trace & Save TorchScript …")
    dummy = torch.randn(1,3,INPUT_SIZE,INPUT_SIZE)
    scripted = torch.jit.trace(model, dummy)
    torch.jit.save(scripted, OUTPUT_SCRIPTMODULE)

    print("\n🎉 Done:", OUTPUT_SCRIPTMODULE)


if __name__=='__main__':
    main()
