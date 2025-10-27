#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import torch
from ultralytics import YOLO

# ========= 可改参数 =========
ARCH_YAML = "/home/visuals/Code/yolo/ultralytics/cfg/models/11/yolo11l-model.yaml"
PRETRAINED_WEIGHTS = "/home/visuals/Code/yolo/runs/classify/train93/weights/best.pt"  # 你当前的ConvAsLinear版best.pt
DATA_YAML = "ppt_dataset13"
RUN_DIR = "/home/visuals/Code/yolo/runs/classify/train_qat_head_U"
IMG_SIZE = 416
BATCH = 64
EPOCHS = 5          # 先 3~5 个epoch，看精度再决定是否加到8~10
LR0 = 5e-5          # 小学习率
DEVICE = "0" if torch.cuda.is_available() else "cpu"
# ===========================

# 兼容不同PyTorch版本命名
try:
    import torch.ao.quantization as tq
except Exception:
    import torch.quantization as tq


def get_head_module(model: torch.nn.Module):
    """
    根据Ultralytics分类模型结构获取分类Head模块。
    一般 model.model 是 nn.Sequential，最后一个就是 Classify。
    """
    head = None
    if hasattr(model, "model") and isinstance(model.model, torch.nn.Sequential):
        cand = model.model[-1]
        # 你的Head里一定有 pool 和 linear(ConvAsLinear: nn.Conv2d)
        if hasattr(cand, "pool") and hasattr(cand, "linear"):
            head = cand

    # 兜底：在所有子模块里找“同时拥有 pool 和 linear 成员”的模块
    if head is None:
        for m in model.modules():
            if hasattr(m, "pool") and hasattr(m, "linear"):
                head = m
                break

    if head is None:
        raise RuntimeError("未找到分类Head模块，请检查模型结构。")

    return head


def build_qconfig():
    """
    官方推荐：激活 per-tensor affine (quint8)，权重 per-channel symmetric (qint8)。
    与后续 mtk_converter INT8 PTQ 的常用配置匹配。
    """
    act = tq.fake_quantize.FakeQuantize.with_args(
        observer=tq.observer.MovingAverageMinMaxObserver,
        dtype=torch.quint8,
        qscheme=torch.per_tensor_affine,
    )
    w = tq.fake_quantize.FakeQuantize.with_args(
        observer=tq.observer.PerChannelMinMaxObserver,
        dtype=torch.qint8,
        qscheme=torch.per_channel_symmetric,
        ch_axis=0,
    )
    return tq.qconfig.QConfig(activation=act, weight=w)


def attach_head_qat(model: torch.nn.Module):
    """
    仅给分类Head挂QAT（Conv2d/ConvAsLinear），其它层 qconfig=None。
    然后对“全模型”执行 prepare_qat（只有有qconfig的子模块会被插桩）。
    """
    head = get_head_module(model)
    qcfg = build_qconfig()

    # 先全部置 None
    for m in model.modules():
        m.qconfig = None

    # 只给 Head 内的 Conv2d（含你的 ConvAsLinear）挂上 qconfig
    for m in head.modules():
        if isinstance(m, torch.nn.Conv2d):
            m.qconfig = qcfg

    # 可选：如果 Head 里存在 PReLU/BN 等你也想量化，可在此按需设置
    # 例如：if isinstance(m, torch.nn.PReLU): m.qconfig = qcfg

    # 插入 observers/fake-quant（eager）
    tq.prepare_qat(model, inplace=True)
    return model


def main():
    os.makedirs(RUN_DIR, exist_ok=True)

    # 1) 构建&加载（保持你训练时的ConvAsLinear结构；训练要用logits）
    y = YOLO(ARCH_YAML, task="classify")
    y.load(PRETRAINED_WEIGHTS)
    net = y.model  # nn.Module

    # 2) 仅Head挂QAT，但【不冻结】任何层参数（U=Unfreeze）
    net.train()
    net = attach_head_qat(net)
    y.model = net  # 回写到YOLO外壳

    # 3) 正常用YOLO训练循环微调。注意：QAT建议关闭AMP
    results = y.train(
        data=DATA_YAML,
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH,
        device=DEVICE,
        workers=8,
        lr0=LR0,
        lrf=0.01,
        optimizer="AdamW",
        momentum=0.937,
        weight_decay=0.0,
        amp=False,                 # 量化微调不要AMP
        hsv_h=0.0, hsv_s=0.0, hsv_v=0.0,
        fliplr=0.5, flipud=0.0,
        mosaic=0.0, mixup=0.0,
        project=os.path.dirname(RUN_DIR),
        name=os.path.basename(RUN_DIR),
        exist_ok=True,
        verbose=True,
        cos_lr=True,
        warmup_epochs=1.0,
        patience=20,
    )
    print("✅ QAT Head 微调完成：", results)


if __name__ == "__main__":
    main()
