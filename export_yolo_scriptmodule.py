#!/usr/bin/env python3
# ==============================================================================
# 脚本A+: 导出YOLO11模型为TorchScript (仅保留Softmax单输出)
# ==============================================================================

import torch
import torch.nn as nn
from ultralytics import YOLO
import os

# ================================ 配置区域 ================================
MODEL_WEIGHTS = "/home/visuals/Code/yolo/runs/classify/train92/weights/best.pt"
OUTPUT_SCRIPTMODULE = "/home/visuals/Code/yolo/yolo11l_cls_scriptmodule_single.pt"
INPUT_SIZE = 416
# ========================================================================


class HardSigmoidReplacement(nn.Module):
    """MTK APU兼容的HardSigmoid替代实现"""
    def __init__(self):
        super().__init__()
        self.alpha = 1.0 / 6.0
        self.beta = 0.5

    def forward(self, x):
        return torch.clamp(x * self.alpha + self.beta, 0.0, 1.0)


def replace_hardsigmoid_in_model(model):
    """递归替换模型中的HardSigmoid"""
    replaced_count = 0
    for name, module in model.named_modules():
        if hasattr(module, 'act') and isinstance(module.act, nn.Hardsigmoid):
            print(f"  ✓ 替换 {name}.act → HardSigmoidReplacement")
            module.act = HardSigmoidReplacement()
            replaced_count += 1
        if isinstance(module, nn.Hardsigmoid):
            parent_name = '.'.join(name.split('.')[:-1])
            attr_name = name.split('.')[-1]
            if parent_name:
                parent_module = dict(model.named_modules())[parent_name]
                setattr(parent_module, attr_name, HardSigmoidReplacement())
                print(f"  ✓ 替换 {name} → HardSigmoidReplacement")
                replaced_count += 1
    return model, replaced_count


def verify_no_hardsigmoid(model):
    """验证模型中是否还有HardSigmoid"""
    for name, module in model.named_modules():
        if isinstance(module, nn.Hardsigmoid) or \
           (hasattr(module, 'act') and isinstance(module.act, nn.Hardsigmoid)):
            print(f"  ⚠️  残留 HardSigmoid: {name}")
            return False
    return True


class SingleOutputWrapper(nn.Module):
    """
    包装原模型，使其只返回 Softmax 输出
    原YOLO分类模型返回 (logits, softmax)，这里强制返回 softmax
    """
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        out = self.model(x)
        # 部分YOLO分类模型返回tuple (logits, softmax)
        if isinstance(out, (tuple, list)):
            # 输出logits
            return out[0]
        return out


def main():
    print("=" * 80)
    print("YOLO11 → TorchScript 导出脚本 (单输出 + MTK优化)")
    print("=" * 80)

    # Step 1: 加载模型
    print(f"\n[1/5] 加载 YOLO 模型...")
    if not os.path.exists(MODEL_WEIGHTS):
        print(f"❌ 权重文件不存在: {MODEL_WEIGHTS}")
        return

    yolo = YOLO(MODEL_WEIGHTS)
    model = yolo.model
    model.eval()
    print(f"  ✅ 模型加载成功: {type(model).__name__}")

    # Step 2: 替换 HardSigmoid
    print(f"\n[2/5] 替换 HardSigmoid...")
    model, replaced = replace_hardsigmoid_in_model(model)
    if verify_no_hardsigmoid(model):
        print(f"  ✅ 已完全移除 HardSigmoid ({replaced} 个替换)")
    else:
        print(f"  ⚠️  部分 HardSigmoid 未被替换")

    # Step 3: 包装为单输出
    print(f"\n[3/5] 包装模型以强制单输出 (Softmax)...")
    model = SingleOutputWrapper(model)

    # Step 4: 测试并导出 TorchScript
    print(f"\n[4/5] 导出 TorchScript...")
    dummy_input = torch.randn(1, 3, INPUT_SIZE, INPUT_SIZE)
    try:
        with torch.no_grad():
            out = model(dummy_input)
        print(f"  ✅ 前向传播成功，输出形状: {tuple(out.shape)}")
    except Exception as e:
        print(f"  ❌ 前向传播失败: {e}")
        return

    try:
        scripted = torch.jit.trace(model, dummy_input)
        print("  ✅ TorchScript trace 成功")
    except Exception as e:
        print(f"  ❌ TorchScript trace 失败: {e}")
        return

    # Step 5: 保存模型
    os.makedirs(os.path.dirname(OUTPUT_SCRIPTMODULE), exist_ok=True)
    torch.jit.save(scripted, OUTPUT_SCRIPTMODULE)
    size_mb = os.path.getsize(OUTPUT_SCRIPTMODULE) / (1024 * 1024)
    print(f"  ✅ 模型已保存: {OUTPUT_SCRIPTMODULE} ({size_mb:.2f} MB)")

    print("\n" + "=" * 80)
    print("🎉 导出完成! (仅单输出 Softmax)")
    print("=" * 80)
    print(f"  • 输入: {MODEL_WEIGHTS}")
    print(f"  • 输出: {OUTPUT_SCRIPTMODULE}")
    print(f"  • 输入尺寸: (1, 3, {INPUT_SIZE}, {INPUT_SIZE})")
    print(f"  • 输出尺寸: {tuple(out.shape)}")
    print(f"  • 替换 HardSigmoid: {replaced}")
    print("=" * 80)
    print("🔜 下一步: 执行 TFLite FP16 转换脚本 (mtk_converter) 即可")
    print("=" * 80)


if __name__ == '__main__':
    main()
