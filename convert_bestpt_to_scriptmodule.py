#!/usr/bin/env python3
"""
将 Ultralytics best.pt 转换为 TorchScript 格式
用于 MTK 转换流程

输入: runs/classify/train88/weights/best.pt (Ultralytics checkpoint)
输出: yolo11_best_scriptmodule.pt (TorchScript)

环境要求: Python 3.8+ 环境，需要安装 ultralytics
"""

import torch
import os
from pathlib import Path

# ================================ 配置区域 ================================
# 输入: Ultralytics best.pt 路径
INPUT_BEST_PT = "/home/visuals/Code/yolo/runs/classify/train88/weights/best.pt"

# 输出: TorchScript 模型路径
OUTPUT_SCRIPTMODULE = "/home/visuals/Code/yolo/yolo11_best_scriptmodule.pt"

# 模型输入配置
INPUT_SHAPE = (1, 3, 416, 416)  # (batch, channels, height, width)
# ========================================================================


def load_ultralytics_model(checkpoint_path):
    """
    从 Ultralytics checkpoint 加载模型
    """
    print(f"正在加载 Ultralytics checkpoint: {checkpoint_path}")
    
    # 加载 checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    # 提取模型
    if isinstance(checkpoint, dict):
        if 'model' in checkpoint:
            model = checkpoint['model']
            print(f"  ✅ 从 checkpoint 字典中提取模型")
        elif 'ema' in checkpoint:
            model = checkpoint['ema']
            print(f"  ✅ 从 checkpoint 中提取 EMA 模型")
        else:
            raise ValueError(f"Checkpoint 中未找到模型，keys: {list(checkpoint.keys())}")
    else:
        model = checkpoint
        print(f"  ✅ Checkpoint 直接是模型")
    
    # 确保模型在 CPU 上并设置为评估模式
    model = model.cpu()
    model.eval()
    
    print(f"  模型类型: {type(model)}")
    print(f"  模型: {model.__class__.__name__}")
    
    return model


def convert_to_scriptmodule(model, input_shape, output_path):
    """
    将模型转换为 TorchScript 格式
    """
    print(f"\n正在转换为 TorchScript...")
    
    # 创建示例输入
    dummy_input = torch.randn(*input_shape)
    print(f"  示例输入形状: {dummy_input.shape}")
    
    # 测试前向传播
    print(f"  测试前向传播...")
    try:
        with torch.no_grad():
            output = model(dummy_input)
        print(f"  ✅ 前向传播成功")
        print(f"  输出类型: {type(output)}")
        
        if isinstance(output, tuple):
            print(f"  输出是 tuple，包含 {len(output)} 个元素")
            for i, out in enumerate(output):
                if hasattr(out, 'shape'):
                    print(f"    元素[{i}]: {out.shape}")
        elif hasattr(output, 'shape'):
            print(f"  输出形状: {output.shape}")
            
    except Exception as e:
        print(f"  ⚠️  前向传播测试出现问题: {e}")
        print(f"  继续尝试转换...")
    
    # 使用 torch.jit.trace 转换
    print(f"\n  使用 torch.jit.trace 进行转换...")
    try:
        scripted_model = torch.jit.trace(model, dummy_input)
        print(f"  ✅ TorchScript 转换成功")
    except Exception as e:
        print(f"  ❌ torch.jit.trace 失败: {e}")
        print(f"\n  尝试使用 torch.jit.script...")
        try:
            scripted_model = torch.jit.script(model)
            print(f"  ✅ torch.jit.script 转换成功")
        except Exception as e2:
            print(f"  ❌ torch.jit.script 也失败: {e2}")
            raise RuntimeError("无法转换模型为 TorchScript")
    
    # 测试转换后的模型
    print(f"\n  测试 TorchScript 模型...")
    with torch.no_grad():
        scripted_output = scripted_model(dummy_input)
    print(f"  ✅ TorchScript 模型推理成功")
    
    # 保存 TorchScript 模型
    print(f"\n  保存 TorchScript 模型到: {output_path}")
    torch.jit.save(scripted_model, output_path)
    
    # 检查文件大小
    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"  ✅ 模型保存成功")
    print(f"  文件大小: {file_size_mb:.2f} MB")
    
    return scripted_model


def verify_scriptmodule(scriptmodule_path, input_shape):
    """
    验证 TorchScript 模型是否可以正确加载和推理
    """
    print(f"\n正在验证 TorchScript 模型...")
    
    # 加载模型
    loaded_model = torch.jit.load(scriptmodule_path)
    print(f"  ✅ TorchScript 模型加载成功")
    
    # 测试推理
    dummy_input = torch.randn(*input_shape)
    with torch.no_grad():
        output = loaded_model(dummy_input)
    print(f"  ✅ 推理成功")
    
    if isinstance(output, tuple):
        print(f"  输出类型: tuple，包含 {len(output)} 个元素")
        for i, out in enumerate(output):
            if hasattr(out, 'shape'):
                print(f"    元素[{i}]: {out.shape}")
    elif hasattr(output, 'shape'):
        print(f"  输出形状: {output.shape}")
    
    return True


def main():
    print("=" * 80)
    print("Ultralytics best.pt → TorchScript 转换工具")
    print("=" * 80)
    
    # 检查输入文件
    if not os.path.exists(INPUT_BEST_PT):
        print(f"\n❌ 错误: 输入文件不存在: {INPUT_BEST_PT}")
        print(f"请检查路径是否正确")
        return
    
    input_size_mb = os.path.getsize(INPUT_BEST_PT) / (1024 * 1024)
    print(f"\n输入文件: {INPUT_BEST_PT}")
    print(f"文件大小: {input_size_mb:.2f} MB")
    print(f"输出文件: {OUTPUT_SCRIPTMODULE}")
    
    try:
        # 步骤1: 加载 Ultralytics 模型
        print(f"\n{'='*80}")
        print(f"步骤 1/3: 加载 Ultralytics 模型")
        print(f"{'='*80}")
        model = load_ultralytics_model(INPUT_BEST_PT)
        
        # 步骤2: 转换为 TorchScript
        print(f"\n{'='*80}")
        print(f"步骤 2/3: 转换为 TorchScript")
        print(f"{'='*80}")
        scripted_model = convert_to_scriptmodule(model, INPUT_SHAPE, OUTPUT_SCRIPTMODULE)
        
        # 步骤3: 验证转换结果
        print(f"\n{'='*80}")
        print(f"步骤 3/3: 验证转换结果")
        print(f"{'='*80}")
        verify_scriptmodule(OUTPUT_SCRIPTMODULE, INPUT_SHAPE)
        
        # 完成
        print(f"\n{'='*80}")
        print(f"🎉 转换完成!")
        print(f"{'='*80}")
        print(f"\n✅ TorchScript 模型已保存到: {OUTPUT_SCRIPTMODULE}")
        print(f"\n下一步:")
        print(f"  1. 在 mtk-env 环境中运行: python mtk_convert_from_scriptmodule.py")
        print(f"  2. 确保 mtk_convert_from_scriptmodule.py 中的 INPUT_SCRIPTMODULE 指向:")
        print(f"     {OUTPUT_SCRIPTMODULE}")
        print(f"{'='*80}")
        
    except Exception as e:
        print(f"\n{'='*80}")
        print(f"❌ 转换失败!")
        print(f"{'='*80}")
        print(f"错误信息: {e}")
        import traceback
        traceback.print_exc()
        return


if __name__ == '__main__':
    main()
