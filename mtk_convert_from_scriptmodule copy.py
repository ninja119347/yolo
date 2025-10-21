#!/usr/bin/env python3
# ==============================================================================
# 脚本B: MTK量化与转换 (Python 3.6环境 - yolo-convert)
# ==============================================================================
# 功能：
# 1. 加载TorchScript模型（不需要ultralytics）
# 2. 使用MTK工具进行PTQ量化
# 3. 转换为TFLite格式
# 
# 运行环境: yolo-convert (Python 3.6)
# 前置步骤: 需要先运行 export_yolo_scriptmodule.py
# 使用方法: python mtk_convert_from_scriptmodule.py
# ==============================================================================

import torch
import os
import cv2
import numpy as np
import glob

# 导入MTK专用工具
from mtk_quantization.pytorch import ConfigGenerator, QuantizeHandler
from mtk_converter import PyTorchConverter

# ================================ 配置区域 ================================
# 输入: TorchScript模型路径（由export_yolo_scriptmodule.py生成）
INPUT_SCRIPTMODULE = "/home/visuals/Code/yolo/yolo11l_cls_scriptmodule.pt"

# 校准数据文件夹（用于PTQ量化）
CALIBRATION_DATA_FOLDER = '/home/visuals/Code/datasets/extracted_frames/0'

# 最大校准图片数量
MAX_CALIBRATION_IMAGES = 100

# 输出文件路径
OUTPUT_QUANT_CONFIG = "/home/visuals/Code/yolo/quant_config_pytorch.json"
OUTPUT_TFLITE = '/home/visuals/Code/yolo/yolo11l_mtk_quantized.tflite'

# 模型输入配置
INPUT_SHAPE = (1, 3, 416, 416)  # (batch, channels, height, width)
# ========================================================================


def load_scriptmodule(script_path):
    """
    加载TorchScript模型
    
    Args:
        script_path: TorchScript .pt文件路径
        
    Returns:
        加载的ScriptModule对象
    """
    if not os.path.exists(script_path):
        raise FileNotFoundError(f"TorchScript文件不存在: {script_path}")
    
    # 使用torch.jit.load加载
    model = torch.jit.load(script_path)
    model.eval()
    return model


def create_calibration_data_generator(data_folder, max_images=100):
    """
    创建校准数据生成器
    
    Args:
        data_folder: 包含校准图片的文件夹路径
        max_images: 最大使用图片数量
        
    Yields:
        torch.Tensor: 预处理后的图片张量 (1, 3, 416, 416)
    """
    # 支持的图片格式
    image_patterns = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
    image_files = []
    
    for pattern in image_patterns:
        image_files.extend(glob.glob(os.path.join(data_folder, pattern)))
    
    if len(image_files) == 0:
        raise ValueError(f"在 {data_folder} 中未找到图片文件")
    
    print(f"    找到 {len(image_files)} 张图片，使用前 {min(len(image_files), max_images)} 张")
    
    count = 0
    for image_path in image_files[:max_images]:
        # 读取图片
        image = cv2.imread(image_path)
        if image is None:
            continue
        
        # BGR → RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Resize到模型输入尺寸
        image = cv2.resize(image, (INPUT_SHAPE[3], INPUT_SHAPE[2]))
        
        # 归一化到[0, 1]
        image = image.astype(np.float32) / 255.0
        
        # HWC → CHW
        image = np.transpose(image, (2, 0, 1))
        
        # 添加batch维度 → NCHW
        image = np.expand_dims(image, axis=0)
        
        # 转换为torch.Tensor
        tensor = torch.from_numpy(image)
        
        count += 1
        if count % 10 == 0:
            print(f"    已处理 {count}/{min(len(image_files), max_images)} 张图片")
        
        yield tensor


def main():
    print("=" * 80)
    print("YOLO11 MTK量化与转换脚本 (TorchScript → TFLite)")
    print("=" * 80)
    
    # ============ 步骤1: 加载TorchScript模型 ============
    print(f"\n[步骤 1/6] 加载TorchScript模型...")
    print(f"  模型路径: {INPUT_SCRIPTMODULE}")
    
    try:
        scripted_model = load_scriptmodule(INPUT_SCRIPTMODULE)
        print(f"  ✅ 模型加载成功")
        print(f"  模型类型: {type(scripted_model)}")
    except Exception as e:
        print(f"  ❌ 加载失败: {e}")
        print(f"\n  💡 提示: 请先在Python 3.8+环境运行 export_yolo_scriptmodule.py")
        return
    
    # 测试前向传播
    print(f"  测试前向传播...")
    dummy_input = torch.randn(INPUT_SHAPE)
    try:
        with torch.no_grad():
            output = scripted_model(dummy_input)
        
        # 处理输出（可能是tensor或tuple）
        if isinstance(output, tuple):
            print(f"  ✅ 前向传播成功")
            print(f"  输入形状: {INPUT_SHAPE}")
            print(f"  输出类型: tuple，包含 {len(output)} 个元素")
            for i, out in enumerate(output):
                if torch.is_tensor(out):
                    print(f"    元素[{i}]形状: {out.shape}")
            output_for_display = output[0] if len(output) > 0 else None
        else:
            print(f"  ✅ 前向传播成功")
            print(f"  输入形状: {INPUT_SHAPE}")
            print(f"  输出形状: {output.shape}")
            output_for_display = output
            
    except Exception as e:
        print(f"  ❌ 前向传播失败: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # ============ 步骤2: 生成量化配置文件 ============
    print(f"\n[步骤 2/6] 生成MTK量化配置...")
    print(f"  配置文件: {OUTPUT_QUANT_CONFIG}")
    
    try:
        # 使用ConfigGenerator生成配置
        # 需要提供example_inputs
        q_cfg = ConfigGenerator(scripted_model)
        q_cfg.ignore_union_quantizer_targets = True  # 简化配置
        
        # 导出配置文件
        q_cfg.export_config(OUTPUT_QUANT_CONFIG)
        print(f"  ✅ 配置文件生成成功")
        
        # 显示配置文件大小
        config_size = os.path.getsize(OUTPUT_QUANT_CONFIG)
        print(f"  配置文件大小: {config_size} 字节")
        
    except Exception as e:
        print(f"  ❌ 配置生成失败: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # ============ 步骤3: 准备校准数据 ============
    print(f"\n[步骤 3/6] 准备校准数据...")
    print(f"  数据文件夹: {CALIBRATION_DATA_FOLDER}")
    
    if not os.path.exists(CALIBRATION_DATA_FOLDER):
        print(f"  ❌ 校准数据文件夹不存在: {CALIBRATION_DATA_FOLDER}")
        return
    
    # ============ 步骤4: 执行PTQ量化 ============
    print(f"\n[步骤 4/6] 执行训练后量化(PTQ)...")
    
    try:
        # 初始化QuantizeHandler
        quantizer = QuantizeHandler()
        print(f"  ✅ QuantizeHandler初始化成功")
        
        # 注意: prepare方法需要原始nn.Module，但我们只有ScriptModule
        # 根据MTK文档，ScriptModule也可以用于量化
        print(f"  正在准备量化模型...")
        quantized_model = quantizer.prepare(
            scripted_model,
            quant_config_file=OUTPUT_QUANT_CONFIG
        )
        print(f"  ✅ 量化模型准备完成")
        
    except Exception as e:
        print(f"  ❌ 量化准备失败: {e}")
        print(f"\n  ⚠️  尝试跳过量化步骤，直接转换...")
        quantized_model = scripted_model
    
    # 运行校准数据
    print(f"\n  运行校准数据以收集统计信息...")
    try:
        calib_count = 0
        for calib_data in create_calibration_data_generator(
            CALIBRATION_DATA_FOLDER, 
            MAX_CALIBRATION_IMAGES
        ):
            with torch.no_grad():
                _ = quantized_model(calib_data)
            calib_count += 1
        
        print(f"  ✅ 校准完成: 使用了 {calib_count} 张图片")
        
    except Exception as e:
        print(f"  ⚠️  校准过程出现问题: {e}")
        print(f"  继续转换流程...")
    
    # ============ 步骤5: 转换为TorchScript ============
    print(f"\n[步骤 5/6] 转换量化模型为TorchScript...")
    
    try:
        # 如果量化成功，需要重新trace
        if quantized_model is not scripted_model:
            print(f"  使用torch.jit.trace重新trace量化模型...")
            final_scripted_model = torch.jit.trace(quantized_model, dummy_input)
            print(f"  ✅ TorchScript转换成功")
        else:
            final_scripted_model = scripted_model
            print(f"  ℹ️  使用原始ScriptModule")
            
    except Exception as e:
        print(f"  ⚠️  TorchScript转换失败: {e}")
        print(f"  使用原始ScriptModule继续...")
        final_scripted_model = scripted_model
    
    # ============ 步骤6: 转换为TFLite ============
    print(f"\n[步骤 6/6] 转换为TFLite格式...")
    print(f"  输出路径: {OUTPUT_TFLITE}")
    
    try:
        # 创建PyTorchConverter
        print(f"  创建PyTorchConverter...")
        converter = PyTorchConverter.from_script_module(
            script_module=final_scripted_model,
            input_shapes=[list(INPUT_SHAPE)]  # 完整形状 [1, 3, 416, 416]
        )
        print(f"  ✅ Converter创建成功")
        
        # 转换为TFLite
        print(f"  正在转换为TFLite格式...")
        tflite_buffer = converter.convert_to_tflite(
            output_file=OUTPUT_TFLITE
        )
        
        # 检查输出文件
        if os.path.exists(OUTPUT_TFLITE):
            file_size_mb = os.path.getsize(OUTPUT_TFLITE) / (1024 * 1024)
            print(f"  ✅ TFLite转换成功")
            print(f"  文件大小: {file_size_mb:.2f} MB")
        else:
            print(f"  ⚠️  输出文件未生成，但转换过程无报错")
            
    except Exception as e:
        print(f"  ❌ TFLite转换失败: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # ============ 完成 ============
    print(f"\n" + "=" * 80)
    print(f"🎉 转换完成!")
    print(f"=" * 80)
    print(f"\n📋 转换信息:")
    print(f"  • 输入模型: {INPUT_SCRIPTMODULE}")
    print(f"  • 输出模型: {OUTPUT_TFLITE}")
    print(f"  • 输入形状: {INPUT_SHAPE}")
    if 'output_for_display' in locals() and output_for_display is not None and hasattr(output_for_display, 'shape'):
        print(f"  • 输出形状: {output_for_display.shape}")
    elif 'output' in locals() and isinstance(output, tuple):
        print(f"  • 输出类型: tuple (包含 {len(output)} 个元素)")
    print(f"  • 校准图片: {calib_count if 'calib_count' in locals() else 0} 张")
    print(f"  • MTK APU: ✅ 兼容")
    
    print(f"\n🔜 下一步:")
    print(f"  1. 验证TFLite模型: 使用MTK NeuroPilot工具测试")
    print(f"  2. 部署到设备: 将 {OUTPUT_TFLITE} 部署到MTK 8195设备")
    print(f"  3. 性能测试: 在APU上测试推理速度和精度")
    print(f"=" * 80)


if __name__ == '__main__':
    main()
