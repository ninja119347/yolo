import torch
import os
import cv2
import numpy as np
import glob
# export CUDA_VISIBLE_DEVICES=""
# 导入MTK专用工具
from mtk_converter import PyTorchConverter

# ================================ 配置区域 ================================
# 输入: TorchScript模型路径
INPUT_SCRIPTMODULE = "/home/visuals/Code/yolo/yolo11l_cls_scriptmodule_single.pt"
# INPUT_SCRIPTMODULE = "/home/visuals/Code/yolo/runs/classify/train88/weights/best.pt"

# 校准数据文件夹
CALIBRATION_DATA_FOLDER = '/home/visuals/Code/datasets/extracted_frames/0'
MAX_CALIBRATION_IMAGES = 100

# 输出文件路径
OUTPUT_TFLITE = '/home/visuals/Code/yolo/yolo11l_mtk_fp16.tflite'

# 模型输入配置
INPUT_SHAPE = (1, 3, 416, 416) # (batch, channels, height, width)
# ========================================================================

def create_calibration_data_generator(data_folder, max_images):
    """
    创建校准数据生成器，为转换器提供样本数据。
    """
    image_files = []
    for pattern in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
        image_files.extend(glob.glob(os.path.join(data_folder, pattern)))

    if len(image_files) == 0:
        raise ValueError(f"在 {data_folder} 中未找到图片文件")

    print(f" 找到 {len(image_files)} 张图片，将使用前 {min(len(image_files), max_images)} 张进行校准")

    for image_path in image_files[:max_images]:
        image = cv2.imread(image_path)
        if image is None:
            continue
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (INPUT_SHAPE[3], INPUT_SHAPE[2]))
        image = image.astype(np.float32) / 255.0
        image = np.transpose(image, (2, 0, 1))
        image = np.expand_dims(image, axis=0)
        yield [image]

def main():
    print("=" * 80)
    print("YOLO11 MTK FP16 TFLite 转换脚本 (带校准)")
    print("=" * 80)

    # ============ 步骤1: 加载TorchScript模型 ============
    print(f"\n[步骤 1/3] 加载TorchScript模型...")
    scripted_model = torch.jit.load(INPUT_SCRIPTMODULE, map_location='cpu')
    scripted_model.eval()
    print(f" ✅ 模型加载成功")

    # ============ 步骤2: 转换为TFLite (FP16模式) ============
    print(f"\n[步骤 2/3] 准备转换为FP16 TFLite格式...")

    try:
        # 创建PyTorchConverter
        converter = PyTorchConverter.from_script_module(
            script_module=scripted_model,
            input_shapes=[list(INPUT_SHAPE)]
        )
        print(f" ✅ Converter创建成功")

        # 设置FP16量化模式
        converter.quantize = True
        converter.default_quantization_bitwidth = 8
        converter.use_weights_symmetric_quantization=False
        converter.use_symmetric_quantization=False
        converter.use_per_output_channel_quantization=False
        converter.use_unsigned_quantization_type=True
        converter.input_value_ranges = [(0.0, 1.0)]
        print(f" ✅ 已设置转换模式为 16-bit (FP16)")
        # converter.convert_unsupported_data_type = True
        # converter.decompose_grouped_conv2d_ops = True
        # converter.use_adaround = False
        # converter.use_bias_correction = False

        # (!!!) 核心修正：添加校准数据生成器 (!!!)
        print(f" 正在准备校准数据生成器...")
        converter.calibration_data_gen = lambda: create_calibration_data_generator(
            CALIBRATION_DATA_FOLDER,
            MAX_CALIBRATION_IMAGES
        )
        print(f" ✅ 校准数据准备就绪")

        # ============ 步骤3: 执行转换 ============
        print(f"\n[步骤 3/3] 正在执行转换 (此步骤会运行校准数据)...")
        _ = converter.convert_to_tflite(output_file=OUTPUT_TFLITE)

        # 检查输出文件
        if os.path.exists(OUTPUT_TFLITE):
            file_size_mb = os.path.getsize(OUTPUT_TFLITE) / (1024 * 1024)
            print(f" ✅ TFLite转换成功")
            print(f" 文件大小: {file_size_mb:.2f} MB")
        else:
            print(f" ⚠️ 输出文件未生成，但转换过程无报错")

    except Exception as e:
        print(f" ❌ TFLite转换失败: {e}")
        import traceback
        traceback.print_exc()
        return

    # ============ 完成 ============
    print(f"\n" + "=" * 80)
    print(f"🎉 FP16 转换完成!")
    print(f" 最终模型: {OUTPUT_TFLITE}")
    print(f"=" * 80)

if __name__ == "__main__":
    main()