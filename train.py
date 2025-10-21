from ultralytics import YOLO
import os
import sys
import torch
import platform


if __name__ == '__main__':
    # 打印环境信息
    
    # 加载模型
    # model = YOLO("yolo11l-cls.pt")
    model = YOLO("/home/visuals/Code/yolo/ultralytics/cfg/models/11/yolo11l-model.yaml", task="classify")
    # model = YOLO("yolov5l-cls.pt")
    # model.load("/home/visuals/Code/yolo/yolov5lu.pt")
    model.load("yolo11l-cls.pt")  
    # results = model.train(data="caltech101", epochs=100, imgsz=416)    
    
    # 使用DEBUG参数开启更多日志
    # 训练模型
    train_results = model.train(
        # data="ppt_dataset",  # 数据集 YAML 路径
        data="ppt_dataset13",  # 数据集 YAML 路径
        # data="/home/visuals/Code/yolo/ultralytics/cfg/datasets/ppt.yaml",  # 数据集 YAML 路径
        batch=80,  # 降低批量大小以减少内存使用和可能的维度问题
        epochs=50,  # 训练轮次
        # imgsz=640,       # 建议比416大，细节更完整
        imgsz=416,   # 验证图像大小
        # rect=False,       # 矩形训练，保持长宽比 + 黑边
        device="0",  # 运行设备
        workers=10,  # 减少工作线程数，避免资源竞争
        verbose=True,  # 启用详细日志
         # 学习率和优化器相关设置
        lr0=0.001,  # 较小的初始学习率
        lrf=0.01,  # 最终学习率为初始的1%
        cos_lr=True,  # 使用余弦学习率调度器
        
        # 正则化和防止过拟合
        weight_decay=0.0005,  # L2正则化
        
        # 学习率预热
        warmup_epochs=5.0,  # 增加预热轮次
        # 数据增强
        multi_scale=False,  # 启用多尺度训练
        # augment=False,      # 关闭所有增强
        # hsv_h=0.0,  
        # hsv_s=0.05,  
        # hsv_v=0.05,  
        # degrees=0.0,  
        # translate= 0.05,    # 看你是否想保留边框严格位置
        # scale=0.05,        # 看你是否容忍分辨率变化
        # shear=0.0,  
        # perspective=0.0,  
        # fliplr=0.0,  
        # bgr=0.0,  
        # mosaic=0.0,  
        # mixup=0.0,  
        # copy_paste=0.0,  
        # copy_paste_mode=None,  
        # auto_augment=None,  
        # erasing=0.05,  

        # 其他优化
        patience=20,  # 早停策略，20个epoch无改善则停止
        plots=True,  # 生成训练图表
        amp=False,              # 先关
    )

    # # 评估模型在验证集上的性能
    # metrics = model.val()
    #
    # # 在图像上执行对象检测
    # results = model("path/to/image.jpg")
    # results[0].show()
    #
    # # 将模型导出为 ONNX 格式
    # path = model.export(format="onnx")  # 返回导出模型的路径]
    print(train_results)