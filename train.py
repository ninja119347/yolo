from ultralytics import YOLO
import os
import sys
import torch
import platform


if __name__ == '__main__':
    # 打印环境信息
    
    # 加载模型
    model = YOLO("ultralytics/cfg/models/11/yolo11l-model.yaml")
    # model = YOLO("yolo11s-cls.yaml")
    model.load("yolo11l-cls.pt")
    # results = model.train(data="caltech101", epochs=100, imgsz=416)    
    
    # 使用DEBUG参数开启更多日志
    # 训练模型
    train_results = model.train(
        data="caltech101",  # 数据集 YAML 路径
        batch=64,  # 降低批量大小以减少内存使用和可能的维度问题
        epochs=100,  # 训练轮次
        imgsz=416,  # 训练图像尺寸
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
        multi_scale=True,  # 启用多尺度训练
        
        # 其他优化
        patience=20,  # 早停策略，20个epoch无改善则停止
        plots=True,  # 生成训练图表
        
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