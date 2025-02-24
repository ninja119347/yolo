from ultralytics import YOLO

# 加载模型
# model = YOLO("/Users/taozc/PycharmProjects/ultralytics/ultralytics/cfg/models/11/yolo11-model.yaml")
model = YOLO("yolo11s-cls.yaml")
model.load("yolo11s-cls.pt")
# 训练模型
train_results = model.train(
    data="mnist160",  # 数据集 YAML 路径
    epochs=20,  # 训练轮次
    imgsz=640,  # 训练图像尺寸
    # device="0",  # 运行设备，例如 device=0 或 device=0,1,2,3 或 device=cpu

)

# # 评估模型在验证集上的性能
# metrics = model.val()
#
# # 在图像上执行对象检测
# results = model("path/to/image.jpg")
# results[0].show()
#
# # 将模型导出为 ONNX 格式
# path = model.export(format="onnx")  # 返回导出模型的路径