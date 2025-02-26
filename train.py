from ultralytics import YOLO
import mlflow
if __name__ == '__main__':
    mlflow.set_tracking_uri('file:./runs')
    # 加载模型
    model = YOLO("ultralytics/cfg/models/11/yolo11s-model.yaml")
    # model = YOLO("yolo11s-cls.yaml")
    model.load("yolo11s-cls.pt")
    # results = model.train(data="caltech101", epochs=100, imgsz=416)
    # 训练模型
    train_results = model.train(
        data="caltech101",  # 数据集 YAML 路径
        batch = '32',
        epochs=50,  # 训练轮次
        imgsz=416,  # 训练图像尺寸
        device="0",  # 运行设备，例如 device=0 或 device=0,1,2,3 或 device=cpu
        workers = 1

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