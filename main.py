from ultralytics import YOLO

if __name__ == "__main__":
    model = YOLO("yolo11_cbam_wiou.pt")

    # results = model.train(
    #     data="data.yaml",
    #     epochs=300,
    #     imgsz=640,
    #     device=0,
    #     batch=16,
    #     name="test",
    #     optimizer='SGD'
    # )

    # Run validation
    metrics = model.val(split = 'val')
    print(f"mAP50: {metrics.box.map50}")
    print(f"mAP50-95: {metrics.box.map}")