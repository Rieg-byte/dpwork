from ultralytics import YOLO

if __name__ == "__main__":

    model = YOLO("cbam_2.yaml")
    results = model.train(
        data="data.yaml",
        epochs=300,
        imgsz=640,
        device=0,
        batch=16,
        name="yolo_cbam",
        optimizer='SGD'
    )

    # validation
    metrics = model.val(split = 'test')
    print(f"mAP50: {metrics.box.map50}")
    print(f"mAP50-95: {metrics.box.map}")
    model.save("yolo_cbam2.pt")