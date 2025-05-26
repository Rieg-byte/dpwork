from ultralytics import YOLO

if __name__ == "__main__":
    model = YOLO("yolo11_cbam_2.yaml")

    results = model.train(
        data="data.yaml",
        epochs=50,
        imgsz=640,
        device=0,
        batch=16,
        name="train_cbam_wlou_2",
        optimizer='SGD'
    )

    # Run validation
    metrics = model.val()
    print(f"mAP50: {metrics.box.map50}")
    print(f"mAP50-95: {metrics.box.map}")
    model.save("yolo11_cbam_wiou_2.pt")