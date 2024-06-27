    #git clone https://github.com/THU-MIG/yolov10.git
    #cd yolov10
    #pip install -e .
    from ultralytics import YOLO
    model = YOLO('yolov10n.pt')
    model.train(data="coco8.yaml", epochs=4, batch=2, imgsz=640)
    model.val(data="coco8.yaml", imgsz=640, batch=2, conf=0.25, iou=0.6)
    model.predict("bus.jpg")
    model.export(format="onnx")

