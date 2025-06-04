from ultralytics import YOLO # type: ignore

# Load model dan lakukan training
model = YOLO("yolov8n.pt")  # model pre-trained
model.train(data="trotoar/data.yaml", epochs=50)