from ultralytics import YOLO

model = YOLO("yolov8n.pt")


model.train(
    data="./toilet-disable/data.yaml",
    epochs=10,
    optimizer="Adam",
    imgsz=640,               
    augment=True,           
    fliplr=0.5,               
    translate=0.1,      
    scale=0.5,           
    degrees=10,             
    batch=8,     
    project="models",  
    name="yolov8n_cadangan",   
    exist_ok=True,          
)


