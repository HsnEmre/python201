from ultralytics import YOLO



model=YOLO("yolov8n.pt")

result=model(source=1,show=True)