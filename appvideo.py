from ultralytics import YOLO

model=YOLO("yolov8n.pt")

result=model("video.mp4",show=True,save=True,project="videolar")