from ultralytics import YOLO
import cv2


model=YOLO("yolov8n.pt")

# result=model("video.mp4",show=True,save=True,project="videolar")


for r in model ("video.mp4",stream=True):
    frame=r.plot()#yolo çizimlerini frame üzerine ekler
    frame=cv2.resize(frame,(640,640))
    
    cv2.imshow("video",frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break