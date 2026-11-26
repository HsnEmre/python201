from ultralytics import YOLO
import cv2


# model=YOLO("yolov8m.pt")
# results=model("https://ultralytics.com/images/bus.jpg")
# results[0].show()


model=YOLO("yolov8n.pt")


results=model("face.jpg",save=True,project="görüntüler",name="human1")


img=results[0].plot()


cv2.imshow("sonuc",img)
cv2.waitKey(0)
cv2.destroyAllWindows()


