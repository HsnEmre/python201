from ultralytics import YOLO
import cv2
import dil
model=YOLO("yolov8n.pt")






labels_tr = dil.Label_tr()
results=model("https://gaiadergi.com/wp-content/uploads/2016/01/Hayvanlar-Efsane-1.jpg",save=True,project="görüntüler",name="human1")
results=results[0]
# img=results[0].plot()
img=results.orig_img.copy()#orjinal veriyi kopyalamak
dil.tr_dil(labels_tr, results, img)
cv2.imshow("sonuc",img)
cv2.waitKey(0)
# cv2.destroyAllWindows()


