from ultralytics import YOLO
import cv2


model=YOLO("yolov8m.pt")
# result=model(source=1,show=True)
cap=cv2.VideoCapture(1)

while True:
    ret, frame=cap.read()
    if not ret:
        break

    result=model(frame,stream=True,classes=[0,43])

    for r in result:
        frame=r.plot()
        count=len(r.boxes)
        cv2.putText(frame,f"Nesne Sayısı: {count}"
                    ,(20,40)
                    ,cv2.FONT_HERSHEY_SIMPLEX
                    ,0.8 
                    ,(0,255,0)
                    ,2)

        #insan tespit edilirse yaz
        for box in r.boxes:
            cls=int(box.cls[0])#kutuların içinde gezerken 0 ı alıcam insan class ı oluyor
            if cls==0:
                cv2.putText(frame,"İnsan tespit edildi"
                            ,(20,80)
                            ,cv2.FONT_HERSHEY_SIMPLEX
                            ,0.9
                            ,(0,0,255)
                            ,3)
                break
            if cls==43:
                cv2.putText(frame,"bıcak tespit edildi"
                                            ,(20,120)
                                            ,cv2.FONT_HERSHEY_SIMPLEX
                                            ,0.9
                                            ,(0,0,255)
                                            ,3)
                break
        #pencere boyutunu hazırla
        frame=cv2.resize(frame,(640,640))    
        cv2.imshow("Akıllı Kamera",frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
cap.release()
cv2.destroyAllWindows()

            
