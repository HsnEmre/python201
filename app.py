from ultralytics import YOLO
import cv2


# model=YOLO("yolov8m.pt")
# results=model("https://ultralytics.com/images/bus.jpg")
# results[0].show()


model=YOLO("yolov8l.pt")


labels_tr = {
    # İnsan ve hayvanlar
    "person": "insan", "bird": "kuş", "cat": "kedi", "dog": "köpek",
    "horse": "at", "sheep": "koyun", "cow": "inek", "elephant": "fil",
    "bear": "ayı", "zebra": "zebra", "giraffe": "zürafa",

    # Taşıtlar
    "bicycle": "bisiklet", "car": "araba", "motorcycle": "motosiklet",
    "airplane": "uçak", "bus": "otobüs", "train": "tren", "truck": "kamyon",
    "boat": "tekne",

    # Sokak
    "traffic light": "trafik ışığı", "fire hydrant": "yangın musluğu",
    "stop sign": "dur işareti", "parking meter": "otopark sayacı", "bench": "bank",

    # Eşyalar
    "backpack": "sırt çantası", "umbrella": "şemsiye", "handbag": "el çantası",
    "tie": "kravat", "suitcase": "valiz",

    # Spor
    "frisbee": "frizbi", "skis": "kayak", "snowboard": "snowboard",
    "sports ball": "top", "kite": "uçurtma", "baseball bat": "beyzbol sopası",
    "baseball glove": "beyzbol eldiveni", "skateboard": "kaykay",
    "surfboard": "sörf tahtası", "tennis racket": "tenis raketi",

    # Mutfak
    "bottle": "şişe", "wine glass": "şarap kadehi", "cup": "fincan",
    "fork": "çatal", "knife": "bıçak", "spoon": "kaşık", "bowl": "kase",

    # Yiyecekler
    "banana": "muz", "apple": "elma", "sandwich": "sandviç", "orange": "portakal",
    "broccoli": "brokoli", "carrot": "havuç", "hot dog": "sosisli",
    "pizza": "pizza", "donut": "donut", "cake": "pasta",

    # Mobilya
    "chair": "sandalye", "couch": "kanepe", "potted plant": "saksı bitkisi",
    "bed": "yatak", "dining table": "yemek masası", "toilet": "klozet",

    # Elektronik
    "tv": "televizyon", "laptop": "dizüstü bilgisayar", "mouse": "fare",
    "remote": "kumanda", "keyboard": "klavye", "cell phone": "cep telefonu",

    # Ev aletleri
    "microwave": "mikrodalga", "oven": "fırın", "toaster": "tost makinesi",
    "sink": "lavabo", "refrigerator": "buzdolabı",

    # Diğer
    "book": "kitap", "clock": "saat", "vase": "vazo", "scissors": "makas",
    "teddy bear": "oyuncak ayı", "hair drier": "saç kurutma makinesi",
    "toothbrush": "diş fırçası",
}
results=model("https://gaiadergi.com/wp-content/uploads/2016/01/Hayvanlar-Efsane-1.jpg",save=True,project="görüntüler",name="human1")
results=results[0]
# img=results[0].plot()
img=results.orig_img.copy()#orjinal veriyi kopyalamak



for box in results.boxes:
 x1,y1,x2,y2=map(int,box.xyxy[0])
 cls=int(box.cls[0])
 conf=float(box.conf[0])
 eng=results.names[cls]
 tr=labels_tr.get(eng,eng)#okunan objenin türkesi yoksa eng yaz
 label=f"{tr} {conf:.2f}"   

 cv2.rectangle(img,(x1,y1),(x2,y2),(255,0,0),2)
 (tw,th),_=cv2.getTextSize(label,cv2.FONT_HERSHEY_SCRIPT_COMPLEX,0.6,2)

 cv2.rectangle(img,(x1,y1-th-10),(x1+tw,y1),(255,0,0),-1)
 cv2.putText(img,label,(x1,y1-5),cv2.FONT_HERSHEY_SCRIPT_COMPLEX,0.6,(255,255,255),2)



cv2.imshow("sonuc",img)
cv2.waitKey(0)
# cv2.destroyAllWindows()


