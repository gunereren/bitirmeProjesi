import math
import time

import cv2
import cvzone  # Opencv'yi daha kullanışlı hale getirmesi için kullanılır. Nesne takibi ve tanıma için kullanılır
from ultralytics import YOLO  # nesne algılama modelini içerir. Bunu YOLO kütüphanesine aktarır.

confidence = 0.3  # Doğruluk oranının kaç olmasını belirleriz.

cap = cv2.VideoCapture(1)  # 0 Pc kamera, 1 harici kamera
cap.set(3, 1280)
cap.set(4, 720)
# cap = cv2.VideoCapture("../Videos/motorbikes.mp4")  # For Video

model = YOLO("C:\\Users\\Eren\\Desktop\\SE\\bitirme\\best.pt")
model = YOLO("C:\\Users\\Eren\\Desktop\\SE\\bitirme\\last.pt")

classNames = ['healthy', 'glaucoma']  # Algılanacak sınıflar


# Zaman ölçümü ve sürekli video akışı için bir döngü oluşturulur.

prev_frame_time = 0
new_frame_time = 0

while True:
    new_frame_time = time.time()
    success, img = cap.read()

    # YOLO ile Nesne Algılama ve Sonuç Görselleştirme

    results = model(img, stream=True, verbose=False)
    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            cv2.rectangle(img,(x1,y1),(x2,y2),(255,0,255),3)
            w, h = x2 - x1, y2 - y1
            # Confidence

            conf = math.ceil((box.conf[0] * 100)) / 100
            # Class Name
            cls = int(box.cls[0])
            if conf > confidence:
                print(classNames[cls])

                if classNames[cls] == 'glaucoma':
                    color = (0, 255, 0)
                else:
                    color = (0, 0, 255)

                cvzone.cornerRect(img, (x1, y1, w, h), colorC=color, colorR=color)
                cvzone.putTextRect(img, f'{classNames[cls].upper()} {int(conf*100)}%',
                                   (max(0, x1), max(35, y1)), scale=1, thickness=1, colorR=color,
                                   colorB=color)

    # FPS Hesaplama ve Ekran Gösterimi

    fps = 1 / (new_frame_time - prev_frame_time)
    prev_frame_time = new_frame_time
    # print(fps)

    cv2.imshow("Image", img)
    cv2.waitKey(1)
