import cv2
from ultralytics import YOLO

# YOLO modelini yükle
model = YOLO('yolo11n.pt')

# Kameradan video akışını başlat
cap = cv2.VideoCapture(0)

while True:
    # Kameradan bir kare al
    ret, frame = cap.read()

    if not ret:
        break


    results = model(frame)


    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box koordinatları
            confidence = box.conf[0]  # Güven oranı
            class_id = int(box.cls[0])  # Sınıf ID'si

            # Sınıf adı ve güven oranını al
            label = f"{model.names[class_id]}: {confidence:.2f}"

            if confidence > 0:
                # Bounding box'u çizin
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                # Sınıf adını ve güven oranını ekleyin
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    # Sonuçları ekranda göster
    cv2.imshow("YOLO Object Detection", frame)

    # 'q' tuşuna basıldığında çık
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Kaynakları serbest bırak
cap.release()
cv2.destroyAllWindows()
