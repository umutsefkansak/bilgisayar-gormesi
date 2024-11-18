import cv2
import numpy as np

# Kamerayı başlat (0, varsayılan kamerayı temsil eder)
cap = cv2.VideoCapture(0)

while True:
    # Kameradan bir kare yakala
    ret, frame = cap.read()

    # Eğer görüntü alındıysa işlemleri uygula
    if not ret:
        break

    image = frame

    # Ortalama filtre uygulama
    mean_filtered = cv2.blur(image, (3, 3))

    # Laplace filtresi uygulama
    laplace_filtered = cv2.Laplacian(mean_filtered, cv2.CV_64F)

    # Sonuçları normalize etme
    laplace_filtered = cv2.convertScaleAbs(laplace_filtered)

    # Görüntüleri yan yana birleştirme
    result_image = np.hstack((image, mean_filtered, laplace_filtered))

    # Sonucu gösterme
    cv2.imshow('Orijinal - Ortalama Filtre - Laplace Filtre', result_image)

    # 'q' tuşuna basıldığında çıkış yap
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


# Kamerayı serbest bırak
cap.release()
cv2.destroyAllWindows()
