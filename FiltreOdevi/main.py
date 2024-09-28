import cv2
import numpy as np
import matplotlib.pyplot as plt


image = cv2.imread('lena.png')

# Ortalama filtre uygulama
mean_filtered = cv2.blur(image, (3, 3))

# Laplace filtresi uygulama
laplace_filtered = cv2.Laplacian(mean_filtered, cv2.CV_64F)

# Sonuçları normalize etme
laplace_filtered = cv2.convertScaleAbs(laplace_filtered)

# Görüntüleri yan yana birleştirme
result_image = np.hstack((image, mean_filtered, laplace_filtered))

# Sonucu gösterme
plt.figure(figsize=(12, 6))
plt.title('Orijinal - Ortalama Filtre - Laplace Filtre')
plt.imshow(cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.show()
