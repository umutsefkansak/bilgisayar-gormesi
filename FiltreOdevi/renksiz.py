import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread('lena.png', cv2.IMREAD_GRAYSCALE)

# Mean filtre uygulama
mean_filtered = cv2.blur(image, (3, 3))

# Laplace filtresi uygulama
laplace_filtered = cv2.Laplacian(mean_filtered, cv2.CV_64F)

# Görüntüleri yan yana birleştirme
result_image = np.hstack((image, mean_filtered, laplace_filtered))

# Sonucu gösterme
plt.figure(figsize=(12, 6))
plt.title('Orijinal - Ortalama Filtre - Laplace Filtre')
plt.imshow(result_image, cmap='gray')
plt.axis('off')
plt.show()
