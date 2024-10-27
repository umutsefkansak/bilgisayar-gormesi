import cv2
import numpy as np
import matplotlib.pyplot as plt

# 1. Görüntüyü yükleyin ve gri seviyeye dönüştürün
image = cv2.imread('dikdortgen.png')  # Görüntüyü okuyun
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Griye dönüştürün

# 2. Yatay türev için kernel tanımlama ve filtreleme
kernel_x = np.array([[-1, 0, 1]])  # Yatay türev matrisi
grad_x = cv2.filter2D(gray_image, cv2.CV_64F, kernel_x)  # Yatay türev uygulama

# 3. Dikey türev için kernel tanımlama ve filtreleme
kernel_y = np.array([[-1], [0], [1]])  # Dikey türev matrisi
grad_y = cv2.filter2D(gray_image, cv2.CV_64F, kernel_y)  # Dikey türev uygulama

# 4. Türevleri normalize etme
grad_x = cv2.normalize(grad_x, None, 0, 255, cv2.NORM_MINMAX)
grad_y = cv2.normalize(grad_y, None, 0, 255, cv2.NORM_MINMAX)

# 5. Türevleri birleştirme (magnitude hesaplama)
magnitude = np.sqrt(grad_x**2 + grad_y**2)  # Hipotenüs hesaplama (magnitude)
magnitude = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)

# Sonuçları kaydedin
cv2.imwrite('grayscale_image.png', gray_image)
cv2.imwrite('horizontal_derivative.png', grad_x)
cv2.imwrite('vertical_derivative.png', grad_y)
cv2.imwrite('magnitude.png', magnitude)

# Sonuçları gösterme
plt.figure(figsize=(10, 8))
plt.subplot(2, 2, 1)
plt.title('Grayscale Image')
plt.imshow(gray_image, cmap='gray')

plt.subplot(2, 2, 2)
plt.title('Horizontal Derivative')
plt.imshow(grad_x, cmap='gray')

plt.subplot(2, 2, 3)
plt.title('Vertical Derivative')
plt.imshow(grad_y, cmap='gray')

plt.subplot(2, 2, 4)
plt.title('Magnitude')
plt.imshow(magnitude, cmap='gray')

plt.show()
