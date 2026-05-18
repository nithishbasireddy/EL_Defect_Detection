import cv2
import matplotlib.pyplot as plt

mask = cv2.imread("data/raw/test_crack/ann/2070.png", 0)

plt.imshow(mask)
plt.colorbar()
plt.show()