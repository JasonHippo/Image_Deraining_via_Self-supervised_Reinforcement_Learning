import cv2
import numpy as np
import os
src_dir = '../dataset/Rain100L/test/RDP_before_binarization/'
binary_mask_dir = '../dataset/Rain100L/test/RDP5/'
os.makedirs(binary_mask_dir, exist_ok=True)

for i in os.listdir(src_dir):
    rain = cv2.imread(os.path.join(src_dir, i), 0)
    rain_binary = np.where(rain > 5, 255, 0)
    cv2.imwrite(os.path.join(binary_mask_dir, i), rain_binary)