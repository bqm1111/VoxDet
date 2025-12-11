import cv2
import numpy as np
img_path = "data/tartanair/CarWelding/Data_easy/P000/depth_lcam_right/000000_lcam_right_depth.png"

depth_rgba = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
print(depth_rgba)
depth = depth_rgba.view("<f4")

print(np.squeeze(depth, axis=-1))