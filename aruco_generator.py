import cv2
import numpy as np
import os

id = 3
size = 300
name = "{0}_{1}x{1}_DICT_4x4_50.png".format(id, size)

tag = np.zeros((size, size, 1), dtype="uint8")
cv2.aruco.drawMarker(cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_50), id, size, tag, 1)
# write the generated ArUCo tag to disk and then display it to our
# screen

cv2.imwrite(os.path.join(os.getcwd(), name), tag)
cv2.imshow("ArUCo Tag", tag)
cv2.waitKey(0)