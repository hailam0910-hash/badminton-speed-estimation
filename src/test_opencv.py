import cv2
import numpy as np

print("OpenCV version:", cv2.__version__)
print("NumPy version:", np.__version__)

img = np.zeros((300, 500, 3), dtype=np.uint8)
cv2.putText(img, "OpenCV OK!", (120, 150),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

cv2.imshow("Test Window", img)
cv2.waitKey(0)
cv2.destroyAllWindows()