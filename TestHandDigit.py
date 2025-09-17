import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread('image.png')
if image is None:
    raise FileNotFoundError("Image not found. Check the path!")

image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
image_gray = cv2.GaussianBlur(image_gray, (5, 5), 0)
_, im_th = cv2.threshold(image_gray, 155, 255, cv2.THRESH_BINARY_INV)

plt.imshow(im_th)

ctrs, _ = cv2.findContours(im_th.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

rects = [cv2.boundingRect(ctr) for ctr in ctrs]

for rect in rects:

    cv2.rectangle(image, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), (0, 0, 255), 3)

    length = int(rect[3] * 1.6)

    pt1 = int(rect[1] + rect[3] // 2 - length // 2)
    pt2 = int(rect[0] + rect[2] // 2 - length // 2)

    roi = im_th[pt1:pt1 + length, pt2:pt2 + length]
    roi = cv2.resize(roi, (28, 28), interpolation=cv2.INTER_AREA)
    print(roi)
    plt.imshow(roi)
plt.imshow(image)