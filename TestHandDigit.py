import cv2
import numpy as np
import matplotlib.pyplot as plt

theta = np.loadtxt('theta.txt')
theta.shape[0]

def sigmoid(s):
    return 1 / (1 + np.exp(-s))

image = cv2.imread('image3.png')

image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
image_gray = cv2.GaussianBlur(image_gray, (5, 5), 0)
_, im_th = cv2.threshold(image_gray, 155, 255, cv2.THRESH_BINARY_INV)

plt.imshow(im_th)
# plt.show()

ctrs, _ = cv2.findContours(im_th.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

rects = [cv2.boundingRect(ctr) for ctr in ctrs] 

for rect in rects:

    cv2.rectangle(image, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), (0, 0, 255), 3)

    length = int(rect[3] * 1.6)

    pt1 = int(rect[1] + rect[3] // 2 - length // 2)
    pt2 = int(rect[0] + rect[2] // 2 - length // 2)

    roi = im_th[pt1:pt1 + length, pt2:pt2 + length]
    roi = cv2.dilate(roi, (3, 3))
    roi = cv2.resize(roi, (28, 28), interpolation=cv2.INTER_AREA)
    # print(roi)
    # plt.imshow(roi)
     
    x = np.array([roi]).reshape(1, 28*28)
    one = np.ones((x.shape[0], 1))
    x = np.concatenate((one, x), axis=1)
    prediction = sigmoid(np.dot(x, theta.T))
    cv2.putText(image, str(int(prediction)), (rect[0], rect[1]), cv2.FONT_HERSHEY_DUPLEX, 1.5, (0, 255, 255), 1)
    
plt.imshow(image)
plt.show()