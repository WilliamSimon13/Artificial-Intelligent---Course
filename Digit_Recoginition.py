import cv2
import numpy as np

# Load tham số đã train
theta = np.loadtxt("theta.txt").reshape(-1, 1)  # (785,1)

def sigmoid(s):
    return 1 / (1 + np.exp(-s))

def predict_digit(roi, theta, low=0.2, high=0.8):
    """
    Nhận diện ROI là 0 hoặc 1, hoặc -1 nếu không chắc chắn
    """
    roi = roi / 255.0  # chuẩn hóa
    x = roi.reshape(1, -1)  # (1,784)
    one = np.ones((1, 1))
    x = np.concatenate((one, x), axis=1)  # (1,785)
    prob = sigmoid(np.dot(x, theta))  # xác suất
    if prob >= high:
        return 1
    elif prob <= low:
        return 0
    else:
        return -1

# Mở camera
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Chuyển grayscale + xử lý nhị phân
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, im_th = cv2.threshold(blur, 155, 255, cv2.THRESH_BINARY_INV)

    # Tìm contours
    ctrs, _ = cv2.findContours(im_th.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    rects = [cv2.boundingRect(ctr) for ctr in ctrs]

    for rect in rects:
        # Vẽ bounding box
        cv2.rectangle(frame, (rect[0], rect[1]), 
                      (rect[0] + rect[2], rect[1] + rect[3]), (0, 0, 255), 2)

        length = int(rect[3] * 1.6)
        pt1 = int(rect[1] + rect[3] // 2 - length // 2)
        pt2 = int(rect[0] + rect[2] // 2 - length // 2)

        roi = im_th[pt1:pt1 + length, pt2:pt2 + length]

        if roi.shape[0] > 0 and roi.shape[1] > 0:
            roi = cv2.resize(roi, (28, 28), interpolation=cv2.INTER_AREA)
            roi = cv2.dilate(roi, (3, 3))

            prediction = predict_digit(roi, theta)

            if prediction != -1:  # chỉ hiển thị khi chắc chắn là 0 hoặc 1
                cv2.putText(frame, str(prediction), (rect[0], rect[1] - 10),
                            cv2.FONT_HERSHEY_DUPLEX, 1, (0, 255, 255), 2)

    cv2.imshow("Digit Recognition (0/1 only)", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
