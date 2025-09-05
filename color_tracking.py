import cv2
import numpy as np
import imutils

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Làm mượt ảnh để giảm nhiễu
    blur = cv2.GaussianBlur(frame, (11, 11), 0)

    # Chuyển sang không gian màu HSV
    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)

    # Ngưỡng màu (ví dụ: đỏ)
    lower = np.array([15, 128, 33])
    upper = np.array([186, 255, 255])

    # Tạo mask nhị phân cho vùng màu trong khoảng [lower, upper]
    mask = cv2.inRange(hsv, lower, upper)

    # Xử lý hình thái học để loại bỏ nhiễu
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)

    # Tìm contours
    ball_cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    ball_cnts = imutils.grab_contours(ball_cnts)

    if len(ball_cnts) > 0:
        # Chọn contour có diện tích lớn nhất
        c = max(ball_cnts, key=cv2.contourArea)
        (x, y), radius = cv2.minEnclosingCircle(c)

        if radius > 10:
            cv2.circle(frame, (int(x), int(y)), int(radius), (0, 0, 255), 2)

    # Hiển thị kết quả
    cv2.imshow("frame", frame)
    cv2.imshow("mask", mask)

    # Nhấn phím 'q' để thoát
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
