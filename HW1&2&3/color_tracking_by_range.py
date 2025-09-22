import cv2
import numpy as np
import imutils

def callback(value):
    pass

# Tạo trackbar để chỉnh HSV
def setup_trackbars(range_filter):
    cv2.namedWindow("Trackbars")

    for i in ["MIN", "MAX"]:
        v = 0 if i == "MIN" else 255
        for j in range_filter:
            cv2.createTrackbar("%s_%s" % (j, i), "Trackbars", v, 255, callback)

def get_trackbar_values(range_filter):
    values = []
    for i in ["MIN", "MAX"]:
        for j in range_filter:
            v = cv2.getTrackbarPos("%s_%s" % (j, i), "Trackbars")
            values.append(v)
    return values

def main():
    range_filter = "HSV"
    cap = cv2.VideoCapture(0)

    setup_trackbars(range_filter)

    while True:
        ret, frame = cap.read()
        if not ret:
            continue   # bỏ qua frame lỗi, không thoát luôn

        # Làm mượt ảnh
        blur = cv2.GaussianBlur(frame, (11, 11), 0)
        hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)

        # Lấy giá trị HSV từ trackbar
        v1_min, v2_min, v3_min, v1_max, v2_max, v3_max = get_trackbar_values(range_filter)
        lower = np.array([v1_min, v2_min, v3_min])
        upper = np.array([v1_max, v2_max, v3_max])

        # Tạo mask
        mask = cv2.inRange(hsv, lower, upper)
        mask = cv2.erode(mask, None, iterations=2)
        mask = cv2.dilate(mask, None, iterations=2)

        # Tìm contours
        cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)

        if len(cnts) > 0:
            c = max(cnts, key=cv2.contourArea)
            (x, y), radius = cv2.minEnclosingCircle(c)

            if radius > 10:
                cv2.circle(frame, (int(x), int(y)), int(radius), (0, 0, 255), 2)

        # Hiển thị
        cv2.imshow("Frame", frame)
        cv2.imshow("Mask", mask)

        # Nhấn 'q' để thoát
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
