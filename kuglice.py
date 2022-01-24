import numpy as np
import cv2
from sklearn import datasets
import matplotlib.pyplot as plt
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
from google.colab.patches import cv2_imshow
from skimage.feature import peak_local_max
from skimage.morphology import watershed
from scipy import ndimage


def detect_lines(img):
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    kernel_size = 5
    blur_gray = cv2.GaussianBlur(gray_img, (kernel_size, kernel_size), 0)

    edges_img = cv2.Canny(blur_gray, 50, 150)
    min_line_length = 50

    line_image = np.copy(img)
    lines = cv2.HoughLinesP(image=edges_img, rho=1, theta=np.pi / 180, threshold=10, lines=np.array([]),
                            minLineLength=min_line_length, maxLineGap=20)
    array_lines = []
    for line in lines:
        for x1, y1, x2, y2 in line:
            if x1 == x2 and abs(y2 - y1) > 350:
                array_lines.append(line)
    array_lines = sorted(array_lines, key=lambda x: x[0][0])
    cv2.line(line_image, (array_lines[1][0][0], array_lines[1][0][1]),
             (array_lines[1][0][2], array_lines[1][0][3]), (255, 0, 0), 5)
    cv2.line(line_image, (array_lines[2][0][0], array_lines[2][0][1]),
             (array_lines[2][0][2], array_lines[2][0][3]), (255, 0, 0), 5)
    lines_edges = cv2.addWeighted(img, 0.8, line_image, 1, 0)
    cv2_imshow(lines_edges)
    return array_lines[2], array_lines[1]


def detect_circle(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    cnts = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]

    array_list = []
    for c in cnts:
        (x, y), r = cv2.minEnclosingCircle(c)
        if 4.3 < r < 5:
            array_list.append([x, y, r])
            cv2.drawContours(image, [c], 0, (0, 252, 0), -1)
    return array_list


def process_video(video_path):
    hit = 0
    frame_num = 0
    cap = cv2.VideoCapture(video_path)
    cap.set(1, frame_num)

    line_left, line_right = [], []
    previous = 0
    while True:
        frame_num += 1
        ret_val, frame = cap.read()
        if not ret_val:
            break
        if frame_num == 1:
            line_coords = detect_lines(frame)
            line_right, line_left = line_coords
            print(f"desno:{line_right},levo:{line_left}")
        circles = detect_circle(frame)
        for circle in circles:
            if (abs(line_right[0][0] - circle[0]) < 20.9 or abs(circle[0] - line_left[0][0]) < 20.9) and (
                    frame_num > (previous + 1)):
                hit += 1
                previous = frame_num
    cap.release()
    return hits


if __name__ == '__main__':
    expected = [7, 18, 21, 18, 10, 32, 13, 15, 14, 24]
    actual = []
    for i in range(1, 11):
        actual.append(process_video(f"/content/video{i}.mp4"))
    print(mean_absolute_error(actual, expected))
