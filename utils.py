from pathlib import Path
import os
from glob import glob
import cv2
import numpy as np

SUPPORTED_EXT = ("*.jpg", "*.jpeg", "*.png", "*.tif", "*.tiff", "*.bmp")

def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)

def find_images(input_path):
    input_path = str(input_path)
    if os.path.isfile(input_path):
        return [input_path]
    paths = []
    for ext in SUPPORTED_EXT:
        paths.extend(glob(os.path.join(input_path, ext)))
    return sorted(paths)

def read_image(path):
    img = cv2.imread(str(path))
    if img is None:
        raise RuntimeError("Goruntu okunamadi: " + str(path))
    return img

def resize_for_screen(img, max_w=1500, max_h=900):
    h, w = img.shape[:2]
    scale = min(max_w / w, max_h / h, 1.0)
    out = cv2.resize(img, None, fx=scale, fy=scale)
    return out, scale

def select_two_points(img, window_name, text):
    points = []
    display, scale = resize_for_screen(img)

    def cb(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(points) < 2:
            points.append((x, y))
            print(window_name + " nokta " + str(len(points)) + ": " + str((x, y)))

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window_name, cb)
    print("")
    print(text)
    print("2 nokta tikla. ESC iptal eder.")

    while True:
        temp = display.copy()
        cv2.putText(temp, text[:95], (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0,255,255), 2)
        for i, p in enumerate(points):
            cv2.circle(temp, p, 6, (0,0,255), -1)
            cv2.putText(temp, str(i+1), (p[0]+8, p[1]-8), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
        if len(points) == 2:
            cv2.line(temp, points[0], points[1], (255,0,0), 2)
        cv2.imshow(window_name, temp)
        key = cv2.waitKey(20) & 0xFF
        if key == 27:
            cv2.destroyWindow(window_name)
            raise RuntimeError("Secim iptal edildi.")
        if len(points) == 2:
            cv2.waitKey(250)
            break

    cv2.destroyWindow(window_name)
    p1 = np.array(points[0], dtype=np.float32) / scale
    p2 = np.array(points[1], dtype=np.float32) / scale
    return p1, p2
