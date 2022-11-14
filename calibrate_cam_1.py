import os.path

import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pickle


nx = 9
ny = 6


def cal_calibrate_params(file_paths):
    object_points = []
    image_points = []
    # 对象点的坐标
    objp = np.zeros((nx * ny, 3), np.float32)
    objp[:, :2] = np.mgrid[0:nx, 0:ny].T.reshape(-1, 2)
    # 检测每幅图像角点坐标
    for file_path in file_paths:
        if file_path.endswith('.'):
            continue
        print (os.path.join(file_paths , file_path))
        img = mpimg.imread(os.path.join(file_paths , file_path))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        rect, corners = cv2.findChessboardCorners(gray, (nx, ny), None)
        if rect == True:
            object_points.append(objp)
            image_points.append(corners)
    # 获取畸变系数
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(object_points, image_points, gray.shape[::-1], None, None)
    return ret, mtx, dist, rvecs, tvecs

cal_calibrate_params('./camera_cal')
