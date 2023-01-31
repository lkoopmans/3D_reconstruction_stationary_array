import cv2 as cv
import glob
import numpy as np
from scipy import linalg
from lib import ImageProcessingFunctions as ip
import os
import scipy


def index_images(image_names, t_interval, delta_t):
    images = []
    c_time = 0

    for image_name in image_names:
        if t_interval[1] >= c_time >= t_interval[0]:
            im = cv.imread(image_name, 1)
            images.append(im)
        c_time += delta_t

    return images


def compute_stereo_params(deployment, extract_new_images_from_video=True):
    print('---------- COMPUTING STEREO PARAMETERS ----------')
    for i in range(len(deployment.calibration_video_names)):
        if '_cal' not in deployment.calibration_video_names[i]:
            temp = deployment.calibration_video_names[i]
            temp = temp.split('.')
            deployment.calibration_video_names[i] = temp[0] + '_cal.' + temp[1]

    delta_t = 1 / 2
    for i in range(2):
        calibration_path = os.path.join(deployment.path_in, deployment.camera_names[i],
                                        deployment.calibration_video_names[i])

        if extract_new_images_from_video:
            ip.extract_frames(calibration_path, delta_t)

    base_name_1 = os.path.splitext(os.path.basename(deployment.calibration_video_names[0]))[0]
    base_name_2 = os.path.splitext(os.path.basename(deployment.calibration_video_names[1]))[0]

    output_directory_1 = os.path.join('images/detection_images', base_name_1, '*')
    output_directory_2 = os.path.join('images/detection_images', base_name_2, '*')

    rows = 8  # number of checkerboard rows.
    columns = 11  # number of checkerboard columns.
    world_scaling = 30  # change this to the real world square size

    # sort list
    output_directory_1 = sorted(glob.glob(output_directory_1))
    output_directory_2 = sorted(glob.glob(output_directory_2))

    images_1 = []
    for image_name in output_directory_1:
        im = cv.imread(image_name, 1)
        images_1.append(im)

    images_2 = []
    for image_name in output_directory_2:
        im = cv.imread(image_name, 1)
        images_2.append(im)

    # criteria used by checkerboard pattern detector.
    # Change this if the code can't find the checkerboard
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # coordinates of squares in the checkerboard world space
    objp = np.zeros((rows * columns, 3), np.float32)
    objp[:, :2] = np.mgrid[0:rows, 0:columns].T.reshape(-1, 2)
    objp = world_scaling * objp

    # frame dimensions. Frames should be the same size.
    width = images_1[0].shape[1]
    height = images_1[0].shape[0]

    # Pixel coordinates of checkerboards
    image_points_1 = []  # 2d points in image plane.
    image_points_2 = []  # 2d points in image plane.

    # coordinates of the checkerboard in checkerboard world space.
    object_points_1 = []  # 3d point in real world space
    object_points_2 = []  # 3d point in real world space

    c1 = 0
    c2 = 0

    min_max = np.array([0, 0])
    c_check_detects = 0

    number_of_images = min(len(images_1), len(images_2))
    both_detected1 = []
    both_detected2 = []

    for c in range(int(number_of_images)):
        gray1 = cv.cvtColor(images_1[c], cv.COLOR_RGB2GRAY)
        gray2 = cv.cvtColor(images_2[c], cv.COLOR_RGB2GRAY)

        # find the checkerboard
        ret1, corners1 = cv.findChessboardCorners(gray1, (rows, columns), None)
        ret2, corners2 = cv.findChessboardCorners(gray2, (rows, columns), None)

        conv_size = (3, 3)

        if ret1 or ret2:
            print('Image', str(c + 1) + '/' + str(number_of_images) + '. Checkerboard detected.')
        else:
            print('Image', str(c + 1) + '/' + str(number_of_images) + '. No checkerboard detected.')

        if ret1:
            corners1 = cv.cornerSubPix(gray1, corners1, conv_size, (-1, -1), criteria)

            # cv.waitKey(500)

            object_points_1.append(objp)
            image_points_1.append(corners1)
            c1 += 1

        if ret2:
            corners2 = cv.cornerSubPix(gray2, corners2, conv_size, (-1, -1), criteria)

            # cv.waitKey(500)

            object_points_2.append(objp)
            image_points_2.append(corners2)
            c2 += 1

        if ret1 and ret2:
            c_check_detects += 1

            if c_check_detects == 1:
                min_max[0] = delta_t * c

            min_max[1] = delta_t * c

            both_detected1.append(c1 - 1)
            both_detected2.append(c2 - 1)

        if not (ret1 or ret2):
            os.remove(output_directory_1[c])
            os.remove(output_directory_2[c])

    ret1, mtx1, dist1, rvecs1, tvecs1 = cv.calibrateCamera(object_points_1, image_points_1, (width, height), None, None)
    ret2, mtx2, dist2, rvecs2, tvecs2 = cv.calibrateCamera(object_points_2, image_points_2, (width, height), None, None)

    stereocalibration_flags = cv.CALIB_FIX_INTRINSIC
    ret, CM1, dist1, CM2, dist2, R, T, E, F = cv.stereoCalibrate([object_points_1[i] for i in both_detected1],
                                                                 [image_points_1[i] for i in both_detected1],
                                                                 [image_points_2[i] for i in both_detected2], mtx1,
                                                                 dist1,
                                                                 mtx2, dist2, (width, height), criteria=criteria,
                                                                 flags=stereocalibration_flags)

    stereo_parameters = {'mtx1': mtx1, 'dist1': dist1, 'mtx2': mtx2, 'dist2': dist2, 'ret': ret, 'ret1': ret1, 'ret2':
        ret2, 'CM1': CM1, 'CM2': CM2, 'R': R, 'T': T, 'E': E, 'F': F}

    return stereo_parameters


def triangulate(x1, x2, mtx1, mtx2, R, T):
    # compute projection matrices
    p1 = mtx1 @ np.concatenate([np.eye(3), [[0], [0], [0]]], axis=-1)
    p2 = mtx2 @ np.concatenate([R, T], axis=-1)  # projection matrix for C2

    p3ds = []
    for uv1, uv2 in zip(x1, x2):
        _p3d = direct_linear_transform(p1, p2, uv1, uv2)
        p3ds.append(_p3d)

    return np.array(p3ds)


def direct_linear_transform(p1, p2, point1, point2):
    A = [point1[1] * p1[2, :] - p1[1, :], p1[0, :] - point1[0] * p1[2, :], point2[1] * p2[2, :] - p2[1, :],
         p2[0, :] - point2[0] * p2[2, :]]
    A = np.array(A).reshape((4, 4))

    B = A.transpose() @ A
    U, s, Vh = linalg.svd(B, full_matrices=False)

    return Vh[3, 0:3] / Vh[3, 3]


def detect_measurement_pole(img):
    imgHSV = cv.cvtColor(img, cv.COLOR_BGR2HSV)

    mask = cv.inRange(imgHSV, (170, 100, 10), (180, 150, 255)) + cv.inRange(imgHSV, (0, 100, 10), (5, 150, 255))

    mask2 = scipy.signal.convolve2d(mask, np.ones((10, 10)), 'same')
    #mask[mask2 < 255 * 2] = 0
    xx, yy = np.meshgrid(np.arange(1, img.shape[1] + 1), np.arange(1, img.shape[0] + 1))

    x = xx[mask == 255]
    y = yy[mask == 255]

    xmin = int(min(x))
    xmax = int(max(x))

    ymin = int(np.mean(y[x == xmin]))
    ymax = int(np.mean(y[x == xmax]))

    p

    return np.array([[xmin, ymin], [xmax, ymax]])
