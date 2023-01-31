import os

import numpy as np
from lib import SyncVideoSet
from lib import ImageProcessingFunctions as ip
from lib import StereoCalibrationFunctions as sc
import cv2 as cv
import matplotlib.pyplot as plt
import matplotlib
from scipy import ndimage
import scipy

matplotlib.use('TkAgg')

path_all = ['/Volumes/Disk_I/06_12_2022/4',
            '/Volumes/Disk_I/06_12_2022/5',
            '/Volumes/Disk_I/07_12_2022/2',
            '/Volumes/Disk_I/07_12_2022/3',
            '/Volumes/Disk_I/07_12_2022/4',
            '/Volumes/Disk_I/07_12_2022/5',
            ]

path_in = '/Volumes/Disk_M/Predator/10_12_2022/2'

deployment = SyncVideoSet(path_in, mode=1)

deployment.detect_calibration_videos()

deployment.get_time_lag(method='custom', number_of_video_chapters_to_evaluate=4)

deployment.get_calibration_videos()

deployment.stereo_parameters = sc.compute_stereo_params(deployment, extract_new_images_from_video=True)

deployment.save()

filenames1 = sorted(os.listdir('images/Stick_images_06_12_2022_5/GH010049/'))
filenames2 = sorted(os.listdir('images/Stick_images_06_12_2022_5/GH010059/'))

mtx1 = deployment.stereo_parameters['mtx1']
mtx2 = deployment.stereo_parameters['mtx2']
R = deployment.stereo_parameters['R']
T = deployment.stereo_parameters['T']
all_lengths = []
z = []

for name1, name2 in zip(filenames1, filenames2):
    try:
        img1 = cv.imread('images/Stick_images_06_12_2022_5/GH010049/' + name1)
        res1 = sc.detect_measurement_pole(img1)

        img2 = cv.imread('images/Stick_images_06_12_2022_5/GH010059/' + name2)
        res2 = sc.detect_measurement_pole(img2)

        res_real_cor = sc.triangulate(res1, res2, mtx1, mtx2, R, T)
        length_pole = np.sum((res_real_cor[0, :] - res_real_cor[1, :])**2)**0.5
        all_lengths.append(length_pole)
        z.append(np.mean(res_real_cor[:, 2]))

        plt.figure()
        plt.subplot(1, 2, 1)
        plt.imshow(img1[:, :, ::-1])
        plt.scatter(res1[:, 0], res1[:, 1])
        plt.subplot(1, 2, 2)
        plt.imshow(img2[:, :, ::-1])
        plt.scatter(res2[:, 0], res2[:, 1])
        plt.title('Length of pole: ' + str(np.round(length_pole / 10, 2)) + 'cm')

    except:
        pass

plt.figure()
plt.scatter(z, all_lengths)
plt.hlines(1001, 0, 5000)
plt.xlim(0, 5000)
plt.ylim(950, 1100)

plt.figure()
plt.subplot(1, 2, 1)
plt.imshow(img1[:, :, ::-1])
plt.scatter(res1[:, 0], res1[:, 1])
plt.subplot(1, 2, 2)
plt.imshow(img2[:, :, ::-1])
plt.scatter(res2[:, 0], res2[:, 1])
plt.title('Length of pole: ' + str(np.round(length_pole/10,2)) + 'cm')

