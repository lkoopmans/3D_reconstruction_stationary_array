import numpy as np
from lib import SyncVideoSet
from lib import StereoCalibrationFunctions as sc
import cv2 as cv
import matplotlib.pyplot as plt

# calibration_video_mode = 0 --> If the calibration video is contained in the first video chapter of the main video
# calibration_video_mode = 1 --> If the calibration video is contained in a single video
path_all = ['/Volumes/Disk_I/06_12_2022/4',
            '/Volumes/Disk_I/06_12_2022/5',
            '/Volumes/Disk_I/07_12_2022/2',
            '/Volumes/Disk_I/07_12_2022/3',
            '/Volumes/Disk_I/07_12_2022/4',
            '/Volumes/Disk_I/07_12_2022/5',
            ]

path_in = '/Volumes/Disk_I/06_12_2022/5'

deployment = SyncVideoSet(path_in, mode=1)

deployment.detect_calibration_videos()

deployment.get_time_lag(method='custom', number_of_video_chapters_to_evaluate=4)

deployment.get_calibration_videos()

# deployment.stereo_parameters = sc.compute_stereo_params(deployment, extract_new_images_from_video=True)

deployment.save()

x1 = np.array([[876, 179], [1566, 118]])
x2 = np.array([[200, 219], [878, 152]])

mtx1 = deployment.stereo_parameters['mtx1']
mtx2 = deployment.stereo_parameters['mtx2']
R = deployment.stereo_parameters['R']
T = deployment.stereo_parameters['T']

p3ds = sc.triangulate(x1, x2, mtx1, mtx2, R, T)

p_length = np.sum((p3ds[0, :] - p3ds[1, :])**2)**0.5/10

print(p_length)
print(p_length-100.1)