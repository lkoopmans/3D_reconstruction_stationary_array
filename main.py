import numpy as np
from lib import SyncVideoSet
from lib import StereoCalibrationFunctions as sc

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