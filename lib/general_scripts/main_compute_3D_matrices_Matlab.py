
from lib import SyncVideoSet

'''import numpy as np
np.load('results/names_all_deployments_bommie.npy')

import os

input_path = '/Volumes/homes/Lars/Fieldwork_Curacou/predator/14_07_22/2/'
deployment = SyncVideoSet(input_path, input_path.replace('2022_copy', 'Synced'), remove_cut_files=True)

print(deployment.lag_matrix)
if not deployment.flag_same_inputs or not deployment.flag_folder_input:
    print('Synchronization of', input_path, 'has stopped')
else:
    # Determines the time lag between the video set
    deployment.get_time_lag(method='custom', number_of_videos_to_evaluate=5)
    print(deployment.lag_matrix)
    deployment.cut_and_merge_videos(merge=False, cut=True)


name1 = 'GX010150_cut_cal.mp4'
name2 = 'GX015330_cut_cal.mp4'

save_folder = 'results/calib_results/predator_14_07_22_2_JK'

print(os.path.exists(name1))
if not os.path.exists(save_folder):
    os.makedirs(save_folder)

os.system('/Applications/MATLAB_R2022a.app/bin/matlab -nodesktop -nosplash -r "python_run_matlab_camera_calibration(\''
          + name1 + '\',\'' + name2 + '\',\'' + save_folder + '\')"')
'''

from lib import FishDetection
from lib import ImageProcessingFunctions as ip
input_path = '/Volumes/2022_copy/predator/14_07_22/2'
deployment = SyncVideoSet(input_path, input_path.replace('2022_copy', 'Synced'), remove_cut_files=True)

ip.cut_video('/Volumes/2022_copy/predator/14_07_22/2''/Volumes/2022_copy/predator/14_07_22/2')

j = 3
fish_detection = FishDetection(input_path + '/' + deployment.camera_names[0] + '/' + deployment.video_names[0][j])

fish_detection.path_to_weights = 'yolo_weights/Jack_detect2.pt'
fish_detection.image_size_detection = '1280'
fish_detection.confidence = '0.8'
fish_detection.save_images = True
fish_detection.time_interval_between_frames = 1/deployment.fps  # in sec
fish_detection.number_of_label_classes = 1
fish_detection.trimmed_clips_output_path = 'results/Jack_clips/'

fish_detection.extract_frames_from_video()
fish_detection.run_detector()