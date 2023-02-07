from lib import FishDetection
from lib import SyncVideoSet
from lib import ImageProcessingFunctions as ip
import os

# Use Matlab to get stereo params
save_folder = 'results/calib_results/test'
squareSizeMM = '30'
frames_per_x_sec = 1

# Path to deployment folder
for i in range(8):
    input_path = '/Volumes/2022_copy/Swimmingpool/'+str(i+1)
    save_folder = 'results/calib_results/Swimmingpool/'+str(i+1)

    # Create a synchronization object containing the deployment's properties
    deployment = SyncVideoSet(input_path, recut_videos=True, single_video_mode=True)

    # Determine time lag between videos
    deployment.get_time_lag(method='custom', number_of_videos_to_evaluate=1)

    # Get calibration videos
    deployment.get_calibration_videos()

    name1 = os.path.join(deployment.path_in, deployment.camera_names[0], deployment.calibration_video_names[0])
    name2 = os.path.join(deployment.path_in, deployment.camera_names[1], deployment.calibration_video_names[1])

    ip.extract_frames(name1, frames_per_x_sec, folder='results/calib_results/Swimmingpool/Frames_from_videos/'+str(i+1)
                                                      + '/' + deployment.camera_names[0])
    ip.extract_frames(name2, frames_per_x_sec, folder='results/calib_results/Swimmingpool/Frames_from_videos/'+str(i+1)
                                                      + '/' + deployment.camera_names[1])
