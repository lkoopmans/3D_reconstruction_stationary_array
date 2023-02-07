from lib import FishDetection
from lib import SyncVideoSet
from lib import ImageProcessingFunctions as ip
import os

# Use Matlab to get stereo params
save_folder = 'results/calib_results/test'
squareSizeMM = '30'

# Path to deployment folder
for i in range(2, 8):
    print(i)
    input_path = '/Volumes/2022_copy/Swimmingpool/'+str(i+1)
    save_folder = 'results/calib_results/Swimmingpool/'+str(i+1)

    # Create a synchronization object containing the deployment's properties
    deployment = SyncVideoSet(input_path, recut_videos=True, single_video_mode=True)

    # Determine time lag between videos
    deployment.get_time_lag(method='custom', number_of_videos_to_evaluate=1)

    # Get calibration videos
    deployment.get_calibration_videos()

    # Generate calibration images
    frames_per_x_sec = 1
    deployment.generate_images_from_calibration_video(frames_per_x_sec)

    deployment.compute_3d_matrices(squareSizeMM, save_folder)

