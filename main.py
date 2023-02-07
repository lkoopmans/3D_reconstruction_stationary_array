"""
Input variables:
    - path_in: path to folder containing videos
    - mode: 0 = the calibration video is
            1= the calibration video is part
    - method: custom - number of video chapters used to sync is set by the user.
              maximum - takes the all video chapters to sync
    - number_of_video_chapters_to_evaluate: number of chapters used to determine the time lag between the cameras.

The following datastructure is assumed:
path_in ---> - Camera_1 --> - Video_1
                            - Video_2
                                .
                                .
                            - Video_M
             - Camera_2
                .
                .
             - Camera_N
"""

# Load libraries
from lib import SyncVideoSet
from lib import StereoCalibrationFunctions as sc

# Load metadata of the video set
deployment = SyncVideoSet(path_in='/Volumes/Disk_M/Predator/10_12_2022/2', mode=1)

# Detect video containing the calibration checkerboards
deployment.detect_calibration_videos()

# Determine the time lag between the cameras
deployment.get_time_lag(method='custom', number_of_video_chapters_to_evaluate=6)

# Cut synced calibration videos used to determine the stereo parameters
deployment.get_calibration_videos()

# Determine the stereo parameters
deployment.stereo_parameters = sc.compute_stereo_params(deployment, extract_new_images_from_video=True)

# Store results
deployment.save()
