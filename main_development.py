
from lib import SyncVideoSet
from lib import StereoCalibrationFunctions as sc


# calibration_video_mode = 0 --> If the calibration video is contained in the first video chapter of the main video
# calibration_video_mode = 1 --> If the calibration video is contained in a single video
path_all = ['/Volumes/Disk_I/06_12_2022/4',
            '/Volumes/Disk_I/06_12_2022/5',
            '/Volumes/Disk_I/07_12_2022/2',
            '/Volumes/Disk_I/07_12_2022/3',
            '/Volumes/Disk_I/07_12_2022/4',
            '/Volumes/Disk_I/07_12_2022/5',
            ]

for path_in in path_all:
    try:
        deployment = SyncVideoSet(path_in, mode=1)

        deployment.detect_calibration_videos()

        deployment.get_time_lag(method='custom', number_of_video_chapters_to_evaluate=4)

        deployment.get_calibration_videos()

        deployment.stereo_parameters = sc.compute_stereo_params(deployment)

        deployment.save()
    except:
        pass





