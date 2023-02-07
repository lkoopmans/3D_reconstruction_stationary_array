from lib.SyncVideoSet import list_all_deployments
from lib import SyncVideoSet
from lib import StereoCalibrationFunctions as sc

path_in = '/Volumes/Disk_M/Predator'
all_deployments = list_all_deployments(path_in)

for name in all_deployments:
    try:
        deployment = SyncVideoSet(name, mode=1)
        deployment.detect_calibration_videos()
        deployment.get_time_lag(method='custom', number_of_video_chapters_to_evaluate=6)
        deployment.get_calibration_videos()
        deployment.stereo_parameters = sc.compute_stereo_params(deployment, extract_new_images_from_video=True)
        deployment.save()
    except:
        print('Error occurred')

