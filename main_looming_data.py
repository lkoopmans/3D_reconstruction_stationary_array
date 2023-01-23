from lib import FishDetection
from lib import SyncVideoSet
from lib import ImageProcessingFunctions as ip

# Path to deployment folder
path_in = '/Volumes/2022_copy/bommie/13_07_22/1'

# Create a synchronization object containing the deployment's properties
deployment = SyncVideoSet(path_in, recut_videos=False, calibration_video_mode=0)

deployment.load()

'''# Determine time lag between videos
deployment.get_time_lag(method='custom', number_of_video_chapters_to_evaluate=4)
deployment.save()

'''