import os.path
from lib import SyncVideoSet
import time
import pickle
from lib import StereoCalibrationFunctions as sc

# calibration_video_mode = 0 --> If the calibration video is contained in the first video chapter of the main video
# calibration_video_mode = 1 --> If the calibration video is contained in a single video

path_in = '/Volumes/Disk_M/Predator'
init_list = os.listdir(path_in)
all_deployments = []

for l in init_list:
    subdir = os.listdir(os.path.join(path_in, l))
    for k in subdir:
        all_deployments.append(os.path.join(path_in, l, k))

t0 = time.time()

for name in all_deployments:
    try:
        deployment = SyncVideoSet(name, mode=1)
        deployment.detect_calibration_videos()
        deployment.get_time_lag(method='custom', number_of_video_chapters_to_evaluate=6)
        deployment.get_calibration_videos()
        deployment.stereo_parameters = sc.compute_stereo_params(deployment, extract_new_images_from_video=True)
        deployment.save()
    except:
        print('Caught')

print(time.time() - t0)

'''
idx_sheet = -1

# Read relevant columns from excel file
ex_file = pd.ExcelFile(input_excel)
current_sheet_name = ex_file.sheet_names[idx_sheet]
current_sheet = pd.read_excel(input_excel, current_sheet_name, usecols=[1, 2, 3, 4, 8])

# Define paths
paths_to_videos = []
data = current_sheet.T.iat

for idx in range(len(current_sheet.index)):
    if data[3, idx] == 'Deployed':
        paths_to_videos.append(os.path.join(HD_name, data[4, idx], data[0, idx], str(data[1, idx]), 'Card_'+data[2, idx]))

base_codes = []
files = os.listdir(paths_to_videos[1])

for name in files:
    if '.mp4' or '.MP4' in name:
        base_codes.append(name[4:8])

values, counts = np.unique(base_codes, return_counts=True)
base_code = values[counts > 5]

'''
