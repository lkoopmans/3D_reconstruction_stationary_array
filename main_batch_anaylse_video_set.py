from lib.SyncVideoSet import list_all_deployments
from lib import SyncVideoSet
from lib import StereoCalibrationFunctions as sc

path_in = '/Volumes/Disk_M/Predator'
all_deployments = list_all_deployments(path_in)

for name in all_deployments:
    try:
        deployment = SyncVideoSet(name, mode=1)
        deployment.run_all(6)
    except:
        print('Error occurred')

