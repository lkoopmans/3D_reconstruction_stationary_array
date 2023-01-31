from lib import FishDetection
from lib import SyncVideoSet
from lib import ImageProcessingFunctions as ip
import numpy as np
import os
import subprocess

# Path to deployment folder
path_in = '/Volumes/Disk_M/Predator/10_12_2022/2'

# Create a synchronization object containing the deployment's properties
deployment = SyncVideoSet(path_in)
output_path = 'results/clips/attack1'

video_name = 'GH113665'
time_interval = np.array([60+38, 60+42])
base_code = video_name[4:8]
pre_code = video_name[:4]
c = 0
cam_number = []

for name in deployment.base_code:
    if name == base_code:
        cam_number = c
    c += 1

offset = deployment.lag_out - deployment.lag_out[cam_number]

for i in range(deployment.number_of_cameras):
    path_input = os.path.join(deployment.path_in, deployment.camera_names[i], pre_code + deployment.base_code[i]
                              + '.MP4')
    print(path_input)

    if not os.path.exists(output_path):
        os.mkdir(output_path)
    ip.cut_video(path_input, os.path.join(output_path, 'cam_' + pre_code + deployment.base_code[i]+'.MP4'),
                 time_interval[0] + offset[i], time_interval[1] + offset[i])

    path_out = os.path.join(output_path, 'cam_' + pre_code + deployment.base_code[i]+'.MP4')

    if not os.path.exists(output_path + '/frames_'+'cam_' + pre_code + deployment.base_code[i]):
        os.mkdir(output_path + '/frames_'+'cam_' + pre_code + deployment.base_code[i])

    subprocess.run('ffmpeg -loglevel error -i ' + path_out + ' ' + output_path + '/frames_'+'cam_' + pre_code + deployment.base_code[i]
                   + '/frame%03d.png', shell=True)

    from scipy import ndimage
    import cv2 as cv

    left = cv.imread('results/clips/attack1/frames_cam_GH020347/frame001.png')
    right = cv.imread('results/clips/attack1/frames_cam_GH023665/frame001.png')
    #left = (left/256).astype('uint8')
    #right = (right/256).astype('uint8')

    cameraMatrix1 = deployment.stereo_parameters['mtx1']
    cameraMatrix2 = deployment.stereo_parameters['dist1']
    distCoeffs1 = deployment.stereo_parameters['mtx2']
    distCoeffs2 = deployment.stereo_parameters['dist2']
    R = deployment.stereo_parameters['R']
    T = deployment.stereo_parameters['T']
    E = deployment.stereo_parameters['E']
    F = deployment.stereo_parameters['F']

    flags = cv.CALIB_ZERO_DISPARITY
    image_size = left.shape[::-1][1:3]

    ''' 
R1, R2, P1, P2, Q, roi1, roi2 = cv.stereoRectify(cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2, image_size,
                                                     rotationMatrix, transVector, flags=flags, newImageSize=image_size)'''

'''    R1, R2, P1, P2, Q, roi1, roi2 = cv.stereoRectify(cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2,
                                                     image_size, R, T, alpha=-1)

    mapx1, mapy1 = cv.initUndistortRectifyMap(K, d, R1, K, train_frame.shape[:2], cv.CV_32F)
    mapx2, mapy2 = cv.initUndistortRectifyMap(K, d, R2, K, query_frame.shape[:2], cv.CV_32F)
    img_rect1 = cv.remap(train_bckp, mapx1, mapy1, cv.INTER_LINEAR)
    img_rect2 = cv.remap(query_bckp, mapx2, mapy2, cv.INTER_LINEAR)

    leftmapX, leftmapY = cv.initUndistortRectifyMap(cameraMatrix1, distCoeffs1, R1, P1, image_size, cv.CV_32FC1)
    rightmapX, rightmapY = cv.initUndistortRectifyMap(cameraMatrix2, distCoeffs2, R2, P2, image_size, cv.CV_32FC1)

    left_remap = cv.remap(left, leftmapX, leftmapY, cv.INTER_LANCZOS4)
    right_remap = cv.remap(right, leftmapX, rightmapY, cv.INTER_LANCZOS4)


    # For some reason, the images get rotated upside down after remapping, and I have to invert them back
    left_remap = ndimage.rotate(left_remap, 180)
    right_remap = ndimage.rotate(right_remap, 180)

    for line in range(0, int(right_remap.shape[0] / 20)):
        left_remap[line * 20, :] = 0
        right_remap[line * 20, :] = 0

    cv.namedWindow('output images', cv.WINDOW_NORMAL)
    cv.imshow('output images', np.hstack([left_remap, right_remap]))
    cv.waitKey(0)
    cv.destroyAllWindows()'''