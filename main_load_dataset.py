from lib import FishDetection
from lib import SyncVideoSet
from lib import ImageProcessingFunctions as ip
import cv2 as cv
import pickle
import matplotlib.pyplot as plt
import numpy as np

filename = 'results/deployments/Disk_I_07_12_2022_4.pkl'

with open(filename, 'rb') as input:
    deployment = pickle.load(input)
    print(deployment.stereo_parameters['ret'])
    print(deployment.stereo_parameters['ret1'])
    print(deployment.stereo_parameters['ret2'])

mat1 = deployment.stereo_parameters['mtx1']
mat2 = deployment.stereo_parameters['mtx2']

img1 = plt.imread('images/detection_images/GH010050_cal/GH010050_cal_00002315.png')
# img2 = plt.imread('images/detection_images/GH010014_cal/GH010014_cal_00002315.png')
img1 = plt.imread('images/detection_images/GH010059_cal/GH010059_cal_00002710.png')

imgYUV = cv.cvtColor(img1, cv.COLOR_RGB2HSV)

# r = cv.selectROI("select the area", imgYUV)

# Crop image
#cropped_image = imgYUV[int(r[1]):int(r[1] + r[3]), int(r[0]):int(r[0] + r[2])]
#temp = np.array([np.median(cropped_image[:, :, 0]), np.median(cropped_image[:, :, 1]), np.median(cropped_image[:, :, 2])])

orange = np.array([0.69803923, 0.40392157, 0.3529412])
# orange = np.array([0.42517257, 0.7645977, 0.33958673])
orange = np.array([13, 0.73431, 0.96471])
low_bound = orange - np.array([5, 0.1, 0.1])
up_bound = orange + np.array([3, 0.1, 0.1])

e1 = low_bound[0] < imgYUV[:, :, 0]
e2 = imgYUV[:, :, 0] < up_bound[0]
e3 = low_bound[1] < imgYUV[:, :, 1]
e4 = imgYUV[:, :, 1] < up_bound[1]
e5 = low_bound[2] < imgYUV[:, :, 2]
e6 = imgYUV[:, :, 2] < up_bound[2]

plt.figure()
temp = img1
temp[:, :, 1] = img1[:, :, 1]+e1*e2
plt.imshow(temp)

'''
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(roi_cropped[:, :, 0].ravel(), roi_cropped[:, :, 1], roi_cropped[:, :, 2])
ax.set_xlabel('R')
ax.set_ylabel('G')
ax.set_zlabel('B')
plt.show()
print(np.median(roi_cropped[:, :, 0]), np.median(roi_cropped[:, :, 1]), np.median(roi_cropped[:, :, 2]))

plt.figure()
plt.subplot(1, 2, 1)
plt.imshow(img1)
plt.subplot(1, 2, 2)
plt.imshow(np.flip(np.flip(img2[:, :, 2], axis=1), axis=0))
plt.show()
'''
# cv.triangulatePoints(mat1, mat2, projPoints1, projPoints2)'''