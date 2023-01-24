from lib import SyncVideoSet
import cv2 as cv
import glob
import numpy as np
import time
from lib import ImageProcessingFunctions as ip
import os
import matplotlib.pyplot as plt

img1 = plt.imread()

gray2 = cv.cvtColor(images_2[c], cv.COLOR_BGR2GRAY)

# find the checkerboard
ret1, corners1 = cv.findChessboardCorners(gray1, (rows, columns), None)