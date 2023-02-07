import os

import numpy as np
from lib import SyncVideoSet
from lib import ImageProcessingFunctions as ip
from lib import StereoCalibrationFunctions as sc
import cv2 as cv
import matplotlib.pyplot as plt
import matplotlib
from scipy import ndimage
import scipy
import pickle
import pandas as pd

matplotlib.use('TkAgg')

def running_mean(x, N):
    cumsum = np.cumsum(x, axis=0)
    return (cumsum[N:] - cumsum[:-N]) / float(N)

filename = 'results/deployments/Predator_10_12_2022_2.pkl'

with open(filename, 'rb') as input:
    deployment = pickle.load(input)

mtx1 = deployment.stereo_parameters['mtx1']
mtx2 = deployment.stereo_parameters['mtx2']
R = deployment.stereo_parameters['R']
T = deployment.stereo_parameters['T']
all_lengths = []
z = []

right = pd.read_excel('results/clips/attack1/Attack1_left.xlsx')
left = pd.read_excel('results/clips/attack1/Attack1_right.xlsx')

centroid_chromis_left = np.array([left['centroid_Chromis_1'], deployment.height - left['centroid_Chromis_2']]).T
centroid_chromis_right = np.array([right['centroid_Chromis_1'], deployment.height - right['centroid_Chromis_2']]).T

Jack_head_left = np.array([left['Head_Jack_1'], deployment.height - left['Head_Jack_2']]).T
Jack_head_right = np.array([right['Head_Jack_1'], deployment.height - right['Head_Jack_2']]).T

Jack_tail_left = np.array([left['Tail_Jack_1'], deployment.height - left['Tail_Jack_2']]).T
Jack_tail_right = np.array([right['Tail_Jack_1'], deployment.height - right['Tail_Jack_2']]).T

t_offset = 1
res_real_cor_Chromis = sc.triangulate(centroid_chromis_left[:-t_offset, :], centroid_chromis_right[t_offset:, :], mtx1, mtx2, R, T)
res_real_cor_Jack_head = sc.triangulate(Jack_head_left[:-t_offset, :], Jack_head_right[t_offset:, :], mtx1, mtx2, R, T)
res_real_cor_Jack_tail = sc.triangulate(Jack_tail_left[:-t_offset, :], Jack_tail_right[t_offset:, :], mtx1, mtx2, R, T)

mov_mean_coff = 5
res_real_cor_Chromis = running_mean(res_real_cor_Chromis[::-1, :], mov_mean_coff)
res_real_cor_Jack_head = running_mean(res_real_cor_Jack_head[::-1, :], mov_mean_coff)
res_real_cor_Jack_tail = running_mean(res_real_cor_Jack_tail[::-1, :], mov_mean_coff)

t = np.arange(0, len(res_real_cor_Chromis))/deployment.fps
off = 0

v_jack = np.sum((np.diff(res_real_cor_Jack_head, axis=0)*deployment.fps/1000)**2, axis=1)**0.5
v_chromis = np.sum((np.diff(res_real_cor_Chromis, axis=0)*deployment.fps/1000)**2, axis=1)**0.5

v_jack_vector = np.diff(res_real_cor_Jack_head, axis=0)*deployment.fps/1000
v_chroms_vector = np.diff(res_real_cor_Chromis, axis=0)*deployment.fps/1000

plt.figure()
plt.plot(centroid_chromis_left[:, 0], centroid_chromis_left[:, 1], label='left cam')
plt.plot(centroid_chromis_right[:, 0], centroid_chromis_right[:, 1], label='right cam')
plt.grid('on')
plt.xlim(0, 1920)
plt.ylim(0, 1080)
plt.legend()

fig = plt.figure()
ax = fig.add_subplot(projection='3d')

ax.plot(res_real_cor_Chromis[:, 0]+off, res_real_cor_Chromis[:, 1]+off, res_real_cor_Chromis[:, 2]+off, 'b', label='Chromis')
ax.scatter(res_real_cor_Chromis[:, 0]+off, res_real_cor_Chromis[:, 1]+off, res_real_cor_Chromis[:, 2]+off, c='b', s=10)

ax.plot(res_real_cor_Jack_head[:, 0]+off, res_real_cor_Jack_head[:, 1]+off, res_real_cor_Jack_head[:, 2]+off, 'r', label='Jack')
ax.scatter(res_real_cor_Jack_head[:, 0]+off, res_real_cor_Jack_head[:, 1]+off, res_real_cor_Jack_head[:, 2]+off, c='r', s=10)

for i in range(len(res_real_cor_Chromis)):
    ax.plot([res_real_cor_Jack_head[i, 0], res_real_cor_Chromis[i, 0]],
            [res_real_cor_Jack_head[i, 1], res_real_cor_Chromis[i, 1]],
            [res_real_cor_Jack_head[i, 2], res_real_cor_Chromis[i, 2]], 'k')

'''for i in range(len(res_real_cor_Chromis)):
    if i % 3 == 0:
        ax.plot([res_real_cor_Jack_head[i, 0], res_real_cor_Jack_tail[i, 0]],
                [res_real_cor_Jack_head[i, 1], res_real_cor_Jack_tail[i, 1]],
                [res_real_cor_Jack_head[i, 2], res_real_cor_Jack_tail[i, 2]], 'k')
        ax.scatter(res_real_cor_Jack_tail[i, 0]+off, res_real_cor_Jack_tail[i, 1]+off, res_real_cor_Jack_tail[i, 2]+off, c='r')
        ax.scatter(res_real_cor_Jack_head[i, 0]+off, res_real_cor_Jack_head[i, 1]+off, res_real_cor_Jack_head[i, 2]+off, c='b')
'''
plt.legend()

plt.figure()
plt.subplot(2, 2, 1)
plt.plot(t, np.sum((res_real_cor_Jack_head-res_real_cor_Chromis)**2, axis=1)**0.5/10)
plt.xlabel('Time [s]')
plt.ylabel('Distance [cm]')
plt.title('Relative distance between predator and prey')

plt.subplot(2, 2, 3)
plt.plot(t, np.abs(res_real_cor_Jack_head[:, 0])/10, label='X')
plt.plot(t, np.abs(res_real_cor_Jack_head[:, 1])/10, label='Y')
plt.plot(t, np.abs(res_real_cor_Jack_head[:, 2])/10, label='Z')
plt.ylabel('Position [cm]')
plt.xlabel('Time [s]')
plt.legend()

plt.subplot(2, 2, 4)
plt.plot(t, np.abs(res_real_cor_Chromis[:, 0])/10, label='X')
plt.plot(t, np.abs(res_real_cor_Chromis[:, 1])/10, label='Y')
plt.plot(t, np.abs(res_real_cor_Chromis[:, 2])/10, label='Z')
plt.xlabel('Time [s]')
plt.ylabel('Position [cm]')
plt.legend()

plt.figure()
plt.subplot(2, 2, 1)
plt.plot(t[:-1], v_jack, label='Jack')
plt.plot(t[:-1], v_chromis, label='Chromis')
plt.xlabel('Time [s]')
plt.ylabel('Speed [m/s]')
plt.legend()

plt.subplot(2, 2, 2)
plt.plot(t[:-1], v_jack)
plt.hlines(1, 0, max(t))
plt.xlabel('Time [s]')
plt.ylabel('Speed [m/s]')

plt.subplot(2, 2, 3)
plt.plot(t[:-2], np.diff(v_jack, axis=0))
plt.subplot(2, 2, 4)
plt.plot(t[:-2], np.diff(v_chromis, axis=0))

length = np.sum((res_real_cor_Jack_head - res_real_cor_Jack_tail)**2, axis=1)**0.5

plt.figure()
plt.scatter(t, length/10)
plt.hlines(np.percentile(length, 95)/10, 0, np.max(t))
plt.text(0, np.percentile(length, 95)/10*1.01, '95th percentile')
plt.xlabel('Time [s]')
plt.ylabel('Length [cm]')
plt.title('Shortest distance between tail and head')

fig = plt.figure()
ax = fig.add_subplot(projection='3d')

for i in range(len(res_real_cor_Chromis)):
    ax.plot([0, res_real_cor_Jack_head[i, 0]-res_real_cor_Jack_tail[i, 0]],
            [0, res_real_cor_Jack_head[i, 1]-res_real_cor_Jack_tail[i, 1]],
            [0, res_real_cor_Jack_head[i, 2]-res_real_cor_Jack_tail[i, 2]], 'k')

x = res_real_cor_Jack_head[:, 0]-res_real_cor_Jack_tail[:, 0]
y = res_real_cor_Jack_head[:, 1]-res_real_cor_Jack_tail[:, 1]
z = res_real_cor_Jack_head[:, 2]-res_real_cor_Jack_tail[:, 2]

theta = np.arctan2(y, x)
phi = np.arccos(z/(x**2 + y**2 + z**2)**0.5)

dt = 1/120

plt.figure()
plt.plot(t, ((theta-theta[0])/np.pi*180), label='polar')
plt.xlabel('Time [s]')
plt.ylabel('Angle [deg]')


x_chromis = running_mean(v_chroms_vector[:, 0], 5)
y_chromis = running_mean(v_chroms_vector[:, 1], 5)
z_chromis = running_mean(v_chroms_vector[:, 2], 5)

x_jack = running_mean(v_jack_vector[:, 0], 5)
y_jack = running_mean(v_jack_vector[:, 1], 5)
z_jack = running_mean(v_jack_vector[:, 2], 5)

theta_jack = np.arctan2(y_jack, x_jack)
theta_chromis = np.arctan2(y_chromis, x_chromis)

plt.plot(((theta_chromis-theta_chromis[0])/np.pi*180), label='polar')
plt.plot(((theta_jack-theta_jack[0])/np.pi*180), label='polar')
plt.xlabel('Time [s]')
plt.ylabel('Angle [deg]')
plt.legend()
