import subprocess
import json
import os
import sys
import cv2
from scipy.io import wavfile
import time
import numpy as np
from scipy.signal import fftconvolve
from lib import ImageProcessingFunctions as ip
import shutil
import pickle
import datetime


class SyncVideoSet:

    def __init__(self, path_in, mode=1):
        print('---------- INITIALIZE SYNCHRONIZATION ----------')
        print('Start analyzing deployment found in', path_in)

        # Define input output paths for the deployment
        temp = path_in.split('/')
        output_name = temp[-3] + '_' + temp[-2] + '_' + temp[-1]
        cd = os.getcwd()

        self.filename = cd + '/results/deployments/' + output_name + '.pkl'
        self.output_name = temp[-3] + '_' + temp[-2] + '_' + temp[-1]
        self.path_in = path_in
        self.path_out = path_in

        self.mode = mode

        self.recut_videos = False
        self.single_video_mode = False

        # Get the camera and video names from the sub-folders
        print('Read video folders...')
        if os.path.isdir(path_in):
            camera_names = np.sort(os.listdir(path_in))
            video_names = [None] * len(camera_names)

            for idx in range(len(camera_names)):
                video_names[idx] = np.sort(os.listdir(path_in + '/' + camera_names[idx]))

            self.flag_folder_input = True

        else:
            print("I/O error: path does not exist")
            sys.exit()

        # Variables to store info on deployment
        self.camera_names = camera_names
        self.number_of_cameras = int(len(camera_names))
        self.number_of_videos = [None] * self.number_of_cameras
        self.base_code = [None] * self.number_of_cameras
        self.calibration_video_names = [None] * self.number_of_cameras
        self.video_names = video_names
        self.fps = [None] * self.number_of_cameras
        self.sample_rate_audio = [None] * self.number_of_cameras
        self.duration = [None] * self.number_of_cameras
        self.width = [None] * self.number_of_cameras
        self.height = [None] * self.number_of_cameras
        self.audio_channels = []
        self.lag_out_cal = []
        self.lag_out = []
        self.lag_matrix = []
        self.lag_matrix_calibration = []
        self.flag_same_inputs = []
        self.calib_interval = []
        self.stereo_parameters = []

        # If analysed before, load data
        if os.path.exists(self.filename):
            self.load()
            print('Data imported from', self.filename)
            return

        # Get base-code from the second, full video and remove all videos not containing this code
        print('Clean folders...')
        get_video_base_code(self)
        remove_additional_videos(self)

        # Get meta deta from videos
        print('Get metadata...')
        load_meta_data(self)

        # Verify input data
        print('Verify input data...')

        verify_input_data(self)

        # After verification of the input data we can reduce list values to single values
        self.duration = self.duration[0]
        self.fps = self.fps[0]
        self.height = self.height[0]
        self.width = self.width[0]
        self.codec_preset = 'ultrafast'

    def get_time_lag(self, method='maximum', number_of_video_chapters_to_evaluate=4, force_recalculating=False):

        print('---------- FIND TIME LAG BETWEEN VIDEOS ----------')

        if any(self.lag_out) and not force_recalculating:
            print('Lag data was imported')
        else:
            self.lag_matrix = get_time_lag_matrix(self, method, number_of_video_chapters_to_evaluate)

            # Determine final lag values
            lag_out_all, lag_out_cal_all = get_lag_vector_from_matrix(self, self.lag_matrix, self.single_video_mode)
            lag_out_all = lag_out_all / self.fps

            if self.mode == 0:
                self.lag_out_cal = lag_out_cal_all / self.fps

            self.lag_out = lag_out_all

        # Get time lag between calibration videos when
        if self.mode == 1 and np.sum(self.lag_matrix_calibration) == 0:
            self.lag_matrix_calibration = get_time_lag_matrix(self, method='calibration_video',
                                                              number_of_video_chapters_to_evaluate=1)
            lag_out_single, lag_out_cal_single = get_lag_vector_from_matrix(self, self.lag_matrix_calibration, True)
            self.lag_out_cal = lag_out_cal_single / self.fps

        self.save()

    def cut_and_merge_videos(self, merge=False):
        print('---------- CUT AND MERGE VIDEO CHAPTERS ----------')

        if self.recut_videos:
            cut_calibration_videos_mode_0(self, False)
        if merge:
            merge_synced_videos(self)

        clean_video_names(self)
        self.save()

    def get_calibration_videos(self):
        print('---------- EXTRACT SYNCED CALIBRATION VIDEOS ----------')
        if self.mode == 0:
            cut_calibration_videos_mode_0(self, True)
            clean_video_names(self)
        else:
            cut_calibration_videos(self)
        self.save()

    def detect_calibration_videos(self):
        print('---------- DETECT CALIBRATION VIDEOS ----------')
        if any(v is None for v in self.calibration_video_names):
            detect_calibration_videos(self)
        else:
            print('Calibration data was imported')
        self.save()

    def compute_3d_matrices(self, square_size_mm='30', save_folder='results/calib_results'):
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)

        if self.single_video_mode:
            compute_3d_matrices_matlab_single_video(self, square_size_mm, save_folder)
        else:
            compute_3d_matrices_matlab(self)
        self.save()

    def generate_images_from_calibration_video(self, frames_per_x_sec):
        generate_calibration_images(self, frames_per_x_sec)

    def load(self):
        'Loads all data from a stucture'
        with open(self.filename, 'rb') as input:
            temp = pickle.load(input)

        self.calib_interval = temp.calib_interval
        self.calibration_video_names = temp.calibration_video_names
        self.lag_out = temp.lag_out
        self.lag_matrix = temp.lag_matrix
        self.lag_out_cal = temp.lag_out_cal
        self.lag_matrix_calibration = temp.lag_matrix_calibration
        self.camera_names = temp.camera_names
        self.number_of_cameras = temp.number_of_cameras
        self.number_of_videos = temp.number_of_videos
        self.base_code = temp.base_code
        self.calibration_video_names = temp.calibration_video_names
        self.video_names = temp.video_names
        self.fps = temp.fps
        self.sample_rate_audio = temp.sample_rate_audio
        self.duration = temp.duration
        self.width = temp.width
        self.height = temp.height
        self.audio_channels = temp.audio_channels
        self.lag_out_cal = temp.lag_out_cal
        self.lag_out = temp.lag_out
        self.lag_matrix = temp.lag_matrix
        self.lag_matrix_calibration = temp.lag_matrix_calibration
        self.flag_same_inputs = temp.flag_same_inputs
        self.calib_interval = temp.calib_interval

        try:
            self.stereo_parameters = temp.stereo_parameters
        except:
            pass

        return self

    def save(self):
        f = open(self.filename, 'wb')
        pickle.dump(self, f, 2)
        f.close()


def get_video_base_code(self):
    if self.single_video_mode:
        find_base_code = "GH01"
    else:
        find_base_code = "GH03"

    for i in range(self.number_of_cameras):
        base_code_video = [s for s in self.video_names[i] if (find_base_code in s and ".MP4" in s and "._G" not in s)]

        if not base_code_video:
            base_code_video = [s for s in self.video_names[i] if
                               (find_base_code.replace('H', 'X') in s and ".MP4" in s and "._G" not in s)]
        if base_code_video:
            self.base_code[i] = base_code_video[0][4:8]
        else:
            self.base_code[i] = []

    return self


def remove_additional_videos(params):
    for i in range(params.number_of_cameras):
        files_to_delete = [s for s in params.video_names[i][:] if (not (".MP4" in s or ".mp4" in s)
                                                                   or "._GH" in s
                                                                   or "._GX" in s)]

        for file_name in files_to_delete:

            if os.path.exists(params.path_in + str('/') + params.camera_names[i] + str('/') + file_name):
                print('REMOVED: ', params.path_in + str('/') + params.camera_names[i] + str('/') + file_name)
                os.remove(params.path_in + str('/') + params.camera_names[i] + str('/') + file_name)

    if os.path.isdir(params.path_in):
        camera_names = np.sort(os.listdir(params.path_in))

        # Replace name list with new names
        for idx in range(len(camera_names)):
            params.video_names[idx] = np.sort(os.listdir(params.path_in + '/' + camera_names[idx]))

    params.number_of_videos = [len(name) for name in params.video_names]

    return params


def load_meta_data(self):
    for idx in range(self.number_of_cameras):
        if self.mode == 0:
            video_file = self.path_in + str('/') + self.camera_names[idx] + str('/') + self.video_names[idx][0]
        else:
            max_idx = len(self.video_names[idx][:]) - 1

            video_file = self.path_in + str('/') + self.camera_names[idx] + str('/') + self.video_names[idx][
                min(10, max_idx)]

        temp_file = os.path.splitext(video_file)[0] + '.json'

        subprocess.run(
            'ffprobe -v quiet -print_format json -show_format -show_streams {} > {}'.format(video_file, temp_file),
            shell=True)

        metadata = json.load(open(temp_file, 'r'))
        os.remove(temp_file)

        fps = np.array(metadata['streams'][0]['r_frame_rate'].split('/')).astype(int)
        self.audio_channels = int(metadata['streams'][1]['channels'])
        self.fps[idx] = fps[0] / fps[1]
        self.sample_rate_audio = int(metadata['streams'][1]['sample_rate'])
        self.duration[idx] = float((metadata['streams'][1]['duration']))
        self.width[idx] = (metadata['streams'][0]['width'])
        self.height[idx] = (metadata['streams'][0]['height'])


def verify_input_data(self):
    flag_fps = len(np.unique(self.fps)) == 1
    flag_resolution = len(np.unique(self.width)) == 1

    self.flag_same_inputs = flag_fps and flag_resolution

    if self.flag_same_inputs:
        print('All ' + str(self.number_of_cameras) + ' cameras have the following settings: ')
        print('Fps: ' + str(np.round(self.fps[0])))
        print('Resolution: ' + str(self.width[0]) + ' X ' + str(self.height[0]))
    else:
        print('The ' + str(self.number_of_cameras) + ' cameras do not have the same settings')
        print('Fps: ' + str(np.round(self.fps)))
        print('Resolution: ' + str(self.width) + ' X ' + str(self.height))

    if np.min(self.number_of_videos) < 1:
        print('One of the cameras has ', str(np.min(self.number_of_videos)), 'video(s). At least 2 are needed. Verify '
                                                                             'if the correct folder is addressed')
        self.flag_folder_input = False


def extract_audio(self, video_file):
    print('Audio extracted from', video_file)

    temp_file = os.path.splitext(video_file)[0] + '.wav'

    subprocess.run('ffmpeg -y -v quiet -i {} -vn -c:a pcm_s16le -ss {} -t {} {}'.format(video_file,
                                                                                        0,
                                                                                        self.duration,
                                                                                        temp_file),
                   shell=True)
    sample_rate, signal = wavfile.read(temp_file)
    os.remove(temp_file)

    return signal.mean(axis=1)


def get_shifted_matrix(mat):
    sx, sy = mat.shape
    out = np.zeros((sx, sy))

    for i in range(sx):
        out[i, :] = mat[i, :] + mat[0, i]

    return out


def get_lag_vector_from_matrix(params, lag_matrix, single_video_mode):
    lag_matrix_frames = np.round(lag_matrix * params.fps)

    rows = np.where(np.arange(len(lag_matrix_frames[:, 0])) % params.number_of_cameras == 1)
    rows = np.tile(rows, 1).transpose()

    lag_out = np.zeros(params.number_of_cameras)

    if not single_video_mode:
        for i in range(1, params.number_of_cameras):
            if i % 2 == 1:
                values, counts = np.unique(
                    np.concatenate([(lag_matrix_frames[:, i]), np.squeeze(lag_matrix_frames[rows, i])]),
                    return_counts=True)
            else:
                values, counts = np.unique(lag_matrix_frames[:, i], return_counts=True)
            lag_out[i] = values[np.argmax(counts)]
    else:
        lag_out = lag_matrix_frames[0, :]

    lag_out_all = lag_out - np.min(lag_out)

    lag_out_calibration = np.zeros(params.number_of_cameras)

    for i in range(int(params.number_of_cameras / 2)):
        lag_out_calibration[i * 2:(i + 1) * 2] = lag_out[i * 2:(i + 1) * 2] - min(lag_out[i * 2:(i + 1) * 2])
    return lag_out_all, lag_out_calibration


def cut_calibration_videos_mode_0(params, only_calibration):
    ts = time.time()
    for i in range(params.number_of_cameras):
        path_in = params.path_in + '/' + params.camera_names[i] + '/' + params.video_names[i][0]

        path_out = params.path_in + '/' + params.camera_names[i] + '/' + os.path.splitext(params.video_names[i][0])[
            0] + '_cut.mp4'
        path_out_cal = params.path_in + '/' + params.camera_names[i] + '/' + os.path.splitext(params.video_names[i][0])[
            0] + '_cut_cal.mp4'

        if not only_calibration:
            print(np.round(time.time() - ts, 2), 's >> Cutting video', path_out, '...')
            ip.trim_video(path_in, path_out, params.lag_out[i], params.duration)

        print(np.round(time.time() - ts, 2), 's >> Cutting video', path_out_cal, '...')
        ip.trim_video(path_in, path_out_cal, params.lag_out_cal[i], params.duration)


def merge_synced_videos(params):
    ts = time.time()
    for idx_camera_num in range(params.number_of_cameras):

        print(np.round(time.time() - ts, 2), 's >> Merging videos for: ',
              params.path_in + '/' + params.camera_names[idx_camera_num])

        # make merge list
        folder_path = params.path_in + '/' + params.camera_names[idx_camera_num]

        filenames = np.sort(os.listdir(folder_path))

        cut_found = False

        with open(folder_path + '/merge_list.txt', 'w') as f:
            for line in filenames:
                if '01' + params.base_code[idx_camera_num] in line:
                    cut_found = True
                    if '_cut' in line and '_cal' not in line:
                        f.write('file \'' + folder_path + '/' + line + '\'')
                        f.write('\n')
                elif params.base_code[idx_camera_num] in line:
                    f.write('file \'' + folder_path + '/' + line + '\'')
                    f.write('\n')

        if not cut_found:
            print('No video name containing _cut found')
            sys.exit()

        if os.path.exists(params.path_out + '/Camera_' + params.camera_names[idx_camera_num] + '.mp4'):
            os.remove(params.path_out + '/Camera_' + params.camera_names[idx_camera_num] + '.mp4')

        if not os.path.exists(params.path_out):
            os.makedirs(params.path_out)

        ip.merge_videos(params.path_out + '/Camera_' + params.camera_names[idx_camera_num] + '.mp4', folder_path +
                        '/merge_list.txt')


def get_time_lag_matrix(params, method, number_of_video_chapters_to_evaluate):
    ts = time.time()

    video_names = []
    for i in range(params.number_of_cameras):
        temp = []
        for name in params.video_names[i][:]:
            if params.base_code[i] in name:
                temp.append(name)
        video_names.append(temp)

    if method == 'maximum':
        itr_max = int(np.min(params.number_of_videos) - 1)
    elif method == 'custom':
        itr_max = min(number_of_video_chapters_to_evaluate, np.min(params.number_of_videos) - 1)
    elif method == 'calibration_video':
        itr_max = 1
        video_names = np.array([[params.calibration_video_names[0]], [params.calibration_video_names[1]]])
    else:
        print('ERROR: no correct method is assigned. Use either ''maximum'' or ''custom''')
        sys.exit()

    if params.single_video_mode:
        itr_max = 1

    out = np.zeros((params.number_of_cameras * itr_max, params.number_of_cameras))

    for itr_video in range(itr_max):
        print(np.round(time.time() - ts, 2), 's >> Analysing audio of video ', itr_video * params.number_of_cameras + 0
              + 1, '/', itr_max * params.number_of_cameras)

        # Pick video file name
        itr_camera = 0
        video_file = params.path_in + str('/') + params.camera_names[itr_camera] + str('/') + \
                     video_names[itr_camera][itr_video]

        res_audio = extract_audio(params, video_file)
        y = np.zeros((len(res_audio), params.number_of_cameras))
        max_length = len(res_audio)
        y[:len(res_audio), 0] = res_audio

        if params.number_of_cameras > 2:
            for i in range(1, params.number_of_cameras):
                print(np.round(time.time() - ts, 2), 's >> Analysing audio of video ',
                      itr_video * params.number_of_cameras + i + 1, '/', itr_max * params.number_of_cameras)

                itr_camera = i
                video_file = params.path_in + str('/') + params.camera_names[itr_camera] + str('/') + \
                             video_names[itr_camera][itr_video]

                res_audio = extract_audio(params, video_file)
                y[:len(res_audio), i] = res_audio
        else:
            print(np.round(time.time() - ts, 2), 's >> Analysing audio of video ',
                  itr_video * params.number_of_cameras + i + 1, '/', itr_max * params.number_of_cameras)

            video_file = params.path_in + str('/') + params.camera_names[1] + str('/') + \
                         video_names[1][itr_video]

            res_audio = extract_audio(params, video_file)
            y[:min(len(res_audio), max_length), i] = res_audio[:min(len(res_audio), max_length)]

        lag = np.zeros((params.number_of_cameras, params.number_of_cameras))

        for k in range(params.number_of_cameras):
            for m in range(params.number_of_cameras):
                corr = fftconvolve(y[:, k], y[::-1, m], mode='full')
                offset = np.argmax(corr)
                lag[k, m] = ((2 * y[:, k].size - 1) // 2 - offset) / int(params.sample_rate_audio)

        out[itr_video * params.number_of_cameras:(itr_video + 1) * params.number_of_cameras, :] = get_shifted_matrix(
            lag)

    return out


def clean_video_names(params):
    # Re-index video names
    for idx in range(params.number_of_cameras):
        params.video_names[idx] = np.sort(os.listdir(params.path_in + '/' + params.camera_names[idx]))

    # Index calibration videos and remove names from original video list
    for i in range(params.number_of_cameras):
        temp = params.video_names[i][:]
        j = 0
        while j < len(temp):
            if 'cut_cal' in temp[j]:
                params.calibration_video_names[i] = temp[j]
            if 'cut' in temp[j]:
                temp = np.delete(temp, j)
                j -= 1
            j += 1
        params.video_names[i] = np.sort(temp)
        params.number_of_videos[i] = j
    return params


def compute_3d_matrices_matlab(params):
    for set_number in range(params.number_of_cameras // 2):
        name1 = params.path_in + '/' + params.camera_names[set_number * 2] + '/' + \
                params.calibration_video_names[set_number * 2]
        name2 = params.path_in + '/' + params.camera_names[set_number * 2 + 1] + '/' + \
                params.calibration_video_names[set_number * 2 + 1]

        folder_names = params.path_in.split('/')
        save_folder = 'results/calib_results/' + folder_names[-3] + '_' + folder_names[-2] + '_' + folder_names[-1] + \
                      '_' + params.camera_names[set_number * 2] + params.camera_names[set_number * 2 + 1]

        if not os.path.exists(save_folder):
            os.makedirs(save_folder)

        os.system(
            params.path_to_matlab + ' -nodesktop -nosplash -r "python_run_matlab_camera_calibration(\'' + name1 +
            '\',\'' + name2 + '\',\'' + save_folder + '\')"')


def generate_calibration_images(params, frames_per_x_sec):
    name1 = os.path.join(params.path_in, params.camera_names[0], params.calibration_video_names[0])
    name2 = os.path.join(params.path_in, params.camera_names[1], params.calibration_video_names[1])

    folder = 'images/calib_images'

    # Clean folders
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))

    ip.extract_frames(name1, frames_per_x_sec, folder=folder)
    ip.extract_frames(name2, frames_per_x_sec, folder=folder)


def compute_3d_matrices_matlab_single_video(params, squareSizeMM, save_folder):
    base_name = os.path.splitext(os.path.basename(params.calibration_video_names[0]))[0]
    output_directory1 = os.path.join('images/calib_images', base_name)

    base_name = os.path.splitext(os.path.basename(params.calibration_video_names[0]))[0]
    output_directory2 = os.path.join('images/calib_images', base_name)

    os.system(
        params.path_to_matlab + ' -nodesktop -nosplash -r "python_run_code_calib_videos(\'' + output_directory1 +
        '\',\'' + output_directory2 + '\',\'' + str(params.height) + '\',\'' + str(
            params.width) + '\',\'' + squareSizeMM +
        '\',\'' + save_folder + '\')"')


def detect_calibration_videos(deployment):
    for filename in os.listdir('images/temp_detect_calib_video'):
        file_path = os.path.join('images/temp_detect_calib_video', filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))

    for i in range(deployment.number_of_cameras):
        files = deployment.video_names[i][:]
        potential_calib_videos = []
        if not deployment.base_code[i]:
            return

        for name in files:
            if ('.mp4' or '.MP4' in name) and (name[4:8] < deployment.base_code[i]):
                path = os.path.join(deployment.path_in, deployment.camera_names[i], name)
                temp_file = os.path.splitext(name)[0] + '.json'
                subprocess.run(
                    'ffprobe -v quiet -print_format json -show_format -show_streams {} > {}'.format(path, temp_file),
                    shell=True)

                metadata = json.load(open(temp_file, 'r'))
                os.remove(temp_file)

                duration = float((metadata['streams'][1]['duration']))

                if duration > 60:
                    potential_calib_videos.append(name)

        delta_t = 4

        for j in range(len(potential_calib_videos)):
            path_in = os.path.join(deployment.path_in, deployment.camera_names[i], potential_calib_videos[j])
            img_folder = ip.extract_frames(path_in, delta_t, folder='images/temp_detect_calib_video')

            rows = 8  # number of checkerboard rows.
            columns = 11  # number of checkerboard columns.

            img_dir_list = os.listdir(img_folder)
            t_detect = []
            t = 0

            for img_name in img_dir_list:
                img = cv2.imread(os.path.join(img_folder, img_name), 0)
                ret, corners = cv2.findChessboardCorners(img, (rows, columns), None)
                if ret:
                    t_detect.append(t)
                t += delta_t

                print('Searching for checkerboards image at time: ', str(datetime.timedelta(seconds=t)))

            if len(t_detect) > 1:
                interval_calib_video = [max(0, min(t_detect) - delta_t), max(t_detect) + delta_t]
                deployment.calibration_video_names[i] = potential_calib_videos[j]
                deployment.calib_interval.append(interval_calib_video)


def cut_calibration_videos(params):
    c = 0
    for name in params.calibration_video_names:
        temp = name.split('.')
        path_input = os.path.join(params.path_in, params.camera_names[c], name)

        if '_cal' not in temp[0]:
            path_output = os.path.join(params.path_in, params.camera_names[c], temp[0] + '_cal.' + temp[1])
            params.calibration_video_names[c] = temp[0] + '_cal.' + temp[1]
            cal_video_name = temp[1]
        else:
            path_output = os.path.join(params.path_in, params.camera_names[c], name)
            cal_video_name = name

        if os.path.exists(path_output):
            print('Calibration video', cal_video_name,
                  'was found. Delete this file to re-sync video')
        else:
            intervals = np.array(params.calib_interval)
            s_start = params.lag_out_cal[c] + min(intervals[:, 0])
            s_end = params.lag_out_cal[c] + max(intervals[:, 1])

            ip.cut_video(path_input, path_output, s_start, s_end)

        c += 1
