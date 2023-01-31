import ffmpeg
import os
import json
import subprocess
import numpy as np
import cv2
from lib import *
import shutil
import datetime
from datetime import datetime
from lib import FishDetection as fd


def parse_metadata(video_file):
    temp_file = os.path.splitext(video_file)[0] + '.json'
    subprocess.run('ffprobe -v quiet -print_format json -show_format -show_streams {} > {}'.format(video_file,
                                                                                                   temp_file),
                   shell=True)
    metadata = json.load(open(temp_file, 'r'))

    os.remove(temp_file)
    fps = np.array(metadata['streams'][0]['r_frame_rate'].split('/')).astype(int)
    channels = int(metadata['streams'][1]['channels'])
    sample_rate = int(metadata['streams'][1]['sample_rate'])
    fps = fps[0] / fps[1]
    duration = (metadata['streams'][1]['duration'])
    width = (metadata['streams'][0]['width'])
    height = (metadata['streams'][0]['height'])

    return {'fps': fps,
            'sample_rate': sample_rate,
            'channels': channels,
            'duration': duration,
            'width': width,
            'height': height}


def extract_frames(video_file, frame_every_x_second, folder='images/detection_images'):
    # writes output as milliseconds in name _########. Maximum timestamp corresponds to

    metadata = parse_metadata(video_file)

    parts = int(float(metadata['duration']) / frame_every_x_second)
    intervals = int((float(metadata['duration']) * 10 // parts))
    interval_list = [(i * intervals, (i + 1) * intervals) for i in range(parts)]
    i = 0

    base_name = os.path.splitext(os.path.basename(video_file))[0]
    output_directory = os.path.join(folder, base_name)

    if not os.path.exists(output_directory):
        os.makedirs(output_directory, exist_ok=True)
    else:
        shutil.rmtree(output_directory)
        os.makedirs(output_directory, exist_ok=True)

    for item in interval_list:
        print('Writing: ', os.path.join(output_directory, '{}_{:08d}.png'.format(base_name, item[1])), '...')
        (
            ffmpeg
            .input(video_file, ss=float(item[1]) / 10)
            .filter('scale', int(metadata['width']), -1)
            .output(os.path.join(output_directory, '{}_{:08d}.png'.format(base_name, item[1])), vframes=1,
                    loglevel="quiet")
            .run()
        )
        i += 1
    return output_directory


def get_base_name(path):
    base_name = os.path.splitext(os.path.basename(path))[0]
    return base_name


def trim_video(path_input, path_output, s_start, s_end):
    # cuts and stores video between timestamp s_start and s_end, both in seconds
    t1 = datetime.fromtimestamp(s_start).strftime("00:%M:%S.%f")
    t2 = datetime.fromtimestamp(s_end).strftime("00:%M:%S.%f")

    if os.path.exists(path_output):
        os.remove(path_output)

    print('Cut from: ', t1, 'to', t2)

    if t1 == 0:
        shutil.copyfile(path_input, path_output)
    else:
        subprocess.run(
            'ffmpeg -loglevel error -ss ' + t1 + ' -i ' + path_input + ' -c:v libx264 -c:a aac -preset ultrafast ' + path_output,
            shell=True)


def cut_video(path_input, path_output, s_start, s_end):
    # cuts and stores video between timestamp s_start and s_end, both in seconds
    t1 = datetime.fromtimestamp(s_start).strftime("00:%M:%S.%f")
    t2 = datetime.fromtimestamp(s_end).strftime("00:%M:%S.%f")

    if os.path.exists(path_output):
        os.remove(path_output)
    print('Cut from', t1, 'to', t2)
    subprocess.run(
        'ffmpeg -loglevel error -ss ' + t1 + ' -to ' + t2 + ' -i ' + path_input + ' -c:v libx264 -c:a aac -preset medium '
        + path_output, shell=True)


def merge_videos(output_file, path_to_merge_list):
    subprocess.run(
        'ffmpeg -loglevel error -f concat -safe 0 -i ' + path_to_merge_list + ' -c:v libx264 -c:a aac -preset '
                                                                              'ultrafast ' + output_file,
        shell=True)


def cut_video_pairs(params, camera_pair, video_chapter, t_start, t_end, input_path, output_path):
    all_intervals = [None] * params.number_of_videos[0]

    for k in range(params.number_of_videos[0]):
        all_intervals[k] = np.concatenate([[float(t_start)], [float(t_end)]], axis=0).reshape(2, 1)
        all_intervals[k].astype(float)

    camera = (camera_pair - 1) * 2

    fish_detection = fd.FishDetection(
        input_path + '/' + params.camera_names[camera] + '/' + params.video_names[camera][video_chapter])
    fish_detection.trimmed_clips_output_path = output_path
    fish_detection.get_video_pairs_for_detected_interval(all_intervals, params, camera, video_chapter)


def get_roi(image_in):
    # select ROI function
    roi = cv2.selectROI(image_in)

    # Crop selected roi from raw image
    roi_cropped = image_in[int(roi[1]):int(roi[1] + roi[3]), int(roi[0]):int(roi[0] + roi[2])]

    return roi_cropped
