import ffmpeg
import numpy as np
import os
import random

path = '/Volumes/Disk_G'
numer_of_images = 100

all_video_files = []
for path, subdirs, files in os.walk(path):
    for name in files:
        if '.MP4' in name and '._' not in name:
            try:
                if float(name[2:4]) > 1:
                    all_video_files.append(os.path.join(path, name))
            except:
                pass

for i in range(numer_of_images):
    try:
        file = all_video_files[random.randint(0, len(all_video_files))]
        t = float(random.randint(0, 8*60+50))
        probe = ffmpeg.probe(file)
        time = float(probe['streams'][0]['duration']) // 2
        width = probe['streams'][0]['width']

        (
            ffmpeg
            .input(file, ss=t)
            .filter('scale', width, -1)
            .output('Jack_image_no_jack/' + file[-12:-4] + '_' + str(i) + '.jpg', vframes=1)
            .run()
        )
    except:
        pass

'''
probe = ffmpeg.probe(path)
time = float(probe['streams'][0]['duration']) // 2
width = probe['streams'][0]['width']

# Set how many spots you want to extract a video from.
parts = 7
start = 9
end = 11
num_of_img = 6

interval_list = np.round(np.linspace(start, end, end-start))
i = 0

for item in interval_list:
    (
        ffmpeg
        .input(path, ss=item)
        .filter('scale', width, -1)
        .output('Jack_image/'+path[-12:-4] +'_'+ str(i) + '.jpg', vframes=1)
        .run()
    )
    i += 1'''
