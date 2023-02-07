import ffmpeg
import numpy as np
path = '/Volumes/Disk_A/22_11_2022/3/Card_LG/GH180024.MP4'

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
    i += 1