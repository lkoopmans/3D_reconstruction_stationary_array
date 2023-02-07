## Summary


## Input variables:
    - path_in: path to folder containing videos
    - mode: 0 = the calibration video is
            1= the calibration video is part
    - method: custom - number of video chapters used to sync is set by the user.
              maximum - takes the all video chapters to sync
    - number_of_video_chapters_to_evaluate: number of chapters used to determine the time lag between the cameras.

## The following datastructure is assumed:
- path_in --> 
  - camera_1 --> 
    - video_1
    - video_2
    - ...
    - video_M
  - camera_2
  - ...
  - camera_N
