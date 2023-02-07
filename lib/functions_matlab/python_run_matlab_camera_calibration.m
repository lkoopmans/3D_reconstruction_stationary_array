function python_run_matlab_camera_calibration(video1, video2, save_folder)

addpath('lib/functions_matlab')
video_paths = {video1, video2};

disp((video_paths{1}))

tic
v = VideoReader(video_paths{1});
options.squareSizeInMM = 27*2;
options.imagesize=[v.Height,v.Width];
options.n_framesout = 300; % target number of calibration frames per camera
options.NumRadialDistortionCoefficients = 3;

% find images with checkerboards
disp('Extracting calibration frames containing checkerboard...')
cal_frames = get_calib_frames(video_paths,options);

% extract paired images from cameras
disp('Wrinting images...')
[cam_images] = extract_paired_images(video_paths,cal_frames);

% get parameters for each camera
disp('Computing camera parameters...')
cam_parms = get_cam_parms(cam_images,options);


% calibrate stereo cameras
disp('Computing stereo parameters...')
[stereoParams, imagepoints] = get_stereo_parms(cam_parms,cam_images,options);

% reformat parameters for use in openCV
[intrinsicMatrix1,distortionCoefficients1,intrinsicMatrix2, ...
   distortionCoefficients2,rotationOfCamera2,translationOfCamera2] =... 
   stereoParametersToOpenCV(stereoParams);

toc

save([save_folder, '/intrinsicMatrix1.mat'], 'intrinsicMatrix1')
save([save_folder, '/intrinsicMatrix2.mat'], 'intrinsicMatrix2')
save([save_folder, '/rotationOfCamera2.mat'], 'rotationOfCamera2')
save([save_folder, '/translationOfCamera2.mat'], 'translationOfCamera2')
save([save_folder, '/distortionCoefficients1.mat'], 'distortionCoefficients1')
save([save_folder, '/distortionCoefficients2.mat'], 'distortionCoefficients2')
save([save_folder, '/stereoParams.mat'], 'stereoParams')

exit
end