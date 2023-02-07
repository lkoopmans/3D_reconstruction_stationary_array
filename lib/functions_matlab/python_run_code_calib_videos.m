function python_run_code_calib_videos(path1, path2, height, width, squareSizeInMM, save_folder)
addpath('lib/functions_matlab/')

options.squareSizeInMM = str2double(squareSizeInMM);
options.NumRadialDistortionCoefficients = 3;
options.imagesize = [str2double(height),str2double(width)];

DirList = dir(fullfile(path1, '*.jpg'));
cam1_images = fullfile(path1, {DirList.name});
DirList = dir(fullfile(path2, '*.jpg'));
cam2_images = fullfile(path2, {DirList.name});

cam_images=[cam1_images;cam2_images];

disp('Computing camera parameters...')
cam_parms = get_cam_parms(cam_images,options);

% calibrate stereo cameras
disp('Computing stereo parameters...')
[stereoParams, imagepoints] = get_stereo_parms(cam_parms,cam_images,options);

% reformat parameters for use in openCV
[intrinsicMatrix1,distortionCoefficients1,intrinsicMatrix2, ...
   distortionCoefficients2,rotationOfCamera2,translationOfCamera2] =... 
   stereoParametersToOpenCV(stereoParams);

save([save_folder, '/intrinsicMatrix1.mat'], 'intrinsicMatrix1')
save([save_folder, '/intrinsicMatrix2.mat'], 'intrinsicMatrix2')
save([save_folder, '/rotationOfCamera2.mat'], 'rotationOfCamera2')
save([save_folder, '/translationOfCamera2.mat'], 'translationOfCamera2')
save([save_folder, '/distortionCoefficients1.mat'], 'distortionCoefficients1')
save([save_folder, '/distortionCoefficients2.mat'], 'distortionCoefficients2')
save([save_folder, '/imagepoints.mat'], 'imagepoints')
save([save_folder, '/stereoParams.mat'], 'stereoParams')

exit
end