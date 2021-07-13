import os
from os import listdir
import sys
import cv2
import numpy as np
import glob

def interpolate(features, features_per_bag):
    feature_size = np.array(features).shape[1]
    interpolated_features = np.zeros((features_per_bag, feature_size))
    interpolation_indicies = np.round(np.linspace(0, len(features) - 1, num=features_per_bag + 1))
    count = 0
    for index in range(0, len(interpolation_indicies)-1):
        start = int(interpolation_indicies[index])
        end = int(interpolation_indicies[index + 1])

        assert end >= start

        if start == end:
            temp_vect = features[start, :]
        else:
            temp_vect = np.mean(features[start:end+1, :], axis=0)

        temp_vect = temp_vect / np.linalg.norm(temp_vect)

        if np.linalg.norm(temp_vect) == 0:
            print("Error")

        interpolated_features[count,:]=temp_vect
        count = count + 1

    return np.array(interpolated_features)

total=0
#containing folder of regular c3d features
path='C:/Users/dalab/Documents/Graduation work4/videos/was training/features/filler/'
files=listdir(path)
files.sort()
for i in files:
    a=np.load(os.path.join(path,i))
    a = interpolate(a, 32)
    video_name=i.split('.')[0]
    print(video_name)
    #ouput c3d-32segments
    np.savetxt('C:/Users/dalab/Documents/Graduation work4/videos/was training/features-32/filler/'+video_name+'.txt',a)
    print(a.shape)
