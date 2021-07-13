import os
from c3d import *
from classifier import *
from utils.visualization_util import *
import sys
import glob

original_stdout=sys.stdout


def run_demo(sample_video_path):

    #video_name = os.path.basename(cfg.sample_video_path).split('.')[0]
    video_name = os.path.basename(sample_video_path).split('.')[0]
    print(video_name)
    # read video
    #video_clips, num_frames = get_video_clips(cfg.sample_video_path)
    video_clips, num_frames = get_video_clips(sample_video_path)

    print("Number of clips in the video : ", len(video_clips))

    # build models
    feature_extractor = c3d_feature_extractor()
    classifier_model = build_classifier_model()

    print("Models initialized")

    # extract features
    rgb_features = []
    for i, clip in enumerate(video_clips):
        clip = np.array(clip)
        #print(params.frame_count)
        if len(clip) < params.frame_count:
            continue

        clip = preprocess_input(clip)
        rgb_feature = feature_extractor.predict(clip)[0]
        rgb_features.append(rgb_feature)

        #print("Processed clip : ", i)

    rgb_features = np.array(rgb_features)

    # bag features
    #rgb_feature_bag = interpolate(rgb_features, 32)
    #np.save('C:/Users/dalab/Documents/Graduation work3/'+video_name+'.npy',rgb_feature_bag)

    np.save('C:/Users/dalab/Documents/Graduation work4/videos/was testing/new-nor/features/'+video_name+'.npy',rgb_features)

    # classify using the trained classifier model
    # predictions = classifier_model.predict(rgb_feature_bag)

    # predictions = np.array(predictions).squeeze()

    # predictions = extrapolate(predictions, num_frames)

    # save_path = os.path.join(cfg.output_folder, video_name + '.mp4')
    # visualize predictions
    print('Executed Successfully - '+video_name + '.gif saved')
    #visualize_predictions(cfg.sample_video_path, predictions, save_path)


if __name__ == '__main__':

    all_files=glob.glob("C:/Users/dalab/Documents/Graduation work4/videos/was testing/new-nor/*.mp4")
    for filepath in all_files:
        run_demo(filepath)