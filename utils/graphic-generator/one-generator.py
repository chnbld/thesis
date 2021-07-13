import os
from c3d import *
from classifier import *
from utils.visualization_util import *
import sys
import glob

original_stdout=sys.stdout


def run_demo():

    video_name = os.path.basename(cfg.sample_video_path).split('.')[0]

    predictions=[0.3942,0.4069,0.4091,0.4062,0.4057,0.3875,0.3846,0.3944,0.3828,0.3949,0.3964,0.4025,0.3976,0.3714,0.3800,0.3755,0.3688,0.3769,0.3836,0.3955,0.3936,0.3599,0.3723,0.3881,0.3681,0.3707,0.3718,0.3898,0.4045,0.3799,0.3867,0.3929]
    predictions = extrapolate(predictions, num_frames)

    save_path = os.path.join(cfg.output_folder, video_name + '.mp4')
    # visualize predictions
    print('Executed Successfully - '+video_name + '.gif saved')
    visualize_predictions(cfg.sample_video_path, predictions, save_path)


if __name__ == '__main__':
    run_demo()