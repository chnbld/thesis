import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from utils.video_util import *


def visualize_clip(clip, convert_bgr=False, save_gif=False, file_path=None):
    num_frames = len(clip)
    fig, ax = plt.subplots()
    fig.set_tight_layout(True)

    def update(i):
        if convert_bgr:
            frame = cv2.cvtColor(clip[i], cv2.COLOR_BGR2RGB)
        else:
            frame = clip[i]
        plt.imshow(frame)
        return plt

    # FuncAnimation will call the 'update' function for each frame; here
    # animating over 10 frames, with an interval of 20ms between frames.
    anim = FuncAnimation(fig, update, frames=np.arange(0, num_frames), interval=1)
    if save_gif:
        anim.save(file_path, dpi=80, writer='imagemagick')
    else:
        # plt.show() will just loop the animation forever.
        plt.show()


def visualize_predictions(video_path, predictions, save_path, video_name):
    video_highlights={}
    video_highlights["01_084"]=[06.84, 11.30,1,27.23, 28.23,0.6]
    video_highlights["01_085"]=[04.41, 05.41,0.5,12.17, 16.11,1,24.62, 26.14,1]
    video_highlights["01_103"]=[0.5, 1,1]

    frames = get_video_frames(video_path)
    print(len(frames))
    print(len(predictions))
    assert len(frames) == len(predictions)

    #fig, ax = plt.subplots(figsize=(5, 5))
    fig, ax = plt.subplots(figsize=(10, 5))
    fig.set_tight_layout(True)

    #fig_frame = plt.subplot(2, 1, 1)
    #fig_prediction = plt.subplot(2, 1, 2)
    fig_prediction = plt.subplot()
    fig_prediction.set_xlim(0, len(frames))
    fig_prediction.set_ylim(0, 1.15) 
    #fig_prediction.broken_barh([(420,69),(900,180),(1680,75),(2430,120),(3300,360),(3939,51),(4050,120),(4200,60),(4320,150),(10110,150),(10410,120),(10755,45),(10920,30),(11220,45),(11490,57),(12090,102),(13260,75),(13500,60),(13680,36),(14364,48),(14880,30),(15027,339)],(1,-1),facecolors='#f9ced3')
    #fig_prediction.broken_barh([(1035,1656)],(0.5,-1),facecolors='#f9ced3')
    # for i in range(0,len(video_highlights[video_name]),3):
    #     fig_prediction.broken_barh([(30*video_highlights[video_name][i],30*video_highlights[video_name][i+1])],(video_highlights[video_name][i+2],-1),facecolors='#f9ced3')
    def update(i):
        #frame = frames[i]
        x = range(0, i)
        y = predictions[0:i]
        fig_prediction.plot(x, y, '-')
        #fig_frame.imshow(frame)
        return plt

    # FuncAnimation will call the 'update' function for each frame; here
    # animating over 10 frames, with an interval of 20ms between frames.

    #anim = FuncAnimation(fig, update, frames=np.arange(0, len(frames), 10), interval=1, repeat=False)
    anim = FuncAnimation(fig, update, frames=np.arange(0, len(frames), 1), interval=1, repeat=False)
    #anim = animation.FuncAnimation(fig, animate, init_func=init, frames=200, interval=20, blit=True)

    if save_path:
        #anim.save(save_path, dpi=200, writer='imagemagick')
        #mywriter = animation.FFMpegWriter(fps=10)
        #anim.save('mymovie.mp4', writer=mywriter)
        anim.save(save_path, fps=30, extra_args=['-vcodec', 'libx264'])
    else:
        plt.show()

    return


