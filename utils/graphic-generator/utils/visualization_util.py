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
    video_highlights['1']=[]
    video_highlights['001']=[]
    video_highlights['002']=[]
    video_highlights['003']=[]
    video_highlights['004']=[]
    video_highlights['005']=[]
    video_highlights['006']=[]
    video_highlights['007']=[]
    video_highlights['008']=[]
    video_highlights['009']=[]
    video_highlights['010']=[]
    video_highlights['011']=[]
    video_highlights['012']=[]
    video_highlights['013']=[]
    video_highlights['014']=[]
    video_highlights['015']=[]
    video_highlights['016']=[]
    video_highlights['017']=[]
    video_highlights['018']=[]
    video_highlights['019']=[]
    video_highlights['020']=[]
    video_highlights['021']=[]
    video_highlights['022']=[]
    video_highlights['023']=[]
    video_highlights['024']=[]
    video_highlights['025']=[]
    video_highlights["026"]=[99,37,364,28]
    video_highlights["027"]=[1,230,341,86,581,36,731,134]
    video_highlights["028"]=[81,88,248,75,740,23]
    video_highlights["029"]=[66,29,194,57]
    video_highlights["030"]=[1,30,167,50]
    video_highlights["031"]=[86,107]
    video_highlights["032"]=[86,31,91,200,341,250]
    video_highlights["033"]=[312,24,566,94,871,39,981,93,1339,21,1400,28,1561,64]
    video_highlights["034"]=[46,67]
    video_highlights["035"]=[1,135]
    video_highlights["036"]=[22,113,171,40,382,31]
    video_highlights["037"]=[694,31,2710,36,]
    video_highlights["038"]=[116,56]
    #video_highlights["039"]=[]
    #video_highlights["040"]=[]
    video_highlights["041"]=[1,2816]
    #video_highlights["042"]=[]
    #video_highlights["043"]=[]
    video_highlights["044"]=[339,187,716,56,966,73]
    video_highlights["045"]=[112,99,]
    #video_highlights["046"]=[]
    video_highlights["047"]=[19,41,260,65]
    #video_highlights["048"]=[]
    #video_highlights["049"]=[]
    video_highlights["050"]=[54,136,279,58]

    
    video_highlights["100"]=[373,162,684,232,1331,180,1547,84,3995,18,4119,13,4190,15,4342,20]    
    video_highlights["104"]=[279,21,320,15,365,17,395,11,457,23,887,49,950,18,2126,19,2278,16,2315,52]
    video_highlights["107"]=[38,25,158,22,220,11,297,34,377,18]
    video_highlights["201"]=[19,17,182,14]
    video_highlights["202"]=[1,480]
    video_highlights["203"]=[547,128,1043,46,1234,72,1584,31,1753,33,1807,44,1921,20,3084,35]
    video_highlights["204"]=[1007,15,1277,15,1437,22]
    video_highlights["210"]=[10,11,40,20,80,65]
    video_highlights["211"]=[48,17,136,25]
    video_highlights["212"]=[8,15,105,21]
    video_highlights["213"]=[75,15,107,22]
    video_highlights["214"]=[36,109]
    video_highlights["215"]=[27,16,85,14]
    video_highlights["216"]=[50,21,76,19]
    video_highlights["217"]=[1,123]
    video_highlights["218"]=[35,82]
    video_highlights["219"]=[37,34,87,74]
    video_highlights["220"]=[23,49]
    video_highlights["221"]=[8,71,138,7]
    video_highlights["222"]=[1,71,98,21,137,8]

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
    #fig_prediction.broken_barh([(361,90)],(1,-1),facecolors='#f9ced3')
    #fig_prediction.broken_barh([],(1,-1),facecolors='#f9ced3')
    for i in range(0,len(video_highlights[video_name]),2):
        fig_prediction.broken_barh([(video_highlights[video_name][i],video_highlights[video_name][i+1])],(1,-1),facecolors='#f9ced3')
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


