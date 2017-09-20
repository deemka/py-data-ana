import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from math import sqrt

np.random.seed(12345)

df = pd.DataFrame([np.random.normal(32000,200000,3650), 
                   np.random.normal(43000,100000,3650), 
                   np.random.normal(43500,140000,3650), 
                   np.random.normal(48000,70000,3650)], 
                  index=[1992,1993,1994,1995])
df['mean'] = df.mean(axis=1)
df['yerr'] = 1.96 * df.std(axis=1) / sqrt(df.shape[1] - 1)

class Range:

    def __init__(self):
        self.r1 = 0
        self.r2 = 0
        self.r1_set = False
        self.done = False

    def set_r1(self, arg):
        self.r1 = arg
        self.r2 = arg
        self.r1_set = True
        self.done = False

    def set_r2(self, arg):
        self.r2 = arg

r = Range()


def calc_inrange(r):

    for idx in df.index:
        ymax = df.loc[idx, 'mean'] + df.loc[idx, 'yerr']
        ymin = df.loc[idx, 'mean'] - df.loc[idx, 'yerr']
        rmax = max(r.r1, r.r2)
        rmin = min(r.r1, r.r2)

        if ( rmin >= ymax or rmax <= ymin):
            df.loc[idx, 'inrange'] = 0

        else:
            s = sorted([rmin, rmax, ymin, ymax])
            df.loc[idx, 'inrange'] = (s[2] - s[1]) / (ymax - ymin)
    return df['inrange']

            
def plotbars(*args, **kwargs):

    plt.cla()
    ax = plt.bar([1992,1993,1994,1995], df['mean'], yerr=df['yerr'], capsize=3)
    for idx in range(4):
        plt.gca().patches[idx].set_alpha(1)

    if 'range' in kwargs.keys():        
        plt.axhline(kwargs['range'].r1, linewidth=1, alpha=.5, color = 'steelblue')
        plt.axhline(kwargs['range'].r2, linewidth=1, alpha=.5, color = 'steelblue')
        plt.fill_between(df.index, kwargs['range'].r1, kwargs['range'].r2, color='steelblue', alpha=.25)
        for idx in range(4):
            plt.gca().patches[idx].set_alpha(max(0.1, calc_inrange(r).tolist()[idx]))
            
    plt.axes().tick_params(axis='both', which='both', length=0)
    plt.xticks(df.index, df.index)
    plt.show()


def onclick(ev):
    if ev.ydata:
        r.set_r1(int(ev.ydata))
        plotbars()

def onmove(ev):
    if ev.ydata and r.r1_set: 
        r.set_r2(int(ev.ydata))
    #plotbars(range=r, colors=calc_inrange(r))

def onrelease(ev):
    if ev.ydata:
        r.set_r2(int(ev.ydata))
        r.done = True
    plotbars(range=r, colors=calc_inrange(r))

plt.gcf().canvas.mpl_connect('button_press_event', onclick)
plt.gcf().canvas.mpl_connect('motion_notify_event', onmove)
plt.gcf().canvas.mpl_connect('button_release_event', onrelease)
plotbars()
plt.show()
