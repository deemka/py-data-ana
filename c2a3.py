import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(12345)

df = pd.DataFrame([np.random.normal(32000,200000,3650), 
                   np.random.normal(43000,100000,3650), 
                   np.random.normal(43500,140000,3650), 
                   np.random.normal(48000,70000,3650)], 
                  index=[1992,1993,1994,1995])
df['mean'] = df.mean(axis=1)


class Range:

    def __init__(self):
        self.r1 = 0
        self.r2 = 0
        self.r1_set = False
        self.done = False

    def set_r1(self, arg):
        self.r1 = arg
        self.r1_set = True
        self.done = False

    def set_r2(self, arg):
        self.r2 = arg
        self.done = True
        self.r1_set = False

r = Range()

def plotbars(*args, **kwargs):
    plt.cla()
    ax = plt.bar([1992,1993,1994,1995], df['mean'])
    plt.xticks(df.index, df.index) 

    if 'range' in kwargs.keys():
        plt.axhline(kwargs['range'].r1, linewidth=1, alpha=.5, color = 'steelblue')
        plt.axhline(kwargs['range'].r2, linewidth=1, alpha=.5, color = 'steelblue')
        plt.fill_between([1991.5, 1993, 1994, 1995.5], kwargs['range'].r1, kwargs['range'].r2, color='steelblue', alpha=.25)
    plt.show()


def onclick(ev):
    if ev.ydata:
        r.set_r1(int(ev.ydata))
        plotbars(range=r)

def onmove(ev):
    if ev.ydata and r.r1_set: 
        r.set_r2(int(ev.ydata))
    plotbars(range=r)

def onrelease(ev):
    if ev.ydata:
        r.set_r2(int(ev.ydata))
    plotbars(range=r)

plt.gcf().canvas.mpl_connect('button_press_event', onclick)
plt.gcf().canvas.mpl_connect('motion_notify_event', onmove)
plt.gcf().canvas.mpl_connect('button_release_event', onrelease)
plotbars()
plt.show()
