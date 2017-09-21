import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from math import sqrt

np.random.seed(12345)


df = pd.DataFrame([np.random.normal(32000, 200000, 3650),
                   np.random.normal(43000, 100000, 3650),
                   np.random.normal(43500, 140000, 3650),
                   np.random.normal(48000, 70000, 3650)],
                  index=[1992, 1993, 1994, 1995])
df['mean'] = df.mean(axis=1)
df['yerr'] = 1.96 * df.std(axis=1) / sqrt(df.shape[1] - 1)


class ModeInfo:

    def __init__(self):
        self.r1 = 0
        self.r2 = 0
        self.mode = None
        self.done = False
        self.but1_down = False
        self.ymax = (df['mean'] + df['yerr']).tolist()
        self.ymin = (df['mean'] - df['yerr']).tolist()
        self.mean = (df['mean']).tolist()

    def set_r1(self, arg):
        self.r1 = arg
        self.mode = 'line'
        self.done = False

    def set_r2(self, arg):
        self.r2 = arg
        self.mode = 'range'

    def reset(self):
        self.mode = None
        self.done = False


mi = ModeInfo()


def calc_inrange(mi):

    res = [1, 1, 1, 1]
    if mi.mode is None:
        return res

    for idx in range(0, 4):

        rmax = max(mi.r1, mi.r2)
        rmin = min(mi.r1, mi.r2)

        if (rmin >= mi.ymax[idx] or rmax <= mi.ymin[idx]):
            res[idx] = 0

        else:
            s = sorted([rmin, rmax, mi.ymin[idx], mi.ymax[idx]])
            if mi.mode == 'range':
                res[idx] = (s[2] - s[1]) / (mi.ymax[idx] - mi.ymin[idx])
            elif mi.mode == 'line':
                res[idx] = 1 - abs((mi.r1 - mi.mean[idx]) / (mi.ymax[idx] - mi.ymin[idx]))
    return res


def plotbars(mi, **kwargs):

    plt.cla()
    ax = plt.bar([1992, 1993, 1994, 1995], df['mean'], yerr=df['yerr'], capsize=3)
    # for idx in range(4):
    #    plt.gca().patches[idx].set_alpha(1)

    if mi.mode == 'line':
        plt.axhline(mi.r1, linewidth=1, alpha=.5, color='steelblue')

    if mi.mode == 'range':
        plt.axhline(mi.r1, linewidth=1, alpha=.5, color='steelblue')
        plt.axhline(mi.r2, linewidth=1, alpha=.5, color='steelblue')
        plt.fill_between(df.index, mi.r1, mi.r2, color='gray', alpha=.25)

    for idx in range(4):
        plt.gca().patches[idx].set_alpha(max(0.1, calc_inrange(mi)[idx]))

    plt.axes().tick_params(axis='both', which='both', length=0)
    plt.xticks(df.index, df.index)
    plt.show()


def onclick(ev):

    if ev.button == 1 and ev.ydata:
        mi.but1_down = True

        mi.reset()
        mi.set_r1(ev.ydata)
        mi.mode = 'line'

    if ev.button == 3:
        mi.reset()

    plotbars(mi)


def onmove(ev):
    if mi.mode is not None and mi.but1_down and ev.ydata:
        mi.set_r2(ev.ydata)

    plotbars(mi)


def onrelease(ev):
    if ev.button == 1:
        if mi.mode == 'range':
            mi.set_r2(ev.ydata)
            mi.done = True
        mi.but1_down = False

    if ev.button == 3:
        mi.reset()

    plotbars(mi)


# Set callbacks
plt.gcf().canvas.mpl_connect('button_press_event', onclick)
plt.gcf().canvas.mpl_connect('motion_notify_event', onmove)
plt.gcf().canvas.mpl_connect('button_release_event', onrelease)


plotbars(mi)
