import pandas as pd
import numpy as np
import matplotlib as mpl
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


def calc_inrange2(mi):

    res = [0.5, 0.5, 0.5, 0.5]
    if mi.mode is None:
        return res
    
    rmax = max(mi.r1, mi.r2)
    rmin = min(mi.r1, mi.r2)

    for idx in range(0, 4):

        if mi.mode == 'line':
            rmax = mi.r1
            if rmax > mi.ymax[idx]:
                res[idx] = 0.
            elif rmax < mi.ymin[idx]:
                res[idx] = 1.
            else:
                res[idx] = (mi.ymax[idx] - rmax)/(mi.ymax[idx] - mi.ymin[idx])
        else:
            res = res
 
    return res
        

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
                if (mi.r1 >= mi.ymax[idx]) or (mi.r1 <= mi.ymin[idx]):
                    res[idx] = 0
                else:
                    res[idx] = 1 - abs((mi.r1 - mi.mean[idx]) / (mi.ymax[idx] - mi.ymin[idx]))
    return res


def plotbars(mi, **kwargs):
    plt.cla()
    ax = plt.bar([1992, 1993, 1994, 1995], df['mean'], yerr=df['yerr'], capsize=3)
    lims = plt.gca().axis()
    # ([1991.2, 1995.7, 0, 52500])
    # for idx in range(4):
    #    plt.gca().patches[idx].set_alpha(1)

    if mi.mode == 'line':
        ax = plt.bar([1992, 1993, 1994, 1995], df['mean'], yerr=df['yerr'], capsize=3,
                     color=list(map(cmap, calc_inrange2(mi))))
        lims = plt.gca().axis()
        plt.axhline(mi.r1, linewidth=1, alpha=.5, color='steelblue')
        for idx in range(4):
            plt.gca().patches[idx].set_color(cmap(calc_inrange2(mi)[idx]))

    if mi.mode == 'range':
        plt.axhline(mi.r1, linewidth=1, alpha=.5, color='steelblue')
        plt.axhline(mi.r2, linewidth=1, alpha=.5, color='steelblue')
        plt.fill_between(lims[0:2], mi.r1, mi.r2, color='steelblue', alpha=.25)
        plt.gca().axis(lims)

        for idx in range(4):
            plt.gca().patches[idx].set_color(cmap(calc_inrange(mi)[idx]))

    plt.axes().tick_params(axis='both', which='both', length=0)
    plt.xticks(df.index, df.index)

    if mi.mode == 'line':
        plt.title('Value of Interest: {}'.format(int(mi.r1)))
        ti = 'Mean value higher than selection'
        cb.set_label(ti)

    if mi.mode == 'range':
        plt.title('Range of Interest: {} < y < {}'.format(int(min(mi.r1, mi.r2)), int(max(mi.r1, mi.r2))))
        ti = 'Mean value is inside selection'
        cb.set_label(ti)
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


fig = plt.figure(figsize=(9, 6))
cmap = mpl.cm.plasma

# Make a phantom plot to create the colorbar
pd = np.zeros((2, 2))
cls = np.linspace(0, 1, num=1000)
pp = plt.contourf(pd, cls, cmap=cmap)
cb = plt.colorbar(pp, orientation='vertical', shrink=0.5, fraction=.12)
cb.set_ticks([0, 1])
plt.cla()

# Set callbacks
plt.gcf().canvas.mpl_connect('button_press_event', onclick)
plt.gcf().canvas.mpl_connect('motion_notify_event', onmove)
plt.gcf().canvas.mpl_connect('button_release_event', onrelease)

mi = ModeInfo()


plotbars(mi)
