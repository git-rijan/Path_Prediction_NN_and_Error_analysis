import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import make_interp_spline


def plot_dict(d, xlabel='', ylabel='', xlim=(0,100), ylim=(0,100), title=''):
    x = np.array(list(d.keys()))
    y = np.array(list(d.values()))
    # plt.plot(x,y)

    # polyfit
    # p = np.polyfit(x, y, 2)
    # y_fit = np.polyval(p, x)
    # plt.plot(x, y_fit)

    # smoothing out by splines
    X_Y_Spline = make_interp_spline(x, y)
    X_ = np.linspace(xlim[0], xlim[1], 500)
    Y_ = X_Y_Spline(X_)

    # plot the smoothed curve
    plt.plot(X_, Y_)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xlim(xlim[0], xlim[1])
    plt.ylim(ylim[0], ylim[1])
    plt.title(title)

    # set the positions of the x and y grid lines
    xticks = np.linspace(xlim[0], xlim[1], 11)
    yticks = np.linspace(ylim[0], ylim[1], 11)
    plt.xticks(xticks)
    plt.yticks(yticks)

    plt.grid()  # add grid linesplt.grid(linestyle=':', linewidth=0.5)
    plt.show()


def plot_dicts(dicts, xlabel='', ylabel='', xlim=(0,100), ylim=(0,100), title=''):
    # loop through each dictionary and plot its (x,y) coordinates
    for i, d in enumerate(dicts):
        x = np.array(list(d.keys()))
        y = np.array(list(d.values()))
        # plt.plot(x, y, label=f'Dataset {i+1}')

        # smoothing out
        # use make_interp_spline to interpolate the data
        X_Y_Spline = make_interp_spline(x, y)
        X_ = np.linspace(xlim[0], xlim[1], 500)
        Y_ = X_Y_Spline(X_)

        # plot the smoothed curve
        plt.plot(X_, Y_, label=f'Dataset {i+1}')

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xlim(xlim[0], xlim[1])
    plt.ylim(ylim[0], ylim[1])
    plt.title(title)

    # set the positions of the x and y grid lines
    xticks = np.linspace(xlim[0], xlim[1], 11)
    yticks = np.linspace(ylim[0], ylim[1], 11)
    plt.xticks(xticks)
    plt.yticks(yticks)

    plt.grid()  # add grid lines
    plt.legend() # add legend for each dictionary
    plt.show()



def plot_graph(data_dict):
    x = list(data_dict.keys())
    y1 = [f for [f, _] in data_dict.values()]
    y2 = [g for [_, g] in data_dict.values()]

    fig, ax = plt.subplots(nrows=2, ncols=1, sharex=False, figsize=(10, 6))

    ax[0].plot(x, y1, '-o', color='blue')
    ax[0].set_ylabel('Percentage error for x')
    ax[0].set_xlabel('eta ( learning rate )')
    ax[0].set_ylim(0,100)  # set limits of y-axis
    ax[1].plot(x, y2, '-o', color='green')
    ax[1].set_ylabel('Percentage error for y')
    ax[1].set_xlabel('eta ( learning rate )')
    ax[1].set_ylim(0,100)  # set limits of y-axis

    fig.align_ylabels()

    plt.show()


def plot_graphs(data_list):
    fig, ax = plt.subplots(nrows=2, ncols=1, sharex=True, figsize=(10, 6))

    colors = ['blue', 'green', 'red', 'purple']  # define list of colors for curves
    labels = [f"Data {i+1}" for i in range(len(data_list))]  # define list of labels for curves

    for i, data_dict in enumerate(data_list):
        x = list(data_dict.keys())
        y1 = [f for [f, _] in data_dict.values()]
        y2 = [g for [_, g] in data_dict.values()]

        ax[0].plot(x, y1, '-o', color=colors[i], label=labels[i])
        ax[1].plot(x, y2, '-o', color=colors[i], label=labels[i])

    ax[0].set_ylabel('Percentage error for x')
    ax[0].set_ylim(0,100)  # set limits of y-axis
    ax[0].legend()  # add legend for multiple curves
    ax[1].set_ylabel('Percentage error for y')
    ax[1].set_xlabel('eta ( learning rate )')
    ax[1].set_ylim(0,100)  # set limits of y-axis
    ax[1].legend()  # add legend for multiple curves

    fig.align_ylabels()

    plt.show()

