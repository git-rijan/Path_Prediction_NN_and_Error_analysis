import matplotlib.pyplot as plt
import numpy as np
import random

# def create_plots(dictionary, n):
#     fig, axs = plt.subplots(n, n, figsize=(12, 12))
#     keys = random.sample(list(dictionary.keys()), n*n)
#     for i in range(n):
#         for j in range(n):
#             key = keys[i*n + j]
#             values = dictionary[key]
#             x, y = zip(*[(key[0], key[1])] + values)
#             axs[i, j].plot(x, y, marker='o', color='black')
#             axs[i, j].grid(True)
#             x_range = np.arange(min(x), max(x)+1)
#             y_range = np.arange(min(y), max(y)+1)
#             axs[i, j].set_xticks(x_range, minor=False)
#             axs[i, j].set_yticks(y_range, minor=False)
#             axs[i, j].set_xticks(np.arange(min(x), max(x), 0.5), minor=True)
#             axs[i, j].set_yticks(np.arange(min(y), max(y), 0.5), minor=True)
#             axs[i, j].grid(which='major', color='#CCCCCC', linestyle='--')
#             axs[i, j].grid(which='minor', color='#CCCCCC', linestyle=':')
#             axs[i, j].set_xlabel('x')
#             axs[i, j].set_ylabel('y')
#             for k in range(len(x)-1):
#                 dx, dy = x[k+1]-x[k], y[k+1]-y[k]
#                 axs[i, j].annotate('', xy=(x[k+1], y[k+1]), xytext=(x[k], y[k]), arrowprops=dict(arrowstyle='->', color='black'), annotation_clip=False)
#     plt.show()

def create_plots(dictionary, n, l, b):
    fig, axs = plt.subplots(n, n, figsize=(12, 12))
    fig.suptitle('Path Co-ordinates', fontsize=16)
    keys = random.sample(list(dictionary.keys()), n*n)
    for i in range(n):
        for j in range(n):
            key = keys[i*n + j]
            values = dictionary[key]
            x, y = zip(*[(key[0], key[1])] + values)
            axs[i, j].plot(x, y, marker='o', color='black')
            axs[i, j].grid(True)
            x_range = np.arange(0, l+1)
            y_range = np.arange(0, b+1)
            axs[i, j].set_xticks(x_range, minor=False)
            axs[i, j].set_yticks(y_range, minor=False)
            axs[i, j].set_xticks(np.arange(0, l, 0.5), minor=True)
            axs[i, j].set_yticks(np.arange(0, b, 0.5), minor=True)
            axs[i, j].grid(which='major', color='#CCCCCC', linestyle='--')
            axs[i, j].grid(which='minor', color='#CCCCCC', linestyle=':')
            axs[i, j].set_xlabel('x (meter)')
            axs[i, j].set_ylabel('y (meter)')
            axs[i, j].set_xlim(0, l)
            axs[i, j].set_ylim(0, b)
            for k in range(len(x)-1):
                dx, dy = x[k+1]-x[k], y[k+1]-y[k]
                axs[i, j].annotate('', xy=(x[k+1], y[k+1]), xytext=(x[k], y[k]), arrowprops=dict(arrowstyle='->', color='black'), annotation_clip=False)
    plt.show()


import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
import random

def create_animations(dictionary, n):
    fig, axs = plt.subplots(n, n, figsize=(12, 12))
    keys = random.sample(list(dictionary.keys()), n*n)
    anims = []
    for i in range(n):
        for j in range(n):
            key = keys[i*n + j]
            values = dictionary[key]
            x, y = zip(*[(key[0], key[1])] + values)
            axs[i, j].set_xlim([min(x)-1, max(x)+1])
            axs[i, j].set_ylim([min(y)-1, max(y)+1])
            axs[i, j].set_xticks(np.arange(min(x)-1, max(x)+1, 1), minor=False)
            axs[i, j].set_yticks(np.arange(min(y)-1, max(y)+1, 1), minor=False)
            axs[i, j].set_xticks(np.arange(min(x), max(x), 0.5), minor=True)
            axs[i, j].set_yticks(np.arange(min(y), max(y), 0.5), minor=True)
            axs[i, j].grid(which='major', color='#CCCCCC', linestyle='--')
            axs[i, j].grid(which='minor', color='#CCCCCC', linestyle=':')
            axs[i, j].set_xlabel('x')
            axs[i, j].set_ylabel('y')
            line, = axs[i, j].plot([], [], marker='o', color='black')
            anim = FuncAnimation(fig, update_line, fargs=(x, y, line), frames=len(x), interval=200, blit=True)
            anims.append(anim)
    plt.show()

def update_line(frame, x, y, line):
    line.set_data(x[:frame+1], y[:frame+1])
    return line,
