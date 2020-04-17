'''
MIT License

Copyright (c) [2020] [Duin BAEK]

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
'''

import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy.ndimage.filters import gaussian_filter

def gaze_position_split(gaze_position):

    x_coord_array = []
    y_coord_array = []

    for position in gaze_position:

        x_coord, y_coord = position.split(',')
        x_coord, y_coord = float(x_coord), float(y_coord)

        if 0 <= x_coord <= 1 and 0 <= y_coord <= 1:

            x_coord_array.append(x_coord)
            y_coord_array.append(y_coord)

    x_coord_array = np.array(x_coord_array)
    y_coord_array = np.array(y_coord_array)

    return x_coord_array, y_coord_array

def plot_heatmap(x_data, y_data, bins, sigma, figure_save_path):

    heatmap, xedges, yedges = np.histogram2d(x_data, y_data, bins = bins)
    heatmap = gaussian_filter(heatmap / heatmap.sum(), sigma = sigma)
    fig, ax = plt.subplots()
    cs = ax.imshow(heatmap, cmap=cm.jet)
    cbar = fig.colorbar(cs)
    cbar.ax.tick_params(labelsize=15)
    plt.xlabel('Yaw in FoV (Degree)', fontsize=18)
    plt.ylabel('Pitch in FoV (Degree)', fontsize=18)
    plt.xticks(np.arange(0, 101, 20), np.linspace(-40, 40, 6, dtype = np.int32), fontsize = 15)
    plt.yticks(np.arange(0, 101, 20), np.linspace(40, -40, 6, dtype = np.int32), fontsize = 15)
    plt.tight_layout()
    plt.savefig(os.path.join(figure_save_path, 'eye_gaze_heatmap.png'))

csv_base_path = 'csv'
figure_save_path = 'figure'
if not os.path.exists(figure_save_path):
    os.makedirs(figure_save_path)

print("PROCESSING")
x_list = []
y_list = []

for csv_idx in range(10):

    for user_name in os.listdir(csv_base_path):

        user_path = os.path.join(csv_base_path, user_name)
        user_data_path = os.path.join(user_path, str(csv_idx) + '.csv')

        user_data = pd.read_csv(user_data_path, delimiter = ';')

        left_gaze_position = user_data['LeftGazePosition']
        right_gaze_position = user_data['RightGazePosition']

        left_x, left_y = gaze_position_split(left_gaze_position)
        right_x, right_y = gaze_position_split(right_gaze_position)

        x_list.append(left_x)
        x_list.append(right_x)
        y_list.append(left_y)
        y_list.append(right_y)

x_list = np.concatenate(x_list)
y_list = np.concatenate(y_list)

plot_heatmap(x_list, y_list, 100, 4, figure_save_path)
print("HEATMAP GENERATED! PLEASE CHECK FIGURE DIRECTORY!")

































#end
