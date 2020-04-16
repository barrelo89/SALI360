import os
import ast
import math
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import sem

def csv2pandas(csv_path):

    data = pd.read_csv(csv_path, delimiter = ';')
    data_columns = list(data.columns)

    dictionarized_data = np.array([ast.literal_eval(line) for line in data[data_columns[0]] if not line in data_columns])
    dictionary_keys = list(dictionarized_data[0].keys())

    pandas_data = []

    for one_dictionary in dictionarized_data:

        line_data = []

        for key in dictionary_keys:

            line_data.append(one_dictionary[key])

        pandas_data.append(line_data)

    pandas_data = pd.DataFrame(pandas_data, columns = dictionary_keys)

    return pandas_data

#visaulize the gaze deviation pattern
base_path = 'data/angle_difference'
duration_units = [2, 5, 10]

overall_pitch = []
overall_yaw = []

for csv_idx in range(10):

    pitch_layer = []
    yaw_layer = []

    for duration in duration_units:

        num_hlines = int(30 / duration)
        content_path = os.path.join(base_path, str(csv_idx))

        yaw_difference = pickle.load(open(os.path.join(content_path,str(duration) + '_yaw_difference.p'), 'rb'))
        pitch_difference = pickle.load(open(os.path.join(content_path,str(duration) + '_pitch_difference.p'), 'rb'))

        size_list = []

        for participant_yaw, participant_pitch in zip(yaw_difference, pitch_difference):

            size_list.append(len(participant_yaw))

        min_size = np.min(size_list)

        yaw_list = []
        pitch_list = []

        for participant_yaw, participant_pitch in zip(yaw_difference, pitch_difference):

            yaw_list.append(np.absolute(participant_yaw)[:min_size])
            pitch_list.append(np.absolute(participant_pitch)[:min_size])


        mean_yaw = np.array(yaw_list).mean(axis = 0)
        mean_pitch = np.array(pitch_list).mean(axis = 0)

        pitch_layer.append(mean_pitch)
        yaw_layer.append(mean_yaw)

    overall_pitch.append(pitch_layer)
    overall_yaw.append(yaw_layer)

overall_pitch = np.array(overall_pitch)
overall_yaw = np.array(overall_yaw)

min_size = min([len(pitch) for pitch in overall_pitch[:, 0]])

row, col = overall_pitch.shape

for col_idx, duration in zip(range(col), duration_units):

    input_yaw = overall_yaw[:, col_idx]
    input_pitch = overall_pitch[:, col_idx]

    input_yaw = np.array([yaw[:min_size] for yaw in input_yaw])
    input_pitch = np.array([pitch[:min_size] for pitch in input_pitch])

    yaw_err = sem(input_yaw)
    pitch_err = sem(input_pitch)

    input_yaw = input_yaw.mean(axis = 0)
    input_pitch = input_pitch.mean(axis = 0)


    num_hlines = int(30 / duration)
    plt.scatter(range(1, len(input_yaw)+1), input_yaw, s = 4, c = 'b')#, label = 'mean'

    for idx in range(1, num_hlines+1):

        plt.vlines(idx*duration*30, ymin = 0, ymax = 121, linestyles = 'dashdot')

    plt.xlabel('Time (Sec)', fontsize = 20)
    plt.ylabel('Deviation (Degree)', fontsize = 20)
    plt.xticks(np.arange(0, 910, 150, dtype = np.int32), [int(item) for item in np.arange(0, 910, 150) / 30], fontsize = 18)
    plt.yticks(range(0, 81, 20), fontsize = 20)
    plt.xlim(0, 910)
    plt.ylim(0, 81)
    plt.tight_layout()
    plt.savefig('yaw_' + str(duration) + '.png')
    plt.close()

    plt.scatter(range(1, len(input_pitch)+1), input_pitch, s = 4, c = 'b')#, label = 'mean'

    for idx in range(1, num_hlines+1):

        plt.vlines(idx*duration*30, ymin = 0, ymax = 81, linestyles = 'dashdot')

    plt.xlabel('Time (Sec)', fontsize = 20)
    plt.ylabel('Deviation (Degree)', fontsize = 20)
    plt.xticks(np.arange(0, 910, 150, dtype = np.int32), [int(item) for item in np.arange(0, 910, 150) / 30], fontsize = 18)
    plt.yticks(range(0, 81, 20), fontsize = 20)
    plt.xlim(0, 910)
    plt.ylim(0, 81)
    plt.tight_layout()
    plt.savefig('pitch_' + str(duration) + '.png')
    plt.close()




































'''
base_path = 'data/angle_difference'
write_base_path = 'data/figure/angle_difference/time_line/absolute'#
duration_units = [2, 5, 10]

for duration in duration_units:

    num_hlines = int(30 / duration)

    yaw_difference = pickle.load(open(os.path.join(base_path,str(duration) + '_yaw_difference.p'), 'rb'))
    pitch_difference = pickle.load(open(os.path.join(base_path,str(duration) + '_pitch_difference.p'), 'rb'))

    duration_path = os.path.join(write_base_path, str(duration))
    if not os.path.exists(duration_path):
        os.makedirs(duration_path)

    user_idx = 1

    for participant_yaw, participant_pitch in zip(yaw_difference, pitch_difference):

        plt.scatter(range(len(participant_yaw)), np.absolute(participant_yaw))
        #plt.scatter(range(len(participant_yaw)), participant_yaw)
        for idx in range(1, num_hlines+1):

            plt.vlines(idx*duration*30, ymin = 0, ymax = participant_yaw.max(), linestyles = 'dashdot')

        plt.title('yaw difference')
        #plt.show()
        plt.tight_layout()
        plt.savefig(os.path.join(duration_path, 'yaw difference distribution_' + str(duration) + '_user_' + str(user_idx) + '.png'))
        plt.close()
        #plt.show()

        plt.scatter(range(len(participant_yaw)), np.absolute(participant_pitch))
        #plt.scatter(range(len(participant_yaw)), participant_pitch)
        for idx in range(1, num_hlines+1):

            plt.vlines(idx*duration*30, ymin = 0, ymax = participant_yaw.max(), linestyles = 'dashdot')

        plt.title('pitch difference')
        plt.tight_layout()
        plt.savefig(os.path.join(duration_path, 'pitch difference distribution_' + str(duration) + '_user_' + str(user_idx) + '.png'))
        plt.close()
        #plt.show()
        user_idx += 1
'''



#end
