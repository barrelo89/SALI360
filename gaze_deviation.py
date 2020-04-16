import os
import ast
import math
import pickle
import numpy as np
import pandas as pd

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

duration = 30
duration_units = [2, 5, 10]

csv_base_path = 'data/csv'
'''
IF YOU HAVE VIDEO SEGMENT DIRECTORY, YOU CAN THE FOLLOWING PATH TO GET Y_P_COMBO
video_base_path = 'video/segments/cube'
Y_P_combo = list(os.listdir(video_base_path))
'''
data_save_path = 'data/angle_difference'

Y_P_combo = pickle.load(open(os.path.join(data_save_path, 'y_p_combo.pkl'), 'rb'))

combo_data = []

print('Making Gaze Deviation Pickle Files: According to Video Contents(csv) and Duration')
for combo in Y_P_combo:

    characters = [char for char in combo]
    p_idx = np.where([item == 'P' for item in characters])[0][0]

    yaw, pitch = int(combo[1:p_idx]), int(combo[p_idx+1:])
    combo_data.append((yaw, pitch))

for duration_unit in duration_units:

    num_videos = int(duration / duration_unit)

    for csv_idx in range(10):

        participant_list = []
        duration_yaw_difference_list = []
        duration_pitch_difference_list = []

        for participant_name in os.listdir(csv_base_path):

            participant_yaw_difference_list = []
            participant_pitch_difference_list = []

            participant_list.append(participant_name)

            participant_csv_path = os.path.join(csv_base_path, participant_name)


            csv_path = os.path.join(participant_csv_path, str(csv_idx) + '.csv')

            pandas_data = csv2pandas(csv_path)
            position_data = (pandas_data['position'] / 1000).astype(np.int32)

            idx = 0

            for yaw_value, pitch_value, position in zip(pandas_data['yaw'], pandas_data['pitch'], position_data):

                if position == idx*duration_unit:

                    starting_idx = np.where(position_data == idx*duration_unit)[0][0]
                    yaw = pandas_data['yaw'][starting_idx]
                    pitch = pandas_data['pitch'][starting_idx]

                    #user yaw, pitch information
                    yaw_degree = (math.degrees(yaw) + 360) % 360
                    pitch_degree = -math.degrees(pitch)

                    radial_distance = [np.linalg.norm(np.array([Y, P]) - np.array([yaw_degree, pitch_degree])) for Y, P in combo_data]

                    front_view_yaw, front_view_pitch = combo_data[np.argmin(radial_distance)][0], combo_data[np.argmin(radial_distance)][1]
                    idx += 1

                yaw_deg = (math.degrees(yaw_value) + 360) % 360
                pitch_deg = -math.degrees(pitch_value)

                yaw_differenece, pitch_difference = np.abs(front_view_yaw - yaw_deg), np.abs(front_view_pitch- pitch_deg)

                if yaw_differenece > 180:
                    yaw_differenece = -(360 - yaw_differenece)

                participant_yaw_difference_list.append(yaw_differenece)
                participant_pitch_difference_list.append(pitch_difference)

            participant_yaw_difference_list = np.array(participant_yaw_difference_list)
            participant_pitch_difference_list = np.array(participant_pitch_difference_list)

            duration_yaw_difference_list.append(participant_yaw_difference_list)
            duration_pitch_difference_list.append(participant_pitch_difference_list)

        participant_list = np.array(participant_list)

        csv_write_path = os.path.join(data_save_path, str(csv_idx))

        if not os.path.exists(csv_write_path):
            os.makedirs(csv_write_path)

        pickle.dump(participant_list, open(os.path.join(csv_write_path, str(duration_unit) + '_user_order.p'), 'wb'))

        duration_yaw_difference_list = np.array(duration_yaw_difference_list)
        duration_pitch_difference_list = np.array(duration_pitch_difference_list)

        pickle.dump(duration_yaw_difference_list, open(os.path.join(csv_write_path, str(duration_unit) + '_yaw_difference.p'), 'wb'))
        pickle.dump(duration_pitch_difference_list, open(os.path.join(csv_write_path, str(duration_unit) + '_pitch_difference.p'), 'wb'))

print('DONE! PLEASE CHECK \'data/angle_difference\' DIRECTORY!')















#end
