import json
import ast
import numpy as np
import os

def deviation_function(input_x):

    return 2*input_x*np.heaviside(input_x, 1) - 2*input_x*np.heaviside(input_x - 150, 1) + 300*np.heaviside(input_x - 150, 1)

def distance_from_current_progress(json_values, current_deviation):

    distance_weight_list = []
    half_side = int(512/2)
    max_distance = 1280

    for dictionary_data in json_values:
        #here 1024 is the length of one cube subframe side
        if dictionary_data['name'] == 'R':
            distance = abs(current_deviation - (int(dictionary_data['column']) + half_side))
        elif dictionary_data['name'] == 'L':
            distance = abs(current_deviation - (1024 - (int(dictionary_data['column']) + half_side)))
        elif dictionary_data['name'] == 'U':
            distance = 1024 - (int(dictionary_data['row']) + 512)
        elif dictionary_data['name']  == 'D':
            distance = int(dictionary_data['row'])
        else:
            distance = abs(current_deviation - (1024 + min([int(dictionary_data['column']) + half_side, 1024 - (int(dictionary_data['column']) + half_side)])))

        distance_weight = np.exp(-2*(distance/max_distance))

        if dictionary_data['name'] in ['U', 'D']:
            distance_weight /= 2

        distance_weight_list.append(distance_weight)

    return np.array(distance_weight_list)

def current_deviation_progress(frame_idx, fps, duration_unit):

    current_progress = frame_idx % (fps*duration_unit)

    return deviation_function(current_progress)

def distance_weight(json_values, frame_idx, fps, duration_unit):

    current_deviation = current_deviation_progress(frame_idx, fps, duration_unit)
    #print(current_deviation)
    distance_weight_array = distance_from_current_progress(json_values, current_deviation)#distance from the current_deviation

    return distance_weight_array

def neighbors(json_values):

    idx = 0
    neighbor_dict = {}
    updated_json_values = []
    hit = 0
    while(idx < len(json_values)):

        if 0 < int(json_values[idx]['width']) < 256:
            if json_values[idx]['name'] == 'B':
                neighbor_dict[str(idx-hit)] = json_values[idx]
                updated_json_values.append(json_values[(idx+1)])

            else:
                neighbor_dict[str(idx-hit)] = json_values[idx+1]
                updated_json_values.append(json_values[(idx)])
            idx += 2
            hit += 1
        else:
            updated_json_values.append(json_values[(idx)])
            idx += 1

    return neighbor_dict, np.array(updated_json_values)

def saliency_score_update(json_path, frame_idx, fps, duration_unit):

    with open(json_path) as json_file:

        json_data = json.load(json_file)
        json_data = ast.literal_eval(json_data)
        json_keys = np.array(list(json_data.keys()))
        json_values = np.array(list(json_data.values()))

    sorted_json_idx = np.argsort([int(key) for key in json_keys])
    json_keys = json_keys[sorted_json_idx]
    json_values = json_values[sorted_json_idx]

    neighbor_dict, json_values = neighbors(json_values)
    #print(json_values)
    #print(neighbor_dict)

    distance_weight_array = distance_weight(json_values, frame_idx, fps, duration_unit)

    saliency_score = []
    #print('saliency score, name, width, column, distance')
    for json_value in json_values:
        #print(json_value['saliency'], json_value['name'], json_value['width'], json_value['column'], distance)
        saliency_score.append(float(json_value['saliency']))

    saliency_score = np.array(saliency_score)

    updated_saliency_score = saliency_score*distance_weight_array
    updated_idx = updated_saliency_score.argsort()[::-1]
    updated_json_values = json_values[updated_idx]
    #print(updated_idx)
    #print('update!')

    updated_json_value_list = []
    neighbor_keys = np.array(list(neighbor_dict.keys()))

    for idx, updated_values in zip(updated_idx, updated_json_values):
        updated_json_value_list.append(updated_values)
        #print(updated_values)

        if str(idx) in neighbor_keys:

            updated_json_value_list.append(neighbor_dict[str(idx)])

    updated_json_values = np.array(updated_json_value_list)
    #print(updated_json_values)
    #for key, value in neighbor_dict.items():

    #    idx = np.where(updated_idx == int(key))[0][0]
    #    #print(idx)
    #    updated_json_values = np.insert(updated_json_values, idx+1, value)

    #print('saliency score, name, width, column')
    #for updated_json_value, updated_saliency in zip(updated_json_values, updated_saliency_score[updated_saliency_score.argsort()[::-1]]):
    updated_json_data = {}

    for json_key, updated_json_value in zip(json_keys, updated_json_values):
        #print(json_key, updated_json_value['name'], updated_json_value['width'], updated_json_value['column'])
        updated_json_data[json_key] = updated_json_value
    return updated_json_data

def json_sort_by_name(json_base_path):
    json_file_list = np.array(os.listdir(json_base_path))
    idx_list = [int(json_file.split('.')[0].split('_')[-1]) for json_file in json_file_list]
    sorted_idx_list = np.argsort(idx_list)
    sorted_json_file_list = json_file_list[sorted_idx_list]

    return sorted_json_file_list

'''
frame_idx = 0
fps = 30
duration_unit = 5
json_base_path = 'Y0P0/'

sorted_json_file_list = json_sort_by_name(json_base_path)

for json_file in sorted_json_file_list:
    json_path = os.path.join(json_base_path, json_file)
    updated_json_data = saliency_score_update(json_path, frame_idx, fps, duration_unit)
    print(updated_json_data.values())
    frame_idx += 1
    #print(updated_json_data)
'''

























#end
