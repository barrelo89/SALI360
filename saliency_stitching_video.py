import json
import os
import numpy as np
import pandas as pd
import cv2
import ast
import math
import multiprocessing
from saliency_update import *

def frame_split_1(frame):
    #return a list of screens in an order of right, left, up, down, front and back
    if len(frame.shape) == 3:
        height, width, _ = frame.shape
    else:
        height, width = frame.shape

    unit_height = int(height / 2)
    unit_width = int(width / 3)

    #1st layer
    right_frame = frame[:unit_height, :unit_width]
    left_frame = frame[:unit_height, unit_width:2*unit_width]
    up_frame = frame[:unit_height, 2*unit_width:]
    #2nd layer
    down_frame = frame[unit_height:, :unit_width]
    front_frame = frame[unit_height:, unit_width:2*unit_width]
    back_frame = frame[unit_height:, 2*unit_width:]

    return [right_frame, left_frame, up_frame, down_frame, front_frame, back_frame]

def frame_split_2(frame, num_row):
    #return a list of screens in an order of right, left, up, down, front and back
    if len(frame.shape) == 3:
        height, width, _ = frame.shape
    else:
        height, width = frame.shape

    unit_width = int(width / 2)

    sub_frame_list = []

    for idx in range(num_row):
        sub_frame_list.append(frame[idx*unit_width:(idx+1)*unit_width, :unit_width])
        sub_frame_list.append(frame[idx*unit_width:(idx+1)*unit_width, unit_width:2*unit_width])

    '''
    #1st layer
    frame_1 = frame[:unit_height, :unit_width]
    frame_2 = frame[:unit_height, unit_width:2*unit_width]
    #2nd layer
    frame_3 = frame[unit_height:2*unit_height, :unit_width]
    frame_4 = frame[unit_height:2*unit_height:, unit_width:2*unit_width]

    #3rd layer
    frame_5 = frame[2*unit_height:3*unit_height, :unit_width]
    frame_6 = frame[2*unit_height:3*unit_height, unit_width:2*unit_width]
    '''

    return sub_frame_list

def frame_merge(frame_list):

    #[right_frame, left_frame, up_frame, down_frame, front_frame, back_frame]
    layer_num = 2
    layer_size = 3

    layer_list = []

    for layer_idx in range(layer_num):

        img_list = []

        for frame_idx in range(layer_size):

            img_list.append(frame_list[layer_idx*layer_size + frame_idx])

        layer_list.append(np.concatenate(img_list, axis = 1))

    return np.concatenate(layer_list, axis = 0)

def json_sort_by_name(y_p_combo_path):
    json_file_list = np.array(os.listdir(y_p_combo_path))
    idx_list = [int(json_file.split('.')[0].split('_')[-1]) for json_file in json_file_list]
    sorted_idx_list = np.argsort(idx_list)
    sorted_json_file_list = json_file_list[sorted_idx_list]

    return sorted_json_file_list

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

def y_p_combo_list(json_base_path):

    Y_P_combo = list(os.listdir(json_base_path))
    combo_data = []

    for combo in Y_P_combo:

        characters = [char for char in combo]
        p_idx = np.where([item == 'P' for item in characters])[0][0]

        yaw, pitch = int(combo[1:p_idx]), int(combo[p_idx+1:])
        combo_data.append((yaw, pitch))

    return combo_data

def json_concatenate(csv_path, duration, duration_unit, j2f_ratio):

    num_videos = int(duration / duration_unit)

    pandas_data = csv2pandas(csv_path)
    position_data = (pandas_data['position'] / 1000).astype(np.int32)

    json_path_list = []

    for idx in range(num_videos):

        starting_idx = np.where(position_data == idx*duration_unit)[0][0]
        yaw = pandas_data['yaw'][starting_idx]
        pitch = pandas_data['pitch'][starting_idx]

        yaw_degree = (math.degrees(yaw) + 360) % 360
        pitch_degree = -math.degrees(pitch)

        radial_distance = [np.linalg.norm(np.array([Y, P]) - np.array([yaw_degree, pitch_degree])) for Y, P in combo_data]
        Y_P_folder = 'Y'+ str(combo_data[np.argmin(radial_distance)][0]) + 'P' + str(combo_data[np.argmin(radial_distance)][1])

        sali_Y_P_path = os.path.join(json_base_path, Y_P_folder)
        duration_folder = str(duration_unit)
        sali_duration_path = os.path.join(sali_Y_P_path, duration_folder)
        sali_idx_folder = Y_P_folder + '_' + str(idx*duration_unit)
        sali_idx_path = os.path.join(sali_duration_path, sali_idx_folder)

        sorted_json_file_list = json_sort_by_name(sali_idx_path)

        for json_file in sorted_json_file_list:
            for _ in range(j2f_ratio):
                json_path_list.append(os.path.join(sali_idx_path, json_file))

    return np.array(json_path_list).ravel()

def saliency_stitching(cube_frame, pyra_frame, json_path, frame_idx, fps, duration_unit, salient_patch_size, tar_num):

    frame_height, frame_width, _ = cube_frame.shape
    side_length = int(frame_width / 3)
    window_size = int(side_length / salient_patch_size) #salient_patch_size!

    name2idx = np.array(['R', 'L', 'U', 'D', 'F', 'B'])

    split_img = frame_split_1(cube_frame) #[right_frame, left_frame, up_frame, down_frame, front_frame, back_frame]
    split_pyra_img = frame_split_1(pyra_frame) #[right_frame, left_frame, up_frame, down_frame, front_frame, back_frame]

    json_data = saliency_score_update(json_path, frame_idx, fps, duration_unit)
    json_keys = np.array(list(json_data.keys()))

    sorted_json_idx = np.argsort([int(key) for key in json_keys])
    json_keys = json_keys[sorted_json_idx]

    img_cluster =[]
    layer_img = []
    threshold = 0
    #newly added
    num_row = 0
    tar_row = 10 # in total, we have 10 rows

    for key in json_keys:
        img = split_img[np.where(name2idx == json_data[key]['name'])[0][0]][int(json_data[key]['row']): int(json_data[key]['row']) + window_size, int(json_data[key]['column']): int(json_data[key]['column']) + int(json_data[key]['width'])]

        layer_img.append(img)
        threshold += int(json_data[key]['width'])
        #print(threshold)

        if threshold == 2*window_size:
            img_cluster.append(np.concatenate(layer_img, axis = 1))
            threshold = 0
            layer_img = []
            num_row += 1
        #print(num_row)
        if num_row == tar_row:
            break
    #print('num', len(img_cluster))
    out_img = np.concatenate(img_cluster, axis = 0) #sali_video_frame
    #cv2.imshow('img', out_img)
    #cv2.waitKey(0)
    split_json_img = frame_split_2(out_img, tar_row)

    row_sum = 0
    split_json_img_idx = 0

    for key in json_keys:

        split_pyra_img_idx = np.where(name2idx == json_data[key]['name'])[0][0]
        row_coord, col_coord = int(json_data[key]['row']), int(json_data[key]['column'])
        width = int(json_data[key]['width'])

        row_sum += width

        #print('cube idx: %d, json_img_idx: %d, json_idx: %d' %(split_pyra_img_idx, split_json_img_idx, json_idx))

        if 0 < row_sum < window_size: #left part of window
            #print('1', split_json_img[split_json_img_idx].shape)
            split_pyra_img[split_pyra_img_idx][row_coord:row_coord + window_size, col_coord:col_coord + width] = split_json_img[split_json_img_idx][:, :width]

        else:#when row_sum is window_size
            if row_sum > 0:
                if row_sum - width == 0: #full window
                    #cv2.imshow('img', split_json_img[split_json_img_idx])
                    #cv2.waitKey(0)
                    #print('2', split_json_img[split_json_img_idx].shape)
                    split_pyra_img[split_pyra_img_idx][row_coord:row_coord + window_size, col_coord:col_coord + width] = split_json_img[split_json_img_idx][:,:]

                else: #right part of window
                    #print('3', split_json_img[split_json_img_idx].shape)
                    split_pyra_img[split_pyra_img_idx][row_coord:row_coord + window_size, col_coord:col_coord + width] = split_json_img[split_json_img_idx][:, -width:]
                row_sum = 0
                split_json_img_idx += 1
        if split_json_img_idx == tar_num: #changed
            break

    stitched_frame = frame_merge(split_pyra_img)

    return stitched_frame

def imwrite(cube_frame, pyra_frame, json_path, write_path, frame_idx, fps, duration_unit, salient_patch_size, tar_num, output):#newly added: frame_idx, fps, duration_unit,
    #if not os.path.exists(write_path):
    stitched_frame = saliency_stitching(cube_frame, pyra_frame, json_path, frame_idx, fps, duration_unit, salient_patch_size, tar_num)#newly added: frame_idx
    cv2.imwrite(write_path, stitched_frame)
    output.put(write_path)

duration = 30
duration_units = [10]
fps = 30
j2f_ratio = 10 #jason to frames ratio
salient_patch_size = 4 # salient_patch_size = cube_side * 1 / 4
tar_nums = [9]#[9, 1, 11, 13, 15]

csv_base_path = 'data/csv'

json_base_path = 'video/sali/json/segment'
combo_data = y_p_combo_list(json_base_path)

frame_base_path = 'video/concatenated'
cube_base_path = os.path.join(frame_base_path, 'cube')
pyra_base_path = os.path.join(frame_base_path, 'pyramid_b')

#write_base_path = os.path.abspath('/media/jihoon/SAMSUNG/6/VR Project/video/concatenated')
for tar_num in tar_nums:

    stitched_write_type_path = os.path.join(frame_base_path, 'stitched_b_' + str(tar_num))

    for duration_unit in duration_units:

        num_videos = int(duration / duration_unit)

        for participant_name in ['Pratik']:#os.listdir(csv_base_path):

            stitched_write_user_path = os.path.join(stitched_write_type_path, participant_name)
            duration_folder = str(duration_unit)
            stitched_write_duration_path = os.path.join(stitched_write_user_path, duration_folder)

            #if os.path.exists(stitched_write_duration_path):
            #    continue
            #else:
            #    os.makedirs(stitched_write_duration_path)

            if not os.path.exists(stitched_write_duration_path):
                os.makedirs(stitched_write_duration_path)

            participant_csv_path = os.path.join(csv_base_path, participant_name)
            csv_path = os.path.join(participant_csv_path, '6.csv')

            user_json_path_array = json_concatenate(csv_path, duration, duration_unit, j2f_ratio)

            cube_user_path = os.path.join(cube_base_path, participant_name)
            pyra_user_path = os.path.join(pyra_base_path, participant_name)

            cube_duration_path = os.path.join(cube_user_path, duration_folder)
            pyra_duration_path = os.path.join(pyra_user_path, duration_folder)

            sorted_frame_name_list = sorted(os.listdir(cube_duration_path))

            stitched_frame_path_list = []
            cube_frame_input_list = []
            pyra_frame_input_list = []
            json_input_list = []
            outputs = []
            process = []

            num_core_2_use = 10
            num_iter = 0
            #newly added
            frame_idx = 0
            frame_idx_list = []
            #newly added

            #use mode to indicate the order of current video frame in the video segment
            #for instance, frame_idx % fps*duration_unit
            #or perhaps we should consider json_idx rather than frame_idx as it is more intuitive and easier to quantify the boundary of consecutive video chunk
            frame_count = len(sorted_frame_name_list)

            for frame_name, json_path in zip(sorted_frame_name_list, user_json_path_array):

                cube_frame = cv2.imread(os.path.join(cube_duration_path, frame_name))
                pyra_frame = cv2.imread(os.path.join(pyra_duration_path, frame_name))

                cube_frame_input_list.append(cube_frame)
                pyra_frame_input_list.append(pyra_frame)
                json_input_list.append(json_path)
                stitched_frame_path_list.append(os.path.join(stitched_write_duration_path, frame_name))
                #newly added
                frame_idx_list.append(frame_idx)
                frame_idx += 1
                #newly added


                if num_iter == int(frame_count / num_core_2_use):
                    num_core_2_use = frame_count - num_core_2_use*num_iter

                if len(cube_frame_input_list) == num_core_2_use:

                    for _ in range(num_core_2_use):
                        outputs.append(multiprocessing.Queue())

                    for idx in range(num_core_2_use):
                        #newly added: frame_idx_list[idx], fps, duration_unit,
                        process.append(multiprocessing.Process(target = imwrite, args = (cube_frame_input_list[idx], pyra_frame_input_list[idx], json_input_list[idx], stitched_frame_path_list[idx], frame_idx_list[idx], fps, duration_unit, salient_patch_size, tar_num, outputs[idx])))

                    for pro in process:
                        pro.start()

                    for idx in range(num_core_2_use):
                        result = outputs[idx].get()
                        print(result)

                    for idx in range(num_core_2_use):
                        outputs[idx].close()

                    for pro in process:
                        pro.terminate()

                    cube_frame_input_list = []
                    pyra_frame_input_list = []
                    json_input_list = []
                    stitched_frame_path_list = []
                    framed_idx_list = []

                    outputs = []
                    process = []

                    num_iter += 1

































'''
fourcc = cv2.VideoWriter_fourcc(*'X264')
video_base_path = 'video/full'
cube_video_base_path = os.path.join(video_base_path, 'rotation')
#pyra_b_video_base_path = os.path.join(video_base_path, 'pyramid_b')
pyra_c_video_base_path = os.path.join(video_base_path, 'pyramid_b')
sali_base_path = os.path.join(video_base_path, 'sali')
sali_stitched_base_path = os.path.join(video_base_path, 'stitched_b')

for video_name in os.listdir(cube_video_base_path):

    cube_video_path = os.path.join(cube_video_base_path, video_name)
    #pyra_b_video_path = os.path.join(pyra_c_video_base_path, video_name)
    pyra_c_video_path = os.path.join(pyra_c_video_base_path, video_name)
    json_folder_path = os.path.join(sali_base_path, os.path.splitext(video_name)[0])

    name2idx = np.array(['R', 'L', 'U', 'D', 'F', 'B'])

    cube_capture = cv2.VideoCapture(cube_video_path)
    pyra_capture = cv2.VideoCapture(pyra_c_video_path)

    json_file_paths = np.array(os.listdir(json_folder_path))
    file_idx = [int(name.split('.')[0].split('_')[-1]) for name in json_file_paths]
    sorted_idx = np.argsort(file_idx)
    sorted_json_file_paths = json_file_paths[sorted_idx]

    #when using the original fps
    fps = round(pyra_capture.get(cv2.CAP_PROP_FPS))
    json_frame_ratio = int((pyra_capture.get(cv2.CAP_PROP_FRAME_COUNT) + 1) / len(sorted_json_file_paths))

    frame_width = int(pyra_capture.get(cv2.CAP_PROP_FRAME_WIDTH)) #width, height of cube video cannot be the values of saliency_video
    frame_height = int(pyra_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    side_length = int(frame_width / 3)
    window_size = int(side_length / 2)

    if not os.path.exists(sali_stitched_base_path):
        os.makedirs(sali_stitched_base_path)

    writer = cv2.VideoWriter(os.path.join(sali_stitched_base_path, video_name), fourcc, fps, (frame_width, frame_height))

    frame_idx = 0
    json_idx = 0
    frame_count = 0

    #when writing the 30fps based video
    while(cube_capture.isOpened() and pyra_capture.isOpened()):

        retval_1, cube_frame = cube_capture.read()
        retval_2, pyra_frame = pyra_capture.read()
        #print(frame_count, json_idx, frame_idx)

        if frame_count == json_frame_ratio:#when writing video in such a way that writes 5 frames based on a json file
            json_idx += 1
            frame_count = 0
            if json_idx == len(sorted_json_file_paths):
                break

        if (retval_1 and retval_2) == True:

            split_img = frame_split_1(cube_frame) #[right_frame, left_frame, up_frame, down_frame, front_frame, back_frame]
            split_pyra_img = frame_split_1(pyra_frame) #[right_frame, left_frame, up_frame, down_frame, front_frame, back_frame]

            file_name = sorted_json_file_paths[json_idx]
            with open(os.path.join(json_folder_path, file_name)) as json_file:

                json_data = json.load(json_file)
                json_data = ast.literal_eval(json_data)
                json_keys = np.array(list(json_data.keys()))

            sorted_json_keys_idx = sorted(range(len(json_keys)), key = json_keys.__getitem__)
            sorted_json_keys = json_keys[sorted_json_keys_idx]

            img_cluster =[]
            layer_img = []
            threshold = 0

            for key in sorted_json_keys:
                img = split_img[np.where(name2idx == json_data[key]['name'])[0][0]][int(json_data[key]['row']): int(json_data[key]['row']) + window_size, int(json_data[key]['column']): int(json_data[key]['column']) + int(json_data[key]['width'])]

                layer_img.append(img)
                threshold += int(json_data[key]['width'])

                if threshold == 2*window_size:
                    img_cluster.append(np.concatenate(layer_img, axis = 1))
                    threshold = 0
                    layer_img = []

            out_img = np.concatenate(img_cluster, axis = 0) #sali_video_frame
            split_json_img = frame_split_2(out_img)

            row_sum = 0
            split_json_img_idx = 0

            for key in sorted_json_keys:
                #print('Name: %s: %d' %(json_data[key]['name'], np.where(name2idx == json_data[key]['name'])[0][0]))
                split_pyra_img_idx = np.where(name2idx == json_data[key]['name'])[0][0]
                row_coord, col_coord = int(json_data[key]['row']), int(json_data[key]['column'])
                width = int(json_data[key]['width'])

                row_sum += width

                print('cube idx: %d, json_img_idx: %d, json_idx: %d' %(split_pyra_img_idx, split_json_img_idx, json_idx))

                if 0 < row_sum < window_size: #left part of window

                    split_pyra_img[split_pyra_img_idx][row_coord:row_coord + window_size, col_coord:col_coord + width] = split_json_img[split_json_img_idx][:, :width]

                else:#when row_sum is window_size
                    if row_sum > 0:
                        if row_sum - width == 0: #full window
                            split_pyra_img[split_pyra_img_idx][row_coord:row_coord + window_size, col_coord:col_coord + width] = split_json_img[split_json_img_idx][:,:]

                        else: #right part of window
                            split_pyra_img[split_pyra_img_idx][row_coord:row_coord + window_size, col_coord:col_coord + width] = split_json_img[split_json_img_idx][:, -width:]

                        row_sum = 0
                        split_json_img_idx += 1


            stitched_frame = frame_merge(split_pyra_img)
            writer.write(stitched_frame)

            frame_count += 1

        else:
            break
    cube_capture.release()
    pyra_capture.release()
    writer.release()
'''
#end
