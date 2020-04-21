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
import cv2
import pickle
import numpy as np
from saliency_update import *

def rotate_img(img, iteration = 1):
    #0: remain same, +n: rotate left in 90*n degree, -n: rotate right in 90*n degree (some part of img is sliced off)
    if len(img.shape) == 3:
        height, width, _ = img.shape
    else:
        height, width = img.shape

    rotated_img = np.zeros(img.shape, dtype = np.uint8)

    if iteration == -1:

        for idx in range(height):
            rotated_img[:, height-1-idx] = img[idx, :]

    elif iteration == 1:

        for idx in range(height):
            rotated_img[:, idx] = np.flip(img[idx, :], axis = 0)

    elif iteration == 0:

        rotated_img = img

    else:

        for idx in range(height):
            rotated_img[height-1-idx, :] = np.flip(img[idx, :], axis = 0)

    return rotated_img

def trapezoid_1_decoding(img): # starting from small frame / 2 + 2

    if len(img.shape) == 3:
        row, col, channel = img.shape

    else:
        row, col = img.shape
        channel = 1

    empty_pallete = np.zeros((row, col, channel), dtype = np.uint8)

    for idx in range(int(col / 4)):

        valid_layer = img[idx:row-idx, idx, :]
        valid_row, valid_col = valid_layer.shape

        layer = cv2.resize(valid_layer.reshape(valid_row, -1, channel), (4, row))

        empty_pallete[:, idx*4:(idx+1)*4] = layer

    return empty_pallete #shape of row, col, channel (1024, 1024, 3)

def trapezoid_2_decoding(img): # starting from small frame / 2

    if len(img.shape) == 3:
        row, col, channel = img.shape

    else:
        row, col = img.shape
        channel = 1

    empty_pallete = np.zeros((row, col, channel), dtype = np.uint8)

    for idx in range(int(row / 4)):

        valid_layer = img[255-idx, idx+1:col-1-idx, :]
        valid_row, valid_col = valid_layer.shape

        layer = cv2.resize(valid_layer.reshape(-1, valid_row, channel), (col, 4))

        empty_pallete[(255-idx)*4: (255-idx+1)*4, :] = layer

    return empty_pallete #shape of row, col, channel (1024, 1024, 3)

def pyramid_b_decoding(img):

    row, col, channel = img.shape

    front_img = img[:, :int(col/2)]
    pyramid_img = img[:, int(col/2):]

    height, width, channel = pyramid_img.shape

    back_img = pyramid_img[int(height/4):int(height/4)+int(height/2), int(width/4):int(width/4)+int(width/2)]
    back_img = cv2.resize(back_img, None, fx = 2, fy = 2)

    right_img = pyramid_img[:, :int(width/4), :]
    right_pallete = np.zeros(pyramid_img.shape, dtype = np.uint8)
    right_pallete[:, :int(width/4), :] = right_img

    right_img = trapezoid_1_decoding(right_pallete)

    left_img = cv2.flip(pyramid_img[:, -int(width/4):, :], 1)
    left_pallete = np.zeros(pyramid_img.shape, dtype = np.uint8)
    left_pallete[:, :int(width/4), :] = left_img

    left_img = trapezoid_1_decoding(left_pallete)
    left_img = cv2.flip(left_img, 1)


    up_img = pyramid_img[:int(height/4), :, :]
    up_pallete = np.zeros(pyramid_img.shape, dtype = np.uint8)
    up_pallete[-int(height/4):, :, :] = up_img
    up_pallete = rotate_img(up_pallete, iteration = 2)

    up_img = trapezoid_2_decoding(up_pallete)

    down_img = pyramid_img[-int(height/4):, :, :]
    down_pallete = np.zeros(pyramid_img.shape, dtype = np.uint8)
    down_pallete[:int(height/4), :, :] = down_img

    down_img = trapezoid_2_decoding(down_pallete)
    down_img = rotate_img(down_img, 2)

    first_layer = np.concatenate([right_img, left_img, up_img], axis = 1)
    second_layer = np.concatenate([down_img, front_img, back_img], axis = 1)
    output = np.concatenate([first_layer, second_layer], axis = 0)

    return output

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

def frame_split_2(frame, salient_patch_size = 4):

    if len(frame.shape) == 3:
        height, width, _ = frame.shape
    else:
        height, width = frame.shape

    unit_width = int(height / salient_patch_size)

    num_row = int(height / unit_width)
    num_col = int(width / unit_width)

    sub_frame_list = []

    for row_idx in range(num_row):
        for col_idx in range(num_col):
            sub_frame_list.append(frame[row_idx*unit_width:(row_idx+1)*unit_width, col_idx*unit_width:(col_idx+1)*unit_width, :])

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

def json_path_generate(path, j2f_ratio):

    json_path_list = []

    sorted_json_file_list = json_sort_by_name(path)

    for json_file in sorted_json_file_list:
        for _ in range(j2f_ratio):
            json_path_list.append(os.path.join(path, json_file))

    return np.array(json_path_list).ravel()

def decode(frame, json_path, frame_idx, fps, duration_unit, salient_patch_size):

    height, width, _ = frame.shape
    pyra = frame[:, :2*height, :]
    sali = frame[:, 2*height:, :]

    window_size = height // salient_patch_size

    s_height, s_width, _ = sali.shape
    tar_num = (s_height*s_width) // (window_size*window_size)

    decoded_pyra = pyramid_b_decoding(pyra)
    split_pyra_img = frame_split_1(decoded_pyra)
    split_json_img = frame_split_2(sali)

    json_data = saliency_score_update(json_path, frame_idx, fps, duration_unit)

    json_keys = np.array(list(json_data.keys()))

    sorted_json_idx = np.argsort([int(key) for key in json_keys])
    json_keys = json_keys[sorted_json_idx]

    name2idx = np.array(['R', 'L', 'U', 'D', 'F', 'B'])

    row_sum = 0
    split_json_img_idx = 0

    for key in json_keys:

        split_pyra_img_idx = np.where(name2idx == json_data[key]['name'])[0][0]
        row_coord, col_coord = int(json_data[key]['row']), int(json_data[key]['column'])
        width = int(json_data[key]['width'])

        row_sum += width

        if 0 < row_sum < window_size: #left part of window
            split_pyra_img[split_pyra_img_idx][row_coord:row_coord + window_size, col_coord:col_coord + width] = split_json_img[split_json_img_idx][:, :width]

        else:#when row_sum is window_size
            if row_sum > 0:
                if row_sum - width == 0: #full window
                    #cv2.imshow('img', np.concatenate([split_pyra_img[split_pyra_img_idx][row_coord:row_coord + window_size, col_coord:col_coord + width], split_json_img[split_json_img_idx][:,:]], axis = 1))
                    #cv2.waitKey(0)
                    split_pyra_img[split_pyra_img_idx][row_coord:row_coord + window_size, col_coord:col_coord + width] = split_json_img[split_json_img_idx][:,:]

                else: #right part of window
                    split_pyra_img[split_pyra_img_idx][row_coord:row_coord + window_size, col_coord:col_coord + width] = split_json_img[split_json_img_idx][:, -width:]

                row_sum = 0
                split_json_img_idx += 1

        if split_json_img_idx == tar_num: #changed
            break

    stitched_frame = frame_merge(split_pyra_img)
    #cv2.imshow('img', stitched_frame)
    #cv2.waitKey(0)
    return stitched_frame

video_path = 'video/segments/sali_encoded'
decoded_path = 'video/segments/sali_decoded'
json_base_path = 'json'

j2f_ratio = 10
salient_patch_size = 4

fourcc = cv2.VideoWriter_fourcc(*'X264')

for y_p_combo in os.listdir(video_path):

    y_p_combo_path = os.path.join(video_path, y_p_combo)
    y_p_write_path = os.path.join(decoded_path, y_p_combo)
    y_p_json_path = os.path.join(json_base_path, y_p_combo)

    for duration in os.listdir(y_p_combo_path):

        #duration_path = os.path.join(os.path.join(y_p_combo_path, duration), 'frames')
        #duration_write_path = os.path.join(os.path.join(y_p_write_path, duration), 'frames')
        duration_path = os.path.join(y_p_combo_path, duration)
        duration_write_path = os.path.join(y_p_write_path, duration)
        duration_json_path = os.path.join(y_p_json_path, duration)

        if not os.path.exists(duration_write_path):
            os.makedirs(duration_write_path)

        for file_name in sorted(os.listdir(duration_path)):

            file_path = os.path.join(duration_path, file_name)
            file_write_path = os.path.join(duration_write_path, file_name)

            json_path = json_path_generate(duration_json_path, j2f_ratio)

            capture = cv2.VideoCapture(file_path)
            fps = round(capture.get(cv2.CAP_PROP_FPS))
            frame_idx = 0

            while(capture.isOpened()):

                retval, frame = capture.read()

                if retval == True:

                    sali_decoded = decode(frame, json_path[frame_idx], frame_idx, fps, duration, salient_patch_size)

                    if frame_idx == 0:
                        frame_height, frame_width, _ = sali_decoded.shape
                        writer = cv2.VideoWriter(file_write_path, fourcc, fps, (frame_width, frame_height))

                    writer.write(sali_decoded)

                    frame_idx += 1
                else:
                    break
            capture.release()
            writer.release()




















































'''
        for frame_idx, file_name in enumerate(sorted(os.listdir(duration_path))):

            print('FRAME IDX:{}'.format(frame_idx))
            file_path = os.path.join(duration_path, file_name)
            file_write_path = os.path.join(duration_write_path, file_name)
            json_path = json_path_generate(duration_json_path, j2f_ratio)

            fps = 30

            result = []

            frame = cv2.imread(file_path)
            sali_decoded = decode(frame, json_path[frame_idx], frame_idx, fps, duration, salient_patch_size)

            cv2.imwrite(file_write_path, sali_decoded)
'''

























#end
