
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
import json
import multiprocessing
from saliency_union import *
from saliency_extraction import *

'''
cube frame structure:
 ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ
|  right  |  Left    |   Up    |
|         |          |         |
|ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ |
|   down  |  Front   |   Back  |
|         |          |         |
 ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ
'''
def encoding_process(video_path, write_path, num_core_2_use, json_frame_ratio, resize_ratio, salient_patch_size):

    video_name = video_path.split('/')[-1]

    cap = cv2.VideoCapture(video_path)

    fourcc = cv2.VideoWriter_fourcc(*'X264')

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = round(cap.get(cv2.CAP_PROP_FPS))

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) #width, height of cube video cannot be the values of saliency_video
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    unit_length = int(frame_height / 2)

    #num_core_2_use x json_frame_ratio should be fps

    #saliency selection parameters
    front_sub_square_num = 30
    up_sub_square_num = 3
    down_sub_square_num = 3
    num_selected_regions = 20

    #returned img will be in a shape of 3x2 (2 == one small frame length)
    frame_width = unit_length
    frame_height = int(1.5*unit_length)
    fps = int(fps / json_frame_ratio)

    original_num_core_2_use = num_core_2_use
    frame_num = 0
    num_iter = 0
    total_iter = int(frame_count / (json_frame_ratio*num_core_2_use))

    target_path = write_path#os.path.splitext(write_path)[0]

    if not os.path.exists(target_path):
        os.makedirs(target_path)

    frame_input_list = []
    outputs = []
    process = []

    while(cap.isOpened()):
        retval, frame = cap.read()

        if retval == True:
            if num_iter == total_iter:
                num_core_2_use = int(np.ceil((frame_count - num_iter*json_frame_ratio*original_num_core_2_use) / json_frame_ratio))

            if frame_num % json_frame_ratio == 0:#
                frame_input_list.append(frame)

            if len(frame_input_list) == num_core_2_use:

                for _ in range(num_core_2_use):
                    outputs.append(multiprocessing.Queue())

                for idx in range(num_core_2_use):
                    print('core idx: %d' %idx)
                    process.append(multiprocessing.Process(target = saliency_encoder, args = (frame_input_list[idx], front_sub_square_num, up_sub_square_num, down_sub_square_num, num_selected_regions, resize_ratio, salient_patch_size, outputs[idx])))

                for pro in process:
                    pro.start()

                for idx in range(num_core_2_use):

                    print(num_iter*original_num_core_2_use + idx)
                    out_img, encoding_info = outputs[idx].get()

                    encoding_info_json = json.dumps(encoding_info)
                    json_name = 'encoding_info_'+ str(num_iter*original_num_core_2_use + idx) + '.json'
                    with open(os.path.join(target_path, json_name), 'w') as f:
                        json.dump(encoding_info_json, f)

                for idx in range(num_core_2_use):
                    outputs[idx].close()

                for pro in process:
                    pro.terminate()
                outputs = []
                process = []

                num_iter += 1
                frame_input_list = []

            frame_num += 1

        else:
            break

    cap.release()

base_path = 'video/segments/cube'
write_base_path = 'json'

if not os.path.exists(write_base_path):
    os.makedirs(write_base_path)

num_core_2_use = multiprocessing.cpu_count() - 2
json_frame_ratio = 10
salient_patch_size = 4
resize_ratio = 4

for y_p_combo in os.listdir(base_path):

    y_p_combo_path = os.path.join(base_path, y_p_combo)
    y_p_write_path = os.path.join(write_base_path, y_p_combo)

    for duration in os.listdir(y_p_combo_path):

        duration_path = os.path.join(y_p_combo_path, duration)
        duration_write_path = os.path.join(y_p_write_path, duration)

        if not os.path.exists(duration_write_path):
            os.makedirs(duration_write_path)

        for y_p_video in os.listdir(duration_path):

            video_path = os.path.join(duration_path, y_p_video)
            encoding_process(video_path, duration_write_path, num_core_2_use, json_frame_ratio, resize_ratio, salient_patch_size)

















#end
