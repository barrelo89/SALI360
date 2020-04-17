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

import cv2
import scipy as sp
import MR #candidate 1
import pySaliencyMap #candidate 2
import pyimgsaliency as psal #candidate 3
from sklearn.preprocessing import MinMaxScaler, StandardScaler #not significant difference between these two scalers

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

def frame_split(frame): #return a list of screens in an order of right, left, up, down, front and back
    height, width, _ = frame.shape #height: 2048 [0~2047], width: 3072 [0~3071]
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

def saliency_1(frame, threshold = 0.5): #requires no frame width, height information
    #candidate 1
    #some 'cracking' detected in the saliency map
    #https://github.com/ruanxiang/mr_saliency

    mr = MR.MR_saliency() # initialization
    sal = mr.saliency(frame).astype(sp.uint8)
    sal = cv2.normalize(sal, None, 0, 255, cv2.NORM_MINMAX)
    idx = (sal >= int(255*threshold))
    sal[idx] = 0
    sal[~idx] = 255
    return sal

def saliency_2(frame, threshold = 0.5):
    #candidate 2
    #its saliency map follows almost the same as the original image frame
    #but no cracking and kind of more plausible saliency map than the candidate 1
    #https://github.com/akisato-/pySaliencyMap

    height, width, _ = frame.shape
    sm = pySaliencyMap.pySaliencyMap(width, height)
    saliency = sm.SMGetSM(frame)
    binarized_map = sm.SMGetBinarizedSM(frame, threshold)

    return saliency, binarized_map

def saliency_3(frame, threshold=0.5):

    #candidate 3
    #https://github.com/yhenon/pyimgsaliency
    #Saliency Optimization from Robust Background Detection, Wangjiang Zhu, Shuang Liang, Yichen Wei and Jian Sun, IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2014

    rbd = psal.get_saliency_rbd(frame).astype('uint8')
    #often, it is desirable to have a binary saliency map
    #check 'binarise.py'
    #the method can be either adaptive or fixed
    #binarise_saliency_map(saliency_map,method='adaptive',threshold=0.5)
    binary_rbd = psal.binarise_saliency_map(rbd, method='adaptive', threshold = threshold)
    #openCV cannot display numpy type 0, so convert to uint8 and scale
    #cv2.imshow('binary',255 * binary_sal.astype('uint8'))

    return rbd, 255*binary_rbd.astype('uint8')

def saliency_4(frame, threshold=0.5):

    #candidate 3
    #https://github.com/yhenon/pyimgsaliency

    #get the saliency maps using the 3 implemented methods
    ft = psal.get_saliency_ft(frame).astype('uint8') #R. Achanta, S. Hemami, F. Estrada and S. Süsstrunk, Frequency-tuned Salient Region Detection, IEEE International Conference on Computer Vision and Pattern Recognition (CVPR 2009), pp. 1597 - 1604, 2009

    #often, it is desirable to have a binary saliency map
    #check 'binarise.py'
    #the method can be either adaptive or fixed
    #binarise_saliency_map(saliency_map,method='adaptive',threshold=0.5)
    binary_ft = psal.binarise_saliency_map(ft, method='adaptive', threshold = threshold)
    return ft, 255*binary_ft.astype('uint8')

def saliency_5(frame, threshold=0.5):

    #candidate 3
    #https://github.com/yhenon/pyimgsaliency

    #get the saliency maps using the 3 implemented methods
    mbd = psal.get_saliency_mbd(frame).astype('uint8') #Jianming Zhang, Stan Sclaroff, Zhe Lin, Xiaohui Shen, Brian Price and Radomír Měch. "Minimum Barrier Salient Object Detection at 80 FPS."

    #often, it is desirable to have a binary saliency map
    #check 'binarise.py'
    #the method can be either adaptive or fixed
    #binarise_saliency_map(saliency_map,method='adaptive',threshold=0.5)
    binary_mbd = psal.binarise_saliency_map(mbd,method='adaptive',threshold=threshold)

    return mbd, 255*binary_mbd.astype('uint8')
