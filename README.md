# SALI360
This is python code for the paper work accepted in ACM MMSys'20. You can access to the paper through this link.

## Prerequisities
- Language: Python
- Required Packages: numpy, pandas, matplotlib, scipy, sklearn, and opencv
- Need to install 'OpenCV' [link](https://www.pyimagesearch.com/2016/10/24/ubuntu-16-04-how-to-install-opencv/)
- Need to install 'transform360' developed by 'facebook' [link](https://github.com/facebook/transform360))
- Need to install 'FFMPEG' [link](https://www.ffmpeg.org/download.html)
- To install the required package, type the following command:

1) Python 2
```
pip install numpy pandas matplotlib scipy sklearn
```
2) Python 3
```
pip3 install numpy pandas matplotlib scipy sklearn
```

## Dataset
We collected head and eye gaze movement dataset from 20 vonlunteers over 10 360-degree VR video.


### Running the code
Plot the eye gaze heat map
```
python3 eye_gaze_heatmap.py
```
![Data Filter](figure/eye_gaze_heatmap.png)

2. Gait Cycle Detection: slice walk cycles from the data sequences
```
python3 cycle_detection.py
```
![Interpolation](figure/)

3. Interpolation: make walk cycles consistent in length
```
python3 interpolation.py
```
![Interpolation](figure/)

4. Cycle Filtering: filter out noisy cycles
```
python3 cycle_filter.py
```
![Cycle Filter](figure/)

5. Classification: DNN (Multi Layer Perceptron), CNN, and RNN (LSTM)
```
python3 DNN.py
python3 CNN.py
python3 RNN.py
```
![Authentication](figure/)
