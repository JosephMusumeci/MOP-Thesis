# Thesis Work (2024)
**MmWave Human Posture Estimation: Deep learning analysis for posture estimation Multi-Optimiser Testing (Mop)**

_**This work was developed for Thesis. It contains the dataset and code used for developing and testing of optimiser types on a CNN in mmWave radar sensors. Further development of a home-made system will be shared on another repository as more model testing and hardware implementation are done with an IWR1443Boost radar.**_

The dataset used for model testing is from MARS [ https://github.com/SizheAn/MARS ] which is free for public access under their MIT license.

**Folder Structure**
The folder structure is described as below.
```
${ROOT}
|-- synced_data
|   |-- wooutlier
|   |   |-- subject1
|   |   |   |-- timesplit
|   |   |-- subject2
|   |   |   |-- timesplit
|   |   |-- subject3
|   |   |   |-- timesplit
|   |   |-- subject4
|   |   |   |-- timesplit
|   |-- woutlier
|   |   |-- subject1
|   |   |   |-- timesplit
|   |   |-- subject2
|   |   |   |-- timesplit
|   |   |-- subject3
|   |   |   |-- timesplit
|   |   |-- subject4
|   |   |   |-- timesplit
|
{Created Files}
|
|-- featurewith
|-- featurewout
|-- results
|   |-- {optimiser_name}
|   |   |-- featurewith
|   |   |-- featurewout
```

**_synced_data_** folder contains all data with outlier/without outlier. Under the subject folder, there are synced kinect_data.mat and radar_data.mat. Under timesplit folder, there are train, validate, the test data and labels for each user. Note that labels here have all 25 joints from Kinect. At this stage, only 19 of them were used. These 19 joints were: Spine(base), Spine(mid), Spine(shoulder), Neck, head, both Shoulder, both Elbow, both Wrist, both Hip, both Knee, both Ankle, both Foot. 
-Both refers to Left {joint} and right {joint}.

**_feature_folders_** folder contains train, validate, the test feature and labels for all users. The features are generated from the synced data.

Dimension of the created radar feature is (frames, 8, 8, 5). The final 5 means x, y, z-axis coordinates, Doppler velocity, and intensity.
Dimension of the reference kinect label is (frames, 57). Where 57 means 19 coordinates in x, y, and z-axis.


**Dependencies**

- Keras 2.3.0
- Python 3.7
- Tensorflow 2.2.0
- matplotlib
- numpy
- pandas

**Run the code**

The code contains load data, compile model, training, and testing. Readers can also choose to load the pretrained model and just do the testing.
```
python kinectlabelcreate.py

python radarfeaturecreate.py

python mop_model.py
```
**License**

Copyright (c) 2024 Joseph Musumeci

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

