This project is the code for Min4-FSMC Project (corresponding to the TheWebConf 2022 paper "Min4-FSMC: Lightweight Frame Scheduling and Model Caching
for Multi-Edge Video Analytics").

Link of TheWebConf 2022: https://www2022.thewebconf.org/   

# This project has three part:  

## (1) SGF Mechanism

The source code is for evaluating the effectiveness of SGF mechanism in this paper. It include similarity calculation and frames filterring.

The **dataset** used for SGF mechanism can be found at: https://www.kaggle.com/akshaybapat04/frames-from-video. This is a dataset that can provide the video frames sampled from video streams.

**Main requirement**: python3, cv2. 

**Location**: Min4-FSMC/SGF. The similarity calculation can be found in SGF/Similarity2.py using the 1.jpg, 2.jpg and 3.jpg to be the examples. The SGF.py first loads the dataset for computing their similarity, and then the writes as a file for saving. The PlotFigure.py reads the similarity results file and decides which frames should be filtered, and finally the plots the result figure.

## (2) edge video analytics measurements

The system source code is for readers to understand how we implemented a measurement for edge video analytics applications. In this measurement, the detectron2 can be used as our object recognition platform, which can be found at: https://github.com/facebookresearch/detectron2. 

After configuration, it is packaged as a docker and can be placed on diverse nodes with different CNN models for measurements. After the measurements, the practical measurement data can be obtained at examples/TargetRecognition/data, which main includes the processing latency, energy cconsumption cost and video analytics accuracy.

The **dataset** used for measurements is the COCO dataset that can be found at: https://cocodataset.org/#download.

There are two reasons for choosing the COCO dataset instead of the video stream-based dataset. On a one hand, the selected CNN models are trained based on the COCO dataset. On the other hand, after filtering by the SGF mechanism, the object of scheduling decision will become video frames. Directly selecting the COCO dataset can simplify our experimental process.

**Main requirements**：python3, opencv and detectron2.

**Location**: Min4-FSMC/edgeVideo_measurements/environment.py. The environment.py is the main file of the measuremts.
The config_name.json notes the name of CNN models and the pic_name.json notes the picture name of video frames. They will be generated during the process of running the object recognition application.

## (3) TGA-FSMC scheme

The source code is for evaluating the TGA-FSMC scheme in this paper, which is based on the measurement dataset in the second part.

The entry file can be found in examples/demo_edgeVideo_targetRecognition.py. Specifically, the crossover.py, mutation.py, ranking.py and selection.py can be found at sko/operations.

**Dataset**: the measurement data can be found at examples/TargetRecognition/data.

**Main requirements**: python3, numpy.

**Location**: This part includes the examples, sko and EdgeVideoClass files. The entry file can be found in examples/demo_edgeVideo_targetRecognition.py. Additionally, the measurement data can be found at examples/TargetRecognition/data and the examples/TargetRecognition.py will read the measurements data and compute the fitness value for TGA-FSMC algorithm. The crossover.py, mutation.py, ranking.py and selection.py can be found at sko/operations and the GA_TR.py is the a class file for GA algorithm, which will evolve from generation to generation to obtain the optimal strategy. Finally, the EdgeVideoClass file includes the Caching.py and User.py, which initialize caching variables and define some parameters of the users.

# Running
First, download the whole source code from the link. Second, enter the directory of the source code and follow the running commands below.

## 1. Running SGF Mechanism
1) Using the 1.jpg, 2.jpg and 3.jpg as the examples to run the SGF/Similarity2.py for illustrating the similarity among different frames.
2) Download the dataset from https://www.kaggle.com/akshaybapat04/frames-from-video.
3) Run the SGF.py for calculating the similarity of the frames in the dataset, and write as a txt file for recording.
4) Run the PlotFigure_SGF.py for filtering the frames.

## 2. edge video analytics measurements
1) Configure OpenCV and detectron2 environment.
2) Download the COCO dataset from the https://cocodataset.org/#download.
3) Select the CNN model and Prepare nodes.
4) Package the whole environment as a docker and place at different nodes for measuring the video analytics performance, including processing delay, energy consumption cost and video analytics accuracy.
6) record as measurement dataset.

## 3. TGA-FSMC scheme
1) prepare the measurement data.
2) run examples/demo_edgeVideo_targetRecognition.py. 







   
   
   
   

