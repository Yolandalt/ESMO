This project is the code for Min^4-FSMC Project (corresponding to the TheWebConf 2022 paper "Min4-FSMC: Lightweight Frame Scheduling and Model Caching
for Multi-Edge Video Analytics").

Link of TheWebConf 2022: https://www2022.thewebconf.org/   

# This project has three part:  

## (1) SGF Mechanism

The source code is for evaluating the effectiveness of SGF mechanism in this paper. It include similarity calculation and frames filterring.

The dataset used for SGF mechanism can be found at: https://www.kaggle.com/akshaybapat04/frames-from-video. This is a dataset that can provide the video frames sampled from video streams.

Main requirement: cv2. 

Location: Min4-FSMC/SGF. The similarity calculation can be found in SGF/Similarity2.py using the 1.jpg, 2.jpg and 3.jpg to be the examples. The SGF.py first loads the dataset for computing their similarity, and then the writes as a file for saving. The PlotFigure.py reads the similarity results file and decides which frames should be filtered, and finally the plots the result figure.

## (2) edge video analytics measurements

The system source code is for readers to understand how we implemented a measurement for edge video analytics applications. In this measurement, the detectron2 can be used as our object recognition platform, which can be found at: https://github.com/facebookresearch/detectron2. 

After configuration, it is packaged as a docker and can be placed on diverse nodes with different CNN models for measurements. After the measurements, the practical measurement data can be obtained at examples/TargetRecognition/data, which main includes the processing latency, energy cconsumption cost and video analytics accuracy.

The dataset used for measurements is the COCO dataset that can be found at: https://cocodataset.org/#download.

There are two reasons for choosing the COCO dataset instead of the video stream-based dataset. On a one hand, the selected CNN models are trained based on the COCO dataset. On the other hand, after filtering by the SGF mechanism, the object of scheduling decision will become video frames. Directly selecting the COCO dataset can simplify our experimental process.

Main requirements: opencv and detectron2.

Location: Min4-FSMC/edgeVideo_measurements.

## (3) TGA-FSMC scheme

The source code is for evaluating the TGA-FSMC scheme in this paper, which is based on the measurement dataset in the second part.

The entry file can be found in examples/demo_edgeVideo_targetRecognition.py. Specifically, the crossover.py, mutation.py, ranking.py and selection.py can be found at sko/operations.

Dataset: the measurement data that can be found at examples/TargetRecognition/data.

Main requirements: numpy.

Location: Min4-FSMC/examples/demo_edgeVideo_targetRecognition.py






   
   
   
   

