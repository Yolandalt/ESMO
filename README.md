This project is the code for Min^4-FSMC Project (corresponding to the TheWebConf 2022 paper "Min4-FSMC: Lightweight Frame Scheduling and Model Caching
for Multi-Edge Video Analytics").

Link of TheWebConf 2022: https://www2022.thewebconf.org/   

# This project has three part:  

## (1) SGF Mechanism

The source code is for evaluating the effectiveness of SGF mechanism in this paper. It include similarity calculation and frames filterring.

The dataset used for SGF mechanism can be found at: https://www.kaggle.com/akshaybapat04/frames-from-video. This is a dataset that can provide the video frames sampled from video streams.

Main requirement: cv2. 

Location: Min4-FSMC/SGF.

## (2) edge video analytics measurements

The system source code is for readers to understand how we implemented a measurement for edge video analytics applications. In this measurement, the detectron2 can be used as our object recognition platform, which can be found at: https://github.com/facebookresearch/detectron2. 

After configuration, it is packaged as a docker and can be placed on diverse nodes with different CNN models for measurements. After the measurements, the practical measurement data can be obtained at examples/TargetRecognition/data, which main includes the processing latency, energy cconsumption cost and video analytics accuracy.

Main requirements: opencv and detectron2.

The dataset used for measurements is the COCO dataset that can be found at: https://cocodataset.org/#download.

There are two reasons for choosing the COCO dataset instead of the video stream-based dataset. On a one hand, the selected CNN models are trained based on the COCO dataset. On the other hand, after filtering by the SGF mechanism, the object of scheduling decision will become video frames. Directly selecting the COCO dataset can simplify our experimental process.

Location: Min4-FSMC/edgeVideo_measurements.

## (3) TGA-FSMC scheme

The source code is for evaluating the TGA-FSMC scheme in this paper, which is based on the measurement dataset in the second part.

The entry file can be found in examples/demo_edgeVideo_targetRecognition.py. Specifically, the crossover.py, mutation.py, ranking.py and selection.py can be found at sko/operations.

Location: Min4-FSMC/examples/demo_edgeVideo_targetRecognition.py



2 Requirements
.Net Framework 4.6.1 or Above.
Unity 2018.2 Pro Edition or Above.
Windows 10 Professional Edition 1709.
Any C++ IDE (Visual Studio 2015 is recommended).
Matlab (R2015a or higher) with Parallel Computing Toolbox and Image Processing Toolbox.
FFMPEG (add the binary folder, e.g. ¡°C:\Program Files\ffmpeg-20190625-dd662bb-win64-static\bin¡±, to system PATH.).

3 Running
Due to limitaion of file size, the whole source code is stored in the BaiduNetdisk. First, download the whole simluation source code from the link under the folder Pano_experiments\Simulation_Statistics. Second, enter the directory of the source code and follow the running commands below.

1) Prepare the programs and data
All the Matlab programs are in root folder "/". The data used are as follows: the videos in "/videos", the pre-calculated depth-of-field data in "/DepthMap", the viewpoint trajectory in "/traj" and "/viewpoint".

2) Split the videos to one-second chunks
Run "/cutChunk.m" to split the videos to one-second chunks in "/videos/setID/videoID/chunkID.mp4".

3) Calculate quality-bitrate efficiency
Run "/getAllTileValueness.m" to calculate the efficiency score of each tile. The efficiency score is stored in "/ratio/setID/videoID/userID/frameID_Value_SMSE.txt".

4) group similar tiles
Run the C++ program "/main.cpp" to group similar tiles. Before running, modify the relative path at line 254/358/380 as appropriate. The result stored in "/tiling1004" consists of the index, the start and end row, the start column and end column of each grouped tile.

5) Run the simulation




   
   
   
   

