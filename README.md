This project is the code for Min^4-FSMC Project (corresponding to the TheWebConf 2022 paper "Min4-FSMC: Lightweight Frame Scheduling and Model Caching
for Multi-Edge Video Analytics").

Link of TheWebConf 2022: https://www2022.thewebconf.org/   

This project has three part:  

## (1) SGF Mechanism

The source code is for evaluating the effectiveness of SGF mechanism in this paper. It include similarity calculation and frames filterring.

The dataset used for SGF mechanism can be found at: https://www.kaggle.com/akshaybapat04/frames-from-video. This is a dataset that can provide the video frames sampled from video streams.

Main requirement: cv2. 

## (2) edge video analytics measurements

The system source code is for readers to understand how we implemented a measurement for edge video analytics applications. In this measurement, the detectron2 can be used as our object recognition platform, which can be found at: https://github.com/facebookresearch/detectron2. 

After configuration, it is packaged as a docker and can be placed on diverse nodes with different CNN models for measurements. After the measurements, the practical measurement data can be obtained at examples/TargetRecognition/data, which main includes the processing latency, energy cconsumption cost and video analytics accuracy.

Main requirements: opencv and detectron2.

The dataset used for measurements is the COCO dataset that can be found at: https://cocodataset.org/#download.

There are two reasons for choosing the COCO dataset instead of the video stream-based dataset. On a one hand, the selected CNN models are trained based on the COCO dataset. On the other hand, after filtering by the SGF mechanism, the object of scheduling decision will become video frames. Directly selecting the COCO dataset can simplify our experimental process.

## (3) TGA-FSMC scheme

The source code is for evaluating the TGA-FSMC scheme in this paper, which is based on the measurement dataset in the second part.




1 Location
The simulation source code is under the folder Pano_experiments\Simulation_Statistics. And now we supply methods of push button running or component running. The system source code is under the folder Pano_system\SystemComponent. And we only supply each component core source code of the Pano system. The following shows the requirements of the system and simulation comparision and the two running method of the simluation source code.

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
Run "/main.m" to simulate grid tiling baseline and Pano client-side mechanism. The PSPNR and bandwidth consumption per user per chunk are stored in "/baselineResult" and "/PanoResult". A PSPNR-bandwidth graph like Fig 18 is drawed based on the simulation result. To observe Pano tiling and allocation result, you should put a breakpoint at "/Pano.m" line 188. When pausing, paste and run the following commands in console:

viewedTilesSize = [];
viewedTileMSE = [];
for i=1:nTiles
if viewed(i)
    viewedTileMSE = [viewedTileMSE,meanMSEreal(user,i,QP(iViewed)-22+1)];
    viewedTilesSize = [viewedTilesSize, tileSize(i,QP(iViewed)-22+1)];
end
end
disp(['set ',num2str(set),' vid ',num2str(vid),' sec ',num2str(sec),' user ',num2str(user),' qp ',num2str(qp)]);
disp('transmitted tile:(start row, end row, start column, end column)');
disp(tiling(viewed));
disp('QP allocation:');
disp(QP);
disp('bitrate of each tile: (kbit/s)');
disp(viewedTilesSize);
disp('PMSE * number of pixels:');
disp(viewedTileMSE);
(2) Automatic Running
Run the batch.bat
4 Contact us if questions
If you have any questions with this project, or you have some questions about the paper, feel free to contact us:

Yu Guan: shanxigy@pku.edu.cn

Chengyuan Zheng: zhengchengyuan@pku.edu.cn

Jinyu Yin : yinjinyu@pku.edu.cn

If you have any suggestion for this code, please also feel free to contact us. We will take our time to update the codes.
   
   
   
   

