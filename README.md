The implementation source code and measurement data of Min^4-FSMC scheme.

This code includes three parts, i.e., SGF, edgeVideo measurements, target recognition (this can be TGA-FSMC scheme).

1. SGF

  requirement:cv2 
  
  including similarity calculation and frames filterring
  
  dataset can be found at: https://www.kaggle.com/akshaybapat04/frames-from-video
  
2. edge video measurements

  requirements:opencv and detectron2
  
  this can be seen in edgeVideo_measurements,and the enviroment.py is the main file. 
  After configuration, it is packaged as a docker and can be placed on nodes with different CPU frequencies for measurements.

  Dataset: COCO dataset, refers to Tsung-Yi Lin, Michael Maire, Serge Belongie, Lubomir Bourdev, Ross Girshick, James Hays, and Pietro Perona. Microsoft coco: Common objects in context, 2015.

3. TGA-FSMC

After measurement, the measurement data set can be obtained. We use the measurement data for evaluating the proposed scheme.
   
The measurement data can be found in examples/TargetRecognition/data.
   
   
   
   

