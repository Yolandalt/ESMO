The implementation source code and measurement data of Min^4-FSMC scheme.

This code includes three parts, i.e., SGF, edgeVideo measurements, target recognition (this can be TGA-FSMC scheme).

1. SGF
  requirement:cv2
  including similarity calculation and frames filterring
  dataset can be found at: https://www.kaggle.com/akshaybapat04/frames-from-video
2. edge video measurements
  requirements:opencv and detectron2
  this can be seen in edgeVideo_measurements,and the enviroment.py is the main file. We need the 
5. TGA-FSMC
