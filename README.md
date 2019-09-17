# The github repository storing source code presented in "Complex Event Analysis for Traffic Risk Prediction based on 3D-CNN with Multi-sources Urban Sensing Data" paper.

### Repository structure ###
* timeseries2raster: the algorithm to convert sensing data stored in time-series format to raster images.
* prep3Draster: the algorithm to individual raster images to video-like data.
* learning_model: 
    * Historical_Average: a statistical model to predict infomation based on its historical data.
        * predict.py: to predict data
    * 2D_CNN: a deep learning model to predict information by preserving spartial orders using 2D-CNN.
        * train.py: to train the model
        * predict.py: to predict data
    * 3D_CNN: a deep learning model predict information by preserving both spartial and temporal relationships using 3D-CNN.
        * train_singlemodal.py: to train the model using only 1 factor (target factor)        
        * predict_singlemodal.py: to predict data using the model produced from train_singlemodal
        * train_multimodal.py: to train the model using many factors
        * predict_multimodal.py: to predict data using the model produced from train_multimodal
