# HRM_NN_Training

Code for training a NN for Sentinel-2 and Google images on [nightlights](https://ngdc.noaa.gov/eog/download.html) (poverty proxy).

We train CNN to predict nightlights intensity from Google Maps and Sentinel-2 satellite images. The trained models are used for extracting features in the [HRM project](https://github.com/WFP-VAM/HRM). The work is based on transfer-learning papers from [N Jean et al.](http://science.sciencemag.org/content/353/6301/790) and [A Head et al.](http://www.jblumenstock.com/files/papers/jblumenstock_2017_ictd_satellites.pdf). 

The process to obtain these models is done in 4 steps:
- get the nightlights intensity as a raster. We use Google Earth Engine to get nightlights layer from [NOAA - VIIRS](https://ncc.nesdis.noaa.gov/VIIRS/) and mask it with the [JRC - GHS settleemnt layer](https://ghsl.jrc.ec.europa.eu/ghs_pop.php). The script can be found here.
- bin the nightlights into 3 classes: high intensity, medium intensity and low intensity. The first 2 steps can be done using the code in [src/nightlights_prep.py](https://github.com/WFP-VAM/HRM_NN_Training/blob/master/scripts/nightlights_prep.py).
- pull the images from Google Maps static API and Sentinel-2. The code that handles this step is [master_getdata.py](https://github.com/WFP-VAM/HRM_NN_Training/blob/master/master_getdata.py).
- train the models using [train.py](https://github.com/WFP-VAM/HRM_NN_Training/blob/master/train.py), GPU may be necessary, we use AWS ec p2 instance. We trained for every datasource both a simple CNN with 3 conv and 1 dense hidden layers and a VGG16 with block 5 and dense layer trainable. The models are in (https://github.com/WFP-VAM/HRM_NN_Training/blob/master/src/models.py).
