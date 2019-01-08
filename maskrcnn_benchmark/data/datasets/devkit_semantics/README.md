###########################################################################
#    THE KITTI VISION BENCHMARK: SEMANTIC/INSTANCE SEGMENATION BENCHMARKS #
#                   Andreas Geiger         Hassan Abu Alhaija             #
#          Max Planck Institute for Intelligent Systems, Tübingen         #
#                          Heidelberg University                          #
#                             www.cvlibs.net                              #
###########################################################################



This file describes the KITTI semantic/instance segmentation 2015 benchmarks,
consisting of 200 training and 200 test image pairs for each task. Ground truth 
has been acquired by manual segmentation by people.


Dataset description:
====================

The Kitti 2015 segmentation format (described below) is used as common format for all datasets. 
The image names are prefixed by the dataset's benchmark name.
Exactly the same image names are used for the input images and the ground truth files.
```
datasets_kitti2015/
   test/
      image_2/
         <dataset>_<img_name>.png
         ...
   training/
      image_2/
         <dataset>_<img_name>.png
         ...
      instance/
         <dataset>_<img_name>.png
         ...
      semantic/
         <dataset>_<img_name>.png
         ...
```

The "semantic" folder contains the semantic segmentation ground truth for the training images. Each file is a single channel uint8 8-bit PNG image with each pixel value representing its semantic label ID. 

The "instance" folder contains the combined instance and semantic segmentation ground truth. 
Each file is a single channel uint16 16-bit PNG image where the lower 8 bits of each pixel value are its instance ID, 
while the higher 8 bits of each pixel value are its semantic labels ID. 
Instance IDs start from 1 for each semantic class (ex. car:1,2,3 ... etc. - buiding:1,2,3 ... etc.). 
Instance ID value of 0 means no instance ground truth is available and should be ignored for instance segmentation. 
An example code for reading the instance and semantic segmentation ground truth from the combined ground truth file in python could look like this :
```
import scipy.misc as sp
instance_semantic_gt = sp.imread('instance/<image name>.png')
instance_gt = instance_semantic_gt  % 256
semantic_gt = instance_semantic_gt // 256

```
The labels IDs, names and instance classes of the Cityscapes dataset are used and can be found [here](https://github.com/mcordts/cityscapesScripts/blob/master/cityscapesscripts/helpers/labels.py)

#### Output ####

The output structure should be analogous to the input.
If your algorithm is called MYALGO, the result files for your instance
or semantic segmentation method can be named and placed as follows:
```
kitti2015_results/
    test/
        MYALGO_instance/
            pred_list/
                <dataset>_<img_name>.txt
                ...
            pred_img/
                <dataset>_<img_name>_000.png
                <dataset>_<img_name>_001.png
                ...
        MYALGO_semantic/
            <dataset>_<img_name>.png
            ...
```

The txt files of the instance segmentation should look as follows:
```
relPathPrediction1 labelIDPrediction1 confidencePrediction1
relPathPrediction2 labelIDPrediction2 confidencePrediction2
relPathPrediction3 labelIDPrediction3 confidencePrediction3
...
```

For example, the Kitti2015_000000_10.txt may contain:
```
../pred_img/Kitti2015_000000_10_000.png 026 0.976347
../pred_img/Kitti2015_000000_10_001.png 026 0.973782
../pred_img/Kitti2015_000000_10_002.png 026 0.973202
...
```

with binary instance masks in `kitti2015_results/test/MYALGO_instance/pred_img/`:
```
Kitti2015_000000_10_000.png
Kitti2015_000000_10_001.png
Kitti2015_000000_10_002.png
...
```


