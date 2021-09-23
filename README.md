# DeTr for Panoptic Segmentation
The objective of this capstone project is to understand and train a custom panoptic segmentation model on the dataset "construction materials". This is doen by using a Transformer based model DeTr. The basic understanding of this model is given [here](https://github.com/MittalNeha/Extensive_Vision_AI6/blob/main/CAPSTONE/Understanding%20DETR.md). In this text I will explain the various steps taken to get the end to end pipeline to train this model. At a very high level, the tasks include creating a ground truth (GT) that complies with the various annotation formats followed for coco, training the detection or bounding box model and training the panoptic segmentation model. 

This repo is being created from the code that was commited to a differnet repo so far: 
https://github.com/MittalNeha/Extensive_Vision_AI6/tree/main/CAPSTONE



1. ## Creating the Ground Truth

   This step includes combining the custom dataset, understanding coco annotation formats and creating these annotations.

   - Custom Dataset: The construction materials dataset that was used for this project is available here. The dataset was created and labelled using CVAT. When downloading the dataset from CVAT, it creates different directories for each class.

2. ## Training Bounding box Model

3. ## Training the Panoptic head



