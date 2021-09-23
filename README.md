# DeTr for Panoptic Segmentation
The objective of this capstone project is to understand and train a custom panoptic segmentation model on the dataset "construction materials". This is doen by using a Transformer based model DeTr. The basic understanding of this model is given [here](https://github.com/MittalNeha/Extensive_Vision_AI6/blob/main/CAPSTONE/Understanding%20DETR.md). In this text I will explain the various steps taken to get the end to end pipeline to train this model. At a very high level, the tasks include creating a ground truth (GT) that complies with the various annotation formats followed for coco, training the detection or bounding box model and training the panoptic segmentation model. 



What is achieved so far is the understanding and the complete flow. However the training of the detr models is still pending

This repo is being created from the code that was commited to a differnet repo so far: 
https://github.com/MittalNeha/Extensive_Vision_AI6/tree/main/CAPSTONE



1. ## Creating the Ground Truth/ Data Preparation

   This step includes combining the custom dataset, understanding coco annotation formats and creating these annotations. All these steps are captured in the [colab file](https://github.com/MittalNeha/detr-panoptic/blob/main/dataset_preparation.ipynb) 

   - Custom Dataset: The construction materials dataset that was used for this project is available [here.](https://drive.google.com/file/d/1IsK268zLnXB2Qq0X2LgNDwBZRuVwvjRx/view?usp=sharing) The dataset was created and labelled using CVAT. When downloading the dataset from CVAT, it creates different directories for each class. CvattoCoco.py combined this data from various categories at large. It performs the following steps to achieve the same:

     - Takes the categories from the coco dataset (things and stuffs) and creates a map map_coco_categories.p such that all the coco things form a new category "misc_stuff". The categories for "materials" i.e. the construction materials is appended further to this categories list.
     - Split the images in each class to 80/20 train/validation dataset, hence creating `materials_train` and `materials_val`
     - Since Detr trains the images with the maximum height of 800. All the source images are converted to "RGB" and downscaled such that the maximum height is 800 and the maximum for the other dimension is 1333.
     - All images that are missing any annotations were moved to the `materials_test` folder.
     - All images are have a incrementing image id and the file_name is same as the image_id.

     On saving these annotations, the directory structure looks like this:

     ```
     parsed_data
     +--	annotations
     |	+-- train_coco.json
     |	+-- val_coco.json
     +-- materials_train
     |	+-- <image_id_1>.jpg
     |	+-- <image_id_2>.jpg
     +-- materials_val
     |	+-- <image_id_1>.jpg
     |	+-- <image_id_2>.jpg
     +-- materials_test
     |	+-- <image_id_1>.jpg
     |	+-- <image_id_2>.jpg
     ```
     
   - Panoptic annotations: Since the final task is to make panoptic predictions, the ground truth should also have the same. In order to create the panoptic mask for the custom dataset, each image is passed through a pretrained detr_panoptic model, with the backbone of resnet50. After these coco class predictions, the mask from materials subcategory needs to be merged on top of the masks. To get this conversion to run in batches, the script was created that follows the detr code refering to the repo https://github.com/facebookresearch/detr. `!python create_panoptic_dataset.py --batch_size 2 --dataset_file coco_materials --data_path parsed_data/ --device cuda --output_dir output/ --coco_panoptic_path coco_panoptic` 

     The post processing step of segmentation consists of the following:

     - ```
       scores > self.threshold
       ```
       
     - Merge Segmentations:
     
     ```
       m_id = masks.transpose(0, 1).softmax(-1)
       m_id = m_id.argmax(-1).view(h, w)
     ```
       The output of softmax here, gives the output such that each query out that satisfies the threshold from the previous step has the value closer to 1.0. Therefore taking the argmax in next step, gives the index corresponsing to the order of predicted classes to each pixel of the mask output.
     
       To trick these operations and give priority to the materials category, the custom_class_mask was concatenated to m_id such that the maximumvalue for the mask was 2.0 instead of 1.0. This way, the regions of overlap between the coco prediction and custom class segmentation will give priority to the custom class.
     
       ```
       custom_mask = cv.normalize(input_segments, None, alpha=0, beta=2, norm_type=cv.NORM_MINMAX, dtype=cv.CV_32F)
       m_id = torch.cat((m_id, custom_mask.to(m_id.device)), 1)
       ```
     
     - Create Segment id: in the panoptic format, the segment_info id is calculated from the RGB values of the segmentation mask image. This is done by using the id2rgb, rgb2id apis. Hence the segment ids were created for each mask. The class id's were multiplied by 1000 just to get a good contrast for the masks
     
      ```
       new_id = cur_classes*1000 + torch.Tensor(range(len(cur_classes)))
      ```
     
     - 
     
       
     
   - kdhfkhaskfhjds
   
2. ## Training Bounding box Model

3. ## Training the Panoptic head



