# Homework 4

## Results and summary

Our best model, [Mask R-CNN](https://arxiv.org/abs/1703.06870) with a 
[ResNeXt-101-64x4d](https://arxiv.org/abs/1611.05431) backbone, 
and using [Dice loss](https://arxiv.org/abs/1606.04797), 
achieves a score of 0.242279.

## Hardware

All experiments were run on workstation with 2080Ti with 11GB of VRAM.

## Reproducing submission

### Installation

Run each of the lines in the `setup.sh` to setup the environment or 
```
chmod +x setup.sh
./setup.sh
```

### Download and prepare dataset

Downloads data from Google Drive, and prepares the data to be used with 
[MMEditing]](https://github.com/open-mmlab/mmediting/) library (renames
folders, resizes images to be divisible by integer factors, rescales them
to low resolution versions by X factors, and creates metadata file).
```
cd data
chmod +x prepare_dataset.sh
./prepare_dataset.sh
```

### Train model

First download pretrained checkpoint and put into checkpoints directory:
```
mkdir checkpoints
cd checkpoints
wget -O mask_rcnn_x101_64x4d_fpn_mstrain-poly_3x_coco.pth 
https://download.openmmlab.com/mmdetection/v2.0/mask_rcnn/mask_rcnn_x101_64x4d_fpn_mstrain-poly_3x_coco/mask_rcnn_x101_64x4d_fpn_mstrain-poly_3x_coco_20210526_120447-c376f129.pth
cd ..
```

Train the best model:

`python tools/train.py configs/custom/edsr_x3c64b16_g1_300k_hw4.py`

### Evaluation

#### Generate COCO submission

Download the pretrained checkpoint of best model from 
[Google Drive](https://drive.google.com/file/d/1UFfsgtLbKcJeia11LlamShPOoXiofghw/view?usp=sharing)
and put it into checkpoints directory:

`python inference.py configs/custom/edsr_x3c64b16_g1_300k_hw4.py checkpoints/edsr_x3c64b16_g1_300k_hw4_iter_65000.pth data/testing_lr_images/testing_lr_images/ results/`

## Reference
* <https://github.com/open-mmlab/mmediting>
* <https://github.com/open-mmlab/mmediting/issues/125>
* <https://github.com/xinntao/BasicSR/blob/master/scripts/data_preparation/generate_meta_info.py>
