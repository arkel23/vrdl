# Homework 3

## Results and summary

Our best model, [Mask R-CNN](https://arxiv.org/abs/1703.06870) with a 
[ResNeXt-101-64x4d](https://arxiv.org/abs/1611.05431) backbone, 
and using [Dice loss](https://arxiv.org/abs/1606.04797), 
achieves a score of 0.242279.

## Hardware

All experiments were run on workstation with RTX 3090 with 24GB VRAM.

## Reproducing submission

### Installation

Run each of the lines in the `setup.sh` to setup the environment or 
```
chmod +x setup.sh
./setup.sh
```

### Download and prepare dataset

Downloads data from Google Drive, then creates the COCO style labels (and
reorganizes folders along the way), and
finally makes a train and val split from the train_val original split. Also,
prepares the test data to be used with 
[mmdetection](https://github.com/open-mmlab/mmdetection/) library (
manually modifies the file with the test image IDs to be compatible).
```
cd data
python download_extract_data.py
python create_cocostyle_labels.py
python cocosplit.py --annotations train_val.json --train train.json \
--test val.json --split 0.9
cd ..
python edit_test.py
```

### Train model

First download pretrained checkpoint and put into checkpoints directory:
# https://drive.google.com/drive/folders/1DMelfiwQ1pEc8pJRW096-BML6w3o2IFU
# https://drive.google.com/file/d/1nKBsZvRQ6RccgCIJRGs4eEdDTyQvgI9J/view?usp=sharing

```
mkdir checkpoints
cd checkpoints
wget -O mask_rcnn_x101_64x4d_fpn_mstrain-poly_3x_coco.pth https://download.openmmlab.com/mmdetection/v2.0/mask_rcnn/mask_rcnn_x101_64x4d_fpn_mstrain-poly_3x_coco/mask_rcnn_x101_64x4d_fpn_mstrain-poly_3x_coco_20210526_120447-c376f129.pth
cd ..
```

Train the best model:

`python tools/train.py configs/custom/mask_rcnn_x101_64x4d_fpn_mstrain-poly_3x_dice_nuclei.py`

### Evaluation

#### Generate COCO submission

Download the pretrained checkpoint of best model from 
[Google Drive](https://drive.google.com/file/d/1XK7YfK1ImlhZXY62CO8omJVful-tGGAV/view?usp=sharing)
and put it into checkpoints directory:

`python tools/test.py configs/custom/mask_rcnn_x101_64x4d_fpn_mstrain-poly_3x_dice_nuclei.py \ 
checkpoints/mask_rcnn_x101_64x4d_fpn_mstrain-poly_3x_dice_nuclei_epoch12.pth \
--format-only --options "jsonfile_prefix=./mask_rcnn_x101_64x4d_fpn_mstrain-poly_3x_dice_nuclei_epoch12"`

## Reference
* <https://github.com/open-mmlab/mmdetection>
* <https://github.com/cocodataset/cocoapi/issues/131>
* <https://github.com/akarazniewicz/cocosplit/blob/master/cocosplit.py>
