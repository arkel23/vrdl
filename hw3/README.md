# Homework 3

## Results and summary

Our best model, [VarifocalNet](https://arxiv.org/abs/2008.13367) with a 
ResNet-101 backbone, achieves mAP of 0.412197, 
inference time per image of 0.2186 s, and 4.58 FPS.

## Hardware

All experiments were run on workstation with either NVIDIA Quadro RTX 8000 48GB 
VRAM, V100 32GB VRAM or RTX 3090 with 24GB VRAM.

## Reproducing submission

### Installation

Run each of the lines in the `setup.sh` to setup the environment or 
```
chmod +x setup.sh
./setup.sh
```

### Download and prepare dataset

Downloads data from Google Drive, then creates the COCO style labels, and
finally makes a train and val split from the train_val original split. Also,
prepares the test data to be used with 
[mmdetection](https://github.com/open-mmlab/mmdetection/) library.

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
wget -O mask_rcnn_r50_fpn_mstrain-poly_3x_coco.pth https://download.openmmlab.com/mmdetection/v2.0/mask_rcnn/mask_rcnn_r50_fpn_mstrain-poly_3x_coco/mask_rcnn_r50_fpn_mstrain-poly_3x_coco_20210524_201154-21b550bb.pth
cd ..
```

Train the best model:

`python tools/train.py configs/custom/vfnet_r101_fpn_mdconv_c3-c5_mstrain_2x_svhn_666x400.py`

### Evaluation

#### Generate COCO submission

Download the pretrained checkpoint of best model from 
[Google Drive](https://drive.google.com/file/d/1XK7YfK1ImlhZXY62CO8omJVful-tGGAV/view?usp=sharing)
and put it into checkpoints directory:

`python tools/test.py configs/custom/vfnet_r101_fpn_mdconv_c3-c5_mstrain_2x_svhn_666x400.py \ 
checkpoints/vfnet_r101_fpn_666x400_svhn_epoch3.pth \
--format-only --options "jsonfile_prefix=./vfnet_r101_fpn_mdconv_c3-c5_mstrain_2x_svhn_666x400_epoch3"`

## Reference
* <https://github.com/open-mmlab/mmdetection>
* <https://github.com/cocodataset/cocoapi/issues/131>
* <https://github.com/akarazniewicz/cocosplit/blob/master/cocosplit.py>
