# Homework 2

## Results and summary

Our best model reaches an accuracy of 91.3617 in the competition using ViT L-16, batch size 4, base_lr 0.08, weight decay 0, 
gradient clipping above 1, and utilizing the whole 3k images for training (skipping validation), and using the last epoch for test.

## Hardware

All experiments were run on workstation with either NVIDIA Quadro RTX 8000 48GB 
VRAM, V100 32GB VRAM or RTX 3090 with 24GB VRAM.

## Reproducing submission

### Installation

Run each of the lines in the setup.sh to setup the environment, 
including downloading ViT pretrained models.

### Download and prepare dataset
```
cd data
python download_extract_data.py
python create_cocostyle_labels.py
python cocosplit.py --annotations train_val.json --train train.json \
--test val.json --split 0.9
cd ..
python tools/dataset_converters/images2coco_modified.py data/test/ \
data/svhn_classes.txt data/test.json
```

### Train model

Train the best model:

`python tools/train.py configs/custom_faster_rcnn_r50_fpn_1x_coco.py --no-validate`

### Evaluation

Download the pretrained checkpoint from [Google Drive](https://drive.google.com/drive/folders/1l1RLUiglv0MHUREi56KBoVFckOulcbVM?usp=sharing)
of best model and put it into folder:

`mv L_16_last.pth save/models/L_16_is448_bs4_blr0.08decay0.0_ptTruefzFalse_trial0_skipTrue/`

`python inference.py --path_backbone save/models/L_16_is448_bs4_blr0.08decay0.0_ptTruefzFalse_trial0_skipTrue/L_16_last.pth`

## Reference
* <https://github.com/open-mmlab/mmdetection>
* <https://github.com/akarazniewicz/cocosplit/blob/master/cocosplit.py>
* <https://github.com/z3588191/NCTU-Selected-Topics-in-Visual-Recognition-using-Deep-Learning-Homework-2/blob/main/create_digit_annot.py>

