# Homework 2

## Results and summary

Our best model, [VarifocalNet](https://arxiv.org/abs/2008.13367) with a ResNet-101 backbone, achieves mAP of 0.412197, inference time per image of 0.2186 s,
and 4.58 FPS.

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
prepares the test data to be used with [mmdetection](https://github.com/open-mmlab/mmdetection/) library.

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

`python tools/train.py configs/custom/vfnet_r101_fpn_mdconv_c3-c5_mstrain_2x_svhn_666x400.py`

### Evaluation

#### Test inference speed

On [Colab](https://colab.research.google.com/drive/1bVP0XQlN5X47KKh07N13lA7GlmtECZuI?usp=sharing)
run all cells to download packages, setup and download required files, and inference model speed.

Alternatively, for local evaluation, run:

`python inference.py --config configs/custom/vfnet_r101_fpn_mdconv_c3-c5_mstrain_2x_svhn_666x400.py \
--ckpt checkpoints/vfnet_r101_fpn_666x400_svhn_epoch3.pth
--data data/test
`

#### Generate COCO submission

Download the pretrained checkpoint of best model from 
[Google Drive](https://drive.google.com/file/d/1XK7YfK1ImlhZXY62CO8omJVful-tGGAV/view?usp=sharing)
and put it into checkpoints directory:

`python tools/test.py configs/custom/vfnet_r101_fpn_mdconv_c3-c5_mstrain_2x_svhn_666x400.py \ 
checkpoints/vfnet_r101_fpn_666x400_svhn_epoch3.pth \
--format-only --options "jsonfile_prefix=./vfnet_r101_fpn_mdconv_c3-c5_mstrain_2x_svhn_666x400_epoch3`


## Reference
* <https://github.com/open-mmlab/mmdetection>
* <https://github.com/akarazniewicz/cocosplit/blob/master/cocosplit.py>
* <https://github.com/z3588191/NCTU-Selected-Topics-in-Visual-Recognition-using-Deep-Learning-Homework-2/blob/main/create_digit_annot.py>

