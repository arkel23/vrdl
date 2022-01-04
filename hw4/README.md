# Homework 4

## Results and summary

Our best model, [EDSR](https://arxiv.org/abs/1707.02921)
achieves a score of 28.2207.

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
[MMEditing](https://github.com/open-mmlab/mmediting/) library (renames
folders, resizes images to be divisible by integer factors, rescales them
to low resolution versions by X factors, and creates metadata file).
```
cd data
chmod +x prepare_dataset.sh
./prepare_dataset.sh
```

### Train model

Download x2 checkpoint from [Drive](https://drive.google.com/file/d/16iqzbRvmzv-m3xm8GEx0EIVfIe0N48U0/view?usp=sharing), 
make folder and put it into it:

`mkdir work_dirs/edsr_x2c64b16_g1_300k_hw4/`

`mv iter_300000.pth work_dirs/edsr_x2c64b16_g1_300k_hw4/`

Train the best model (requires continuing from checkpoint of x2):

`python tools/train.py configs/custom/edsr_x3c64b16_g1_300k_hw4_continue.py`

### Evaluation

#### Find best checkpoint

`python find_best.py --path_log_json work_dirs/edsr_x3c64b16_g1_300k_hw4_continue/202YMMDD_HHMMSS.log.json`

#### Generate high-resolution images

Download the pretrained checkpoint of best model from 
[Google Drive](https://drive.google.com/file/d/1XDDpbUlhBlDT4vl3sHlqLQo7Fsc3Da7N/view?usp=sharing)
and put it into checkpoints directory:

`python inference.py configs/custom/edsr_x3c64b16_g1_300k_hw4_continue.py checkpoints/EDSR_x3_continue_iter_10000.pth data/testing_lr_images/testing_lr_images/ results/`

#### Prepare submission

`cd results`

`zip sample_submission.zip *.png`

## Reference
* <https://github.com/open-mmlab/mmediting>
* <https://github.com/open-mmlab/mmediting/issues/125>
* <https://github.com/xinntao/BasicSR/blob/master/scripts/data_preparation/generate_meta_info.py>
