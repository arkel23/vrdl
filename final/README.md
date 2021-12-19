# Final Project

## Results and summary

Our best model reaches an accuracy of 91.3617 in the competition using ViT L-16, batch size 4, base_lr 0.08, weight decay 0, 
gradient clipping above 1, and utilizing the whole 3k images for training (skipping validation), and using the last epoch for test.

## Hardware

The experiments with L-16 were run on a workstation with NVIDIA Quadro RTX 8000 with 48GB VRAM. The rest were run on servers 
with either V100 32GB VRAM or RTX 3090 with 24GB VRAM.

## Reproducing submission

### Installation

Run each of the lines in the setup.sh to setup the environment, including downloading ViT pretrained models.

### Download and prepare dataset

Go into data_skipeval and run `prepare_dataset.sh`. It will download the dataset, extract it into its corresponding folders, and make the 
required dataset files for reproducing the best results. If you would prefer the version with validation, then do the same but go into the 
data folder and run `prepare_dataset.sh`.

### Train model

Train the best model:

`python train.py --model L_16 --base_lr 0.08 --batch_size 4 --pretrained --weight_decay 0 --clip_grad 1.0 --skip_eval --dataset_path data_skipeval`

### Evaluation

Download the pretrained checkpoint from [Google Drive](https://drive.google.com/drive/folders/1l1RLUiglv0MHUREi56KBoVFckOulcbVM?usp=sharing)
of best model and put it into folder:

`mv L_16_last.pth save/models/L_16_is448_bs4_blr0.08decay0.0_ptTruefzFalse_trial0_skipTrue/`

`python inference.py --path_checkpoint save/models/L_16_is448_bs4_blr0.08decay0.0_ptTruefzFalse_trial0_skipTrue/L_16_last.pth --print_freq 1000`

Then upload directly to Kaggle for evaluation with:
`kaggle competitions submit -c the-nature-conservancy-fisheries-monitoring -f submission.csv -m "Message"`

## Reference
* <https://github.com/HobbitLong/RepDistiller>
* <https://github.com/arkel23/IntermediateFeaturesAugmentedRepDistiller>
* <https://github.com/arkel23/PyTorch-Pretrained-ViT>
* <https://github.com/lukemelas/PyTorch-Pretrained-ViT>

