# Final Project

## Results and summary

Our best model reaches an accuracy of 1.61553 (top 35) in the competition using ViT B-16, batch size 16, base_lr 0.08, weight decay 1e-4, 
gradient clipping above 1, with a split of 0.9 to 0.1 for training and validation, respectively, and using the best checkpoint according to the validation accuracy.

## Hardware

The experiments with L-16 and larger image sizes (640) were run on a workstation with NVIDIA Quadro RTX 8000 with 48GB VRAM. The rest were run on servers 
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

`python train.py --model B_16 --base_lr 0.08 --batch_size 16 --pretrained --weight_decay 0.0001 --clip_grad 1.0`

### Evaluation

Download the pretrained checkpoint from [Google Drive](https://drive.google.com/file/d/10juQWGykD8wmWlEGw6aP75l4Cuu1SGOL/view?usp=sharing)
of best model and put it into folder:

`mkdir checkpoints/B_16_False_cls_False_is448_bs16_blr0.08decay0.0001_ptTrue_skipFalse`

`mv B_16_best.pth checkpoints/B_16_False_cls_False_is448_bs16_blr0.08decay0.0001_ptTrue_skipFalse`

`python inference.py --path_checkpoint checkpoints/B_16_False_cls_False_is448_bs16_blr0.08decay0.0001_ptTrue_skipFalse/B_16_best.pth --print_freq 1000`

Then upload directly to Kaggle for evaluation with:

`kaggle competitions submit -c the-nature-conservancy-fisheries-monitoring -f submission.csv -m "B_16_False_cls_False_is448_bs16_blr0.08decay0.0001_ptTrue_skipFalse/B_16_best.pth"`

## Reference
* <https://github.com/HobbitLong/RepDistiller>
* <https://github.com/arkel23/IntermediateFeaturesAugmentedRepDistiller>
* <https://github.com/arkel23/PyTorch-Pretrained-ViT>
* <https://github.com/lukemelas/PyTorch-Pretrained-ViT>

