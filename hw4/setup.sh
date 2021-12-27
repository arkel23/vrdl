conda create --name 0860802_hw4 -y
conda activate 0860802_hw4
# may need to install according to CUDA version
# https://github.com/open-mmlab/mmediting/blob/master/docs/en/install.md
conda install pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 cudatoolkit=11.0 -c pytorch
# https://download.openmmlab.com/mmcv/dist/cuYYY/torchX.X.X/index.html
pip install mmcv-full==1.4.1 -f https://download.openmmlab.com/mmcv/dist/cu110/torch1.7.0/index.html
pip install -r requirements.txt
pip install -v -e .