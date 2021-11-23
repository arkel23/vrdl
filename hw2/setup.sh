conda create --name 0860802_hw1
conda activate 0860802_hw1
conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch-nightly
conda install pandas scipy
pip install -r requirements.txt
cd fgvr/models/pretrained_vit
pip install -e .
python download_convert_models.py
cd ../../../
