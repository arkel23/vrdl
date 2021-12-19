conda create --name 0860802_hw1 -y
conda activate 0860802_hw1
conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch-nightly -y
conda install pandas scipy -y
pip install -r requirements.txt
cd fgvr/models/pretrained_vit
pip install -e .
python download_convert_models.py
cd ../../../
