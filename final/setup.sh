conda create --name 0860802_final -y
conda activate 0860802_final
conda install pytorch torchvision torchaudio cudatoolkit=X.XX -c pytorch
conda install pandas scipy -y
pip install -r requirements.txt
cd fgvr/models/pretrained_vit
pip install -e .
python download_convert_models.py
cd ../../../
