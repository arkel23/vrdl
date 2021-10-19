wget -O data.zip https://competitions.codalab.org/my/datasets/download/83f7141a-641e-4e32-8d0c-42b482457836
unzip data.zip

mkdir train
mkdir test
unzip training_images.zip -d train
unzip testing_images.zip -d test

python prepare_dataset.py
