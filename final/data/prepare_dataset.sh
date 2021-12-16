pip install kaggle
# put kaggle.json into ~/.kaggle/kaggle.json (if ubuntu) or C:\Users\<Windows-username>\.kaggle\kaggle.json (Windows)
chmod 600 ~/.kaggle/kaggle.json
kaggle competitions download -c the-nature-conservancy-fisheries-monitoring
unzip the-nature-conservancy-fisheries-monitoring.zip
unzip train.zip
unzip sample_submission_stg1.csv.zip
unzip sample_submission_stg2.csv.zip
unzip test_stg1.zip
sudo apt install p7zip-full p7zip-rar
7z x test_stg2.7z
rm *.zip *.7z
mv train fish

python make_data_dic_imagenetstyle.py --path fish
python renumber_classes.py --df_files fish.csv --df_id_name classid_classname.csv
python data_split.py --fn fish.csv
