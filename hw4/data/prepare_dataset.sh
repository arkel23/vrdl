python download_extract_data.py

mv training_hr_images hw4
python resize_dataset.py --path_images hw4/training_hr_images/
python ../tools/data/super-resolution/div2k/make_annot.py \
--gt_folder hw4/HW4_train_hr_divby6/ --meta_info_txt hw4/meta_info_HW4_GT.txt

mv Set5 val_set5
cd val_set5
mv GTmod12 Set5_mod12
mv LRbicx2 Set5_bicLRx2
mv LRbicx3 Set5_bicLRx3
mv LRbicx4 Set5_bicLRx4
