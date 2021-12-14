import os
import json
import shutil
import numpy as np
from PIL import Image
from pycocotools import mask
from skimage import measure


def binary_to_coco(gt_binary_mask):
    # https://github.com/cocodataset/cocoapi/issues/131
    fortran_ground_truth_binary_mask = np.asfortranarray(gt_binary_mask)
    encoded_ground_truth = mask.encode(fortran_ground_truth_binary_mask)
    ground_truth_area = mask.area(encoded_ground_truth)
    ground_truth_bounding_box = mask.toBbox(encoded_ground_truth)
    contours = measure.find_contours(gt_binary_mask, 0.5)

    annotation = {
            "segmentation": [],
            "area": ground_truth_area.tolist(),
            "iscrowd": 0,
            "bbox": ground_truth_bounding_box.tolist(),
            "category_id": 1,
        }

    for contour in contours:
        contour = np.flip(contour, axis=1)
        segmentation = contour.ravel().tolist()
        annotation["segmentation"].append(segmentation)

    return annotation


def read_img(fn, ret_annot=True):
    img = Image.open(fn).convert('L')
    w, h = img.size
    img = np.array(img, dtype=np.uint8)
    if ret_annot:
        annot = binary_to_coco(img)
        return w, h, annot
    return w, h


def check_copy(path_src, path_tgt, file=True):
    try:
        if file:
            shutil.copy(path_src, path_tgt)
        else:
            shutil.copytree(path_src, path_tgt)
    except FileExistsError:
        pass


def main():
    train_path_old = os.path.join('dataset', 'train')
    test_path_old = os.path.join('dataset', 'test')

    train_path_new = os.path.join('train')
    test_path_new = os.path.join('test')

    os.makedirs('nuclei', exist_ok=True)
    os.makedirs(train_path_new, exist_ok=True)

    check_copy(test_path_old, test_path_new, file=False)

    dic = {}
    dic['categories'] = [{'id': i, 'name': str(i)} for i in range(1, 2)]
    dic['images'] = []
    dic['annotations'] = []
    img_id = 0
    annot_id = 0

    for it in os.scandir(train_path_old):
        if it.is_dir():
            img_path = os.listdir(os.path.join(it, 'images'))[0]

            old_fp = os.path.join(it, 'images', img_path)
            new_fp = os.path.join(train_path_new, img_path)
            check_copy(old_fp, new_fp)

            w, h = read_img(old_fp, ret_annot=False)
            temp_img = {'file_name': img_path, 'height': h, 'width': w,
                        'id': img_id}
            dic['images'].append(temp_img)

            mask_paths = os.listdir(os.path.join(it, 'masks'))
            mask_paths = [m for m in mask_paths if m.lower().endswith('png')]

            for mp in mask_paths:
                w, h, temp_annot = read_img(os.path.join(it, 'masks', mp))
                temp_annot['image_id'] = img_id
                temp_annot['id'] = annot_id
                dic['annotations'].append(temp_annot)
                annot_id += 1

            print(img_id, annot_id, len(mask_paths), temp_img)
            img_id += 1

    with open('train_val.json', 'w') as outfile:
        outfile.write(json.dumps(dic))


if __name__ == '__main__':
    main()
