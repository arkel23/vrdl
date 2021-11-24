import os
from PIL import Image
import h5py
import json


def get_img_name(idx, struct):
    names = struct['digitStruct']['name']
    # transform object to string
    img_name = ''.join(map(chr, struct[names[idx][0]][:].flatten()))
    return img_name


def make_annotations(id, img_id, cat_id, bbox):
    # https://github.com/facebookresearch/Detectron/issues/48
    area = int(bbox[2] * bbox[3])
    temp_dic = {'segmentation': [[]], 'area': area, 'iscrowd': 0,
                'id': id, 'image_id': img_id, 'category_id': cat_id,
                'bbox': bbox}
    return temp_dic


def get_img_boxes(idx, struct):
    # left, top, width, height is COCO bounding box standard format
    # read idx-th bbox
    bboxs = struct['digitStruct']['bbox']
    box = struct[bboxs[idx][0]]

    # shape: (n, 4), n is number of digits in idx-th image
    n = box['label'].shape[0]
    bbox_list = []

    if n == 1:
        bbox = []
        for attr, key in enumerate(['left', 'top', 'width', 'height']):
            bbox.append(int(box[key][0][0]))

        category_id = int(box['label'][0][0])
        if category_id == 10:
            category_id = 0
        bbox_list.append({'category_id': category_id, 'bbox': bbox})

    else:
        for i in range(n):
            bbox = []
            for attr, key in enumerate(['left', 'top', 'width', 'height']):
                bbox.append(int(struct[box[key][i][0]][()].item()))

            category_id = int(struct[box['label'][i][0]][()].item())
            if category_id == 10:
                category_id = 0
            bbox_list.append({'category_id': category_id, 'bbox': bbox})

    return bbox_list


def main():
    struct = h5py.File("train/digitStruct.mat", 'r')

    dic = {}
    dic['categories'] = [{'id': i, 'name': str(i)} for i in range(10)]
    dic['images'] = []
    dic['annotations'] = []

    j = 0
    for i in range(struct['/digitStruct/bbox'].shape[0]):

        img_name = get_img_name(i, struct)
        img_id = int(os.path.splitext(os.path.basename(img_name))[0])
        img = Image.open(os.path.join('train', img_name))
        w, h = img.size
        temp_img = {'file_name': img_name, 'height': h, 'width': w,
                    'id': img_id}
        dic['images'].append(temp_img)

        annotations = get_img_boxes(i, struct)
        for annot in annotations:
            temp_annot = make_annotations(j, img_id, annot['category_id'],
                                          annot['bbox'])
            dic['annotations'].append(temp_annot)
            j += 1

        if i % 1000 == 0:
            print(i, j, temp_img, temp_annot)

    with open('train_val.json', 'w') as outfile:
        outfile.write(json.dumps(dic))


if __name__ == '__main__':
    main()
