import os


import numpy as np
import pandas as pd
from PIL import Image
import torch
from torchvision import transforms

from fgvr.dataset.build_dataloaders import build_dataloaders
from fgvr.utils.parser import parse_option_inference
from fgvr.utils.model_utils import load_model_inference
from fgvr.utils.misc_utils import set_seed


def prepare_img(img_path, img_size):
    transform = transforms.Compose([
        transforms.Resize(img_size+32),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5],
                             std=[0.5, 0.5, 0.5])
    ])
    
    img = Image.open(img_path).convert('RGB')
    img = transform(img).unsqueeze(0)
    return img


def main():
    args = parse_option_inference()
    set_seed(args.seed, args.rank)

    # dataloader
    _, _, n_cls = build_dataloaders(args)

    # model
    if not args.path_classifier:
        model = load_model_inference(
            args.path_backbone, args.model, n_cls, args.image_size, 
            args.pretrained, 'last_only')
    else:
        model = load_model_inference(
            args.path_backbone, args.model, n_cls, args.image_size, 
            args.pretrained, 'default')
    # load the classifier head if needed and add an if in the model output
    
    model.to(args.device)
    model.eval()
    
    # all the testing images
    with open(os.path.join(args.dataset_path, 'testing_img_order.txt')) as f:
        test_images = f.readlines()
        test_images = [line.rstrip() for line in test_images]
    
    id2name_path = os.path.join(args.dataset_path, 'id2name_dic.csv')
    df_id2name = pd.read_csv(id2name_path)
    
    submission = []
    for i, img_path in enumerate(test_images):
        img_path_full = os.path.join(args.dataset_path, 'test', img_path)
        img = prepare_img(img_path_full, args.image_size).to(args.device)
        
        with torch.no_grad():
            outputs = model(img).squeeze(0)
        
        for idx in torch.topk(outputs, k=1).indices.tolist():
            predicted_class_id = idx
            
        predicted_class_name = df_id2name[
            df_id2name['class_id']==predicted_class_id]['class_name'].values[0]
        
        submission.append([img_path, predicted_class_name])
        if i % 100 == 0:
            print(img_path, predicted_class_name, predicted_class_id)
        
    np.savetxt('answer.txt', submission, fmt='%s')


if __name__ == '__main__':
    main()