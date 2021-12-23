import os

from PIL import Image
import torch
from torchvision import transforms

from fgvr.dataset.build_dataloaders import build_dataloaders
from fgvr.utils.parser import parse_option_inference
from fgvr.utils.model_utils import load_model_inference
from fgvr.utils.misc_utils import set_seed

MEAN = [93.31786143, 107.12554859, 105.98322504]
STD = [19.44589962, 11.72930063, 15.76595482]
DEFAULT = [0.5, 0.5, 0.5]


def prepare_img(img_path, args):
    if args.custom_mean_std:
        mean = MEAN
        std = STD
    else:
        mean = DEFAULT
        std = DEFAULT

    transform = transforms.Compose([
        transforms.Resize(
            args.img_size+32,
            interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(args.img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean,
                             std=std)
    ])

    img = Image.open(img_path).convert('RGB')
    img = transform(img).unsqueeze(0)
    return img


def main():
    args = parse_option_inference()
    set_seed(args.seed)

    # dataloader
    _, _ = build_dataloaders(args)

    # model
    model = load_model_inference(args)
    model.eval()

    # all the testing images
    with open(os.path.join(args.dataset_path, 'test.csv')) as f:
        test_images = f.readlines()
        test_images = [line.rstrip() for line in test_images]

    f = open('submission.csv', 'w')
    f.write('image,ALB,BET,DOL,LAG,NoF,OTHER,SHARK,YFT\n')

    for i, img_path in enumerate(test_images):
        fn = os.path.basename(os.path.normpath(img_path))
        img_path_full = os.path.join(args.dataset_path, 'test', fn)
        img = prepare_img(img_path_full, args).to(args.device)

        with torch.no_grad():
            outputs = model(img).squeeze(0)
            # https://discuss.pytorch.org/t/cnn-results-negative-when-using-log-softmax-and-nll-loss/16839
            outputs = torch.exp(outputs).cpu().numpy()

        f.write(img_path)
        for v in outputs:
            f.write(',{}'.format(v))
        f.write('\n')

        if i % args.print_freq == 0:
            print('{}/{}: {} | {} | {}'.format(i, len(test_images),
                  fn, img_path, outputs))

    f.close()


if __name__ == '__main__':
    main()
