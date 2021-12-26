import ast
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_log_json', type=str, help='log json file')
    args = parser.parse_args()

    iter_psnr = {}
    with open(args.path_log_json) as f:
        lines = f.readlines()
        for line in lines:
            if "val" in line:
                temp = ast.literal_eval(line)
                iter_psnr['{}'.format(temp['iter'])] = temp['PSNR']

    max_key = max(iter_psnr, key=iter_psnr.get)
    print(f'Best iter: {max_key}\t PSNR: {iter_psnr[max_key]}')
    last_key = list(iter_psnr)[-1]
    print(f'Last iter: {last_key}\t PSNR: {iter_psnr[last_key]}')


main()
